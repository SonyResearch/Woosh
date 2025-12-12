import typing as tp

import torch
from typing import Any
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from sfxfm.module.model.diffusion_blocks import FourierFeaturesTime
from sfxfm.module.model.transformer import (
    ConformerModule,
    FeedForward,
    LayerNorm,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    checkpoint,
    or_reduce,
)

from packaging import version
from torch import nn, einsum
from torch.cuda.amp import autocast
import logging

# get logger
log = logging.getLogger(__name__)
# try:
#     from flash_attn import flash_attn_func, flash_attn_kvpacked_func

#     #  q must have shape (batch_size, seqlen_q, num_heads, head_size_og)
#     # q = torch.randn(2, 64, 8, 64).to(torch.float16).to("cuda")
#     # make flash attention fail:
#     q = torch.randn(2, 64, 64).to(torch.float16).to("cuda")
#     flash_attn_func(q, q, q, causal=False)
# except ImportError as e:
#     log.warning(e)
#     log.warning("flash_attn not installed, disabling Flash Attention")
#     flash_attn_kvpacked_func = None
#     flash_attn_func = None
# except RuntimeError as e:
#     log.warning(e)
#     log.warning("flash_attn not working at runtime, disabling Flash Attention")
#     flash_attn_kvpacked_func = None
#     flash_attn_func = None
flash_attn_kvpacked_func = None
flash_attn_func = None
# Natten is disabled!
natten = None


class DiffiT(nn.Module):
    """
    Adapted from dit.py file
    Adds time-dependant self-attention as in DiffiT paper
    +
    Vectorized timestep embedding

    Simplify also when possible
    """

    def __init__(
        self,
        condition_processor: Any,
        io_channels=32,
        patch_size=1,
        embed_dim=768,
        project_cond_tokens=True,
        depth=12,
        num_heads=8,
        **kwargs,
    ):
        super().__init__()
        # extract info from condition processor
        self.condition_processor = condition_processor

        cond_token_dim = condition_processor.cross_attn_cond_dim
        global_cond_dim = condition_processor.global_embed_dim
        input_concat_dim = condition_processor.input_concat_dim
        prepend_cond_dim = condition_processor.prepend_cond_dim

        # Timestep embeddings
        timestep_features_dim = 256

        self.timestep_features = FourierFeaturesTime(1, timestep_features_dim)

        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
            # TODO was added!
            nn.SiLU(),
        )

        if cond_token_dim > 0:
            # Conditioning tokens

            cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim
            self.to_cond_embed = nn.Sequential(
                nn.Linear(cond_token_dim, cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim, bias=False),
            )
        else:
            cond_embed_dim = 0

        if global_cond_dim > 0:
            # Global conditioning
            self.to_global_embed = nn.Sequential(
                nn.Linear(global_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False),
            )

        if prepend_cond_dim > 0:
            # Prepend conditioning
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False),
            )

        self.input_concat_dim = input_concat_dim

        dim_in = io_channels + self.input_concat_dim

        self.patch_size = patch_size

        # Transformer
        # global cond type is always adaln
        # The global conditioning is projected to the embed_dim already at this point
        global_dim = embed_dim

        self.transformer = DiffiTContinuousTransformer(
            dim=embed_dim,
            depth=depth,
            dim_heads=embed_dim // num_heads,
            dim_in=dim_in * patch_size,
            dim_out=io_channels * patch_size,
            cross_attend=cond_token_dim > 0,
            cond_token_dim=cond_embed_dim,
            global_cond_dim=global_dim,
            **kwargs,
        )

        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

    def _forward(
        self,
        x,
        t,
        mask=None,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        input_concat_cond=None,
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        **kwargs,
    ):
        if cross_attn_cond is not None:
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)

        if global_embed is not None:
            # Project the global conditioning to the embedding dimension
            global_embed = self.to_global_embed(global_embed)

        prepend_inputs = None
        prepend_mask = None
        prepend_length = 0
        if prepend_cond is not None:
            # Project the prepend conditioning to the embedding dimension
            prepend_cond = self.to_prepend_embed(prepend_cond)

            prepend_inputs = prepend_cond
            if prepend_cond_mask is not None:
                prepend_mask = prepend_cond_mask

        if input_concat_cond is not None:
            # Interpolate input_concat_cond to the same length as x
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(
                    input_concat_cond, (x.shape[2],), mode="nearest"
                )

            x = torch.cat([x, input_concat_cond], dim=1)

        # Get the batch of timestep embeddings
        timestep_embed = self.to_timestep_embed(
            self.timestep_features(t[:, None])
        )  # (b, embed_dim)

        # WARNING change here:
        #
        # Timestep embedding is NOT considered a global embedding. Add to the global conditioning if it exists
        # will break in the future
        # was:
        # if global_embed is not None:
        #     global_embed = global_embed + timestep_embed
        # else:
        #     global_embed = timestep_embed

        x = self.preprocess_conv(x) + x

        x = rearrange(x, "b c t -> b t c")

        extra_args = {}

        extra_args["global_cond"] = global_embed

        if len(timestep_embed.size()) == 2:
            # expand missing time dim
            timestep_embed = timestep_embed.unsqueeze(1).expand(
                timestep_embed.size(0), x.size(1), timestep_embed.size(1)
            )

        if self.patch_size > 1:
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)
            timestep_embed = rearrange(
                timestep_embed, "b (t p) c -> b t (c p)", p=self.patch_size
            )

        output = self.transformer(
            x,
            timestep_embed,
            prepend_embeds=prepend_inputs,
            context=cross_attn_cond,
            context_mask=cross_attn_cond_mask,
            mask=mask,
            prepend_mask=prepend_mask,
            **extra_args,
            **kwargs,
        )

        output = rearrange(output, "b t c -> b c t")[:, :, prepend_length:]

        if self.patch_size > 1:
            output = rearrange(output, "b (c p) t -> b c (t p)", p=self.patch_size)

        output = self.postprocess_conv(output) + output

        return output

    def forward(
        self,
        x,
        t,
        cond: tp.Dict[str, torch.Tensor],
        # cross_attn_cond=None,
        # cross_attn_cond_mask=None,
        # global_embed=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        input_concat_cond=None,
        negative_global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        cfg_scale=1.0,  # TODO remove all this
        cfg_dropout_prob=0.0,
        causal=False,
        scale_phi=0.0,
        mask=None,
        **kwargs,
    ):
        # extract conditions
        cross_attn_cond = cond.get("cross_attn_cond")
        cross_attn_cond_mask = cond.get("cross_attn_cond_mask")
        global_embed = cond.get("global_embed")

        assert not causal, "Causal mode is not supported for DiffusionTransformer"

        if cross_attn_cond_mask is not None:
            cross_attn_cond_mask = cross_attn_cond_mask.bool()

            # WARNING: I removed the following line to enable cross_attn_cond
            # cross_attn_cond_mask = None  # Temporarily disabling conditioning masks due to kernel issue for flash attention

        if prepend_cond_mask is not None:
            prepend_cond_mask = prepend_cond_mask.bool()

        # CFG dropout
        if cfg_dropout_prob > 0.0:
            if cross_attn_cond is not None:
                null_embed = torch.zeros_like(
                    cross_attn_cond, device=cross_attn_cond.device
                )
                dropout_mask = torch.bernoulli(
                    torch.full(
                        (cross_attn_cond.shape[0], 1, 1),
                        cfg_dropout_prob,
                        device=cross_attn_cond.device,
                    )
                ).to(torch.bool)
                cross_attn_cond = torch.where(dropout_mask, null_embed, cross_attn_cond)

            if prepend_cond is not None:
                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)
                dropout_mask = torch.bernoulli(
                    torch.full(
                        (prepend_cond.shape[0], 1, 1),
                        cfg_dropout_prob,
                        device=prepend_cond.device,
                    )
                ).to(torch.bool)
                prepend_cond = torch.where(dropout_mask, null_embed, prepend_cond)

        if cfg_scale != 1.0 and (
            cross_attn_cond is not None or prepend_cond is not None
        ):
            # Classifier-free guidance
            # Concatenate conditioned and unconditioned inputs on the batch dimension
            batch_inputs = torch.cat([x, x], dim=0)
            batch_timestep = torch.cat([t, t], dim=0)

            if global_embed is not None:
                batch_global_cond = torch.cat([global_embed, global_embed], dim=0)
            else:
                batch_global_cond = None

            if input_concat_cond is not None:
                batch_input_concat_cond = torch.cat(
                    [input_concat_cond, input_concat_cond], dim=0
                )
            else:
                batch_input_concat_cond = None

            batch_cond = None
            batch_cond_masks = None

            # Handle CFG for cross-attention conditioning
            if cross_attn_cond is not None:
                null_embed = torch.zeros_like(
                    cross_attn_cond, device=cross_attn_cond.device
                )

                # For negative cross-attention conditioning, replace the null embed with the negative cross-attention conditioning
                if negative_cross_attn_cond is not None:
                    # If there's a negative cross-attention mask, set the masked tokens to the null embed
                    if negative_cross_attn_mask is not None:
                        negative_cross_attn_mask = negative_cross_attn_mask.to(
                            torch.bool
                        ).unsqueeze(2)

                        negative_cross_attn_cond = torch.where(
                            negative_cross_attn_mask,
                            negative_cross_attn_cond,
                            null_embed,
                        )

                    batch_cond = torch.cat(
                        [cross_attn_cond, negative_cross_attn_cond], dim=0
                    )

                else:
                    batch_cond = torch.cat([cross_attn_cond, null_embed], dim=0)

                if cross_attn_cond_mask is not None:
                    batch_cond_masks = torch.cat(
                        [cross_attn_cond_mask, cross_attn_cond_mask], dim=0
                    )

            batch_prepend_cond = None
            batch_prepend_cond_mask = None

            if prepend_cond is not None:
                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)

                batch_prepend_cond = torch.cat([prepend_cond, null_embed], dim=0)

                if prepend_cond_mask is not None:
                    batch_prepend_cond_mask = torch.cat(
                        [prepend_cond_mask, prepend_cond_mask], dim=0
                    )

            if mask is not None:
                batch_masks = torch.cat([mask, mask], dim=0)
            else:
                batch_masks = None

            batch_output = self._forward(
                batch_inputs,
                batch_timestep,
                cross_attn_cond=batch_cond,
                cross_attn_cond_mask=batch_cond_masks,
                mask=batch_masks,
                input_concat_cond=batch_input_concat_cond,
                global_embed=batch_global_cond,
                prepend_cond=batch_prepend_cond,
                prepend_cond_mask=batch_prepend_cond_mask,
                **kwargs,
            )

            cond_output, uncond_output = torch.chunk(batch_output, 2, dim=0)
            cfg_output = uncond_output + (cond_output - uncond_output) * cfg_scale

            if scale_phi != 0.0:
                cond_out_std = cond_output.std(dim=1, keepdim=True)
                out_cfg_std = cfg_output.std(dim=1, keepdim=True)

                return (
                    scale_phi * (cfg_output * (cond_out_std / out_cfg_std))
                    + (1 - scale_phi) * cfg_output
                )

            else:
                return cfg_output

        else:
            return self._forward(
                x,
                t,
                cross_attn_cond=cross_attn_cond,
                cross_attn_cond_mask=cross_attn_cond_mask,
                input_concat_cond=input_concat_cond,
                global_embed=global_embed,
                prepend_cond=prepend_cond,
                prepend_cond_mask=prepend_cond_mask,
                mask=mask,
                **kwargs,
            )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_heads=64,
        cross_attend=False,
        dim_context=None,
        global_cond_dim=None,
        causal=False,
        zero_init_branch_outputs=True,
        conformer=False,
        layer_ix=-1,
        remove_norms=False,
        attn_kwargs={},
        ff_kwargs={},
        norm_kwargs={},
    ):
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        self.cross_attend = cross_attend
        self.dim_context = dim_context
        self.causal = causal

        self.pre_norm = (
            LayerNorm(dim, **norm_kwargs) if not remove_norms else nn.Identity()
        )

        self.self_attn = TimeDependentAttention(
            dim,
            dim_heads=dim_heads,
            causal=causal,
            zero_init_output=zero_init_branch_outputs,
            **attn_kwargs,
        )

        if cross_attend:
            self.cross_attend_norm = (
                LayerNorm(dim, **norm_kwargs) if not remove_norms else nn.Identity()
            )
            self.cross_attn = TimeDependentAttention(
                dim,
                dim_heads=dim_heads,
                dim_context=dim_context,
                causal=causal,
                zero_init_output=zero_init_branch_outputs,
                **attn_kwargs,
            )

        self.ff_norm = (
            LayerNorm(dim, **norm_kwargs) if not remove_norms else nn.Identity()
        )
        self.ff = FeedForward(
            dim, zero_init_output=zero_init_branch_outputs, **ff_kwargs
        )

        self.layer_ix = layer_ix

        self.conformer = (
            ConformerModule(dim, norm_kwargs=norm_kwargs) if conformer else None
        )

        self.global_cond_dim = global_cond_dim

        if global_cond_dim is not None:
            self.to_scale_shift_gate = nn.Sequential(
                nn.SiLU(), nn.Linear(global_cond_dim, dim * 6, bias=False)
            )

            nn.init.zeros_(self.to_scale_shift_gate[1].weight)
            # nn.init.zeros_(self.to_scale_shift_gate_self[1].bias)

    def forward(
        self,
        x,
        timestep_embed,
        context=None,
        global_cond=None,
        mask=None,
        context_mask=None,
        rotary_pos_emb=None,
    ):
        if (
            self.global_cond_dim is not None
            and self.global_cond_dim > 0
            and global_cond is not None
        ):
            scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = (
                self.to_scale_shift_gate(global_cond).unsqueeze(1).chunk(6, dim=-1)
            )

            # self-attention with adaLN
            residual = x
            x = self.pre_norm(x)
            x = x * (1 + scale_self) + shift_self
            x = self.self_attn(
                x, timestep_embed, mask=mask, rotary_pos_emb=rotary_pos_emb
            )
            x = x * torch.sigmoid(1 - gate_self)
            x = x + residual

            if context is not None:
                x = x + self.cross_attn(
                    self.cross_attend_norm(x),
                    timestep_embed,
                    context=context,
                    context_mask=context_mask,
                )

            if self.conformer is not None:
                x = x + self.conformer(x)

            # feedforward with adaLN
            residual = x
            x = self.ff_norm(x)
            x = x * (1 + scale_ff) + shift_ff
            x = self.ff(x)
            x = x * torch.sigmoid(1 - gate_ff)
            x = x + residual

        else:
            x = x + self.self_attn(
                self.pre_norm(x),
                timestep_embed,
                mask=mask,
                rotary_pos_emb=rotary_pos_emb,
            )

            if context is not None:
                x = x + self.cross_attn(
                    self.cross_attend_norm(x),
                    timestep_embed,
                    context=context,
                    context_mask=context_mask,
                )

            if self.conformer is not None:
                x = x + self.conformer(x)

            x = x + self.ff(self.ff_norm(x))

        return x


class DiffiTContinuousTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        *,
        dim_in=None,
        dim_out=None,
        dim_heads=64,
        cross_attend=False,
        cond_token_dim=None,
        global_cond_dim=None,
        causal=False,
        rotary_pos_emb=True,
        zero_init_branch_outputs=True,
        conformer=False,
        use_sinusoidal_emb=False,
        use_abs_pos_emb=False,
        abs_pos_emb_max_length=10000,
        **kwargs,
    ):
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.causal = causal
        self.layers = nn.ModuleList([])

        self.project_in = (
            nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        )
        self.project_out = (
            nn.Linear(dim, dim_out, bias=False)
            if dim_out is not None
            else nn.Identity()
        )

        if rotary_pos_emb:
            self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))
        else:
            self.rotary_pos_emb = None

        self.use_sinusoidal_emb = use_sinusoidal_emb
        if use_sinusoidal_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)

        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            self.pos_emb = AbsolutePositionalEmbedding(dim, abs_pos_emb_max_length)

        for i in range(depth):
            self.layers.append(
                TransformerBlock(
                    dim,
                    dim_heads=dim_heads,
                    cross_attend=cross_attend,
                    dim_context=cond_token_dim,
                    global_cond_dim=global_cond_dim,
                    causal=causal,
                    zero_init_branch_outputs=zero_init_branch_outputs,
                    conformer=conformer,
                    layer_ix=i,
                    **kwargs,
                )
            )

    def forward(
        self,
        x,
        timestep_embed,
        mask=None,
        prepend_embeds=None,
        prepend_mask=None,
        global_cond=None,
        **kwargs,
    ):
        batch, seq, device = *x.shape[:2], x.device

        x = self.project_in(x)

        if prepend_embeds is not None:
            prepend_length, prepend_dim = prepend_embeds.shape[1:]

            assert prepend_dim == x.shape[-1], (
                "prepend dimension must match sequence dimension"
            )

            x = torch.cat((prepend_embeds, x), dim=-2)

            if prepend_mask is not None or mask is not None:
                mask = (
                    mask
                    if mask is not None
                    else torch.ones((batch, seq), device=device, dtype=torch.bool)
                )
                prepend_mask = (
                    prepend_mask
                    if prepend_mask is not None
                    else torch.ones(
                        (batch, prepend_length), device=device, dtype=torch.bool
                    )
                )

                mask = torch.cat((prepend_mask, mask), dim=-1)

        # Attention layers

        if self.rotary_pos_emb is not None:
            rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1])
        else:
            rotary_pos_emb = None

        if self.use_sinusoidal_emb:
            x = x + self.pos_emb(x)

        # Iterate over the transformer layers
        for layer in self.layers:
            # x = layer(x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, **kwargs)
            x = checkpoint(
                layer,
                x,
                timestep_embed,
                rotary_pos_emb=rotary_pos_emb,
                global_cond=global_cond,
                **kwargs,
            )

        x = self.project_out(x)

        return x


class TimeDependentAttention(nn.Module):
    def __init__(
        self,
        dim,
        time_embed_dim=None,
        dim_heads=64,
        dim_context=None,
        causal=False,
        zero_init_output=True,
        qk_norm=False,
        natten_kernel_size=None,
    ):
        """
        if dim_context = None all q,k,v are time dependent
        otherwise (like cross attention), only q
        """
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        self.causal = causal

        dim_kv = dim_context if dim_context is not None else dim
        time_embed_dim = time_embed_dim if time_embed_dim is not None else dim

        self.num_heads = dim // dim_heads
        self.kv_heads = dim_kv // dim_heads

        if dim_context is not None:
            self.to_q = nn.Linear(dim + time_embed_dim, dim, bias=False)
            self.to_kv = nn.Linear(dim_kv, dim_kv * 2, bias=False)
        else:
            self.to_qkv = nn.Linear(dim + time_embed_dim, dim * 3, bias=False)

        self.to_out = nn.Linear(dim, dim, bias=False)

        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)

        self.qk_norm = qk_norm

        # Using 1d neighborhood attention
        self.natten_kernel_size = natten_kernel_size
        if natten_kernel_size is not None:
            return

        self.use_pt_flash = torch.cuda.is_available() and version.parse(
            torch.__version__
        ) >= version.parse("2.0.0")

        self.use_fa_flash = torch.cuda.is_available() and flash_attn_func is not None

        self.sdp_kwargs = dict(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        )

    def flash_attn(self, q, k, v, mask=None, causal=None):
        batch, heads, q_len, _, k_len, device = *q.shape, k.shape[-2], q.device
        kv_heads = k.shape[1]
        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if heads != kv_heads:
            # Repeat interleave kv_heads to match q_heads
            heads_per_kv_head = heads // kv_heads
            k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim=1), (k, v))

        if k.ndim == 3:
            k = rearrange(k, "b ... -> b 1 ...").expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, "b ... -> b 1 ...").expand_as(q)

        causal = self.causal if causal is None else causal

        if q_len == 1 and causal:
            causal = False

        if mask is not None:
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

        # handle kv cache - this should be bypassable in updated flash attention 2

        if k_len > q_len and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device=device)
            if mask is None:
                mask = ~causal_mask
            else:
                mask = mask & ~causal_mask
            causal = False

        # manually handle causal mask, if another mask was given

        row_is_entirely_masked = None

        if mask is not None and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device=device)
            mask = mask & ~causal_mask

            # protect against an entire row being masked out

            row_is_entirely_masked = ~mask.any(dim=-1)
            mask[..., 0] = mask[..., 0] | row_is_entirely_masked

            causal = False

        with torch.backends.cuda.sdp_kernel(**self.sdp_kwargs):
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, is_causal=causal
            )

        # for a row that is entirely masked out, should zero out the output of that row token

        if row_is_entirely_masked is not None:
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.0)

        return out

    def forward(
        self,
        x,
        timestep_embed,
        context=None,
        mask=None,
        context_mask=None,
        rotary_pos_emb=None,
        causal=None,
    ):
        h, kv_h, has_context = self.num_heads, self.kv_heads, context is not None

        # concat x and time_embed
        x = torch.cat([x, timestep_embed], dim=2)

        kv_input = context if has_context else x

        if hasattr(self, "to_q"):
            # Use separate linear projections for q and k/v
            q = self.to_q(x)
            q = rearrange(q, "b n (h d) -> b h n d", h=h)

            k, v = self.to_kv(kv_input).chunk(2, dim=-1)

            k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=kv_h), (k, v))
        else:
            # Use fused linear projection
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(
                lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v)
            )

        # Normalize q and k for cosine sim attention
        if self.qk_norm:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)

        if rotary_pos_emb is not None and not has_context:
            freqs, _ = rotary_pos_emb

            q_dtype = q.dtype
            k_dtype = k.dtype

            q = q.to(torch.float32)
            k = k.to(torch.float32)
            freqs = freqs.to(torch.float32)

            q = apply_rotary_pos_emb(q, freqs)
            k = apply_rotary_pos_emb(k, freqs)

            q = q.to(q_dtype)
            k = k.to(k_dtype)

        input_mask = context_mask

        if input_mask is None and not has_context:
            input_mask = mask

        # determine masking
        masks = []
        final_attn_mask = None  # The mask that will be applied to the attention matrix, taking all masks into account

        if input_mask is not None:
            input_mask = rearrange(input_mask, "b j -> b 1 1 j")
            masks.append(~input_mask)

        # Other masks will be added here later

        if len(masks) > 0:
            final_attn_mask = ~or_reduce(masks)

        n, device = q.shape[-2], q.device

        causal = self.causal if causal is None else causal

        if n == 1 and causal:
            causal = False

        if self.natten_kernel_size is not None:
            if natten is None:
                raise ImportError(
                    "natten not installed, please install natten to use neighborhood attention"
                )

            dtype_in = q.dtype
            q, k, v = map(lambda t: t.to(torch.float32), (q, k, v))

            attn = natten.functional.natten1dqk(
                q, k, kernel_size=self.natten_kernel_size, dilation=1
            )

            if final_attn_mask is not None:
                attn = attn.masked_fill(final_attn_mask, -torch.finfo(attn.dtype).max)

            attn = F.softmax(attn, dim=-1, dtype=torch.float32)

            out = natten.functional.natten1dav(
                attn, v, kernel_size=self.natten_kernel_size, dilation=1
            ).to(dtype_in)

        # Prioritize Flash Attention 2
        elif self.use_fa_flash:
            assert final_attn_mask is None, (
                "masking not yet supported for Flash Attention 2"
            )
            # Flash Attention 2 requires FP16 inputs
            fa_dtype_in = q.dtype
            q, k, v = map(
                lambda t: rearrange(t, "b h n d -> b n h d").to(torch.float16),
                (q, k, v),
            )

            out = flash_attn_func(q, k, v, causal=causal)

            out = rearrange(out.to(fa_dtype_in), "b n h d -> b h n d")

        # Fall back to PyTorch implementation
        elif self.use_pt_flash:
            out = self.flash_attn(q, k, v, causal=causal, mask=final_attn_mask)

        else:
            # Fall back to custom implementation

            if h != kv_h:
                # Repeat interleave kv_heads to match q_heads
                heads_per_kv_head = h // kv_h
                k, v = map(
                    lambda t: t.repeat_interleave(heads_per_kv_head, dim=1), (k, v)
                )

            scale = 1.0 / (q.shape[-1] ** 0.5)

            kv_einsum_eq = "b j d" if k.ndim == 3 else "b h j d"

            dots = einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

            i, j, dtype = *dots.shape[-2:], dots.dtype

            mask_value = -torch.finfo(dots.dtype).max

            if final_attn_mask is not None:
                dots = dots.masked_fill(~final_attn_mask, mask_value)

            if causal:
                causal_mask = self.create_causal_mask(i, j, device=device)
                dots = dots.masked_fill(causal_mask, mask_value)

            attn = F.softmax(dots, dim=-1, dtype=torch.float32)
            attn = attn.type(dtype)

            out = einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        # merge heads
        out = rearrange(out, " b h n d -> b n (h d)")

        # Communicate between heads

        with autocast(enabled=False):
            out_dtype = out.dtype
            out = out.to(torch.float32)
            out = self.to_out(out).to(out_dtype)

        if mask is not None:
            mask = rearrange(mask, "b n -> b n 1")
            out = out.masked_fill(~mask, 0.0)

        return out
