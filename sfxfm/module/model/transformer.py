from functools import reduce, partial
import os
import sys
from packaging import version

from einops import rearrange
from einops.layers.torch import Rearrange
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.cuda.amp import autocast
from typing import Callable, Literal
from sfxfm.utils.dist import rank

import logging

rank = rank()

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
# try:
#     import natten
# except ImportError:
#     natten = None

natten = None

DEBUG_NAN = os.environ.get("DEBUG_NAN", False)

if DEBUG_NAN:
    log.warning("DEBUG_NAN is enabled. Might cause slower training.")

# access to scaler params
plmodule = None


def set_pl_module(pl_module):
    global plmodule
    plmodule = pl_module


print_counter = 200


def register_backward_nan_check_hook(x, desc):
    """
    Register a hook to check for NaN in the backward pass.
    Use environment variable `DEBUG_NAN` to enable this.
    """
    if not DEBUG_NAN:
        return None
    global print_counter
    if print_counter > 0:
        # bypass sanity validation
        if x.requires_grad:
            print_counter -= 1
        print(f"rank{rank}, {desc} dtype={x.dtype}", file=sys.stderr, flush=True)
    if not x.isfinite().all():
        print(
            f"rank{rank}, {desc} input is not finite. {x}", file=sys.stderr, flush=True
        )
        raise ValueError(f"rank{rank}, {desc} input is not finite. {x}")
    max_val = x.abs().max().item()
    min_val = x.abs().min().item()
    # print(
    #     f"\nrank{rank}, {desc} max: {max_val}, min: {min_val}, {x.dtype} \n",
    #     file=sys.stderr,
    #     flush=True,
    # )

    def backward_nan_check_hook(grad):
        if grad is not None:
            if torch.isnan(grad).any():
                grad_per_sample = grad.view(grad.shape[0], -1).sum(dim=1)
                print(
                    f"rank{rank}, {desc} grad({grad.dtype}) has NaN. Input max: {max_val}, min: {min_val} \n  \n Grad per sample: {grad_per_sample}",
                    file=sys.stderr,
                    flush=True,
                )

    if x.requires_grad:
        x.register_hook(backward_nan_check_hook)

    x = None  # circular reference causes memory leak

    return backward_nan_check_hook


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


# Copied and modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/attend.py under MIT License
# License can be found in LICENSES/LICENSE_XTRANSFORMERS.txt


def create_causal_mask(i, j, device):
    return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)


def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head


# positional embeddings


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim**-0.5
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos=None, seq_start_pos=None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, (
            f"you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}"
        )

        if pos is None:
            pos = torch.arange(seq_len, device=device)

        if seq_start_pos is not None:
            pos = (pos + seq_start_pos[..., None]).clamp(min=0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return pos_emb


class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        assert (dim % 2) == 0, "dimension must be divisible by 2"
        self.scale = nn.Parameter(torch.ones(1) * dim**-0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta**-freq_seq
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, pos=None, seq_start_pos=None):
        seq_len, device = x.shape[1], x.device

        if pos is None:
            pos = torch.arange(seq_len, device=device)

        if seq_start_pos is not None:
            pos = pos - seq_start_pos[..., None]

        emb = einsum("i, j -> i j", pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb * self.scale


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos=False,
        scale_base=512,
        interpolation_factor=1.0,
        base=10000,
        base_rescale_factor=1.0,
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        assert interpolation_factor >= 1.0
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer("scale", None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer("scale", scale)

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = torch.arange(seq_len, device=device)
        return self.forward(t)

    @autocast(enabled=False)
    def forward(self, t):
        device = self.inv_freq.device

        t = t.to(torch.float32)

        t = t / self.interpolation_factor

        freqs = torch.einsum("i , j -> i j", t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)

        if self.scale is None:
            return freqs, 1.0

        power = (
            torch.arange(seq_len, device=device) - (seq_len // 2)
        ) / self.scale_base
        scale = self.scale ** rearrange(power, "n -> n 1")
        scale = torch.cat((scale, scale), dim=-1)

        return freqs, scale


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


@autocast(enabled=False)
def apply_rotary_pos_emb(t, freqs, scale=1):
    out_dtype = t.dtype

    # cast to float32 if necessary for numerical stability
    dtype = reduce(torch.promote_types, (t.dtype, freqs.dtype, torch.float32))
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs, t = freqs.to(dtype), t.to(dtype)
    freqs = freqs[-seq_len:, :]

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, "b n d -> b 1 n d")

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)

    t, t_unrotated = t.to(out_dtype), t_unrotated.to(out_dtype)

    return torch.cat((t, t_unrotated), dim=-1)


# norms
class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False, fix_scale=False):
        """
        bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
        """
        super().__init__()

        if fix_scale:
            self.register_buffer("gamma", torch.ones(dim))
        else:
            self.gamma = nn.Parameter(torch.ones(dim))

        if bias:
            self.beta = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], weight=self.gamma, bias=self.beta)


class InstanceNorm(nn.Module):
    def __init__(self, dim, bias=False, fix_scale=False):
        """
        bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
        """
        super().__init__()

        if fix_scale:
            self.gamma = None
        else:
            self.gamma = nn.Parameter(torch.ones(dim))

        if bias:
            self.beta = nn.Parameter(torch.zeros(dim))
        else:
            self.beta = None
        self.norm = nn.InstanceNorm1d(
            dim,
        )

    def forward(self, x):
        with torch.autocast(device_type="cuda", enabled=False):
            x = rearrange(x, "b n d -> b d n")
            x = self.norm(x)
            x = rearrange(x, "b d n -> b n d")
            if self.gamma is not None:
                x = x * self.gamma
            if self.beta is not None:
                x = x + self.beta
            return x


# feedforward
class GLU(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        activation: Callable,
        use_conv=False,
        conv_kernel_size=3,
    ):
        super().__init__()
        self.act = activation
        self.proj = (
            nn.Linear(dim_in, dim_out * 2)
            if not use_conv
            else nn.Conv1d(
                dim_in, dim_out * 2, conv_kernel_size, padding=(conv_kernel_size // 2)
            )
        )
        self.use_conv = use_conv

    def forward(self, x):
        if self.use_conv:
            x = rearrange(x, "b n d -> b d n")
            x = self.proj(x)
            x = rearrange(x, "b d n -> b n d")
        else:
            x = self.proj(x)

        x, gate = x.chunk(2, dim=-1)
        return x * self.act(gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        mult=4,
        no_bias=False,
        glu=True,
        use_conv=False,
        conv_kernel_size=3,
        zero_init_output=True,
    ):
        super().__init__()
        inner_dim = int(dim * mult)

        # Default to SwiGLU

        activation = nn.SiLU()

        dim_out = dim if dim_out is None else dim_out

        if glu:
            linear_in = GLU(dim, inner_dim, activation)
        else:
            linear_in = nn.Sequential(
                Rearrange("b n d -> b d n") if use_conv else nn.Identity(),
                (
                    nn.Linear(dim, inner_dim, bias=not no_bias)
                    if not use_conv
                    else nn.Conv1d(
                        dim,
                        inner_dim,
                        conv_kernel_size,
                        padding=(conv_kernel_size // 2),
                        bias=not no_bias,
                    )
                ),
                Rearrange("b n d -> b d n") if use_conv else nn.Identity(),
                activation,
            )

        linear_out = (
            nn.Linear(inner_dim, dim_out, bias=not no_bias)
            if not use_conv
            else nn.Conv1d(
                inner_dim,
                dim_out,
                conv_kernel_size,
                padding=(conv_kernel_size // 2),
                bias=not no_bias,
            )
        )

        # init last linear layer to 0
        if zero_init_output:
            nn.init.zeros_(linear_out.weight)
            if not no_bias:
                nn.init.zeros_(linear_out.bias)

        self.ff = nn.Sequential(
            linear_in,
            Rearrange("b d n -> b n d") if use_conv else nn.Identity(),
            linear_out,
            Rearrange("b n d -> b d n") if use_conv else nn.Identity(),
        )

    def forward(self, x):
        return self.ff(x)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return ((x * rrms) * self.scale).to(dtype=x_dtype)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_heads=64,
        dim_context=None,
        causal=False,
        zero_init_output=True,
        qk_norm: Literal["rms", "l2", "ln", "none"] = "none",
        natten_kernel_size=None,
    ):
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        self.causal = causal

        dim_kv = dim_context if dim_context is not None else dim

        self.num_heads = dim // dim_heads
        self.kv_heads = dim_kv // dim_heads

        if dim_context is not None:
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_kv = nn.Linear(dim_kv, dim_kv * 2, bias=False)
        else:
            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.to_out = nn.Linear(dim, dim, bias=False)

        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)

        self.qk_norm = qk_norm
        if self.qk_norm == "ln":
            self.q_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1.0e-6)
            self.k_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1.0e-6)
        elif self.qk_norm == "rms":
            self.q_norm = RMSNorm(dim_heads)
            self.k_norm = RMSNorm(dim_heads)
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
            register_backward_nan_check_hook(out, "sdpa")

        # for a row that is entirely masked out, should zero out the output of that row token

        if row_is_entirely_masked is not None:
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.0)

        return out

    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        rotary_pos_emb=None,
        causal=None,
    ):
        h, kv_h, has_context = self.num_heads, self.kv_heads, context is not None

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
        if self.qk_norm == "l2":
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        elif self.qk_norm == "ln" or self.qk_norm == "rms":
            q = self.q_norm(q)
            k = self.k_norm(k)

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


class ConformerModule(nn.Module):
    def __init__(
        self,
        dim,
        norm_kwargs={},
    ):
        super().__init__()

        self.dim = dim

        self.in_norm = LayerNorm(dim, **norm_kwargs)
        self.pointwise_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.glu = GLU(dim, dim, nn.SiLU())
        self.depthwise_conv = nn.Conv1d(
            dim, dim, kernel_size=17, groups=dim, padding=8, bias=False
        )
        self.mid_norm = LayerNorm(
            dim, **norm_kwargs
        )  # This is a batch norm in the original but I don't like batch norm
        self.swish = nn.SiLU()
        self.pointwise_conv_2 = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.in_norm(x)
        x = rearrange(x, "b n d -> b d n")
        x = self.pointwise_conv(x)
        x = rearrange(x, "b d n -> b n d")
        x = self.glu(x)
        x = rearrange(x, "b n d -> b d n")
        x = self.depthwise_conv(x)
        x = rearrange(x, "b d n -> b n d")
        x = self.mid_norm(x)
        x = self.swish(x)
        x = rearrange(x, "b n d -> b d n")
        x = self.pointwise_conv_2(x)
        x = rearrange(x, "b d n -> b n d")

        return x


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
        sequence_cond_dim=None,
        unbounded_gate=False,
        instance_norm=False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        self.cross_attend = cross_attend
        self.dim_context = dim_context
        self.causal = causal
        # unbounded gate is a variant of the gated transformer where the gate is not bounded between 0 and 1
        self.unbounded_gate = unbounded_gate
        if unbounded_gate:
            log.info(f"Using unbounded gate for block {layer_ix}.")

        self.instance_norm = instance_norm
        norm_layer = LayerNorm
        if instance_norm:
            norm_layer = InstanceNorm
        self.pre_norm = (
            norm_layer(dim, **norm_kwargs) if not remove_norms else nn.Identity()
        )

        self.self_attn = Attention(
            dim,
            dim_heads=dim_heads,
            causal=causal,
            zero_init_output=zero_init_branch_outputs,
            **attn_kwargs,
        )

        if cross_attend:
            self.cross_attend_norm = (
                norm_layer(dim, **norm_kwargs) if not remove_norms else nn.Identity()
            )
            self.cross_attn = Attention(
                dim,
                dim_heads=dim_heads,
                dim_context=dim_context,
                causal=causal,
                zero_init_output=zero_init_branch_outputs,
                **attn_kwargs,
            )

        self.ff_norm = (
            norm_layer(dim, **norm_kwargs) if not remove_norms else nn.Identity()
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
                nn.SiLU(), nn.Linear(global_cond_dim, dim * 6, bias=self.unbounded_gate)
            )
            if not self.unbounded_gate:
                nn.init.zeros_(self.to_scale_shift_gate[1].weight)
            # nn.init.zeros_(self.to_scale_shift_gate_self[1].bias)

        self.sequence_cond_dim = sequence_cond_dim
        if sequence_cond_dim is not None:
            self.to_sequence_scale_shift_gate = nn.Sequential(
                nn.SiLU(),
                nn.Linear(sequence_cond_dim, dim * 3, bias=self.unbounded_gate),
            )
            if not self.unbounded_gate:
                nn.init.zeros_(self.to_sequence_scale_shift_gate[1].weight)
            self.seqc_norm = (
                norm_layer(dim, **norm_kwargs) if not remove_norms else nn.Identity()
            )
            self.seqc_ff = FeedForward(
                dim,
                zero_init_output=zero_init_branch_outputs,
                mult=1,
            )

    def forward(
        self,
        x,
        context=None,
        global_cond=None,
        mask=None,
        context_mask=None,
        rotary_pos_emb=None,
        sequence_cond=None,
    ):
        if (
            self.global_cond_dim is not None
            and self.global_cond_dim > 0
            and global_cond is not None
        ):
            scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = (
                self.to_scale_shift_gate(global_cond).unsqueeze(1).chunk(6, dim=-1)
            )

            register_backward_nan_check_hook(x, f"x_block{self.layer_ix}")

            register_backward_nan_check_hook(scale_self, f"scale_self_{self.layer_ix}")

            # self-attention with adaLN
            with autocast(enabled=False):
                x = x.float()
                residual = x
                x = self.pre_norm(x)
                register_backward_nan_check_hook(x, f"pre_normx_{self.layer_ix}")
                x = x * (1 + scale_self.float()) + shift_self.float()

            x = self.self_attn(x, mask=mask, rotary_pos_emb=rotary_pos_emb)

            register_backward_nan_check_hook(x, f"self_attn_{self.layer_ix}")

            if self.unbounded_gate:
                with autocast(enabled=False):
                    x = residual + x.float() * gate_self.float()
            else:
                with autocast(enabled=False):
                    x = x.float() * torch.sigmoid(1 - gate_self.float())
                    x = x + residual

            if context is not None:
                x = x + self.cross_attn(
                    self.cross_attend_norm(x),
                    context=context,
                    context_mask=context_mask,
                )

            if self.conformer is not None:
                x = x + self.conformer(x)

            # feedforward with adaLN
            with autocast(enabled=False):
                x = x.float()
                residual = x
                x = self.ff_norm(x)
                x = x * (1 + scale_ff.float()) + shift_ff.float()

            x = self.ff(x)

            if self.unbounded_gate:
                with autocast(enabled=False):
                    x = residual + x.float() * gate_ff.float()
            else:
                with autocast(enabled=False):
                    x = x.float() * torch.sigmoid(1 - gate_ff.float())
                    x = x + residual

        else:
            x = x + self.self_attn(
                self.pre_norm(x), mask=mask, rotary_pos_emb=rotary_pos_emb
            )

            if context is not None:
                x = x + self.cross_attn(
                    self.cross_attend_norm(x),
                    context=context,
                    context_mask=context_mask,
                )

            if self.conformer is not None:
                x = x + self.conformer(x)

            x = x + self.ff(self.ff_norm(x))
        if (
            sequence_cond is not None
            and self.sequence_cond_dim is not None
            and self.sequence_cond_dim > 0
        ):
            scale_seq, shift_seq, gate_seq = self.to_sequence_scale_shift_gate(
                sequence_cond
            ).chunk(3, dim=-1)
            residual = x
            x = self.seqc_norm(x)
            x = x * (1 + scale_seq) + shift_seq
            x = self.seqc_ff(x)
            if self.unbounded_gate:
                x = x * gate_seq
            else:
                x = x * torch.sigmoid(1 - gate_seq)
            x = x + residual
        return x


class ContinuousTransformer(nn.Module):
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
        sequence_cond_dim=None,
        causal=False,
        rotary_pos_emb=True,
        zero_init_branch_outputs=True,
        conformer=False,
        use_sinusoidal_emb=False,
        use_abs_pos_emb=False,
        abs_pos_emb_max_length=10000,
        n_memory_tokens=0,
        n_context_memory_tokens=0,
        non_checkpoint_layers=0,
        max_abs_pos_length=None,
        **kwargs,
    ):
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.causal = causal
        # the layers that are not checkpointed, memory/ speed tradeoff
        if non_checkpoint_layers > 0:
            log.info(
                f"checkpointing speed/memory trade-off, checkpoint_layers: {depth - non_checkpoint_layers} non_checkpoint_layers: {non_checkpoint_layers}"
            )
        self.non_checkpoint_layers = non_checkpoint_layers

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
                    sequence_cond_dim=sequence_cond_dim,
                    causal=causal,
                    zero_init_branch_outputs=zero_init_branch_outputs,
                    conformer=conformer,
                    layer_ix=i,
                    **kwargs,
                )
            )
        self.memory_tokens = None
        self.n_memory_tokens = n_memory_tokens
        if n_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(
                torch.randn(1, n_memory_tokens, dim) * (dim**-0.5), requires_grad=True
            )
            log.info(f"DIT: Using {n_memory_tokens} memory tokens")

        self.n_context_memory_tokens = n_context_memory_tokens
        self.context_memory_tokens = None
        if n_context_memory_tokens > 0:
            self.context_memory_tokens = nn.Parameter(
                torch.randn(1, n_context_memory_tokens, dim) * (dim**-0.5),
                requires_grad=True,
            )
            log.info(f"DIT: Using {n_context_memory_tokens} CONTEXT memory tokens")
        self.max_abs_pos_length = max_abs_pos_length
        if max_abs_pos_length is not None:
            log.info(f"DIT: Using max_abs_pos_length {max_abs_pos_length}")

    def forward(
        self,
        x,
        mask=None,
        prepend_embeds=None,
        prepend_mask=None,
        global_cond=None,
        **kwargs,
    ):
        batch, seq, device = *x.shape[:2], x.device

        register_backward_nan_check_hook(x, "x")

        with torch.autocast(device_type="cuda", enabled=False):
            x = self.project_in(x.float())

        register_backward_nan_check_hook(x, "project_in(x)")

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

        if self.use_abs_pos_emb:
            seq_start_pos = None
            # if training sample a starting pos for abs pos encoding sequence
            if self.training and self.max_abs_pos_length is not None:
                seq_start_pos = torch.randint(
                    0, self.max_abs_pos_length - x.shape[1], (batch,), device=device
                )
            x = x + self.pos_emb(x, pos=None, seq_start_pos=seq_start_pos)

        register_backward_nan_check_hook(x, "abspos(x)")

        # add memory tokens
        if self.memory_tokens is not None:
            x = torch.cat((self.memory_tokens.expand(batch, -1, -1), x), dim=1)
            if mask is not None:
                mask = torch.cat(
                    (
                        torch.ones(
                            (batch, self.n_memory_tokens),
                            device=device,
                            dtype=torch.bool,
                        ),
                        mask,
                    ),
                    dim=-1,
                )
        register_backward_nan_check_hook(x, "memory_tokens(x)")

        # add context  memory tokens
        if self.context_memory_tokens is not None and "context" in kwargs:
            kwargs["context"] = torch.cat(
                (self.context_memory_tokens.expand(batch, -1, -1), kwargs["context"]),
                dim=1,
            )

            # update context_mask
            if "context_mask" in kwargs and kwargs["context_mask"] is not None:
                kwargs["context_mask"] = torch.cat(
                    (
                        torch.ones(
                            (batch, self.n_context_memory_tokens),
                            device=device,
                            dtype=torch.bool,
                        ),
                        kwargs["context_mask"],
                    ),
                    dim=-1,
                )

        non_checkpoint_layers = getattr(self, "non_checkpoint_layers", 0)
        # the number of layers that are not checkpointed, memory/ speed tradeoff
        checkpoint_layers = self.depth - non_checkpoint_layers

        # Iterate over the transformer layers
        for lidx, layer in enumerate(self.layers):
            if lidx < checkpoint_layers:
                x = checkpoint(
                    layer,
                    x,
                    rotary_pos_emb=rotary_pos_emb,
                    global_cond=global_cond,
                    **kwargs,
                )
            else:
                x = layer(
                    x, rotary_pos_emb=rotary_pos_emb, global_cond=global_cond, **kwargs
                )
                register_backward_nan_check_hook(x, f"TransformerBlock_{lidx}(x)")

        # remove memory tokens
        if self.memory_tokens is not None:
            x = x[:, self.n_memory_tokens :, :]

        with torch.autocast(device_type="cuda", enabled=False):
            x = self.project_out(x.float())

        return x
