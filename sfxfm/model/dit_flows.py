from typing import Dict, Optional
from einops import rearrange
import torch
from torch import nn
from sfxfm.model.dit_pipeline import DiTPipeline, DiTFlowMapPipeline
from sfxfm.model.dit_types import DiTArgs, DictTensor, MMDiTArgs

from sfxfm.model.dit_blocks import (
    FixedFourierFeaturesTime,
    FourierFeaturesTime,
    MMMBlock,
    ModalityBlock,
    MultimodalitySingleStreamBlock,
    precompute_freqs_cis,
)


class InputProcessing(nn.Module):
    """
    sends to DictTensor
    """

    def __init__(self, args: DiTArgs):
        super().__init__()
        cond_token_dim = args.cond_token_dim

        input_padding_size = (-args.max_seq_len) % args.patch_size
        self.patch_size = args.patch_size
        if input_padding_size > 0:
            self.input_padding = nn.Parameter(
                torch.randn(args.io_channels, input_padding_size), requires_grad=True
            )

        # === Timestep
        self.timestep_features = (
            FixedFourierFeaturesTime(1, args.timestep_features_dim)
            if args.fixed_timestep_features
            else FourierFeaturesTime(1, args.timestep_features_dim)
        )

        self.to_timestep_embed = nn.Sequential(
            nn.Linear(
                args.timestep_features_dim,
                args.inter_dim,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(args.inter_dim, args.dim, bias=True),
            # last SiLU is included in post_timestep_embed
        )

        self.post_timestep_embed = nn.SiLU()

        # === x
        self.project_in = nn.Linear(
            args.io_channels * args.patch_size, args.dim, bias=True
        )

        # === condition
        self.to_cond_embed = nn.Sequential(
            nn.Linear(cond_token_dim, args.inter_dim, bias=True),
            nn.SiLU(),
            nn.Linear(args.inter_dim, args.dim, bias=True),
        )

        # === memory tokens

        # Only one memory token; like relative positional embeddings:
        self.n_memory_tokens_rope = args.n_memory_tokens_rope
        self.memory_tokens_rope = nn.Parameter(
            torch.randn(1, 1, args.dim), requires_grad=True
        )
        # === precompute rope embeddings
        freqs_cis = precompute_freqs_cis(args)
        # concat downsampled frequencies of memory tokens rope to freqs_cis
        if args.n_memory_tokens_rope > 0:
            downsampling_factor = freqs_cis.size(0) // args.n_memory_tokens_rope
            freqs_cis = torch.cat(
                [freqs_cis[downsampling_factor // 2 :: downsampling_factor], freqs_cis],
                dim=0,
            )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # === Memory tokens for description

        self.n_memory_tokens_description = args.n_memory_tokens_description
        self.memory_tokens_description = nn.Parameter(
            torch.randn(1, args.n_memory_tokens_description, args.dim),
            requires_grad=True,
        )

        if args.n_multimodal_layers > 0:
            # constant freqs_cis for text
            self.register_buffer(
                "freqs_cis_description",
                precompute_freqs_cis(
                    args.model_copy(
                        update={
                            "max_seq_len": args.max_description_length
                            + args.n_memory_tokens_description,
                        }
                    )
                )[:1, :].expand(
                    args.max_description_length + args.n_memory_tokens_description, -1
                ),
                persistent=False,
            )
        else:
            self.register_buffer("freqs_cis_description", None, persistent=False)

        # === Estimation of logvar(t)
        self.estimate_logvar = args.estimate_logvar
        if args.estimate_logvar:
            self.timestep_logvar = FourierFeaturesTime(1, args.timestep_features_dim)
            self.to_logvar = nn.Sequential(
                nn.Linear(args.timestep_features_dim, 128, bias=True),
                nn.SiLU(),
                nn.Linear(128, 1, bias=True),
            )

        self.no_description_mask = args.no_description_mask
        if self.no_description_mask:
            # if no description mask, we use replace the masked tokens with a learnable parameter
            self.description_pad = nn.Parameter(
                torch.randn(args.max_description_length, args.cond_token_dim),
            )

    def pad_description(
        self, description: torch.Tensor, description_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pads the description with a learnable parameter if no_description_mask is True.
        """
        if self.no_description_mask:
            # replace masked tokens with a learnable parameter
            description = torch.where(
                description_mask.unsqueeze(-1).bool(),
                description,
                self.description_pad[None, :],
            )
        return description

    def embed_x(self, x):
        """
        Embeds the input tensor x by rearranging it into patches and projecting it.
        If input_padding is defined, pads the input with learnable parameters.
        """
        if hasattr(self, "input_padding"):
            # pad the input with learnable parameters
            x = torch.cat(
                [
                    self.input_padding[None, :, :].expand(x.size(0), -1, -1),
                    x,
                ],
                dim=2,
            )
        # rearrange x into patches
        x = rearrange(x, "b c (t p) -> b t (p c)", p=self.patch_size)
        return self.project_in(x)

    # Copies signature from forward method of dit.DiffusionTransformer
    def forward(
        self,
        x: torch.Tensor,
        t,
        cond: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> DictTensor:
        batch_size = x.size(0)
        m_plus = self.to_timestep_embed(self.timestep_features(t[:, None]))  # (b, c)
        d = dict(
            # memory tokens + embed(x)
            x=torch.cat(
                [
                    self.memory_tokens_rope.expand(
                        batch_size, self.n_memory_tokens_rope, -1
                    ),
                    self.embed_x(x),
                ],
                dim=1,
            ),
            x_mask=mask,
            m_plus=m_plus,
            t=self.post_timestep_embed(m_plus),
            description=(
                torch.cat(
                    [
                        self.memory_tokens_description.expand(x.size(0), -1, -1),
                        (
                            self.to_cond_embed(cond["cross_attn_cond"])
                            if not self.no_description_mask
                            else self.to_cond_embed(
                                self.pad_description(
                                    cond["cross_attn_cond"],
                                    cond["cross_attn_cond_mask"],
                                )
                            )
                        ),
                    ],
                    dim=1,
                )
                if "cross_attn_cond" in cond
                else None
            ),
            description_tids=cond["text_tids"] if "text_tids" in cond else None,
            description_tembs=cond["text_tembs"] if "text_tembs" in cond else None,
            description_mask=(
                torch.cat(
                    [
                        torch.ones(batch_size, self.n_memory_tokens_description).to(
                            x.device
                        ),
                        (
                            torch.ones_like(cond["cross_attn_cond_mask"])
                            if self.no_description_mask
                            else cond["cross_attn_cond_mask"]
                        ),
                    ],
                    dim=1,
                )
                if "cross_attn_cond" in cond
                else None
            ),
            freqs_cis=self.freqs_cis,
            freqs_cis_description=self.freqs_cis_description,
            logvar=(
                self.to_logvar(self.timestep_logvar(t[:, None]))[:, 0]  # (b,)
                if self.estimate_logvar
                else None
            ),
        )

        # DictTensor is not supposed to contain None values
        return d  # type: ignore


class PostProcessing(nn.Module):
    """
    Simple linear preceded by AdaLN if adaln_last_layer
    """

    def __init__(self, args: DiTArgs):
        super().__init__()
        self.patch_size = args.patch_size

        self.adaln_last_layer = args.adaln_last_layer
        self.adaln_last_layer_nomod = args.adaln_last_layer_nomod
        if self.adaln_last_layer_nomod:
            print("\nadaln_last_layer_nomod is True\n")
        if self.adaln_last_layer:
            self.norm = nn.LayerNorm(args.dim, elementwise_affine=False, eps=1e-6)
            if not self.adaln_last_layer_nomod:
                self.mod_proj = nn.Linear(args.dim, args.dim * 2, bias=True)

        self.linear = nn.Linear(args.dim, args.io_channels * self.patch_size, bias=True)
        self.n_memory_tokens_rope = args.n_memory_tokens_rope
        self.input_padding_size = (-args.max_seq_len) % args.patch_size

    def forward(self, d: DictTensor) -> DictTensor:
        main_key = "x"
        mod_key = "t"
        x = d[main_key]
        # Strip memory tokens rope
        x = x[:, self.n_memory_tokens_rope :]

        # Last layer
        if self.adaln_last_layer:
            if self.adaln_last_layer_nomod:
                x = self.norm(x)
            else:
                bias, scale = self.mod_proj(d[mod_key].unsqueeze(1)).chunk(2, dim=-1)
                x = (1 + scale) * self.norm(x) + bias

        x = self.linear(x)
        x = rearrange(x, "b t (p c) -> b c (t p)", p=self.patch_size)
        # and eventually remove padding tokens after depatching
        d[main_key] = x[:, :, self.input_padding_size :]
        return d


# ------------------------------------
# -- For MeanFlow with 2nd timestep --
# ------------------------------------


class InputProcessingMeanFlow(InputProcessing):
    """
    InputProcessing adapted for MeanFlow with second timestep arg r.
    """

    def __init__(self, args: DiTArgs):
        super().__init__(args)

        timestep_embed_dim = args.timestep_features_dim * 2

        # Timestep encoding of (t, r)
        self.timestep_features_t = FourierFeaturesTime(1, args.timestep_features_dim)
        self.timestep_features_r = FourierFeaturesTime(1, args.timestep_features_dim)
        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_embed_dim, args.inter_dim, bias=True),
            nn.SiLU(),
            nn.Linear(args.inter_dim, args.dim, bias=True),
            nn.SiLU(),
        )

        # Estimation of logvar from (t, r)
        self.estimate_logvar = args.estimate_logvar
        if args.estimate_logvar:
            self.timestep_logvar_t = FourierFeaturesTime(1, args.timestep_features_dim)
            self.timestep_logvar_r = FourierFeaturesTime(1, args.timestep_features_dim)
            self.to_logvar = nn.Sequential(
                nn.Linear(timestep_embed_dim, 128, bias=True),
                nn.SiLU(),
                nn.Linear(128, 1, bias=True),
            )

    # Copies signature from forward method of dit.DiffusionTransformer
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> DictTensor:
        batch_size = x.size(0)

        # Encode t_embed as a MLP applied t and r Fourier feats concatenated
        t_embed = self.to_timestep_embed(  # (b, c)
            torch.cat(
                [
                    self.timestep_features_t(t[:, None]),
                    self.timestep_features_r(r[:, None]),
                ],
                dim=-1,
            )
        )
        # Encode t_logvar as a MLP applied t and r Fourier feats concatenated
        t_logvar = (
            self.to_logvar(  # (b,)
                torch.cat(
                    [
                        self.timestep_logvar_t(t[:, None]),
                        self.timestep_logvar_r(r[:, None]),
                    ],
                    dim=-1,
                )
            )[:, 0]
            if self.estimate_logvar
            else None
        )
        d = dict(
            # memory tokens + embed(x)
            x=torch.cat(
                [
                    self.memory_tokens_rope.expand(
                        batch_size, self.n_memory_tokens_rope, -1
                    ),
                    self.embed_x(x),
                ],
                dim=1,
            ),
            x_mask=mask,
            t=t_embed,
            description=torch.cat(
                [
                    self.memory_tokens_description.expand(x.size(0), -1, -1),
                    self.to_cond_embed(cond["cross_attn_cond"])
                    if not self.no_description_mask
                    else self.to_cond_embed(
                        self.pad_description(
                            cond["cross_attn_cond"], cond["cross_attn_cond_mask"]
                        )
                    ),
                ],
                dim=1,
            )
            if "cross_attn_cond" in cond
            else None,
            description_mask=torch.cat(
                [
                    torch.ones(batch_size, self.n_memory_tokens_description).to(
                        x.device
                    ),
                    (
                        torch.ones_like(cond["cross_attn_cond_mask"])
                        if self.no_description_mask
                        else cond["cross_attn_cond_mask"]
                    ),
                ],
                dim=1,
            )
            if "cross_attn_cond" in cond
            else None,
            freqs_cis=self.freqs_cis,
            freqs_cis_description=self.freqs_cis_description,
            logvar=t_logvar,
        )

        # DictTensor is not supposed to contain None values
        return d  # type: ignore


class SFXFlow(DiTPipeline):
    """
    Same as Flux, but uses MMMBlocks only
    Adds singlestream blocks

    Does not rely on apply_rope argument
    """

    def __init__(self, args: MMDiTArgs):
        """ """
        # TODO remove ref to MMMFlux
        assert args.no_description_mask, "MMMFlux requires no description mask"
        preprocessing = InputProcessing(args)
        postprocessing = PostProcessing(args)

        layers = torch.nn.Sequential()
        assert args.num_sinks == 0, "MMMSSFlux requires num_sinks to be 0"
        for layer_id in range(args.n_layers):
            if layer_id < args.n_multimodal_layers:
                layers.append(
                    MMMBlock(
                        layer_id,
                        modality_block_dict=dict(
                            x=ModalityBlock(
                                args,
                                x_key="x",
                                mod_key="t",
                                freqs_cis_key="freqs_cis",
                                mask_key=None,
                            ),
                            description=ModalityBlock(
                                args,
                                x_key="description",
                                mod_key="t",
                                mask_key=None,
                                freqs_cis_key="freqs_cis_description",
                            ),
                        ),
                    ),
                )

            else:
                layers.append(
                    MultimodalitySingleStreamBlock(
                        layer_id=layer_id,
                        args=args,
                        x_keys=["x", "description"],
                        mod_key="t",
                        freqs_cis_keys=["freqs_cis", "freqs_cis_description"],
                        mask_key=None,
                    ),
                )

        super().__init__(
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            layers=layers,
            non_checkpoint_layers=args.non_checkpoint_layers,
            mask_out_before=args.mask_out_before,
        )


# -------------------------------------------
# -- For MeanFlow models with 2nd timestep --
# -------------------------------------------


class SFXFlowMap(DiTFlowMapPipeline):
    """
    Adapted to 2nd timestep arg (meanflow).
    """

    def __init__(self, args: MMDiTArgs):
        """ """
        assert args.no_description_mask, "MMMFlux requires no description mask"
        preprocessing = InputProcessingMeanFlow(args)
        postprocessing = PostProcessing(args)

        layers = torch.nn.Sequential()
        for layer_id in range(args.n_layers):
            if layer_id < args.n_multimodal_layers:
                layers.append(
                    MMMBlock(
                        layer_id,
                        modality_block_dict=dict(
                            x=ModalityBlock(
                                args,
                                x_key="x",
                                mod_key="t",
                                freqs_cis_key="freqs_cis",
                                mask_key=None,
                            ),
                            description=ModalityBlock(
                                args,
                                x_key="description",
                                mod_key="t",
                                mask_key=None,
                                freqs_cis_key=None,
                            ),
                        ),
                    )
                )
            else:
                layers.append(
                    ModalityBlock(
                        args,
                        x_key="x",
                        mod_key="t",
                        freqs_cis_key="freqs_cis",
                        mask_key=None,
                    ),
                )

        super().__init__(
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            layers=layers,
            non_checkpoint_layers=args.non_checkpoint_layers,
            mask_out_before=args.mask_out_before,
        )
