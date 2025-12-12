"""
LoRA (Low-Rank Adaptation) for Latent Diffusion Model (LDM)
"""

from typing import Annotated, Dict, List, Mapping, Union
from einops import rearrange
from pydantic import BaseModel, Discriminator, Tag
import torch
from torch import nn


from sfxfm.model.ditv2 import (
    DiTArgs,
    DiTPipeline,
    DictTensor,
)
from sfxfm.model.ldm import (
    LatentDiffusionModel,
    LatentDiffusionModelConfig,
    LatentDiffusionModelPipeline,
)


from sfxfm.model.lora import LoraArgs, replace_linear_with_lora
from sfxfm.module.components.base import (
    BaseComponent,
    ComponentConfig,
    LoadConfig,
    _is_load_config,
)
from sfxfm.module.components.conditioners import ConditionConfig, DiffusionConditioner


class InpaintingFinetuneArgs(ComponentConfig):
    ldm: LatentDiffusionModelConfig

    use_lora: bool = False
    lora: LoraArgs
    non_checkpoint_layers: int = 0
    zero_linear_bias: bool = False
    two_masks: bool = True
    set_ground_truth: bool = True


InpaintingFinetuneConfig = Annotated[
    Union[
        Annotated[LoadConfig, Tag("load_config")],
        Annotated[InpaintingFinetuneArgs, Tag("component_args")],
    ],
    Discriminator(discriminator=_is_load_config),
]


class ZeroLinear(nn.Linear):
    """
    Linear layer initialized with zero weight and zero bias
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class CopyCondPreprocessing(nn.Module):
    """
    Simply runs the former preprocessing and adds
    the precomputed conditioners
    """

    def __init__(self, old_preprocessing, keys):
        super().__init__()
        self.old_preprocessing = old_preprocessing
        self.keys = keys

    def forward(self, x, t, cond, mask):
        d = self.old_preprocessing(x, t, cond, mask)
        d.update({k: cond[k] for k in self.keys})
        return d


class XMaskedPreprocessing(nn.Module):
    """
    Project x_masked (x_masked + concatenated binary mask) to dim and
    """

    def __init__(self, old_preprocessing, args: InpaintingFinetuneArgs):
        super().__init__()
        self.old_preprocessing = old_preprocessing

        dit_args: DiTArgs = LatentDiffusionModel.resolve_config(args.ldm).dit  # type: ignore
        num_masks = int(args.two_masks) + 1

        self.project_in_masked_x = ZeroLinear(
            dit_args.io_channels + num_masks, dit_args.dim, bias=args.zero_linear_bias
        )

    def forward(self, x, t, cond, mask):
        # first preprocess x_masked (in cond)
        d = self.old_preprocessing(x, t, cond, mask)
        x_masked = cond["x_masked"]
        batch_size, io_channel_plus_one, length = x_masked.size()

        x_masked_proj = torch.cat(
            [
                torch.zeros(
                    (
                        batch_size,
                        self.old_preprocessing.n_memory_tokens_rope,
                        self.project_in_masked_x.out_features,
                    ),
                    device=x_masked.device,
                ),
                self.project_in_masked_x(rearrange(x_masked, "b c t -> b t c")),
            ],
            dim=1,
        )
        d["x"] = d["x"] + x_masked_proj

        # also copy target_F_x and inpainting mask (different from x_mask)
        d["target_F_x"] = cond["target_F_x"]
        d["mask"] = cond["mask"]
        return d

    # add a few methods to access the old_preprocessing attributes
    @property
    def to_timestep_embed(self):
        return self.old_preprocessing.to_timestep_embed

    @property
    def timestep_features(self):
        return self.old_preprocessing.timestep_features


class MaskingConditioner(nn.Module, DiffusionConditioner):
    """
    x_masked = cat(mask(x), mask)
    """

    def __init__(self, args: InpaintingFinetuneArgs):
        super().__init__()

        dit_args: DiTArgs = LatentDiffusionModel.resolve_config(args.ldm).dit  # type: ignore
        self.io_channels = dit_args.io_channels
        self.two_masks = args.two_masks

    @property
    def output(self) -> Mapping[str, ConditionConfig]:
        return {
            f"x_masked": ConditionConfig(
                id=f"x_masked",
                shape=[self.io_channels + 1],
                type=f"x_masked",
            ),
            f"mask": ConditionConfig(
                id=f"mask",
                shape=[self.io_channels],
                type=f"mask",
            ),
            f"x_original": ConditionConfig(
                id=f"x_original",
                shape=[self.io_channels],
                type=f"x_original",
            ),
        }

    def forward(
        self, batch, condition_dropout=0.0, no_cond=False, device=None, **kwargs
    ) -> DictTensor:
        """ """
        output_cond_dict = {}
        # compute previous conditioners
        # as we need them to compute the controlnet
        x = rearrange(batch["x_original"], "b c t -> b t c")

        mask = batch["mask"].int().unsqueeze(2).expand(-1, -1, x.size(2))
        batch_size, original_length, dim = x.size()
        # mask_tokens = self.mask_token[None, None, :].expand(
        #     batch_size, original_length, dim
        # )
        # x = x * (1 - mask) + mask_tokens * mask
        # mask token == 0
        x = x * (1 - mask)

        # only one channel for mask
        mask = mask[:, :, :1]

        x = torch.cat([x, mask.float()], dim=2)
        if self.two_masks:
            x = torch.cat([x, 1 - mask.float()], dim=2)

        x = rearrange(x, "b t c -> b c t")

        output_cond_dict["x_masked"] = x
        output_cond_dict["mask"] = batch["mask"]
        output_cond_dict["x_original"] = batch["x_original"]

        return output_cond_dict


class SetMaskedGroundTruthPostProcessing(nn.Module):
    """
    Sets the ground truth for masked values
    These are masked in the loss but can be used at inference time
    """

    def __init__(self, old_post_processing):
        super().__init__()
        self.old_post_processing = old_post_processing

    def forward(self, d: DictTensor) -> DictTensor:
        d = d.copy()
        d = self.old_post_processing(d)
        # after old_post_processing, x is (b c t)

        x = d["x"]
        target_F_x = d["target_F_x"].to(d["x"].dtype)

        opposite_mask = (
            (1 - d["mask"]).int().unsqueeze(1).expand(-1, d["x"].size(1), -1)
        )
        x[opposite_mask == 1] = target_F_x[opposite_mask == 1]
        d["x"] = x
        return d


class InpaintingFinetune(nn.Module, BaseComponent, LatentDiffusionModelPipeline):
    """
    Simple LoRA model for Latent Diffusion Model.

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """

    config_class = InpaintingFinetuneArgs

    def __init__(self, config: InpaintingFinetuneConfig):
        # Step 1: init of nn.Module
        super().__init__()

        # Step 2: init of BaseComponent
        self.init_from_config(config)
        # now we use self.config and we know it has been validated
        self.config: InpaintingFinetuneArgs

        # ========= part to fill starts here ==========
        # Step 3: init of LatentDiffusionModelPipeline

        # init of dit pipeline
        ldm = LatentDiffusionModel(self.config.ldm)
        dit_config: DiTArgs = ldm.config.dit

        if self.config.use_lora:
            assert not self.config.ldm.trainable, "Cannot use LoRA with trainable LDM"

        # Preprocess
        new_preprocessing = XMaskedPreprocessing(
            ldm.dit.preprocessing, args=self.config
        )

        new_conditioners = nn.ModuleDict(
            {
                **ldm.conditioners,
                "controlnet_conditioner": MaskingConditioner(args=self.config),
            }
        )

        if self.config.use_lora:
            layers: nn.ModuleList | nn.Sequential = replace_linear_with_lora(
                nn.ModuleList(ldm.dit.layers), args=self.config.lora
            )  # type: ignore
        else:
            layers = ldm.dit.layers

        if self.config.set_ground_truth:
            postprocessing = SetMaskedGroundTruthPostProcessing(
                old_post_processing=ldm.dit.postprocessing
            )
        else:
            postprocessing = ldm.dit.postprocessing

        # Creates new dit pipeline
        dit = DiTPipeline(
            preprocessing=new_preprocessing,
            postprocessing=postprocessing,
            layers=layers,
            non_checkpoint_layers=self.config.non_checkpoint_layers,
            mask_out_before=dit_config.mask_out_before,
        )

        # Creates new LDM pipeline
        self.init_pipeline(
            dit=dit,
            autoencoder=ldm.autoencoder,
            conditioners=new_conditioners,
            sigma_data=ldm.sigma_data,
            pred_type=ldm.pred_type,
        )

        # Step 4 : Register subcomponents
        self.register_subcomponent(
            "backbone_ldm",
            subcomponent=ldm,
        )

        # ========= part to fill ends here ==========
        # After registering all subcomponents, we can finally
        # load the state dict from its internal _weights_path
        self.load_from_config()
