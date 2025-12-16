"""
LoRA (Low-Rank Adaptation) for Latent Diffusion Model (LDM)
"""

import copy
from typing import Annotated, Union
from pydantic import BaseModel, Discriminator, Tag
import torch
from torch import nn

from sfxfm.model.dit_pipeline import DiTPipeline
from sfxfm.model.dit_types import DiTArgs
from sfxfm.model.ldm import (
    LatentDiffusionModel,
    LatentDiffusionModelConfig,
    LatentDiffusionModelPipeline,
)
from sfxfm.components.base import (
    BaseComponent,
    ComponentConfig,
    LoadConfig,
    _is_load_config,
)


class LoraArgs(BaseModel):
    """
    General LoRA configuration
    """

    rank: int
    bias: bool = True
    scale: float = 1.0
    singlora: bool = False
    warmup: int = 1
    use_lora: bool = True


class LoraLDMArgs(ComponentConfig):
    ldm: LatentDiffusionModelConfig
    lora: LoraArgs

    non_checkpoint_layers: int = 0


LoraLDMConfig = Annotated[
    Union[
        Annotated[LoadConfig, Tag("load_config")],
        Annotated[LoraLDMArgs, Tag("component_args")],
    ],
    Discriminator(discriminator=_is_load_config),
]


def shallow_copy(module: nn.Module) -> nn.Module:
    """
    Returns a shallow copy of module.
    """
    new_module = copy.copy(module)  # shallow copy
    new_module.__dict__ = copy.copy(module.__dict__)
    new_module.__dict__["_modules"] = copy.copy(module.__dict__.get("_modules"))
    new_module._parameters = copy.copy(module._parameters)
    new_module._buffers = copy.copy(module._buffers)
    new_module._non_persistent_buffers_set = copy.copy(
        module._non_persistent_buffers_set
    )
    return new_module


def replace_linear_with_lora(
    module: nn.Module,
    args: LoraArgs,
) -> nn.Module:
    """
    Returns a shallow copy of module where Linear layers are replaced with LoRA layers.
    """
    new_module = shallow_copy(module)  # shallow copy
    wrapper_class = (
        SingLoRAWrapper if args.singlora else LoRAWrapper
    )  # choose wrapper class based on args

    def bp_hook(module, grad_input, grad_output):
        """
        Backpropagation hook to set the value of u in SingLoRAWrapper.
        """
        if hasattr(module, "update_u"):
            module.update_u()

    for name, child in new_module.named_children():
        if isinstance(child, nn.Linear):
            new_child = wrapper_class(
                linear=child,
                args=args,
            )
            new_child.register_full_backward_hook(bp_hook)
        else:
            new_child = replace_linear_with_lora(
                module=child,
                args=args,
            )
        setattr(new_module, name, new_child)
        # new_module.set_submodule(name, new_child)
    return new_module


class LoRAWrapper(nn.Module):
    def __init__(self, linear: nn.Linear, args: LoraArgs) -> None:
        super().__init__()

        assert isinstance(linear, nn.Linear), "linear must be a nn.Linear"
        self.scale = args.scale
        rank = args.rank

        in_features = linear.in_features
        out_features = linear.out_features
        if rank > (new_rank := min(out_features, in_features)):
            rank = new_rank

        self.linear = linear
        self.lora_A = nn.Linear(
            in_features=in_features,
            out_features=rank,
            bias=False,
        )
        self.lora_B = nn.Linear(
            in_features=rank,
            out_features=out_features,
            bias=args.bias,
        )
        self.lora_B.weight.data.zero_()
        if args.bias:
            self.lora_B.bias.data.zero_()

        self.use_lora = args.use_lora

    def set_scale(self, scale: float) -> None:
        assert isinstance(scale, float), "scalar value must be a float"
        self.scale = scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(input)
        if not self.use_lora:
            return base_out

        _lora_out_B = self.lora_B(self.lora_A(input))
        lora_update = _lora_out_B * self.scale

        return base_out + lora_update


class SingLoRAWrapper(nn.Module):
    def __init__(self, linear: nn.Linear, args: LoraArgs) -> None:
        super().__init__()

        assert isinstance(linear, nn.Linear), "linear must be a nn.Linear"
        self.scale = args.scale
        rank = args.rank

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        if rank > (new_rank := min(self.out_features, self.in_features)):
            rank = new_rank

        max_features = max(self.in_features, self.out_features)

        self.linear = linear
        self.lora_A = nn.Linear(
            in_features=max_features,
            out_features=rank,
            bias=False,
        )

        self.u = nn.Parameter(
            torch.zeros(
                1,
            ),
            requires_grad=False,
        )
        self.warmup = args.warmup

        self.use_lora = args.use_lora

    def update_u(self) -> None:
        self.u.data = torch.min(
            self.u.data + 1 / self.warmup, torch.tensor([1.0], device=self.u.device)
        )

    def set_scale(self, scale: float) -> None:
        assert isinstance(scale, float), "scalar value must be a float"
        self.scale = scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(input)
        if not self.use_lora:
            return base_out

        _lora_out_A = torch.matmul(input, self.lora_A.weight.t()[: self.in_features])
        _lora_out_B = torch.matmul(
            _lora_out_A, self.lora_A.weight[:, : self.out_features]
        )
        lora_update = self.u * _lora_out_B * self.scale

        return base_out + lora_update


class LoraLDM(nn.Module, BaseComponent, LatentDiffusionModelPipeline):
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

    config_class = LoraLDMArgs

    def __init__(self, config: LoraLDMConfig):
        # Step 1: init of nn.Module
        super().__init__()

        # Step 2: init of BaseComponent
        self.init_from_config(config)
        # now we use self.config and we know it has been validated
        self.config: LoraLDMArgs

        # ========= part to fill starts here ==========
        # Step 3: init of LatentDiffusionModelPipeline
        # keep ldm as ref
        ldm = LatentDiffusionModel(self.config.ldm)
        dit_config: DiTArgs = ldm.config.dit

        new_layers: nn.ModuleList = replace_linear_with_lora(
            nn.ModuleList(ldm.dit.layers), args=self.config.lora
        )  # type: ignore

        # Creates new dit pipeline
        dit = DiTPipeline(
            preprocessing=ldm.dit.preprocessing,
            postprocessing=ldm.dit.postprocessing,
            layers=new_layers,
            non_checkpoint_layers=self.config.non_checkpoint_layers,
            mask_out_before=dit_config.mask_out_before,
        )

        # Creates new LDM pipeline
        self.init_pipeline(
            dit=dit,
            autoencoder=ldm.autoencoder,
            conditioners=ldm.conditioners,
            sigma_data=ldm.sigma_data,
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
