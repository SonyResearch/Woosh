"""
class MeanFlowFromPretrained:

    MeanFlowFromPretrained is a wrapper for a pretrained Latent Diffusion Model (LDM)
    with meanflow-specific modifications, such as custom preprocessing, LoRA support
    and optional discriminator for adversarial training.

"""

import copy
from pydantic import Discriminator, Tag
from typing import Annotated, Literal, Union

import torch
from torch import nn

from sfxfm.model.dit_blocks import (
    FourierFeaturesTime,
    FixedFourierFeaturesTime,
    ModalityAttentionMFWrapper,
    ModalityAttention,
)
from sfxfm.model.dit_pipeline import DiTFlowMapPipeline

from sfxfm.model.dit_types import DiTArgs, DictTensor
from sfxfm.model.ldm import (
    LatentDiffusionModel,
    LatentDiffusionModelConfig,
    LatentDiffusionModelMeanFlowPipeline,
)

from sfxfm.model.lora import LoraArgs, shallow_copy
from sfxfm.components.base import (
    BaseComponent,
    ComponentConfig,
    LoadConfig,
    _is_load_config,
)


# ----------------------------------
# -- MeanFlowFromPretrained utils --
# ----------------------------------


class FlowMapPretrainedArgs(ComponentConfig):
    ldm: LatentDiffusionModelConfig
    pretrained_model_type: Literal["ldm"] = "ldm"
    use_lora: bool = False
    lora: LoraArgs
    pretrain_embeddings: bool = True
    num_steps_emb_pretraining: int = 20000
    distill_cfg: bool = False
    add_discriminator: bool = False


FlowMapPretrainedConfig = Annotated[
    Union[
        Annotated[LoadConfig, Tag("load_config")],
        Annotated[FlowMapPretrainedArgs, Tag("component_args")],
    ],
    Discriminator(discriminator=_is_load_config),
]


def add_extra_mod_key_to_modality_attns(
    module: nn.Module, extra_mod_key: str
) -> nn.Module:
    """
    Returns a shallow copy of module where ModalityAttention layers are replaced with
    ModalityAttentionMFWrapper layers that include an extra_mod_key for meanflow processing.
    """
    new_module = shallow_copy(module)  # shallow copy
    wrapper_class = ModalityAttentionMFWrapper
    for name, child in new_module.named_children():
        if isinstance(child, ModalityAttention):
            new_child = wrapper_class(
                attn=child,
                extra_mod_key=extra_mod_key,
            )
        else:
            new_child = add_extra_mod_key_to_modality_attns(
                module=child,
                extra_mod_key=extra_mod_key,
            )
        setattr(new_module, name, new_child)
        # new_module.set_submodule(name, new_child)
    return new_module


class FlipSignPostprocessing(nn.Module):
    """
    Flips the sign of the teacher postprocessing output to match
    the MeanFlow student.
    """

    def __init__(self, args: FlowMapPretrainedArgs, old_postprocessing):
        super().__init__()
        self.old_postprocessing = old_postprocessing

    def forward(self, d: DictTensor) -> DictTensor:
        # Apply old postprocessing and flip sign
        d = self.old_postprocessing(d)
        d["x"] = -d["x"]
        return d


class MeanFlowPreprocessing(nn.Module):
    """
    Adds meanflow-specific modules to an old ldm.dit.preprocessing
      - init new fixed Fourier features and MLP for t and r.
      - init new trainable Fourier features and MLP for logvar.
      - forward replaces d['t'] and d['logvar'] with the new ones.
    """

    def __init__(
        self, args: FlowMapPretrainedArgs, dit_args: DiTArgs, old_preprocessing
    ):
        """
        Args:
            args (MeanFlowPretrainedArgs): Configuration arguments for MeanFlow wrapper
            dit_args (DiTArgs): Configuration arguments for the pretrained DiT model.
            old_preprocessing (nn.Module): The original preprocessing module from the pretrained model.

        Note:
            dit_args can be obtained directly from args, but its location depends on the model type.
            We pass it here to avoid redundant calls to resolve_config and

        """
        super().__init__()
        self.old_preprocessing = old_preprocessing
        self.distill_cfg = args.distill_cfg

        # Define new timestep embedding modules
        self.timestep_features_t = FixedFourierFeaturesTime(
            1, dit_args.timestep_features_dim, time_factor=1.0
        )
        self.timestep_features_r = FixedFourierFeaturesTime(
            1, dit_args.timestep_features_dim, time_factor=1.0
        )

        if self.distill_cfg:
            self.cfg_features = FixedFourierFeaturesTime(
                1, dit_args.timestep_features_dim, time_factor=1.0
            )
            cfg_features_dim = dit_args.timestep_features_dim
        else:
            cfg_features_dim = 0

        self.to_timestep_embed = nn.Sequential(
            nn.Linear(
                dit_args.timestep_features_dim * 2 + cfg_features_dim,
                dit_args.inter_dim,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(dit_args.inter_dim, dit_args.dim, bias=True),
            nn.SiLU(),
        )

        # Define new timestep embedding modules for logvar
        self.timestep_logvar = FourierFeaturesTime(1, dit_args.timestep_features_dim)
        self.to_logvar = nn.Sequential(
            nn.Linear(dit_args.timestep_features_dim * 2, 128, bias=True),
            nn.SiLU(),
            nn.Linear(128, 1, bias=True),
        )
        # Additional one for CTM logvar loss
        self.to_logvar_ctm = nn.Sequential(
            nn.Linear(dit_args.timestep_features_dim * 2, 128, bias=True),
            nn.SiLU(),
            nn.Linear(128, 1, bias=True),
        )

    def forward(self, x, t, r, cond, mask) -> DictTensor:
        # first preprocess x_masked (in cond)
        # this is an old preprocessing without r:
        d = self.old_preprocessing(x, t, cond, mask)

        # Merge and discard previous t, logvar and logvar_ctm
        if self.distill_cfg:
            d["t"] = self.to_timestep_embed(
                torch.cat(
                    [
                        self.timestep_features_t(t[:, None]),
                        self.timestep_features_r(r[:, None]),
                        self.cfg_features(cond["cfg"][:, None]),
                    ],
                    dim=-1,
                )
            )
        else:
            d["t"] = self.to_timestep_embed(
                torch.cat(
                    [
                        self.timestep_features_t(t[:, None]),
                        self.timestep_features_r(r[:, None]),
                    ],
                    dim=-1,
                )
            )

        d["logvar"] = self.to_logvar(  # (b,)
            torch.cat(
                [
                    self.timestep_logvar(t[:, None]),
                    self.timestep_logvar(r[:, None]),
                ],
                dim=-1,
            )
        )[:, 0]
        d["logvar_ctm"] = self.to_logvar_ctm(  # (b,)
            torch.cat(
                [
                    self.timestep_logvar(t[:, None]),
                    self.timestep_logvar(r[:, None]),
                ],
                dim=-1,
            )
        )[:, 0]

        return d


# ----------------------------------------
# -- Main MeanFlowFromPretrained module --
# ----------------------------------------


class MeanFlowFromPretrained(
    nn.Module, BaseComponent, LatentDiffusionModelMeanFlowPipeline
):
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

    config_class = FlowMapPretrainedArgs

    def init_pretrained_model(self):
        """
        Initializes the pretrained model pipeline based on the configuration.
        Depending on the value of `self.config.pretrained_model_type`, this method will:
        - Instantiate a LatentDiffusionModel if the type is 'ldm', ensuring the model type is supported.
        - Instantiate an InpaintingFinetune model if the type is 'inpainting-finetune', resolving its DiT configuration.
        - Raise a ValueError for unsupported or unknown model types.
        Returns:
            Tuple:
                - ldm: The instantiated pretrained model (LatentDiffusionModel or InpaintingFinetune).
                - dit_config: The DiT configuration associated with the model.
        Raises:
            ValueError: If the model type is unsupported or unknown.
        """
        if self.config.pretrained_model_type == "ldm":
            ldm = LatentDiffusionModel(self.config.ldm)
            dit_config: DiTArgs = ldm.config.dit
            supported_models = ["mmmflux", "mmmssflux"]
            if ldm.config.dit.model_type not in supported_models:
                raise ValueError(
                    f"MeanFlowPretrained only supports {supported_models}, got {ldm.config.dit.model_type}"
                )
        else:
            raise ValueError(
                f"Unknown pretrained_model_type {self.config.pretrained_model_type}"
            )
        return ldm, dit_config

    def __init__(self, config: FlowMapPretrainedConfig):
        # ==== Step 1: init of nn.Module
        super().__init__()

        # ==== Step 2: init of BaseComponent
        self.init_from_config(config)
        self.config: FlowMapPretrainedArgs  # now we use self.config knowing it has been validated

        # ==== Step 3: init student's LDM pipeline with adapted pre and postprocessing
        ldm, dit_config = self.init_pretrained_model()

        ############### TO DEBUG
        # print("REMOVE HEREEEEEEEEEEE")
        # ldm.dit.layers = ldm.dit.layers[:2]
        ############################

        # Define student preprocessing (adapted for 2nd time step r)
        new_preprocessing = MeanFlowPreprocessing(
            args=self.config,
            dit_args=dit_config,
            old_preprocessing=ldm.dit.preprocessing,
        )

        # Define student postprocessing (flip sign of teacher output)
        new_postprocessing = FlipSignPostprocessing(self.config, ldm.dit.postprocessing)

        # Cast v in float32 for JVP compatibility
        ldm.dit.set_cast_v(cast_v=True)
        new_layers = ldm.dit.layers

        # Creates new DiT and LDM pipelines for student
        dit = DiTFlowMapPipeline(
            preprocessing=new_preprocessing,
            postprocessing=new_postprocessing,
            layers=new_layers,
            non_checkpoint_layers=len(new_layers) + 1,  # we can't use checkpoints
            mask_out_before=dit_config.mask_out_before,
        )
        self.init_pipeline(
            dit=dit,
            autoencoder=ldm.autoencoder,
            conditioners=ldm.conditioners,
            sigma_data=ldm.sigma_data,
            pred_type=ldm.pred_type,
        )

        # ==== Step 4 : Register subcomponents
        self.register_subcomponent(
            "backbone_ldm",
            subcomponent=ldm,
        )

        # Load state dict from its internal _weights_path
        self.load_from_config()

        # Save frozen copy of pretrained teacher model for distillation targets
        self.teacher_ldm = copy.deepcopy(ldm)
        self.teacher_ldm.autoencoder = None
        self.teacher_ldm.conditioners = ldm.conditioners  # to sample from teacher model
        self.teacher_ldm.requires_grad_(False)
        self.teacher_ldm.eval()
