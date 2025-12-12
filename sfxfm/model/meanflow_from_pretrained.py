"""
class MeanFlowFromPretrained:

    MeanFlowFromPretrained is a wrapper for a pretrained Latent Diffusion Model (LDM)
    with meanflow-specific modifications, such as custom preprocessing, LoRA support
    and optional discriminator for adversarial training.

"""

import copy
from einops import rearrange
from itertools import chain
from pydantic import Discriminator, Tag
from tqdm import trange
from typing import Annotated, Literal, Union, Dict, List

import torch
from torch import nn

from sfxfm.model.dit_blocks import (
    FourierFeaturesTime,
    FixedFourierFeaturesTime,
    ModalityAttentionMFWrapper,
    ModalityAttention,
)
from sfxfm.model.dit_pipeline import DiTMeanFlowPipeline, DiTPipeline
from sfxfm.model.ditv2 import DiTArgs, DictTensor
from sfxfm.model.inpainting_finetune import (
    InpaintingFinetune,
    InpaintingFinetuneConfig,
)
from sfxfm.model.ldm import (
    LatentDiffusionModel,
    LatentDiffusionModelConfig,
    LatentDiffusionModelMeanFlowPipeline,
)
from sfxfm.model.discriminator_sana import DiscHeadNetwork
from sfxfm.model.lora import LoraArgs, shallow_copy
from sfxfm.module.components.base import (
    BaseComponent,
    ComponentConfig,
    LoadConfig,
    _is_load_config,
)


# ----------------------------------
# -- MeanFlowFromPretrained utils --
# ----------------------------------


class MeanFlowPretrainedArgs(ComponentConfig):
    ldm: LatentDiffusionModelConfig | InpaintingFinetuneConfig
    # try to keep same keys as in extract_component.py
    pretrained_model_type: Literal["ldm", "inpainting-finetune"] = "ldm"
    use_lora: bool = False
    lora: LoraArgs
    pretrain_embeddings: bool = True
    num_steps_emb_pretraining: int = 20000
    distill_cfg: bool = False
    add_discriminator: bool = False


MeanFlowPretrainedConfig = Annotated[
    Union[
        Annotated[LoadConfig, Tag("load_config")],
        Annotated[MeanFlowPretrainedArgs, Tag("component_args")],
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

    def __init__(self, args: MeanFlowPretrainedArgs, old_postprocessing):
        super().__init__()
        self.old_postprocessing = old_postprocessing

    def forward(self, d: DictTensor) -> DictTensor:
        # Apply old postprocessing and flip sign
        d = self.old_postprocessing(d)
        d["x"] = -d["x"]
        return d


class DiscHead(nn.Module):
    """
    Discriminator head
    should be inserted after layer_id in backbone model
    concatenates features with key in x_keys and apply a small NN

    adds disc_head_{layer_id} key to the DictTensor
    """

    def __init__(
        self,
        args: MeanFlowPretrainedArgs,
        dit_args: DiTArgs,
        x_keys: List[str],
        layer_id: int,
    ):
        super().__init__()
        # must access pretrained dit args
        # pretrained_args: DiTArgs = LatentDiffusionModel.resolve_config(args.ldm).dit
        pretrained_args: DiTArgs = dit_args

        # TODO Do better like spectral norm + conv
        self.x_keys = x_keys

        # TODO try more complex network
        self.linear = DiscHeadNetwork(channels=pretrained_args.dim, c_dim=0)

        self.layer_id = layer_id

    def forward(self, d: DictTensor) -> DictTensor:
        # (b, t, c)
        features = torch.cat([d[k] for k in self.x_keys], dim=1)
        features = rearrange(features, "b t c -> b c t")

        # TODO use cond?
        d[f"disc_head_{self.layer_id}"] = self.linear(features, c=None)[:, 0]
        # disc_head is (batch, length)
        return d


class DiscriminatorFromTeacher(nn.Module):
    def __init__(
        self,
        args: MeanFlowPretrainedArgs,
        dit_args: DiTArgs,
        pretrained: LatentDiffusionModel,
    ) -> None:
        super().__init__()
        # args of the teacher model
        # pretrained_args: DiTArgs = LatentDiffusionModel.resolve_config(args.ldm).dit
        pretrained_args: DiTArgs = dit_args

        insert_head_after = [1, 5, 9, 11]
        # 2, 8, 14, 20, 27 in SANA-Sprint

        new_layers = []
        trainable_parameters = []
        for layer_id, layer in enumerate(pretrained.dit.layers):
            layer.requires_grad_(False)
            new_layers.append(layer)
            if layer_id in insert_head_after:
                # TODO add description?!
                # TODO add special head on description only?
                disc_head = DiscHead(
                    args=args, dit_args=pretrained_args, x_keys=["x"], layer_id=layer_id
                )
                new_layers.append(disc_head)
                trainable_parameters.append(disc_head)

        new_layers = nn.ModuleList(new_layers)

        self.backbone = DiTPipeline(
            preprocessing=pretrained.dit.preprocessing,
            postprocessing=pretrained.dit.postprocessing,
            layers=new_layers,
            non_checkpoint_layers=len(new_layers) + 1,  # we can't use checkpoints
            mask_out_before=pretrained_args.mask_out_before,
        )

        self.trainable_parameters = list(
            chain(*[p.parameters() for p in trainable_parameters])
        )

    def forward(
        self,
        x: torch.Tensor,
        t,
        cond: Dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        d = self.backbone.forward(x, t, cond, mask)
        # concat disc_heads
        heads = [d[key] for key in d.keys() if key.startswith("disc_head_")]
        return torch.stack(heads, dim=1)

    def unfreeze(self) -> None:
        """Unfreeze trainable parameters for training."""
        for param in self.trainable_parameters:
            param.requires_grad = True
        self.train()

    def freeze(self) -> None:
        r"""Freeze all params for inference."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()


class MeanFlowPreprocessing(nn.Module):
    """
    Adds meanflow-specific modules to an old ldm.dit.preprocessing
      - init new fixed Fourier features and MLP for t and r.
      - init new trainable Fourier features and MLP for logvar.
      - forward replaces d['t'] and d['logvar'] with the new ones.
    """

    def __init__(
        self, args: MeanFlowPretrainedArgs, dit_args: DiTArgs, old_preprocessing
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

    config_class = MeanFlowPretrainedArgs

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
        elif self.config.pretrained_model_type == "inpainting-finetune":
            ldm = InpaintingFinetune(self.config.ldm)
            dit_config: DiTArgs = LatentDiffusionModel.resolve_config(
                ldm.config.ldm
            ).dit
        else:
            raise ValueError(
                f"Unknown pretrained_model_type {self.config.pretrained_model_type}"
            )
        return ldm, dit_config

    def __init__(self, config: MeanFlowPretrainedConfig):
        # ==== Step 1: init of nn.Module
        super().__init__()

        # ==== Step 2: init of BaseComponent
        self.init_from_config(config)
        self.config: MeanFlowPretrainedArgs  # now we use self.config knowing it has been validated

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
        dit = DiTMeanFlowPipeline(
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

        # Optionally init discriminator
        self.discriminator = (
            DiscriminatorFromTeacher(
                args=self.config,
                dit_args=dit_config,
                pretrained=self.teacher_ldm,
            )
            if self.config.add_discriminator
            else None
        )

        # Run warmup pretraining of new timestep embeddings to match old ones
        if self.config.pretrain_embeddings:
            self._init_embeddings()

    # TODO to remove
    def _init_embeddings(self, batch_size=256):
        """
        Train only new timestep/logvar embeddings on a sampled set of
        (t, r) values such that new_encoding(t, r) = old_encoding(t).
        """
        if rank() == 0:
            # Push embeddings to CUDA first
            device = "cuda"
            self.dit.preprocessing.to_timestep_embed = (
                self.dit.preprocessing.to_timestep_embed.to(device=device)
            )
            self.dit.preprocessing.timestep_features_t = (
                self.dit.preprocessing.timestep_features_t.to(device=device)
            )
            self.dit.preprocessing.timestep_features_r = (
                self.dit.preprocessing.timestep_features_r.to(device=device)
            )
            if self.config.distill_cfg:
                self.dit.preprocessing.cfg_features = (
                    self.dit.preprocessing.cfg_features.to(device=device)
                )
            self.dit.preprocessing.old_preprocessing = (
                self.dit.preprocessing.old_preprocessing.to(device=device)
            )

            # Define optimizer with appropriate set of parameters to train
            params_to_train = chain(
                self.dit.preprocessing.to_timestep_embed.parameters(),
                self.dit.preprocessing.timestep_features_t.parameters(),
                self.dit.preprocessing.timestep_features_r.parameters(),
                self.dit.preprocessing.cfg_features.parameters()
                if self.config.distill_cfg
                else iter([]),
            )
            optimizer = torch.optim.AdamW(params_to_train, lr=1e-4, weight_decay=1e-2)

            # Training loop
            pbar = trange(
                self.config.num_steps_emb_pretraining,
                desc="Timestep embedding warmup",
                ncols=100,
            )
            for _ in pbar:
                optimizer.zero_grad()
                loss = 0.0

                with torch.no_grad():
                    # Sample (t, r)
                    t = torch.rand(batch_size, device=device)
                    r = torch.rand(batch_size, device=device)

                    # Compute old timestep embedding
                    old_fourier_t = (
                        self.dit.preprocessing.old_preprocessing.timestep_features(
                            t[:, None]
                        )
                    )
                    if isinstance(
                        self.dit.preprocessing.old_preprocessing.to_timestep_embed[-1],
                        nn.SiLU,
                    ):
                        old_t_embedding = (
                            self.dit.preprocessing.old_preprocessing.to_timestep_embed[
                                :-1
                            ](old_fourier_t)
                        )

                    else:
                        old_t_embedding = (
                            self.dit.preprocessing.old_preprocessing.to_timestep_embed(
                                old_fourier_t
                            )
                        )

                # Compute new timestep embedding
                new_fourier_t = self.dit.preprocessing.timestep_features_t(t[:, None])
                new_fourier_r = self.dit.preprocessing.timestep_features_r(r[:, None])

                if self.config.distill_cfg:
                    # cfg = torch.rand_like(t) * 2 + 2
                    cfg = torch.ones_like(t) * 3
                    new_fourier_cfg = self.dit.preprocessing.cfg_features(cfg[:, None])
                    new_t_r_embedding = self.dit.preprocessing.to_timestep_embed[:-1](
                        torch.cat(
                            [new_fourier_t, new_fourier_r, new_fourier_cfg], dim=-1
                        )
                    )
                else:
                    new_t_r_embedding = self.dit.preprocessing.to_timestep_embed[:-1](
                        torch.cat([new_fourier_t, new_fourier_r], dim=-1)
                    )

                # Compute loss
                loss = torch.nn.functional.mse_loss(
                    new_t_r_embedding, old_t_embedding, reduction="mean"
                )
                pbar.set_postfix({"loss": loss.item()})

                # Backprop
                loss.backward()
                optimizer.step()
