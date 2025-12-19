"""
Defines full Latent Diffusion Model (LDM) with Autoencoder,
DiT and CLAP conditioner.
"""

import logging
from pydantic import Discriminator, Tag
from typing import Annotated, Dict, Literal, Union

import torch
from torch import nn

from sfxfm.model.dit_pipeline import SFXFlow
from sfxfm.model.dit_types import DictTensor, DiTArgs, MMDiTArgs
from sfxfm.components.autoencoders import AudioAutoEncoder
from sfxfm.components.base import (
    BaseComponent,
    ComponentConfig,
    LoadConfig,
    _is_load_config,
)
from sfxfm.components.clap_conditioners import SFXCLAPTextConditioner
from sfxfm.components.conditioners import ConditionConfig, DiffusionConditioner

# Get logger
log = logging.getLogger(__name__)


class LatentDiffusionModelArgs(ComponentConfig):
    model_type: Literal["LatentDiffusionModel"] = "LatentDiffusionModel"
    dit: DiTArgs
    conditioners: Dict[str, LoadConfig]
    autoencoder: LoadConfig


LatentDiffusionModelConfig = Annotated[
    Union[
        Annotated[LoadConfig, Tag("load_config")],
        Annotated[LatentDiffusionModelArgs, Tag("component_args")],
    ],
    Discriminator(discriminator=_is_load_config),
]


class LatentDiffusionModelPipeline:
    def __init__(self) -> None:
        super().__init__()
        assert isinstance(self, nn.Module)

    def init_pipeline(self, dit, autoencoder, conditioners) -> None:
        self.dit: SFXFlow = dit
        self.autoencoder: AudioAutoEncoder = autoencoder
        self.conditioners: nn.ModuleDict = conditioners

    def get_cond(
        self, batch, condition_dropout=0.0, no_dropout=False, no_cond=False, **kwargs
    ):
        """
        Returns conditioning dict (possibly empty).
        Dropout removes the descriptions, so use no_dropout=True for validation.
        """
        cond_dict = {}
        cond: DiffusionConditioner
        for cond_name, cond in self.conditioners.items():
            res = cond(
                batch,
                condition_dropout=0.0 if no_dropout else condition_dropout,
                no_cond=no_cond,
                **kwargs,
            )

            v: ConditionConfig
            for res_k, v in cond.output.items():
                if v.type in cond_dict:
                    log.warning(
                        f"Conditioner {cond_name} overwrote key {res_k} in cond_dict"
                    )
                if res_k in res:
                    cond_dict[v.type] = res[res_k]
                else:
                    log.warning(
                        f"Conditioner {cond_name} did not return the expected key {res_k}"
                    )
        # Copy inference keys
        for k in [
            "focus_tids",
            "delta_mod_weight_local",
            "delta_mod_weight_global",
            "category",
            "tora_scale",
        ]:
            if k in batch:
                cond_dict[k] = batch[k]

        return cond_dict

    def _denoise_dict_no_param(self, x_t, t, cond=None, mask=None) -> DictTensor:
        """
        Method used for denoising, returns the whole DictTensor.
        """
        assert cond is not None
        if mask is None:
            mask = torch.ones_like(x_t[:, 0, :])
        d = self.dit(x_t, t=t, cond=cond, mask=mask)
        d["x_hat"] = d["x"]
        return d


class LatentDiffusionModelFlowMapPipeline(LatentDiffusionModelPipeline):
    """
    Same as LatentDiffusionModelPipeline, but adapted for FlowMap, where
    the _denoise_dict_no_param method uses a 2nd timestep argument r.
    """

    def __init__(self, dit, autoencoder, conditioners):
        super().__init__()

    def _denoise_dict_no_param(self, x_t, t, r, cond=None, mask=None) -> DictTensor:
        assert cond is not None
        if mask is None:
            mask = torch.ones_like(x_t[:, 0, :])
        d = self.dit(x_t, t=t, r=r, cond=cond, mask=mask)
        d["x_hat"] = d["x"]
        return d


class LatentDiffusionModel(nn.Module, BaseComponent, LatentDiffusionModelPipeline):
    """
    Latent Diffusion Model for the pretrained Flow teacher.
    """

    config_class = LatentDiffusionModelArgs

    def __init__(self, config: LatentDiffusionModelConfig):
        # Step 1: init of nn.Module
        super().__init__()

        # Step 2: init of BaseComponent
        self.init_from_config(config)
        # now we use self.config and we know it has been validated
        self.config: LatentDiffusionModelArgs

        # Step 3: init of LatentDiffusionModelPipeline
        dit = SFXFlow(MMDiTArgs.model_validate(self.config.dit, strict=True))
        autoencoder = AudioAutoEncoder(self.config.autoencoder)
        # TODO should be a more general DiffusionConditioner builder
        conditioners = nn.ModuleDict(
            {
                k: SFXCLAPTextConditioner(conditioner_config)
                for k, conditioner_config in self.config.conditioners.items()
            }
        )

        self.init_pipeline(dit, autoencoder, conditioners)

        # Step 4 : Register subcomponents
        self.register_subcomponent("autoencoder", self.autoencoder)
        self.register_subcomponent_dict("conditioners", self.conditioners)

        # After registering all subcomponents, we can finally
        # load the state dict from its internal _weights_path
        self.load_from_config()
