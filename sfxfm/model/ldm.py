import logging
from typing import Annotated, Dict, Literal, Union

from pydantic import Discriminator, Tag
import torch
from torch import nn
from sfxfm.model.dit_class import DiT
from sfxfm.model.dit_types import DictTensor
from sfxfm.model.dit_types import DiTArgs
from sfxfm.components.autoencoders import AudioAutoEncoder
from sfxfm.components.base import (
    BaseComponent,
    ComponentConfig,
    LoadConfig,
    _is_load_config,
)
from sfxfm.components.clap_conditioners import SFXCLAPTextConditioner

from sfxfm.components.conditioners import ConditionConfig, DiffusionConditioner

# get logger
log = logging.getLogger(__name__)


class LatentDiffusionModelArgs(ComponentConfig):
    model_type: Literal["LatentDiffusionModel"] = "LatentDiffusionModel"
    dit: DiTArgs
    conditioners: Dict[str, LoadConfig]  # torch.nn.ModuleDict
    autoencoder: LoadConfig  # torch.nn.Module
    sigma_data: float = 1

    pred_type: Literal["v_pred", "x_pred"] = (
        "v_pred"  # if dit is expected to return v directly or x
    )


LatentDiffusionModelConfig = Annotated[
    Union[
        Annotated[LoadConfig, Tag("load_config")],
        Annotated[LatentDiffusionModelArgs, Tag("component_args")],
    ],
    Discriminator(discriminator=_is_load_config),
]


class LatentDiffusionModelPipeline:
    """
    This is NOT an torch.nn.Module
    Should not be used as main module being trained
    """

    def __init__(self) -> None:
        super().__init__()
        assert isinstance(self, nn.Module)

    def init_pipeline(
        self,
        dit,
        autoencoder,
        conditioners: nn.ModuleDict,
        sigma_data,
        pred_type="v_pred",
    ) -> None:
        self.dit: DiT = dit
        self.autoencoder: AudioAutoEncoder = autoencoder
        self.conditioners: nn.ModuleDict = conditioners
        self.sigma_data = sigma_data
        self.pred_type = pred_type

    # Returns dict a (possibly empty) dict
    def get_cond(
        self, batch, condition_dropout=0.0, no_dropout=False, no_cond=False, **kwargs
    ):
        """
        no_dropout=True for validation
        if drop=True return the unconditional
        """
        # TODO should be cleaned, too many unused cond names
        cond_dict = {}
        cond: DiffusionConditioner
        for cond_name, cond in self.conditioners.items():  # type: ignore
            if cond_name == "dae_features" and "dae_features" in batch:
                # TODO remove this hack
                # it is used when called by no_cond.
                res = {
                    "dae_features": batch["dae_features"],
                }
            # TDO really remove this
            elif cond_name == "masked_conditioner":
                # TODO remove this hack
                # it is used when called by no_cond. for inpainting
                res = {
                    "x_original": batch["x_original"],
                    "mask": batch["mask"],
                    "x_masked": cond(
                        batch,
                        condition_dropout=0 if no_dropout else condition_dropout,
                        no_cond=no_cond,
                        **kwargs,
                    ).get("x_masked"),
                }
            else:
                res = cond(
                    batch,
                    condition_dropout=0.0 if no_dropout else condition_dropout,
                    no_cond=no_cond,
                    **kwargs,
                )

            v: ConditionConfig
            for res_k, v in cond.output.items():
                # cond_dict[v.type].append(res[res_k])
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
        # copy inference keys
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

    def no_cond(self, x, cond=None):
        """
        return the no cond tokens by setting description to None
        If a cond is provided copies all fields except description
        """
        if cond is None:
            no_cond = {
                "audio": x,  # TODO This makes no sense
                "description": [
                    None,
                ]
                * x.size(0),
            }
        else:
            no_cond = cond.copy()
            no_cond["description"] = [
                None,
            ] * x.size(0)
            # TODO hack
            # we add the latents as audio
            # if missing
            # as this can be used to get the device
            if "audio" not in cond:
                no_cond["audio"] = x

        cond_from_conditioners = self.get_cond(
            no_cond,
            no_cond=True,
        )

        # additionally copy elements used as external cond
        # like target_F_x
        KEYS_TO_COPY = ["target_F_x"]
        for key in KEYS_TO_COPY:
            if cond is not None and key in cond:
                cond_from_conditioners[key] = cond[key]

        return cond_from_conditioners

    # @TODO refactor these two methods
    def _batched_cond_denoise(self, x_t, t, mask=None, cond=None, cond_batched=None):
        """
        Returns denoised_cond and denoised_nocond

        if cond_batched is provided, does not use cond
        At least one cond or cond_batched must be provided

        Computation is done in one call

        """
        # we compute no cond and concatenate if cond_batched is not precomputed
        if cond_batched is None:
            assert cond is not None
            cond_batched = self._batch_cond_nocond(x_t=x_t, cond=cond)
        # cond_batched is tuple means, condm nocond coud not be batched
        if type(cond_batched) is tuple:
            return self.denoise(x_t=x_t, t=t, cond=cond_batched[0]), self.denoise(
                x_t=x_t, t=t, cond=cond_batched[1]
            )

        x_t_batched = torch.cat([x_t, x_t], dim=0)

        denoised = self.denoise(x_t=x_t_batched, t=t, cond=cond_batched)

        denoised_cond, denoised_nocond = torch.chunk(denoised, chunks=2, dim=0)
        return denoised_cond, denoised_nocond

    # TODO refactor, should be general
    # Cond must be computed from batch
    def _batch_cond_nocond(self, x_t, cond):
        """
        returns a dict of the concatenations (along the batch dimension) of cond and no_cond

        cond is a dictionary of conditioning vectors or sequences

        if batching fails, returns a tuple, to maintain compitabilty with samplers
        """

        no_cond = self.no_cond(x_t, cond=cond)
        # @TODO remove this filtering list
        for k in (
            "global_embed",
            "cross_attn_cond",
            "cross_attn_cond_mask",
            "seq_embed",
            "envelope",
            "envelope_mask",
            "spcenter",
            "spcenter_mask",
            "video_cond",
            "video_cond_mask",
            "video_cond_scale",
            "spflatness",
            "spflatness_mask",
            "spflux",
            "spflux_mask",
            "harmo",
            "harmo_mask",
            "video_features",
            "target_F_x",
        ):
            if (k in cond) != (k in no_cond):  # xor
                # we cannot batch if no_cond doesn't have the same keys as cond, mainly for seq_embed
                return (cond, no_cond)
        # in the case of cross atten and size mismatch
        # @TODO move this to CLAP of OURCLAP CP
        k = "cross_attn_cond"
        mk = "cross_attn_cond_mask"
        if k in no_cond:
            if cond[k].shape != no_cond[k].shape:
                assert (
                    cond[k].shape[0] == no_cond[k].shape[0]
                    and cond[k].shape[-1] == no_cond[k].shape[-1]
                ), (
                    f"can not pad, cond shape {cond[k].shape},   no_cond shape {no_cond[k].shape}"
                )
                pdiff = cond[k].shape[1] - no_cond[k].shape[1]
                assert pdiff > 0, "cannot negative pad"
                no_cond[k] = torch.nn.functional.pad(
                    no_cond[k], (0, 0, 0, pdiff), "constant", 0
                )
                no_cond[mk] = torch.nn.functional.pad(
                    no_cond[mk], (0, pdiff), "constant", 0
                )

        k = "seq_embed"
        if k in cond:
            if cond[k].shape != no_cond[k].shape:
                no_cond[k] = no_cond[k].expand_as(cond[k])

        # TODO this is unbearable
        # TODO this is still unbearable
        try:
            cond_batched = {
                k: torch.cat([cond[k], no_cond[k]], dim=0)
                for k in cond
                if k
                in (
                    "global_embed",
                    "cross_attn_cond",
                    "cross_attn_cond_mask",
                    "seq_embed",
                    "dea_features",
                    "x_original",
                    "x_masked",
                    "mask",
                    "envelope",
                    "envelope_mask",
                    "spcenter",
                    "spcenter_mask",
                    "spflatness",
                    "spflatness_mask",
                    "spflux",
                    "spflux_mask",
                    "harmo",
                    "harmo_mask",
                    "redux_audio",
                    "video_cond",
                    "video_cond_mask",
                    "video_cond_scale",
                    "x2",
                    "video_features",
                    "target_F_x",
                )
                and k in no_cond
            }
        except Exception as e:
            log.error(
                f"Error while batching cond {cond} and no_cond {no_cond} with exception {e}"
            )
            print("cond", cond)
            print("no_cond", no_cond)
            raise e
        return cond_batched

    def denoise(self, x_t, t, cond=None, mask=None) -> torch.Tensor:
        if "mask" in cond:
            mask = cond["mask"]
        return self._denoise_dict(x_t, t, cond=cond, mask=mask)["x_hat"]

    def _denoise_dict(self, x_t, t, cond=None, mask=None) -> DictTensor:
        """
        version of denoise that returns the whole DictTensor
        """
        assert cond is not None
        # TODO absurd, we shouldn't mix x_loss mask and inpainting masks
        if mask is None:
            if "mask" in cond:
                mask = cond["mask"]
            else:
                mask = torch.ones_like(x_t[:, 0, :])
        with torch.autocast(device_type="cuda", enabled=False):
            if isinstance(t, float) or len(t.size()) == 0:
                batch_size = x_t.size(0)
                t = torch.Tensor([t] * batch_size).double().to(x_t.device)
            sigma = t
            c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
            c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
            c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
            c_noise = sigma.log() / 4

            # broadcasts:
            # c_noise is not broadcasted
            c_in = c_in[:, None, None]
            c_out = c_out[:, None, None]
            c_skip = c_skip[:, None, None]

            # model inputs
            x_in = (c_in * x_t).float()
            c_noise = c_noise.float()
            # this is the value to be predicted by the dit
            # before reparameterization
            if "x_original" in cond:
                cond["target_F_x"] = (cond["x_original"] - c_skip * x_t) / c_out

        d = self.dit(x_in, t=c_noise, cond=cond, mask=mask)

        # Extract parameters to compute the loss
        # if mask is provided, F_x only providing
        # meaningful info when mask == 1
        F_x = d["x"]

        with torch.autocast(device_type="cuda", enabled=False):
            c_skip = c_skip.float()
            c_out = c_out.float()
            D_x = c_skip * x_t + c_out * F_x.float()
        # adds x_hat key
        d["x_hat"] = D_x
        return d

    def _denoise_dict_no_param(self, x_t, t, cond=None, mask=None) -> DictTensor:
        """
        version of denoise_dict that returns the whole DictTensor
        AND
        does NOT use a specific parameterization vs _denoise_dict which uses the EDM parameterization
        """
        assert cond is not None
        # TODO absurd, we shouldn't mix x_loss mask and inpainting masks
        if mask is None:
            if "mask" in cond:
                mask = cond["mask"]
            else:
                mask = torch.ones_like(x_t[:, 0, :])
        with torch.autocast(device_type="cuda", enabled=False):
            if isinstance(t, float) or len(t.size()) == 0:
                batch_size = x_t.size(0)
                t = torch.Tensor([t] * batch_size).double().to(x_t.device)

            # model inputs
            x_in = (x_t).float()
            t = t.float()

        d = self.dit(x_in, t=t, cond=cond, mask=mask)

        if self.pred_type == "x_pred":
            # if dit is x-pred; we must convert it to v:
            # in our case,
            # target = real - noise
            # x_t = (1 - t) * real + t * noise
            # (x - x_t) / t => real - noise
            d["x"] = (d["x"] - x_t) / torch.clip(t[:, None, None], min=0.05)

        # adds x_hat key
        d["x_hat"] = d["x"]
        return d

    @staticmethod
    def from_diffusion_module(
        diffusion_module,
        ema: bool = True,
    ) -> "LatentDiffusionModelPipeline":
        """
        returns a LatentDiffusionModelPipeline from a diffusion v2 module
        """

        ldm = diffusion_module.ldm
        if ema:
            ldm.dit.load_state_dict(diffusion_module.ldm_ema.dit.state_dict())

        return ldm


class LatentDiffusionModel(nn.Module, BaseComponent, LatentDiffusionModelPipeline):
    config_class = LatentDiffusionModelArgs

    def __init__(self, config: LatentDiffusionModelConfig):
        # Step 1: init of nn.Module
        super().__init__()

        # Step 2: init of BaseComponent
        self.init_from_config(config)
        # now we use self.config and we know it has been validated
        self.config: LatentDiffusionModelArgs

        # Step 3: init of LatentDiffusionModelPipeline
        dit = DiT(self.config.dit)
        autoencoder = AudioAutoEncoder(self.config.autoencoder)
        # TODO should be a more general DiffusionConditioner builder
        conditioners = nn.ModuleDict(
            {
                k: SFXCLAPTextConditioner(conditioner_config)
                for k, conditioner_config in self.config.conditioners.items()
            }
        )

        sigma_data = self.config.sigma_data
        pred_type = self.config.pred_type

        self.init_pipeline(
            dit, autoencoder, conditioners, sigma_data, pred_type=pred_type
        )

        # Step 4 : Register subcomponents
        self.register_subcomponent(
            "autoencoder",
            self.autoencoder,
        )
        self.register_subcomponent_dict(
            "conditioners",
            self.conditioners,
        )

        # After registering all subcomponents, we can finally
        # load the state dict from its internal _weights_path
        self.load_from_config()


# --------------------------
# -- Classes for MeanFlow --
# --------------------------


class LatentDiffusionModelMeanFlowPipeline(LatentDiffusionModelPipeline):
    """
    A LatentDiffusionModelPipeline with MeanFlow specific methods
    Only redefines denoise_dict_no_param method to have a 2nd timestep arg r.
    """

    def __init__(self, dit, autoencoder, conditioners, sigma_data):
        super().__init__()

    def _denoise_dict_no_param(self, x_t, t, r, cond=None, mask=None) -> DictTensor:
        """
        version of denoise_dict that returns the whole DictTensor
        AND
        does NOT use a specific parameterization vs _denoise_dict which uses the EDM parameterization
        The possibility for a second timestep r is added for meanflow.
        """
        assert cond is not None
        # TODO absurd, we shouldn't mix x_loss mask and inpainting masks
        if mask is None:
            if "mask" in cond:
                mask = cond["mask"]
            else:
                mask = torch.ones_like(x_t[:, 0, :])

        with torch.autocast(device_type="cuda", enabled=False):
            if isinstance(t, float) or len(t.size()) == 0:
                batch_size = x_t.size(0)
                t = torch.Tensor([t] * batch_size).double().to(x_t.device)

            # model inputs
            x_in = (x_t).float()
            t = t.float()

        d = self.dit(x_in, t=t, r=r, cond=cond, mask=mask)

        # adds x_hat key
        d["x_hat"] = d["x"]
        return d


class LatentDiffusionModelMeanFlow(
    nn.Module, BaseComponent, LatentDiffusionModelMeanFlowPipeline
):
    config_class = LatentDiffusionModelArgs

    def __init__(self, config: LatentDiffusionModelConfig):
        # Step 1: init of nn.Module
        super().__init__()

        # Step 2: init of BaseComponent
        self.init_from_config(config)
        # now we use self.config and we know it has been validated
        self.config: LatentDiffusionModelArgs

        # Step 3: init of LatentDiffusionModelPipeline
        dit = DiT(self.config.dit)
        autoencoder = AudioAutoEncoder(self.config.autoencoder)
        # TODO should be a more general DiffusionConditioner builder
        conditioners = nn.ModuleDict(
            {
                k: SFXCLAPTextConditioner(conditioner_config)
                for k, conditioner_config in self.config.conditioners.items()
            }
        )

        sigma_data = self.config.sigma_data

        self.init_pipeline(dit, autoencoder, conditioners, sigma_data)

        # Step 4 : Register subcomponents
        self.register_subcomponent(
            "autoencoder",
            self.autoencoder,
        )
        self.register_subcomponent_dict(
            "conditioners",
            self.conditioners,
        )

        # After registering all subcomponents, we can finally
        # load the state dict from its internal _weights_path
        self.load_from_config()
