import torchmetrics

from sfxfm.module.metrics.multimodal_metric_collection import MultiModalMetricCollection
import sfxfm.module.model.transformer
from .diffusion_module import DiffusionModule
import math
import torch

import torch.nn.functional as F
import numpy as np


class EDMModule(DiffusionModule):
    """
    Elucidating Diffusion Models Module
    Args:
    """

    def __init__(
        self,
        diffusion_model,
        preprocessor,
        ema_decay,
        optim,
        quantizer,
        t_schedule,
        mask_sampler=None,
        fixed_encoder=False,
        normalize_audio=True,
        condition_dropout=0.0,
        minsnr_gamma=-1,
        sigma_data=0.5,
        sigma_lo=0.0,
        sigma_hi=np.inf,
        beta=False,
        metric_collection_train=torchmetrics.MetricCollection([]),
        metric_collection_val: MultiModalMetricCollection = MultiModalMetricCollection(
            []
        ),
        gen_edm_kwargs={},
        finetune_checkpoint=None,
        **kwargs,
    ):
        """
        minsnr_gamma: gamma value in the MinSNR-gamma weighting strategy
        as described in
        Efficient Diffusion Training via Min-SNR Weighting Strategy https://arxiv.org/pdf/2303.09556
        if negative, we use the standard loss weighting from EDM
        """
        super().__init__(
            diffusion_model,
            preprocessor,
            ema_decay,
            optim,
            quantizer,
            t_schedule,
            mask_sampler=mask_sampler,
            condition_dropout=condition_dropout,
            fixed_encoder=fixed_encoder,
            normalize_audio=normalize_audio,
            metric_collection_train=metric_collection_train,
            metric_collection_val=metric_collection_val,
            gen_edm_kwargs=gen_edm_kwargs,
            finetune_checkpoint=finetune_checkpoint,
        )
        sfxfm.module.model.transformer.set_pl_module(self)

        self.sigma_data = sigma_data
        self.sigma_lo = sigma_lo
        self.sigma_hi = sigma_hi
        self.minsnr_gamma = minsnr_gamma

    def round_sigma(self, value: float):
        return torch.as_tensor(value)

    # base sampling function associated to DiffusionModule
    # used in callbacks for instance
    def sample(
        self,
        noise,
        *args,
        sampler="heun",
        **kwargs,
    ):
        sampler_fn = getattr(self, sampler, None)
        noise_scheduler = kwargs.pop("noise_scheduler", None)
        if noise_scheduler is not None:
            t_steps = self.get_t_steps(noise_scheduler, noise.device, **kwargs)
            kwargs["t_steps"] = t_steps

        if callable(sampler_fn):
            return sampler_fn(noise, *args, **kwargs)
        raise ValueError(f"Sampler {sampler} not found!")

    def denoise(self, x_t, t, cond=None, mask=None, ema=False):
        """
        Wrapper around network
        Selects ema or non-ema weights
        Can be reimplemented for preconditioning
        """
        if mask is None:
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

        if ema and hasattr(self, "diffusion_ema"):
            self.diffusion_ema.eval()
            # cast since sampling can be done in float64
            F_x = self.diffusion_ema(x_in, t=c_noise, cond=cond, mask=mask)
        else:
            F_x = self.diffusion(x_in, t=c_noise, cond=cond, mask=mask)
        # if mask is provided, F_x is of size (batch, channels, number_of_unmasked_tokens)

        with torch.autocast(device_type="cuda", enabled=False):
            c_skip = c_skip.float()
            c_out = c_out.float()
            D_x = (
                c_skip
                * x_t[mask.unsqueeze(1).expand(*x_t.size()) == 1].view(
                    x_t.size(0), x_t.size(1), -1
                )
                + c_out * F_x.float()
            )
        return D_x

    def loss_fn(self, reals, cond, noise, t, mask, ema=False):
        with torch.autocast(device_type="cuda", enabled=False):
            # sigma equals t in ALL EDM
            # t is not broadcasted
            sigma = t[:, None, None]

            # We clip only if minsnr_gamma is positive
            # minsnr_gammma <= 0 amounts to "classic" EDM
            # contrary to the paper, we perform clipping over EDM weighting,
            # not 1 / sigma**2. The two are close for small sigmas anyway
            weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
            if self.minsnr_gamma > 0:
                weight = torch.clip(weight, max=self.minsnr_gamma)

            # Combine the ground truth images and the noise
            x_t = reals + sigma * noise
            targets = reals[mask.unsqueeze(1).expand(*reals.size()) == 1].view(
                reals.size(0), reals.size(1), -1
            )

        x_hat = self.denoise(x_t, t=t, cond=cond, mask=mask, ema=ema)

        with torch.autocast(device_type="cuda", enabled=False):
            reco_loss = F.mse_loss(x_hat.float(), targets, reduce=False)
            mse_loss = reco_loss * weight
            loss = mse_loss.mean()
            # save weight for logging
            self.last_weight = weight.detach().reshape(-1)

        return loss, reco_loss

    def _batched_cond_denoise(
        self, x_t, t, ema, mask=None, cond=None, cond_batched=None
    ):
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
            return self.denoise(
                x_t=x_t, t=t, cond=cond_batched[0], ema=ema
            ), self.denoise(x_t=x_t, t=t, cond=cond_batched[1], ema=ema)

        x_t_batched = torch.cat([x_t, x_t], dim=0)

        # TODO cast here?
        # it was already cast
        # denoised = self.denoise(x_t=x_t_batched, t=t, cond=cond_batched, ema=ema).to(
        #     torch.float64
        # )
        denoised = self.denoise(x_t=x_t_batched, t=t, cond=cond_batched, ema=ema)

        denoised_cond, denoised_nocond = torch.chunk(denoised, chunks=2, dim=0)
        return denoised_cond, denoised_nocond

    def _batch_cond_nocond(self, x_t, cond):
        """
        returns a dict of the concatenations (along the batch dimension) of cond and no_cond

        cond is a dictionary of conditioning vectors or sequences

        if batching fails, returns a tuple, to maintain compitabilty with samplers
        """

        no_cond = self.no_cond(x_t)
        for k in (
            "global_embed",
            "cross_attn_cond",
            "cross_attn_cond_mask",
            "seq_embed",
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

        k = "original_x"
        # we copy x_original to no_cond if present in cond
        if k in cond and k not in no_cond:
            no_cond[k] = cond[k]
            if cond[k].shape != no_cond[k].shape:
                no_cond[k] = no_cond[k].expand_as(cond[k])

        cond_batched = {
            k: torch.cat([cond[k], no_cond[k]], dim=0)
            for k in cond
            if k
            in (
                "global_embed",
                "cross_attn_cond",
                "cross_attn_cond_mask",
                "seq_embed",
                "original_x",
            )
        }
        return cond_batched

    def linear_scheduler(
        self, num_steps: int, sigma_min: float, sigma_max: float, device
    ) -> torch.Tensor:
        t_steps = torch.linspace(
            sigma_max, sigma_min, num_steps, dtype=torch.float64, device=device
        )
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        return t_steps

    def karras_scheduler(
        self,
        num_steps: int,
        sigma_min: float,
        sigma_max: float,
        rho: float,
        device,
        add_zero: float = True,
    ) -> torch.Tensor:
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        if add_zero:
            t_steps = torch.cat(
                [self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
            )
        return t_steps

    def sigmoid_scheduler(
        self,
        num_steps: int,
        sigma_min: float,
        sigma_max: float,
        rho: float,
        device,
        add_zero: float = True,
    ) -> torch.Tensor:
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        t_steps = sigma_min + (sigma_max - sigma_min) / (
            1 + torch.exp(-rho * (step_indices - num_steps / 2))
        )
        t_steps = t_steps.flip(0)
        if add_zero:
            t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        return t_steps

    def cosine_scheduler(
        self,
        num_steps: int,
        sigma_min: float,
        sigma_max: float,
        device,
        add_zero: bool = True,
    ) -> torch.Tensor:
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        cos_values = (torch.cos((step_indices / (num_steps - 1)) * math.pi) + 1) / 2
        t_steps = sigma_min + (sigma_max - sigma_min) * cos_values
        if add_zero:
            t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        return t_steps

    def get_t_steps(self, noise_scheduler, device, **kwargs):
        if noise_scheduler is None:
            return None, kwargs
        args_scheduler = {
            "num_steps": kwargs["num_steps"],
            "sigma_min": kwargs["sigma_min"],
            "sigma_max": kwargs["sigma_max"],
            "device": device,
            "add_zero": kwargs.pop("add_zero", True),
        }

        if noise_scheduler == "karras":
            args_scheduler["rho"] = kwargs.pop("rho", 1.0)
            t_steps = self.karras_scheduler(**args_scheduler)
        elif noise_scheduler == "linear":
            args_scheduler.pop("rho", None)
            args_scheduler.pop("add_zero", None)
            t_steps = self.linear_scheduler(**args_scheduler)
        elif noise_scheduler == "sigmoid":
            args_scheduler["rho"] = kwargs.pop("rho", 1.0)
            t_steps = self.sigmoid_scheduler(**args_scheduler)
        elif noise_scheduler == "linear":
            t_steps = self.linear_scheduler(**args_scheduler)
        elif noise_scheduler == "cosine":
            t_steps = self.cosine_scheduler(**args_scheduler)
        else:
            raise ValueError(f"Scheduler {noise_scheduler} not found!")

        return t_steps

    # ======== putting signatures of samplers just for reference ============
    # check .edm_modules/edm_sampler.py for the implementations
    @torch.no_grad()
    def heun(
        self,
        noise,
        cond=None,
        num_steps=18,
        sigma_min=1e-5,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
        noise_scheduler=None,
        ema=True,
        cfg=1,
        progress_tqdm=None,
        sigma_lo=0,
        sigma_hi=float("inf"),
        beta=False,
        t_steps=None,
        alpha_param=3.0,
        beta_param=3.0,
        callback=None,
    ):
        raise NotImplementedError

    def cfgpp(
        self,
        noise,
        cond=None,
        num_steps=18,
        sigma_min=1e-5,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=0,
        ema=True,
        cfg=1,
        progress_tqdm=None,
        progress_disable=False,
        second_order_correction=False,
    ):
        raise NotImplementedError

    def multidiffusion_cfgpp(
        self,
        noise,
        overlap=0,
        cond=None,
        num_steps=18,
        sigma_min=1e-5,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=0,
        ema=True,
        cfg=1,
        progress_tqdm=None,
        loop=False,
        aggregate_on_noisy=True,
        smooth=False,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def cfgpp_inversion(
        self,
        x_0,
        cond=None,
        num_steps=18,
        sigma_min=1e-5,
        sigma_max=80,
        rho=7,
        ema=True,
        cfg=1,
        progress_tqdm=None,
    ):
        raise NotImplementedError

    def mcg_dps(
        self,
        noise,
        ref,
        alpha=1.0,
        project=True,
        mask=None,
        num_steps=18,
        sigma_min=1e-5,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=0,
        ema=True,
        progress_tqdm=None,
    ):
        raise NotImplementedError

    def mcg_cm(
        self,
        noise,
        ref,
        quantizer,
        alpha=1.0,
        mask=None,
        num_steps=18,
        sigma_min=1e-5,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=0,
        ema=True,
        progress_tqdm=None,
    ):
        raise NotImplementedError

    def psld_cfgpp(
        self,
        noise,
        ref,
        omega=1,
        gamma=1,
        squared_error=True,
        cond=None,
        mask=None,
        num_steps=18,
        sigma_min=1e-5,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=0,
        ema=True,
        cfg=1,
        second_order_correction=False,
        progress_tqdm=None,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def lms(
        self,
        noise,
        cond=None,
        num_steps=18,
        sigma_min=1e-5,
        sigma_max=80,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
        ema=True,
        cfg=1,
        progress_tqdm=None,
        sigma_lo=0,
        sigma_hi=float("inf"),
        beta=False,
        alpha_param=3.0,
        beta_param=3.0,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def klms(
        self,
        noise,
        cond=None,
        num_steps=18,
        sigma_min=1e-5,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
        ema=True,
        cfg=1,
        progress_tqdm=None,
        sigma_lo=0,
        sigma_hi=float("inf"),
        beta=False,
        alpha_param=3.0,
        beta_param=3.0,
        order=4,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def dpmpp2m(
        self,
        noise,
        cond=None,
        num_steps=18,
        sigma_min=1e-5,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
        ema=True,
        cfg=1,
        progress_tqdm=None,
        sigma_lo=0,
        sigma_hi=float("inf"),
        beta=False,
        alpha_param=3.0,
        beta_param=3.0,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def dpmpp2msde(
        self,
        noise,
        cond=None,
        num_steps=18,
        sigma_min=1e-5,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
        ema=True,
        cfg=1,
        progress_tqdm=None,
        sigma_lo=0,
        sigma_hi=float("inf"),
        beta=False,
        alpha_param=3.0,
        beta_param=3.0,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def rungekutta3(
        self,
        noise,
        cond=None,
        num_steps=18,
        sigma_min=1e-5,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
        ema=True,
        cfg=1,
        progress_tqdm=None,
        sigma_lo=0,
        sigma_hi=float("inf"),
        beta=False,
        alpha_param=3.0,
        beta_param=3.0,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def euler(
        self,
        noise,
        cond=None,
        num_steps=18,
        sigma_min=1e-5,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
        ema=True,
        cfg=1,
        progress_tqdm=None,
        sigma_lo=0,
        sigma_hi=float("inf"),
        beta=False,
        alpha_param=3.0,
        beta_param=3.0,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def fasteuler(
        self,
        noise,
        cond=None,
        num_steps=18,
        sigma_min=1e-5,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
        ema=True,
        cfg=1,
        progress_tqdm=None,
        sigma_lo=0,
        sigma_hi=float("inf"),
        beta=False,
        alpha_param=3.0,
        beta_param=3.0,
        fast_factor=0.7,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def rungekutta(
        self,
        noise,
        cond=None,
        num_steps=18,
        sigma_min=1e-5,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
        ema=True,
        cfg=1,
        progress_tqdm=None,
        sigma_lo=0,
        sigma_hi=float("inf"),
        beta=False,
        alpha_param=3.0,
        beta_param=3.0,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def dpmpp3msde(
        self,
        noise,
        cond=None,
        num_steps=18,
        sigma_min=1e-5,
        sigma_max=80,
        rho=7,
        ema=True,
        cfg=1,
        progress_tqdm=None,
        sigma_lo=0,
        sigma_hi=float("inf"),
        beta=False,
        t_steps=None,
        alpha_param=3.0,
        beta_param=3.0,
        eta=1.0,
        noise_scheduler=None,
    ):
        raise NotImplementedError
