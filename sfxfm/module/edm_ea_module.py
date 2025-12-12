from .diffusion_ae_module import DiffusionAutoEncoderModule

import torch

import torch.nn.functional as F
import numpy as np


class EDMAutoEncoderModule(DiffusionAutoEncoderModule):
    """
    Elucidating Diffusion Models Module
    Args:
    """

    def __init__(
        self,
        diffusion_model,
        encoder,
        preprocessor,
        ema_decay,
        optim,
        quantizer,
        t_schedule,
        condition_dropout=0.0,
        fixed_encoder=False,
        normalize_audio=True,
        sigma_data=0.5,
        **kwargs,
    ):
        super().__init__(
            diffusion_model,
            encoder,
            preprocessor,
            ema_decay,
            optim,
            quantizer,
            t_schedule,
            condition_dropout=condition_dropout,
            fixed_encoder=fixed_encoder,
            normalize_audio=normalize_audio,
        )
        self.sigma_data = sigma_data

    def round_sigma(self, value: float):
        return torch.as_tensor(value)

    # base sampling function associated to DiffusionModule
    # used in callbacks for instance
    @torch.no_grad()
    def sample(
        self,
        noise,
        cond,
        n_q=None,
        num_steps=18,
        sigma_min=1e-5,
        sigma_max=80,
        rho=7,
        S_churn=0,
        S_min=0,
        S_max=float("inf"),
        S_noise=1,
        ema=True,
    ):
        # TODO grid search on parameters
        """
        Returns data BEFORE post processing
        """
        if type(n_q) == int:
            n_q = torch.Tensor([n_q] * noise.size(0)).long().to(noise.device)
        # sigma_min = max(sigma_min, self.sigma_min)
        # sigma_max = min(sigma_max, self.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=noise.device)

        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat(
            [self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        )  # t_N = 0

        # Main sampling loop.
        x_next = noise.to(torch.float64) * t_steps[0]

        for i, (t_cur, t_next) in enumerate(
            zip(t_steps[:-1], t_steps[1:])
        ):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = self.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (
                t_hat**2 - t_cur**2
            ).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            denoised = self.denoise(x_t=x_hat, t=t_hat, cond=cond, n_q=n_q, ema=ema)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.denoise(
                    x_t=x_next, t=t_next, cond=cond, n_q=n_q, ema=ema
                ).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next.float()

    def denoise(self, x_t, t, cond, n_q, ema=False):
        """
        Wrapper around network
        Selects ema or non-ema weights
        Can be reimplemented for preconditioning
        """
        if type(n_q) == int:
            batch_size = x_t.size(0)
            n_q = torch.Tensor([n_q] * batch_size).long().to(x_t.device)

        if type(t) == float or len(t.size()) == 0:
            batch_size = x_t.size(0)
            t = torch.Tensor([t] * batch_size).double().to(x_t.device)

        original_dtype = x_t.type()
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

        if ema:
            self.diffusion_ema.eval()
            # cast since sampling can be done in float64
            F_x = self.diffusion_ema(
                (c_in * x_t).float(), t=c_noise.float(), cond=cond.float(), n_q=n_q
            ).type(original_dtype)
        else:
            F_x = self.diffusion(
                (c_in * x_t).float(), t=c_noise.float(), cond=cond.float(), n_q=n_q
            ).type(original_dtype)

        D_x = c_skip * x_t + c_out * F_x
        return D_x

    # def sample_t(self, x):
    #     """
    #     We return SIGMA
    #     """
    #     rnd_normal = torch.randn(x.size(0)).to(x.device)
    #     sigma = (rnd_normal * self.P_std + self.P_mean).exp()
    #     return sigma

    def loss_fn(self, reals, cond, n_q, noise, t, quantization_loss, ema=False):
        # sigma equals t in ALL EDM
        # t is not broadcasted
        sigma = t[:, None, None]
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # Combine the ground truth images and the noise
        x_t = reals + sigma * noise
        targets = reals

        x_hat = self.denoise(x_t, t=t, cond=cond, n_q=n_q, ema=ema)
        reco_loss = F.mse_loss(x_hat, targets, reduce=False)
        mse_loss = reco_loss * weight
        loss = mse_loss.mean() + quantization_loss.mean()

        return loss, reco_loss

    # class EDMLoss:

    #     def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
    #         self.P_mean = P_mean
    #         self.P_std = P_std
    #         self.sigma_data = sigma_data

    #     def __call__(self, net, images, labels=None, augment_pipe=None):
    #         rnd_normal = torch.randn([images.shape[0], 1, 1, 1],
    #                                  device=images.device)
    #         sigma = (rnd_normal * self.P_std + self.P_mean).exp()
    #         weight = (sigma**2 + self.sigma_data**2) / (sigma *
    #                                                     self.sigma_data)**2
    #         y, augment_labels = augment_pipe(
    #             images) if augment_pipe is not None else (images, None)
    #         n = torch.randn_like(y) * sigma
    #         D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
    #         loss = weight * ((D_yn - y)**2)
    #         return loss

    # def forward(self,
    #             x,
    #             sigma,
    #             class_labels=None,
    #             force_fp32=False,
    #             **model_kwargs):
    #     x = x.to(torch.float32)
    #     sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
    #     class_labels = None if self.label_dim == 0 else torch.zeros(
    #         [1, self.label_dim],
    #         device=x.device) if class_labels is None else class_labels.to(
    #             torch.float32).reshape(-1, self.label_dim)
    #     dtype = torch.float16 if (self.use_fp16 and not force_fp32 and
    #                               x.device.type == 'cuda') else torch.float32

    #     c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
    #     c_out = sigma * self.sigma_data / (sigma**2 +
    #                                        self.sigma_data**2).sqrt()
    #     c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
    #     c_noise = sigma.log() / 4

    #     F_x = self.model((c_in * x).to(dtype),
    #                      c_noise.flatten(),
    #                      class_labels=class_labels,
    #                      **model_kwargs)
    #     assert F_x.dtype == dtype
    #     D_x = c_skip * x + c_out * F_x.to(torch.float32)
    #     return D_x
