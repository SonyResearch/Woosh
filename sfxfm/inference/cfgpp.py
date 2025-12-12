import torch
from tqdm import tqdm
import numpy as np

from sfxfm.model.ldm import LatentDiffusionModelPipeline

from .schedulers import karras_scheduler, round_sigma


@torch.inference_mode()
def cfgpp(
    ldm: LatentDiffusionModelPipeline,
    noise,
    cond=None,
    num_steps=18,
    sigma_min=1e-5,
    sigma_max=80.0,
    rho=7.0,
    S_churn=0.0,
    S_min=0.0,
    S_max=float("inf"),
    S_noise=0.0,
    ema=True,
    cfg=1.0,
    normalized_noise_input=True,
    progress_tqdm=None,
    second_order_correction=False,
    callback=None,
    call_every=10,
    t_steps=None,
):
    """
    CFG++ algorithm adapted to EDM sampler

    There are many ways to devise the second order correction

    Warning: noise is expected to be randn-like
    it'll be multiplied by sigma_max in this method
    if you want to use this


    Returns data BEFORE post processing
    """
    # sigma_min = max(sigma_min, self.sigma_min)
    # sigma_max = min(sigma_max, self.sigma_max)
    if progress_tqdm is None:
        progress_tqdm = tqdm
    # Time step discretization.
    if t_steps is None:
        t_steps = karras_scheduler(num_steps, sigma_min, sigma_max, rho, noise.device)

    # Main sampling loop.
    x_next = noise.to(torch.float64)
    # if input noise is normalized, we need to scale it by sigma_max
    if normalized_noise_input:
        x_next = x_next * t_steps[0]

    # compute concatenation of cond with nocond only once
    cond_batched = ldm._batch_cond_nocond(x_t=x_next, cond=cond)

    for i, (t_cur, t_next) in progress_tqdm(
        enumerate(zip(t_steps[:-1], t_steps[1:])), total=num_steps, desc="Sampling"
    ):  # 0, ..., N-1
        x_cur = x_next
        if callback is not None:
            callback(
                {
                    "x": x_cur,
                    "i": i,
                    "sigma": t_cur,
                    "sigma_hat": None,
                    "denoised": None,
                    "stage": "start",
                }
            )
        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        )

        t_hat = round_sigma(t_cur + gamma * t_cur)  # type: ignore
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * torch.randn_like(x_cur)

        # compute cond and unconditional denoised in one pass
        denoised_cond, denoised_nocond = ldm._batched_cond_denoise(
            x_t=x_hat, t=t_hat, cond=None, cond_batched=cond_batched
        )
        delta_t = -(t_next - t_hat)
        epsilon_nocond = (x_hat - denoised_nocond) / t_hat
        denoised_lambda_diff = cfg * (denoised_cond - denoised_nocond)

        # ---- originally ----:
        # (unconditional model and CFG)
        # normal step without conditioning
        # x_next = x_hat - delta_t * espilon_nocond
        # or
        # x_next = x_hat - delta_t * espilon_lambda
        # for original CFG where
        # epsilon_lambda = (1 - cfg) * epsilon_nocond + cfg * epsilon_cond

        # ------ CFG++ --------
        # First formulation:
        # x_next = (t_hat - delta_t) * epsilon_nocond + denoised_lambda
        # with
        # denoised_lambda = (1 - cfg) * denoised_nocond + cfg * denoised_cond
        # delta_t = -(t_next - t_hat)
        #
        # Second formulation:
        # It is
        # should be equal to
        x_next = x_hat - delta_t * epsilon_nocond + denoised_lambda_diff

        # # Apply 2nd order correction.
        if i < num_steps - 1 and second_order_correction:
            denoised_cond, denoised_nocond = ldm._batched_cond_denoise(
                x_t=x_next,
                t=t_next,
                cond_batched=cond_batched,
                cond=None,
            )

            epsilon_nocond_prime = (x_next - denoised_nocond) / t_next
            denoised_lambda_diff_prime = cfg * (denoised_cond - denoised_nocond)

            x_next = (
                x_hat
                - delta_t * (0.5 * epsilon_nocond + 0.5 * epsilon_nocond_prime)
                + (0.5 * denoised_lambda_diff + 0.5 * denoised_lambda_diff_prime)
            )
            # x_next = (
            #     x_hat
            #     - delta_t * (0.5 * epsilon_nocond + 0.5 * epsilon_nocond_prime)
            #     + (1 * denoised_lambda_diff + 0 * denoised_lambda_diff_prime)
            # )
            if (callback is not None) and (
                (i % call_every) == 0 or (i == num_steps - 1)
            ):
                callback(t_step=i, x=x_next.float())

    return x_next.float()
