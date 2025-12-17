"""
This is where the FlowMap sampler is defined for denoising using
the distilled model.
"""

import torch
from sfxfm.model.ldm import LatentDiffusionModelFlowMapPipeline


@torch.inference_mode()
def sample_euler(
    model: LatentDiffusionModelFlowMapPipeline,
    noise,
    cond,
    num_steps=4,
    renoise=0.0,
    cfg=4.0,
):
    """
    Sampling with Euler integration. Input and output tensors have shape (batch, feats, steps).

    Arguments:
    -----------
    model: LatentDiffusionModelFlowMapPipeline
        diffusion model to sample from.
    noise: torch.Tensor
        initial Gaussian noise tensor to start denoising from.
    cond: Dict
        conditioning dictionary for the model containing audio description.
    num_steps: int
        number of denoising steps (best values should be 4-8).
    renoise: float or list
        amount of noise to add at each step, each value must be in [0, 1].
    cfg: float
        classifier-free guidance scale (between 0 and 9).
    """
    device = noise.device
    batch_size = noise.size(0)
    cond["cfg"] = cfg * torch.ones((batch_size,), device=device)

    # Define renoise schedule
    if isinstance(renoise, (float, int)):
        renoise_schedule = [renoise] * num_steps
    elif isinstance(renoise, (list, tuple)) and len(renoise) == num_steps:
        renoise_schedule = renoise
    else:
        raise TypeError("renoise must be a float or a list with num_steps values.")

    # Define linear step schedule and reshape as (batch_size, num_steps + 1)
    t_vals = torch.linspace(1, 0, num_steps + 1)
    t_vals = t_vals.unsqueeze(0).repeat(batch_size, 1).to(device)

    # Denoising steps using Euler
    for i in range(num_steps):
        t, r = t_vals[:, i], t_vals[:, i + 1]
        renoise_i = renoise_schedule[i]

        # Increase noise temporarily.
        if renoise_i > 0:
            gamma = renoise_i * (t - r)
            t_hat = torch.clamp(t + gamma, max=1.0)

            scale_ = (1 - t_hat) / (1 - t + 1e-12)
            std_ = (t_hat**2 - (t * scale_) ** 2).sqrt()[:, None, None]
            new_noise = scale_[:, None, None] * noise + std_ * torch.randn_like(noise)

            # Only renoise for t_hat > t (otherwise we lose the original noise when t=t_hat=1)
            mask_ = t_hat > t
            noise = torch.where(mask_[:, None, None], new_noise, noise)
            t = torch.where(mask_, t_hat, t)

        u = model._denoise_dict_no_param(x_t=noise, t=t, r=r, cond=cond)["x_hat"]
        noise = noise - (t - r)[:, None, None] * u

    return noise
