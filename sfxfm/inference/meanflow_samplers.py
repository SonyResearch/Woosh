"""
This is where the MeanFlow sampler is defined.

Alexandre Bittar, 2025
"""

from sfxfm.model.ldm import LatentDiffusionModelMeanFlowPipeline
import torch


@torch.inference_mode()
def sample_euler(
    model: LatentDiffusionModelMeanFlowPipeline,
    noise,
    cond,
    num_steps=4,
    step_schedule="linear",
    renoise=0.0,
    cfg=3.0,
):
    """
    Sampling using Euler integration with dynamic step size (step scheduler).
    Input and output tensors have shape (B, F, T).
    """
    device = noise.device
    batch_size = noise.size(0)

    # Define renoise schedule
    if isinstance(renoise, (float, int)):
        renoise_schedule = [renoise] * num_steps
    elif hasattr(renoise, "__len__") and len(renoise) == num_steps:
        renoise_schedule = renoise
    else:
        raise TypeError("renoise must be a float or a list with num_steps values.")

    # Define step schedule
    if step_schedule == "linear":
        t_vals = torch.linspace(1, 0, num_steps + 1)
    elif hasattr(step_schedule, "__len__") and len(step_schedule) == num_steps + 1:
        t_vals = torch.tensor(step_schedule)
    else:
        raise NotImplementedError(
            f"step_schedule must be linear, or a list of num_steps+1 values got {step_schedule}."
        )

    # Reshape as (batch_size, num_steps + 1)
    t_vals = t_vals.unsqueeze(0).repeat(batch_size, 1).to(device)

    # Use classifier free guidance
    cond["cfg"] = cfg * torch.ones((batch_size,), device=device)

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
