# Naive CFG
import torch
from torchdiffeq import odeint

from woosh.model.ldm import LatentDiffusionModel


def flowmatching_integrate(
    ldm: LatentDiffusionModel,
    noise: torch.Tensor,
    cond: dict,
    cond_neg: dict = None,
    negative_text_only: bool = True,
    cfg: float = 4.5,
    device: str = "cuda",
    method="dopri5",
    rtol=1e-3,
    atol=1e-3,
    return_steps=False,
    **fm_kwargs,
):
    """
    Integrate a flow-matching trajectory using ODE-based sampling with classifier-free guidance (CFG).

    Runs numerical ODE integration from noise (t=0) to data (t=1) using the provided latent
    diffusion model as the velocity field. Supports optional negative conditioning for CFG,
    including a mode that replaces only the text cross-attention conditioning.

    Returns data BEFORE post processing (i.e., in latent space).

    Args:
        ldm (LatentDiffusionModel): The latent diffusion model used to predict the denoised output
            at each ODE step.
        noise (torch.Tensor): Initial noise tensor of shape ``(batch_size, *latent_shape)``
            from which integration begins.
        cond (dict): Conditioning dictionary containing the positive conditioning signals
            (e.g., text embeddings, audio features) passed to the model.
        cond_neg (dict, optional): Negative conditioning dictionary used for CFG. If ``None``,
            an unconditional (dropout) forward pass is used as the negative condition.
            Defaults to ``None``.
        negative_text_only (bool, optional): If ``True`` and ``cond_neg`` is provided, only the
            cross-attention text conditioning is replaced with the negative counterpart, leaving
            other conditioning signals from the unconditional pass intact. Defaults to ``False``.
        cfg (float, optional): Classifier-free guidance scale. The guidance is applied as
            ``pred = pred_cond + cfg * (pred_cond - pred_uncond)``. Defaults to ``4.5``.
        device (str, optional): Device on which to run conditioning computation. Defaults to
            ``"cuda"``.
        method (str, optional): ODE solver method. Defaults to ``"dopri5"``.
        rtol (float, optional): Relative tolerance for ODE solver. Defaults to ``1e-3``.
        atol (float, optional): Absolute tolerance for ODE solver. Defaults to ``1e-3``.
        **fm_kwargs: Additional keyword arguments forwarded to ``torchdiffeq.odeint``.

    Returns:
        tuple[torch.Tensor, int]:
            - **fakes** (``torch.Tensor``): Integrated output tensor of the same shape as
              ``noise``, representing the generated latent samples.
            - **steps** (``int``): Total number of ODE function evaluations performed.

    Example:
        >>> fakes, n_steps = flowmatching_integrate(
        ...     ldm, noise, cond, method='dopri5', rtol=1e-5, atol=1e-5
        ... )
    """

    batch_size = noise.size(0)

    no_cond = ldm.get_cond(
        {"audio": noise, **cond},
        no_dropout=True,
        device=device,  # we must provide device if we don't provide x
        no_cond=True,
    )
    if cond_neg is not None:
        if negative_text_only:
            # only replace text conditioning
            # print("Using negative TEXT-ONLY conditioning")
            no_cond["cross_attn_cond"] = cond_neg["cross_attn_cond"]
            no_cond["cross_attn_cond_mask"] = cond_neg["cross_attn_cond_mask"]
        else:
            # print("Using negative conditioning")
            no_cond = cond_neg
    step = 0

    def f(t, y):
        c = cond
        nc = no_cond
        nonlocal step
        step += 1

        res = ldm._denoise_dict_no_param(y, 1 - t, c)["x_hat"]
        res_nc = ldm._denoise_dict_no_param(y, 1 - t, nc)["x_hat"]

        res = res + cfg * (res - res_nc)
        return res

    # t = [0, 1]
    t = torch.linspace(0, 1, steps=2, device=noise.device)

    fakes = odeint(f, noise, t, atol=atol, rtol=rtol, method=method, options=fm_kwargs)[
        -1
    ]

    # print(f"Integrating finished in {step + 1} steps")
    if return_steps:
        return fakes, step + 1
    return fakes
