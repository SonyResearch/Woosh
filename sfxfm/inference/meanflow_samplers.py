"""
This is where the MeanFlow sampler is defined.

Alexandre Bittar, 2025
"""

from einops import rearrange
import numpy as np
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


def find_latent(
    model: LatentDiffusionModelMeanFlowPipeline,
    x,
    autoencoder,
    cond,
    num_iterations=100,
    cfg=3.0,
    renormalize=False,
    one_step=True,
    kl_reg=0.0,
    ac_reg=0.0,
):
    """
    Optimize noise to fit a certain condition
    """
    assert x.size(0) == 1, "Only batch size 1 is supported for now."
    x = x[:1, :, :].detach()
    batch_size = 2
    x_repeat = x.expand(batch_size, -1, -1)
    with torch.no_grad():
        audio_target = autoencoder.inverse(x_repeat)

    param = torch.nn.Parameter(torch.randn_like(x, device=x.device), requires_grad=True)

    ex_noise = param.expand(batch_size, -1, -1)
    t = torch.ones_like(ex_noise[:, 0, 0], device=x.device)
    # r = torch.zeros_like(ex_noise[:, 0, 0], device=x.device) + 0.05
    r = torch.zeros_like(ex_noise[:, 0, 0], device=x.device)
    s = torch.rand_like(ex_noise[:, 0, 0], device=x.device)

    t_b = t[:, None, None]
    r_b = r[:, None, None]
    s_b = s[:, None, None]

    # Use classifier free guidance
    cond["cfg"] = cfg * torch.ones((batch_size,), device=x.device)
    optimizer = torch.optim.Adam([param], lr=1e-2, betas=(0.9, 0.99))

    # Autocorrelation function for noise with delta lag
    # def autocorr(seq, max_lag=10):
    #     # seq: (batch, channels, time)
    #     seq = seq - seq.mean(dim=-1, keepdim=True)
    #     ac = []
    #     for lag in range(1, max_lag):
    #         ac_lag = (
    #             (seq[:, :, :-lag] * seq[:, :, lag:])
    #             .mean(dim=-1)
    #             .mean(dim=0)
    #             .mean(dim=0)
    #         )
    #         ac.append(ac_lag)
    #     return torch.stack(ac)

    # random lag
    def autocorr(seq, max_lag=10):
        # seq: (batch, channels, time)
        seq = seq - seq.mean(dim=-1, keepdim=True)
        ac = []
        for _ in range(max_lag):
            lag = np.random.randint(1, 50)
            ac_lag = (
                ((seq[:, :, :-lag] * seq[:, :, lag:]).mean(dim=-1) ** 2)
                .mean(dim=0)
                .mean(dim=0)
            )
            ac.append(ac_lag)
        return torch.stack(ac)

    for i in range(num_iterations):
        optimizer.zero_grad()

        noise = param.expand(batch_size, -1, -1)

        # renormalize at each step seems to work
        if renormalize:
            noise = (noise - noise.mean()) / (noise.std() + 1e-5)  # renormalize

        t = t.detach()
        r = r.detach()
        s = s.detach()
        t_b = t_b.detach()
        r_b = r_b.detach()
        s_b = s_b.detach()

        # 1-step prediction
        if one_step:
            z = (
                noise
                - (t_b - r_b)
                * model._denoise_dict_no_param(x_t=noise, t=t, r=r, cond=cond)["x_hat"]
            )

        # 2-step
        else:
            z_mid = (
                noise
                - (t_b - s_b)
                * model._denoise_dict_no_param(x_t=noise, t=t, r=s, cond=cond)["x_hat"]
            )
            z = (
                z_mid
                - (s_b - r_b)
                * model._denoise_dict_no_param(x_t=noise, t=s, r=r, cond=cond)["x_hat"]
            )

        # ---------- Losses -----------------
        # try audio based loss:

        # == Inpainting:
        # x_target = x_repeat.detach() * (1 - r_b) + r_b * torch.randn_like(noise)
        # main_loss = ((z - x_target) ** 2)[:, :, :250].mean()
        # == Reconstruction
        # x_target = x_repeat.detach() * (1 - r_b) + r_b * torch.randn_like(noise)
        # main_loss = ((z - x_target) ** 2).mean()

        # == Reconstruction on audio
        # audio = autoencoder.inverse(z)  #  b c t

        # # Spectrogram loss
        # spec_transform = torchaudio.transforms.Spectrogram(
        #     n_fft=2048, hop_length=512
        # ).to(audio.device)
        # # spec_audio = spec_transform(audio)
        # # spec_target = spec_transform(audio_target).detach()
        # # main_loss = ((spec_audio - spec_target) ** 2).mean()

        # spec_audio = spec_transform(audio - audio_target)
        # main_loss = ((spec_audio) ** 2).mean()

        # Melspectrogram loss
        # mel_transform = torchaudio.transforms.MelSpectrogram(
        #     sample_rate=48000, n_fft=2048, hop_length=512, n_mels=128
        # ).to(audio.device)

        # mel_audio = mel_transform(audio)
        # mel_target = mel_transform(audio_target)
        # melspec_loss = ((mel_audio - mel_target) ** 2).mean()
        # main_loss = melspec_loss

        # main_loss = ((audio - audio_target) ** 2).mean()

        # == RMS
        audio = autoencoder.inverse(z)  #  b c t

        a = audio
        normalize_audio = False
        window: int = 4800 * 2  # the window size for the audio to extract the envelope
        hop_length: int = 480
        if normalize_audio:
            a = a / torch.clamp(
                a.max(dim=-1, keepdim=True).values, min=10e-3
            )  # avoid division by zero

        a2 = a**2

        rms = torch.nn.functional.avg_pool1d(
            a2,
            window,
            hop_length,
            padding=window // 2,
        )

        # if normalize_energy:
        #     rms = rms / torch.clamp(
        #         rms.max(dim=-1, keepdim=True).values, min=10e-3
        #     )  # avoid division by zero
        rms = rms.sqrt()
        rms = 20 * torch.log10(rms + 1e-8)

        # going up
        # target_rms = torch.arange(-60.0, -10.0, 50.0 / rms.size(-1), device=rms.device)[
        #     None, None, :
        # ]

        # Up and down envelope shaping
        # Create an envelope that ramps up then down
        envelope = torch.linspace(0, 1, rms.size(-1) // 2 + 1, device=rms.device)
        envelope = torch.cat(
            [envelope, envelope.flip(0)],
            dim=0,
        )[: rms.size(-1)]
        envelope = envelope[None, None, :]

        target_rms = -60.0 + 50.0 * envelope

        main_loss = ((rms - target_rms) ** 2).mean()

        # --------------- Regularization terms ----------------
        # KL divergence between noise and centered normal distribution
        noise_reg = rearrange(noise, "b c t -> (b t) c")
        sigma = noise_reg.std(dim=0)
        mu = noise_reg.mean(dim=0)
        kl_div = 0.5 * ((sigma**2 + mu**2) - 1 - sigma.log() * 2).mean()

        ac_noise = autocorr(noise, max_lag=10).mean()

        loss = main_loss + ac_reg * ac_noise + kl_reg * kl_div
        # loss = main_loss + 0.01 * kl_div
        # loss = (
        #     main_loss + 0.1 * (noise.std() - 1) ** 2 + 0.1 * noise.mean() ** 2
        #     # + 0.1 * ((z - x_target).abs().mean())
        # )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)
        optimizer.step()

        print(
            f"Iter {i}, loss: {loss.item():.4f}, main_loss: {main_loss.item():.4f}, kl_div: {kl_div.item():.4f}, ac: {ac_noise.item():.4f}"
        )

    return noise[:1]
