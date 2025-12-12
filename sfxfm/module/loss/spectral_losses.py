from einops import rearrange
import torch
from torch import nn
import typing as tp
import torchaudio


def get_window(window_type, window_length):
    if window_type == "rectangular":
        window = torch.ones(window_length)
    elif window_type == "hann":
        window = torch.hann_window(window_length)
    elif window_type == "sqrt_hann":
        window = torch.hann_window(window_length).pow(0.5)
    else:
        raise NotImplementedError(
            f"""Wrong argument in get_window: allowed window_type values are rectangular | hann | sqrt_hann, got {window_type} """
        )
    return window


class STFTParams(tp.TypedDict):
    window_length: int
    hop_length: int
    window_type: str


class MultiScaleSTFTLoss(nn.Module):
    """Computes the multi-scale STFT loss from [1].

    Parameters
    ----------
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    References
    ----------

    1.  Engel, Jesse, Chenjie Gu, and Adam Roberts.
        "DDSP: Differentiable Digital Signal Processing."
        International Conference on Learning Representations. 2019.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        window_lengths: tp.List[int] = [2048, 512],
        loss_fn: tp.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        weight: float = 1.0,
        window_type: str = "rectangular",  # rectangular, hann, sqrt_hann
        center: bool = False,
    ):
        super().__init__()
        self.window_lengths = window_lengths
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.loss_fn = loss_fn
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.weight = weight
        self.center = center

        for window_length in window_lengths:
            self.register_buffer(
                f"window_{window_length}", get_window(window_type, window_length)
            )

    @property
    def windows(self):
        return [
            self.get_buffer(f"window_{window_length}")
            for window_length in self.window_lengths
        ]

    def forward(self, x, y):
        """Computes multi-scale STFT between an estimate and a reference
        signal.

        Parameters
        ----------
        x : Estimate signal
        y : Reference signal

        Returns
        -------
        torch.Tensor
            Multi-scale STFT loss.
        """
        loss = 0.0
        # removes channel dim
        assert x.size(1) == 1

        x = x[:, 0]
        y = y[:, 0]

        for p, window in zip(self.stft_params, self.windows):
            x_spec = torch.stft(
                x,
                n_fft=p["window_length"],
                hop_length=p["hop_length"],
                win_length=p["window_length"],
                window=window,
                center=self.center,
                return_complex=True,
            )
            y_spec = torch.stft(
                y,
                n_fft=p["window_length"],
                hop_length=p["hop_length"],
                win_length=p["window_length"],
                window=window,
                center=self.center,
                return_complex=True,
            )

            if self.log_weight > 0:
                loss = loss + self.log_weight * self.loss_fn(
                    x_spec.abs().add(self.clamp_eps).log(),
                    y_spec.abs().add(self.clamp_eps).log(),
                )

            # if self.log_weight > 0:
            #     # 1e-5 in DAC
            #     # 1e-7 in Vocos
            #     loss = loss + self.log_weight * self.loss_fn(
            #         x_spec.abs().clamp(min=1e-5).log(),
            #         y_spec.abs().clamp(min=1e-5).log(),
            #     )

            if self.mag_weight > 0:
                # abs dist
                loss = loss + self.mag_weight * self.loss_fn(x_spec.abs(), y_spec.abs())

        return loss


class MelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [150, 80],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : tp.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        sample_rate,
        n_mels: tp.List[int] = [150, 80],
        window_lengths: tp.List[int] = [2048, 512],
        loss_fn: tp.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        weight: float = 1.0,
        mel_fmin: tp.List[float] = [0.0, 0.0],
        mel_fmax: tp.List[tp.Optional[float]] = [None, None],
        mel_scale: str = "slaney",  # slaney or htk, slaney is used by dac
        window_type: str = "hann",  # rectangular, hann, sqrt_hann
        mel_norm="slaney",
        center=True,
        pad_mode="reflect",
        pow: float = 1.0,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sample_rate = sample_rate
        self.mel_scale = mel_scale

        if window_type == "hann":
            window_fn = torch.hann_window
        elif window_type == "rectangular":
            window_fn = lambda window_length: torch.ones(window_length)  # noqa: E731
        elif window_type == "sqrt_hann":
            window_fn = lambda window_length: torch.hann_window(window_length).pow(0.5)  # noqa: E731
        else:
            raise NotImplementedError(
                f"""Wrong argument in window_type: allowed window_type values are rectangular | hann | sqrt_hann, got {window_type} """
            )
        self.mel_transforms = nn.ModuleList(
            [
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=w,
                    hop_length=w // 4,
                    f_min=fmin,
                    f_max=fmax,
                    n_mels=nmel,
                    window_fn=window_fn,
                    power=1,  # matching dac
                    center=center,
                    pad_mode=pad_mode,
                    norm=mel_norm,  # slaney matching dac
                    mel_scale=mel_scale,  # slaney matching dac
                )
                for nmel, w, fmin, fmax in zip(
                    n_mels, window_lengths, mel_fmin, mel_fmax
                )
            ]
        )
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.pow = pow

    def forward(self, x, y):
        """Computes mel loss between an estimate and a reference
        signal.

        Parameters
        ----------
        x : Estimate signal
        y : Reference signal

        Returns
        -------
        torch.Tensor
            Mel loss.
        """
        loss = 0.0
        loss = 0.0
        for mel in self.mel_transforms:
            x_mels = mel(x)
            y_mels = mel(y)

            loss += self.log_weight * self.loss_fn(
                x_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_mels, y_mels)

        return loss
