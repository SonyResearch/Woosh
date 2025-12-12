# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""MS-STFT discriminator, provided here for reference."""

import typing as tp

import torchaudio
import torch
from torch import nn
from einops import rearrange
from torch.nn.utils.parametrizations import spectral_norm, weight_norm
from sfxfm.utils.audio import find_max_mel_bands, find_max_bark_bands, barkscale_fbanks
from ..model.filter import FreqPreemphasis


def get_2d_padding(
    kernel_size: tp.Tuple[int, int], dilation: tp.Tuple[int, int] = (1, 1)
):
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )


def apply_parametrization_norm(module: nn.Module, norm: str = "none") -> nn.Module:
    # assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        return weight_norm(module)
    elif norm == "spectral_norm":
        return spectral_norm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module


class NormConv2d(nn.Module):
    """Wrapper around Conv2d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
        # self.norm = get_norm_module(self.conv, causal=False, norm=norm, **norm_kwargs)
        # self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        # x = self.norm(x)
        return x


class ComplexSTFTDiscriminator(nn.Module):
    """STFT sub-discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_fft (int): Size of FFT for each scale. Default: 1024
        hop_length (int): Length of hop between STFT windows for each scale. Default: 256
        kernel_size (tuple of int): Inner Conv2d kernel sizes. Default: ``(3, 9)``
        stride (tuple of int): Inner Conv2d strides. Default: ``(1, 2)``
        dilations (list of int): Inner Conv2d dilation on the time dimension. Default: ``[1, 2, 4]``
        win_length (int): Window size for each scale. Default: 1024
        normalized (bool): Whether to normalize by magnitude after stft. Default: True
        norm (str): Normalization method. Default: `'weight_norm'`
        activation (str): Activation function. Default: `'LeakyReLU'`
        activation_params (dict): Parameters to provide to the activation function.
        growth (int): Growth factor for the filters. Default: 1

        kernel_size=(kernel_size_time, kernel_size_freq) and stride=(stride_time,stride_freq)
        dilation is a list on dilation factors for each conv layer in the time axis
        skip_n_first_fmaps (int): Do not return the n first fmaps, 0 returns all feature maps, even the one following the first Conv
    """

    def __init__(
        self,
        filters: int = 32,
        in_channels: int = 1,
        out_channels: int = 1,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        max_filters: int = 1024,
        filters_scale: int = 1,
        kernel_size: tp.Tuple[int, int] = (3, 9),
        dilations: tp.List = [1, 2, 4],
        stride: tp.Tuple[int, int] = (1, 2),
        normalized: bool = True,
        norm: str = "weight_norm",
        activation: str = "LeakyReLU",
        activation_params: dict = {"negative_slope": 0.2},
        power: int = 1,
        n_bands: int = 1,
        log_scale: bool = True,
        epsilon_log: float = 1e-5,
        fbank: str = None,
        fbank_filters: int = None,
        sample_rate: int = None,
        preemphasis: FreqPreemphasis = None,
        skip_n_first_fmaps: int = 0,
    ):
        super().__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        assert power in [None, 1, 2]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters = filters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_filters = max_filters
        self.filters_scale = filters_scale
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.stride = stride
        self.normalized = normalized
        self.norm = norm
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.n_bands = n_bands
        self.power = power
        self.log_scale = log_scale
        self.epsilon_log = epsilon_log
        self.spec_channels = (
            2 * self.in_channels if self.power is None else self.in_channels
        )
        # Only the power = None version (complex spectrogram) is implemented
        assert self.power is None

        self.fbank = fbank
        self.fbank_filters = fbank_filters
        self.sample_rate = sample_rate

        self.init_spec()
        self.band_convs = nn.ModuleList([self.convs_() for _ in range(self.n_bands)])
        out_chs = min(
            (self.filters_scale ** (len(self.dilations) + 1)) * self.filters,
            self.max_filters,
        )
        self.conv_post = NormConv2d(
            out_chs,
            self.out_channels,
            kernel_size=(kernel_size[0], kernel_size[0]),
            padding=get_2d_padding((kernel_size[0], kernel_size[0])),
            norm=norm,
        )

        self.preemphasis = (
            preemphasis(n_fft=self.n_fft) if preemphasis is not None else None
        )
        self.skip_n_first_fmaps = skip_n_first_fmaps

    def spec_transform(self, x):
        # Only the power = None version (complex spectrogram) is implemented
        assert self.power is None
        # Only mono signals
        assert x.size(1) == 1
        return torch.stft(
            x[:, 0],
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False,
            return_complex=True,
            window=torch.hann_window(self.win_length, device=x.device),
        ).unsqueeze(1)

    def init_spec(self):
        self.fbank_transform = None

        if self.fbank is not None:
            if self.fbank == "linear":
                # linear-scaled filterbank
                n_linears = self.fbank_filters
                if n_linears is not None:
                    # limit fbank_filters to n_fft//2+1
                    n_linears = (
                        n_linears
                        if n_linears <= self.n_fft // 2 + 1
                        else self.n_fft // 2 + 1
                    )
                else:
                    # force no default fbank filters
                    raise ValueError(
                        "fbank_filters needs to be specified for fbank=='linear'"
                    )
                self.fbank_transform = torchaudio.functional.linear_fbanks(
                    n_freqs=self.n_fft // 2 + 1,
                    f_min=0,
                    f_max=self.sample_rate / 2,
                    n_filter=n_linears,
                    sample_rate=self.sample_rate,
                )

            elif self.fbank == "mel":
                # mel-scaled filterbank
                n_mels = self.fbank_filters
                n_mels_max = find_max_mel_bands(
                    self.n_fft,
                    self.sample_rate,
                    mel_scale="slaney",
                )
                if n_mels is not None:
                    # limit fbank_filters to max mel filters given fft size
                    n_mels = n_mels if n_mels <= n_mels_max else n_mels_max
                else:
                    n_mels = n_mels_max

                self.fbank_transform = torchaudio.functional.melscale_fbanks(
                    n_freqs=self.n_fft // 2 + 1,
                    f_min=0,
                    f_max=self.sample_rate / 2,
                    n_mels=n_mels,
                    sample_rate=self.sample_rate,
                    mel_scale="slaney",
                )

            elif self.fbank == "bark":
                # bark-scaled filterbank
                n_barks = self.fbank_filters
                n_barks_max = find_max_bark_bands(
                    self.n_fft,
                    self.sample_rate,
                    bark_scale="traunmuller",
                )
                if n_barks is not None:
                    # limit fbank_filters to max mel filters given fft size
                    n_barks = n_barks if n_barks <= n_barks_max else n_barks_max
                else:
                    n_barks = n_barks_max

                self.fbank_transform = barkscale_fbanks(
                    n_freqs=self.n_fft // 2 + 1,
                    f_min=0,
                    f_max=self.sample_rate / 2,
                    n_barks=n_barks,
                    sample_rate=self.sample_rate,
                    bark_scale="traunmuller",
                )

    def compute_spec(self, x: torch.Tensor):
        z = self.spec_transform(x)  # [B, 1, F, T]
        if self.preemphasis is not None:
            z = self.preemphasis(z)
        z = rearrange(z, "b c w t -> b c t w")  # [B,1,T,F]

        # if z is complex
        if self.power is None:
            if self.fbank_transform is not None:
                # move fbank_transform to z.device
                if self.fbank_transform.device != z.device:
                    self.fbank_transform = self.fbank_transform.to(z.device)

                # transform each fft in stft using fbank_transform
                # z = [B,1,T,FB]
                z = torch.complex(
                    torch.matmul(z.real, self.fbank_transform),
                    torch.matmul(z.imag, self.fbank_transform),
                )

            if self.log_scale:
                # compute complex log for (r,i)=r*exp(phi) as log(r)*exp(phi)
                # simply to have log magnitude and keep complex values
                with torch.no_grad():
                    logr = torch.log(self.epsilon_log + (z * z.conj()).real) / 2

                    phi = torch.atan2(z.imag, z.real)
                    z = torch.complex(
                        logr * torch.cos(phi),
                        logr * torch.sin(phi),
                    )
            # concat real and imaginary parts as two different channels: [B, 2, T, F]
            z = torch.cat([z.real, z.imag], dim=1)

        else:
            if self.fbank_transform is not None:
                if self.fbank_transform.device != z.device:
                    self.fbank_transform = self.fbank_transform.to(z.device)

                z = torch.matmul(z, self.fbank_transform)

            if self.log_scale:
                z = torch.log(z + self.epsilon_log)

        return z

    def convs_(self):
        convs = []
        convs.append(
            NormConv2d(
                self.spec_channels,
                self.filters,
                kernel_size=self.kernel_size,
                padding=get_2d_padding(self.kernel_size),
            )
        )
        in_chs = min(self.filters_scale * self.filters, self.max_filters)
        for i, dilation in enumerate(self.dilations):
            out_chs = min(
                (self.filters_scale ** (i + 1)) * self.filters, self.max_filters
            )
            convs.append(
                NormConv2d(
                    in_chs,
                    out_chs,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    dilation=(dilation, 1),
                    padding=get_2d_padding(self.kernel_size, (dilation, 1)),
                    norm=self.norm,
                )
            )
            in_chs = out_chs
        out_chs = min(
            (self.filters_scale ** (len(self.dilations) + 1)) * self.filters,
            self.max_filters,
        )

        convs.append(
            NormConv2d(
                in_chs,
                out_chs,
                kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                padding=get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
                norm=self.norm,
            )
        )

        return nn.ModuleList(convs)

    def forward(self, x: torch.Tensor):
        fmap = []

        z = self.compute_spec(x)
        # The network acts on (b, c, t, f) tensors
        # reshape for outputs only
        reshape = lambda x: rearrange(x, "b c t f -> b c f t")

        z_bands = torch.chunk(z, self.n_bands, dim=-1)

        b = []
        for band, stack in zip(z_bands, self.band_convs):
            for k, layer in enumerate(stack):
                band = layer(band)
                # only use fmaps after MLP
                if k >= self.skip_n_first_fmaps:
                    fmap.append(reshape(band))
                band = self.activation(band)
            b.append(reshape(band))

        b = torch.cat(b, dim=-1)
        b = self.conv_post(b)
        # fmap.append(b)

        return b, fmap
