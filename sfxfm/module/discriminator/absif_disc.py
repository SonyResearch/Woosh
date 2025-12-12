"""AbsIF discriminator"""

import typing as tp

import torch
from torch import nn
from einops import rearrange
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from sfxfm.module.discriminator import Discriminator
from sfxfm.module.loss.spectral_losses import STFTParams


def get_2d_padding(
    kernel_size: tp.Tuple[int, int], dilation: tp.Tuple[int, int] = (1, 1)
):
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )


def get_1d_padding(kernel_size: int, dilation: int = 1):
    return (kernel_size - 1) * dilation // 2


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
        **kwargs
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
        # self.norm = get_norm_module(self.conv, causal=False, norm=norm, **norm_kwargs)
        # self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        # x = self.norm(x)
        return x


class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)

    def forward(self, x):
        x = self.conv(x)
        return x


class AbsIFDiscriminator(Discriminator):
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
    ):
        super().__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.stft_params = STFTParams(
            window_length=self.n_fft,
            hop_length=self.hop_length,
            window_type=None,
        )

        spec_channels = 2 * self.in_channels
        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(
                spec_channels,
                self.filters,
                kernel_size=kernel_size,
                padding=get_2d_padding(kernel_size),
                norm=norm,
            )
        )
        in_chs = min(self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * self.filters, max_filters)
            self.convs.append(
                NormConv2d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=(dilation, 1),
                    padding=get_2d_padding(kernel_size, (dilation, 1)),
                    norm=norm,
                )
            )
            in_chs = out_chs
        out_chs = min(
            (filters_scale ** (len(dilations) + 1)) * self.filters, max_filters
        )
        self.convs.append(
            NormConv2d(
                in_chs,
                out_chs,
                kernel_size=(kernel_size[0], kernel_size[0]),
                padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                norm=norm,
            )
        )
        self.conv_post = NormConv2d(
            out_chs,
            self.out_channels,
            kernel_size=(kernel_size[0], kernel_size[0]),
            padding=get_2d_padding((kernel_size[0], kernel_size[0])),
            norm=norm,
        )

        self.gn = nn.GroupNorm(
            num_groups=2,
            num_channels=2 * (n_fft // 2 + 1),
            affine=True
            # , track_running_stats=True
        )

    def forward(self, x: torch.Tensor):
        fmap = []
        z = self.absif_repr(x)

        z = rearrange(z, "b c w t -> b c t w")
        reshape = lambda x: rearrange(x, "b c t w -> b c w t")
        for layer in self.convs:
            z = layer(z)
            z = self.activation(z)
            fmap.append(reshape(z))
        z = self.conv_post(z)

        z = reshape(z)
        return z, fmap

    def absif_repr(self, x: torch.Tensor):
        assert x.size(1) == 1
        spectrogramme = torch.stft(
            x[:, 0],
            n_fft=self.stft_params["window_length"],
            hop_length=self.stft_params["hop_length"],
            win_length=self.stft_params["window_length"],
            center=False,
            return_complex=True,
        ).unsqueeze(1)

        # rotation = spectrogramme[:, :, :, :-1].conj() / (
        #     spectrogramme[:, :, :, :-1].abs() + 1e-8
        # )

        # # detach?
        # rotation = rotation.detach()

        rotation = spectrogramme[:, :, :, :-1].conj()
        absif = spectrogramme[:, :, :, 1:] * rotation

        absif = torch.cat([absif.real, absif.imag], dim=1)

        if self.gn is not None:
            absif = rearrange(absif, "b c w t -> b (c w) t")
            absif = self.gn(absif)
            absif = rearrange(absif, "b (c w) t -> b c w t", c=2)
        return absif


class AbsIFDiscriminator1d(nn.Module):
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
    """

    def __init__(
        self,
        filters: int = 128,
        in_channels: int = 1,
        out_channels: int = 1,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        max_filters: int = 1024,
        filters_scale: int = 1,
        kernel_size: int = 3,
        dilations: tp.List = [1, 2, 4],
        stride: int = 1,
        normalized: bool = True,
        norm: str = "weight_norm",
        activation: str = "LeakyReLU",
        activation_params: dict = {"negative_slope": 0.2},
    ):
        super().__init__()
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.stft_params = STFTParams(
            window_length=self.n_fft,
            hop_length=self.hop_length,
            window_type=None,
        )

        spec_channels = 2 * (n_fft // 2 + 1)
        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv1d(
                spec_channels,
                self.filters,
                kernel_size=kernel_size,
                padding=get_1d_padding(kernel_size),
                norm=norm,
            )
        )
        in_chs = min(filters_scale * self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * self.filters, max_filters)
            self.convs.append(
                NormConv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=get_1d_padding(kernel_size, dilation),
                    norm=norm,
                )
            )
            in_chs = out_chs
        out_chs = min(
            (filters_scale ** (len(dilations) + 1)) * self.filters, max_filters
        )
        self.convs.append(
            NormConv1d(
                in_chs,
                out_chs,
                kernel_size=kernel_size,
                padding=get_1d_padding(kernel_size),
                norm=norm,
            )
        )
        self.conv_post = NormConv1d(
            out_chs,
            self.out_channels,
            kernel_size=kernel_size,
            padding=get_1d_padding(kernel_size),
            norm=norm,
        )

    def forward(self, x: torch.Tensor):
        fmap = []
        z = self.absif_repr(x)

        z = rearrange(z, "b c w t -> b (c w) t")
        for layer in self.convs:
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)
        z = self.conv_post(z)
        return z, fmap

    def absif_repr(self, x: torch.Tensor):
        assert x.size(1) == 1
        spectrogramme = torch.stft(
            x[:, 0],
            n_fft=self.stft_params["window_length"],
            hop_length=self.stft_params["hop_length"],
            win_length=self.stft_params["window_length"],
            center=False,
            return_complex=True,
        ).unsqueeze(1)

        rotation = spectrogramme[:, :, :, :-1].conj() / (
            spectrogramme[:, :, :, :-1].abs() + 1e-8
        )

        # detach?
        rotation = rotation.detach()
        absif = spectrogramme[:, :, :, 1:] * rotation

        absif = torch.cat([absif.real, absif.imag], dim=1)

        return absif
