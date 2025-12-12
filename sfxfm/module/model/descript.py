import math
from typing import Callable, List, Optional


import numpy as np
import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from sfxfm.module.model.autoencoder import AutoEncoder
import logging

from sfxfm.module.model.blocks import FourierFeatures
from .updown_activation import UpDown1d
from sfxfm.utils.dist import rank

rank = rank()

# get logger
log = logging.getLogger(__name__)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        try:
            nn.init.constant_(m.bias, 0)
        except:
            pass


class ResidualUnit(nn.Module):
    def __init__(
        self,
        dim: int = 16,
        dilation: int = 1,
        bias: bool = True,
        updown_activation: bool = False,
    ):
        """
        Only bias in the last conv
        """
        super().__init__()
        pad = ((7 - 1) * dilation) // 2

        self.block = nn.Sequential(
            UpDown1d(Snake1d(dim)) if updown_activation else Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            UpDown1d(Snake1d(dim)) if updown_activation else Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1, bias=bias),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1, bias: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1, bias=bias),
            ResidualUnit(dim // 2, dilation=3, bias=bias),
            ResidualUnit(dim // 2, dilation=9, bias=bias),
            Snake1d(dim // 2) if bias else nn.Identity(),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                bias=bias,
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
        bias: bool = True,
        use_fourier_features: bool = False,
    ):
        super().__init__()

        embedding = (
            FourierFeatures(1, d_model)
            if use_fourier_features
            else WNConv1d(1, d_model, kernel_size=7, padding=3)
        )
        # Create first convolution
        self.block = [embedding]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride, bias=bias)]

        # Create last convolution
        self.block += [
            Snake1d(d_model) if bias else nn.Identity(),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1, bias=bias),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        bias: bool = True,
        updown_activation: bool = False,
    ):
        super().__init__()

        if bias:
            act = (
                UpDown1d(Snake1d(input_dim))
                if updown_activation
                else Snake1d(input_dim)
            )
        else:
            act = nn.Identity()

        self.block = nn.Sequential(
            act,
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                bias=bias,
            ),
            ResidualUnit(
                output_dim,
                dilation=1,
                bias=bias,
                updown_activation=updown_activation,
            ),
            ResidualUnit(
                output_dim,
                dilation=3,
                bias=bias,
                updown_activation=updown_activation,
            ),
            ResidualUnit(
                output_dim,
                dilation=9,
                bias=bias,
                updown_activation=updown_activation,
            ),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
        bias: bool = True,
        updown_activation: bool = False,
    ):
        super().__init__()

        # Add first conv layer
        layers = [
            WNConv1d(input_channel, channels, kernel_size=7, padding=3, bias=bias)
        ]

        # Add upsampling + MRF blocks
        assert len(rates) > 0
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [
                DecoderBlock(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    stride=stride,
                    bias=bias,
                    updown_activation=updown_activation,
                )
            ]

        # Add final conv layer
        if bias:
            act = (
                UpDown1d(Snake1d(output_dim))
                if updown_activation
                else Snake1d(output_dim)
            )
        else:
            act = nn.Identity()
        layers += [
            act,
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3, bias=bias),
            # nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DACAutoEncoder(AutoEncoder):
    def __init__(
        self,
        channels: int = 1,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        z_dim: Optional[int] = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        bias: bool = True,
        latent_noise: float = 0,
        updown_activation: bool = False,
        use_fourier_features: bool = False,
    ) -> None:
        """AutoEncoder from Descript (without quantization)

        Args:
            channels (int, optional): number of input channels.
            Defaults to 1.
            encoder_dim (int, optional): base dim for encoder.
            Defaults to 64.
            encoder_rates (List[int], optional): list of downscaling rates.
            Defaults to [2, 4, 8, 8].
            z_dim (Optional[int], optional): list of upscaling rates.. Defaults to None.
            decoder_dim (int, optional): base dim for decoder (max dim at the beginning, right after latent dim) Defaults to 1536.
            decoder_rates (List[int], optional): _description_. Defaults to [8, 8, 4, 2].
        """
        assert channels == 1
        if z_dim is None:
            z_dim = encoder_dim * (2 ** len(encoder_rates))
        assert type(z_dim) == int

        self.encoder_rates = encoder_rates

        encoder = Encoder(
            d_model=encoder_dim,
            strides=encoder_rates,
            d_latent=z_dim,
            bias=bias,
            use_fourier_features=use_fourier_features,
        )

        decoder = Decoder(
            input_channel=z_dim,
            channels=decoder_dim,
            rates=decoder_rates,
            bias=bias,
            updown_activation=updown_activation,
        )

        super().__init__(encoder=encoder, decoder=decoder, latent_noise=latent_noise)
        if rank == 0:
            log.info(
                f"""Using DACAutoEncoder:
                     time_downscaling: {np.prod(encoder_rates)}
                     num_channels_in: {channels}
                     z_dim: {z_dim}
                     Total compression factor {np.prod(encoder_rates) / z_dim * channels}
                     """
            )

    def fix_input_length(self, x):
        """
        the input samples should be a multiple of the product of downsampling factors
        """
        assert len(x.shape) == 3
        down_factor = np.prod(self.encoder_rates)
        # make sure the input length is a multiple of the overall downsampling factor
        x = x[:, :, : x.shape[2] - (x.shape[2] % down_factor)]
        return x.contiguous()
