"""
RAVE-like autoencoder.
Decoder has a noise generator
The rest is like the DACAutoencoder 
"""

from typing import Callable, List, Optional
import torch
from torch import nn
import math
from sfxfm.utils.hashing import log_dict
import numpy as np
from sfxfm.module.model.autoencoder import AutoEncoder

from sfxfm.module.model.descript import DecoderBlock, Encoder, Snake1d, WNConv1d
from sfxfm.module.model.updown_activation import UpDown1d
import logging
from sfxfm.utils.dist import rank

rank = rank()

# get logger
log = logging.getLogger(__name__)


def mod_sigmoid(x):
    return 2 * torch.sigmoid(x) ** 2.3 + 1e-7


def amp_to_impulse_response(amp, target_size):
    """
    transforms frequency amps to ir on the last dimension
    """
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = torch.fft.irfft(amp)

    filter_size = amp.shape[-1]

    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

    amp = amp * win

    amp = nn.functional.pad(
        amp,
        (0, int(target_size) - int(filter_size)),
    )
    amp = torch.roll(amp, -filter_size // 2, -1)

    return amp


def fft_convolve(signal, kernel):
    """
    convolves signal by kernel on the last dimension
    """
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = torch.fft.irfft(torch.fft.rfft(signal) * torch.fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2 :]

    return output


class NoiseGeneratorV2(nn.Module):
    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        data_size: int,
        ratios: List[int],
        noise_bands: int,
        n_channels: int = 1,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(0.2),
    ):
        super().__init__()
        net = []
        self.n_channels = n_channels
        channels = [in_size]
        channels.extend((len(ratios) - 1) * [hidden_size])
        channels.append(data_size * noise_bands * n_channels)

        for i, r in enumerate(ratios):
            net.append(
                WNConv1d(
                    channels[i],
                    channels[i + 1],
                    2 * r,
                    padding=math.ceil(r / 2),
                    stride=r,
                )
            )
            if i != len(ratios) - 1:
                net.append(activation(channels[i + 1]))

        self.net = nn.Sequential(*net)
        self.data_size = data_size

        self.register_buffer(
            "target_size",
            torch.tensor(np.prod(ratios)).long(),
        )

    def forward(self, x):
        amp = mod_sigmoid(self.net(x) - 5)
        amp = amp.permute(0, 2, 1)
        amp = amp.reshape(
            amp.shape[0], amp.shape[1], self.n_channels * self.data_size, -1
        )

        ir = amp_to_impulse_response(amp, self.target_size)
        noise = torch.rand_like(ir) * 2 - 1

        noise = fft_convolve(noise, ir).permute(0, 2, 1, 3)
        noise = noise.reshape(noise.shape[0], noise.shape[1], -1)
        return noise


class RAVEDecoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
        bias: bool = True,
        updown_activation: bool = False,
        noise_generator_hidden_size: int = 128,
        noise_generator_ratios: List[int] = [2, 2, 2],
        noise_bands: int = 5,
    ):
        """Like Descript Decoder but with a NoiseGeneratorV2 layer and amplitude modification to generate the waveform"""
        super().__init__()

        # Add first conv layer
        layers = [
            WNConv1d(input_channel, channels, kernel_size=7, padding=3, bias=bias)
        ]

        # Add upsampling + MRF blocks
        assert len(rates) > 0
        output_dim = -1
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
        assert output_dim > 0

        # Add final conv layer
        if bias:
            act = (
                UpDown1d(Snake1d(output_dim))
                if updown_activation
                else Snake1d(output_dim)
            )
        else:
            act = nn.Identity()
        layers += [act]

        self.model = nn.Sequential(*layers)

        self.noise = NoiseGeneratorV2(
            output_dim,
            hidden_size=noise_generator_hidden_size,
            data_size=d_out,
            ratios=noise_generator_ratios,
            noise_bands=noise_bands,
        )

        self.to_waveform = WNConv1d(
            output_dim, d_out * 2, kernel_size=7, padding=3, bias=bias
        )

    def forward(self, x):
        h = self.model(x)

        noise = self.noise(h)
        y, amp = self.to_waveform(h).chunk(2, dim=1)

        return torch.tanh(y * torch.sigmoid(amp) + noise)


class RAVEAutoEncoder(AutoEncoder):
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
        noise_generator_hidden_size: int = 128,
        noise_generator_ratios: List[int] = [2, 2, 2],
        noise_bands: int = 5,
        use_fourier_features: bool = False,
    ) -> None:
        """AutoEncoder from Descript (without quantization)
        with RAVE decodder's final layer

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

        decoder = RAVEDecoder(
            input_channel=z_dim,
            channels=decoder_dim,
            rates=decoder_rates,
            bias=bias,
            updown_activation=updown_activation,
            noise_generator_hidden_size=noise_generator_hidden_size,
            noise_generator_ratios=noise_generator_ratios,
            noise_bands=noise_bands,
        )

        super().__init__(encoder=encoder, decoder=decoder, latent_noise=latent_noise)
        if rank == 0:
            log.info(f"Using RAVEAutoEncoder:")
            log.info(f"  time_downscaling: {np.prod(encoder_rates)}")
            log.info(f"  num_channels_in: {channels}")
            log.info(f"  z_dim: {z_dim}")
            log.info(
                f"  compression factor: {np.prod(encoder_rates) / z_dim * channels}"
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
