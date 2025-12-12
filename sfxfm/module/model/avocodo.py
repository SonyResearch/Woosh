from typing import List, Optional
import logging

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from torch.nn import Conv1d, ConvTranspose1d  # AvgPool1d, Conv2d
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.weight_norm import remove_weight_norm

from sfxfm.module.model.autoencoder import AutoEncoder
from sfxfm.module.model.descript import Encoder
from sfxfm.utils.dist import rank
from sfxfm.module.model.rave import NoiseGeneratorV2

rank = rank()

log = logging.getLogger(__name__)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.2)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.2)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for _l in self.convs1:
            remove_weight_norm(_l)
        for _l in self.convs2:
            remove_weight_norm(_l)


class AvocodoDecoder(torch.nn.Module):
    def __init__(
        self,
        input_dim=32,
        n_resblock: int = 1,
        upsample_rates: List[List[int]] = [[16], [8], [2], [2]],
        upsample_kernel_sizes: List[List[int]] = [[32], [16], [4], [4]],
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        projection_filters: List[int] = [0, 1, 1, 1],
        projection_kernels: List[int] = [0, 5, 7, 11],
        noise_module: bool = True,
        bias: bool = True,
    ):
        super(AvocodoDecoder, self).__init__()
        self.n_resblock = n_resblock
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(
            Conv1d(input_dim, upsample_initial_channel, 7, 1, padding=3)
        )
        self.projection_filters = projection_filters
        self.projection_kernels = projection_kernels
        self.noise_module = noise_module
        resblock = ResBlock

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            _ups = nn.ModuleList()
            for _i, (_u, _k) in enumerate(zip(u, k)):
                in_channel = upsample_initial_channel // (2**i)
                out_channel = upsample_initial_channel // (2 ** (i + 1))
                _ups.append(
                    weight_norm(
                        ConvTranspose1d(
                            in_channel, out_channel, _k, _u, padding=(_k - _u) // 2
                        )
                    )
                )
            self.ups.append(_ups)

        self.resblocks = nn.ModuleList()
        self.conv_post = nn.ModuleList()
        for i in range(self.num_upsamples):
            ch = upsample_initial_channel // (2 ** (i + 1))
            temp = nn.ModuleList()
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                temp.append(resblock(ch, k, d))
            self.resblocks.append(temp)

            if self.projection_filters[i] != 0:
                self.conv_post.append(
                    weight_norm(
                        Conv1d(
                            ch,
                            self.projection_filters[i],
                            self.projection_kernels[i],
                            1,
                            padding=self.projection_kernels[i] // 2,
                        )
                    )
                )
            else:
                self.conv_post.append(torch.nn.Identity())

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        output_dim = upsample_initial_channel // (2 ** (self.num_upsamples))

        if self.noise_module:
            self.noise = NoiseGeneratorV2(
                in_size=output_dim,
                hidden_size=128,
                data_size=1,
                ratios=[2, 2, 2],
                noise_bands=5,
                n_channels=1,
            )

            self.conv_post_noise = weight_norm(
                Conv1d(output_dim, 2, kernel_size=7, padding=3)
            )

    def forward(self, x):
        outs = []
        x = self.conv_pre(x)
        for i, (ups, resblocks, conv_post) in enumerate(
            zip(self.ups, self.resblocks, self.conv_post)
        ):
            x = F.leaky_relu(x, 0.2)
            for _ups in ups:
                x = _ups(x)
            xs = None
            for j, resblock in enumerate(resblocks):
                if xs is None:
                    xs = resblock(x)
                else:
                    xs += resblock(x)
            x = xs / self.num_kernels

            if i == self.num_upsamples - 1 and self.noise_module:
                noise = self.noise(x)
                y, amp = self.conv_post_noise(x).chunk(2, dim=1)
                _x = torch.tanh(y * torch.sigmoid(amp) + noise)
                outs.append(_x)
                return outs

            if i >= (self.num_upsamples - 3):
                _x = F.leaky_relu(x)
                _x = conv_post(_x)
                _x = torch.tanh(_x)
                outs.append(_x)
            else:
                x = conv_post(x)

        return outs

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for ups in self.ups:
            for _l in ups:
                remove_weight_norm(_l)
        for resblock in self.resblocks:
            for _l in resblock:
                _l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        for _l in self.conv_post:
            if not isinstance(_l, torch.nn.Identity):
                remove_weight_norm(_l)


class AvocodoAutoEncoder(AutoEncoder):
    def __init__(
        self,
        channels: int = 1,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        z_dim: Optional[int] = None,
        decoder_dim: int = 512,
        decoder_rates: List[List[int]] = [[8], [8], [4], [2]],
        bias: bool = True,
        latent_noise: float = 0,
        use_fourier_features: bool = False,
        n_resblock: int = 1,
        upsample_kernel_sizes: List[List[int]] = [[16], [16], [4], [4]],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        projection_filters: List[int] = [0, 1, 1, 1],
        projection_kernels: List[int] = [0, 5, 7, 11],
        noise_module: bool = True,
    ) -> None:
        """AutoEncoder from Avocodo
        Descript Encoder
        Avocodo Decoder

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
        assert isinstance(z_dim, int)

        encoder = Encoder(
            d_model=encoder_dim,
            strides=encoder_rates,
            d_latent=z_dim,
            bias=bias,
            use_fourier_features=use_fourier_features,
        )

        decoder = AvocodoDecoder(
            input_dim=z_dim,
            n_resblock=n_resblock,
            resblock_kernel_sizes=resblock_kernel_sizes,
            upsample_rates=decoder_rates,
            upsample_initial_channel=decoder_dim,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            projection_filters=projection_filters,
            projection_kernels=projection_kernels,
            noise_module=noise_module,
            bias=bias,
        )

        super().__init__(encoder=encoder, decoder=decoder, latent_noise=latent_noise)
        if rank == 0:
            log.info(
                f"""Using AvocodoAutoEncoder:
                    time_downscaling: {np.prod(encoder_rates)}
                    num_channels_in: {channels}
                    z_dim: {z_dim}
                     Total compression factor {np.prod(encoder_rates) / z_dim * channels}
                    """
            )
