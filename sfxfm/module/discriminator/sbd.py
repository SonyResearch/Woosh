from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils.weight_norm import weight_norm
from torch.nn.utils.spectral_norm import spectral_norm

from sfxfm.module.discriminator.pqmf import PQMF
from sfxfm.module.discriminator.combd import get_padding


class MDC(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        strides,
        kernel_size,
        dilations,
        use_spectral_norm=False,
    ):
        super(MDC, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.d_convs = nn.ModuleList()
        for _k, _d in zip(kernel_size, dilations):
            self.d_convs.append(
                norm_f(
                    Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=_k,
                        dilation=_d,
                        padding=get_padding(_k, _d),
                    )
                )
            )
        self.post_conv = norm_f(
            Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=strides,
                padding=get_padding(_k, _d),
            )
        )
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        _out = None
        for _l in self.d_convs:
            _x = torch.unsqueeze(_l(x), -1)
            _x = F.leaky_relu(_x, 0.2)
            if _out is None:
                _out = _x
            else:
                _out = torch.cat([_out, _x], axis=-1)
        x = torch.sum(_out, dim=-1)
        x = self.post_conv(x)
        x = F.leaky_relu(x, 0.2)  # @@

        return x


class SBDBlock(torch.nn.Module):
    def __init__(
        self,
        segment_dim,
        strides,
        filters,
        kernel_size,
        dilations,
        use_spectral_norm=False,
        input_noise=0.01,
    ):
        super(SBDBlock, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.input_noise = input_noise
        self.convs = nn.ModuleList()
        filters_in_out = [(segment_dim, filters[0])]
        for i in range(len(filters) - 1):
            filters_in_out.append([filters[i], filters[i + 1]])

        for _s, _f, _k, _d in zip(strides, filters_in_out, kernel_size, dilations):
            self.convs.append(
                MDC(
                    in_channels=_f[0],
                    out_channels=_f[1],
                    strides=_s,
                    kernel_size=_k,
                    dilations=_d,
                    use_spectral_norm=use_spectral_norm,
                )
            )
        self.post_conv = norm_f(
            Conv1d(
                in_channels=_f[1],
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=3 // 2,
            )
        )  # @@

    def forward(self, x):
        num_maps = len(self.convs)
        outs = []
        fmap = {i: [] for i in range(num_maps)}
        for chunk in x:
            for i, _l in enumerate(self.convs):
                chunk = _l(chunk + self.input_noise * torch.randn_like(chunk))
                fmap[i].append(chunk)
            outs.append(self.post_conv(chunk))  # @@

        outs = torch.mean(torch.stack(outs), dim=0)
        fmap = [torch.mean(torch.stack(fmap[i]), dim=0) for i in range(num_maps)]
        return outs, fmap


class MDCDConfig:
    def __init__(
        self,
        pqmf_config,
        filters,
        kernel_sizes,
        dilations,
        strides,
        band_ranges,
        transpose,
        segment_size,
        use_spectral_norm,
        input_noise,
    ):
        self.pqmf_params = pqmf_config["sbd"]
        self.f_pqmf_params = pqmf_config["fsbd"]
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.strides = strides
        self.band_ranges = band_ranges
        self.transpose = transpose
        self.segment_size = segment_size
        self.use_spectral_norm = use_spectral_norm
        self.input_noise = input_noise


class SBD(torch.nn.Module):
    def __init__(
        self,
        sbd_config,
    ):
        super(SBD, self).__init__()
        self.config = MDCDConfig(
            pqmf_config=sbd_config.pqmf_config,
            filters=sbd_config.filters,
            kernel_sizes=sbd_config.kernel_sizes,
            dilations=sbd_config.dilations,
            strides=sbd_config.strides,
            band_ranges=sbd_config.band_ranges,
            transpose=sbd_config.transpose,
            segment_size=sbd_config.segment_size,
            use_spectral_norm=sbd_config.use_spectral_norm,
            input_noise=sbd_config.input_noise,
        )
        self.pqmf = PQMF(*self.config.pqmf_params)
        if True in sbd_config.transpose:
            self.f_pqmf = PQMF(*self.config.f_pqmf_params)
        else:
            self.f_pqmf = None

        self.discriminators = torch.nn.ModuleList()

        for _f, _k, _d, _s, _br, _tr in zip(
            self.config.filters,
            self.config.kernel_sizes,
            self.config.dilations,
            self.config.strides,
            self.config.band_ranges,
            self.config.transpose,
        ):
            if _tr:
                self.segment_dim = self.config.segment_size // _br[1] - _br[0]
            else:
                self.segment_dim = _br[1] - _br[0]

            self.discriminators.append(
                SBDBlock(
                    segment_dim=self.segment_dim,
                    filters=_f,
                    kernel_size=_k,
                    dilations=_d,
                    strides=_s,
                    use_spectral_norm=self.config.use_spectral_norm,
                    input_noise=self.config.input_noise,
                )
            )

    def forward(self, y, y_hat):
        outs_real = {}
        outs_fake = {}
        f_maps_real = {}
        f_maps_fake = {}
        y_in = self.pqmf.analysis(y)
        y_hat_in = self.pqmf.analysis(y_hat)
        if self.f_pqmf is not None:
            y_in_f = self.f_pqmf.analysis(y)
            y_hat_in_f = self.f_pqmf.analysis(y_hat)

        for d, br, tr in zip(
            self.discriminators, self.config.band_ranges, self.config.transpose
        ):
            if tr:
                # br[1] = segment_size // _br[1] - _br[0]
                _y_in = y_in_f[:, br[0] : br[1], :]
                _y_hat_in = y_hat_in_f[:, br[0] : br[1], :]
                _y_in_length = _y_in.size(-1)
                # Divide into chunks to handle variable length inputs
                _y_in = list(
                    torch.split(
                        torch.transpose(_y_in, 1, 2),
                        split_size_or_sections=self.segment_dim,
                        dim=1,
                    )
                )
                _y_hat_in = list(
                    torch.split(
                        torch.transpose(_y_hat_in, 1, 2),
                        split_size_or_sections=self.segment_dim,
                        dim=1,
                    )
                )
                if _y_in_length % self.segment_dim != 0:
                    _y_in = _y_in[:-1]
                    _y_hat_in = _y_hat_in[:-1]
            else:
                _y_in = [y_in[:, br[0] : br[1], :]]
                _y_hat_in = [y_hat_in[:, br[0] : br[1], :]]

            out_real, fmap_real = d(_y_in)
            out_fake, fmap_fake = d(_y_hat_in)

            prefix = "f" if tr else "t"
            outs_real.update({f"{prefix}-{br[1]}-sbd": out_real})
            outs_fake.update({f"{prefix}-{br[1]}-sbd": out_fake})
            f_maps_real.update({f"{prefix}-{br[1]}-sbd": fmap_real})
            f_maps_fake.update({f"{prefix}-{br[1]}-sbd": fmap_fake})

        return outs_real, outs_fake, f_maps_real, f_maps_fake
