"""
Collaborative Multi Band Discriminator
Copied from https://github.com/ncsoft/avocodo/blob/main/avocodo/models/CoMBD.py
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils.weight_norm import weight_norm
from torch.nn.utils.spectral_norm import spectral_norm

#from sfxfm.module.discriminator.pqmf import PQMF

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class CoMBDBlock(torch.nn.Module):
    def __init__(
        self,
        h_u: List[int], # channel sizes
        d_k: List[int], # kernel size
        d_s: List[int], # stride
        d_d: List[int], # dilation
        d_g: List[int], # groups
        d_p: List[int], # padding
        op_f: int, # projection channels
        op_k: int, # projection kernlels
        op_g: int, # projection groups
        use_spectral_norm=False,
    ):
        super(CoMBDBlock, self).__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm

        self.convs = nn.ModuleList()
        filters = [[1, h_u[0]]]
        for i in range(len(h_u) - 1):
            filters.append([h_u[i], h_u[i + 1]])
        for _f, _k, _s, _d, _g, _p in zip(filters, d_k, d_s, d_d, d_g, d_p):
            self.convs.append(norm_f(
                Conv1d(
                    in_channels=_f[0],
                    out_channels=_f[1],
                    kernel_size=_k,
                    stride=_s,
                    dilation=_d,
                    groups=_g,
                    padding=_p
                )
            ))
        self.projection_conv = norm_f(
            Conv1d(
                in_channels=filters[-1][1],
                out_channels=op_f,
                kernel_size=op_k,
                groups=op_g
            )
        )

    def forward(self, x):
        fmap = []
        for block in self.convs:
            x = block(x)
            x = F.leaky_relu(x, 0.2)
            fmap.append(x)
        x = self.projection_conv(x)
        return x, fmap


class CoMBD(torch.nn.Module):
    def __init__(self,
                combd_config,
                pqmf_list,
                use_spectral_norm=False,
                use_hierarchical=True,
                use_multi_scale=True,
                input_noise=0.01,
        ):
        super(CoMBD, self).__init__()
        
        self.pqmf = pqmf_list
        self.use_hierarchical = use_hierarchical
        self.use_multi_scale = use_multi_scale
        assert self.use_hierarchical is True or self.use_multi_scale is True
        assert input_noise >= 0
        self.input_noise = input_noise

        self.blocks = nn.ModuleList()
        for _h_u, _d_k, _d_s, _d_d, _d_g, _d_p, _op_f, _op_k, _op_g in zip(
            combd_config.combd_h_u,
            combd_config.combd_d_k,
            combd_config.combd_d_s,
            combd_config.combd_d_d,
            combd_config.combd_d_g,
            combd_config.combd_d_p,
            combd_config.combd_op_f,
            combd_config.combd_op_k,
            combd_config.combd_op_g,
        ):
            self.blocks.append(CoMBDBlock(
                _h_u,
                _d_k,
                _d_s,
                _d_d,
                _d_g,
                _d_p,
                _op_f,
                _op_k,
                _op_g,
                use_spectral_norm,
            ))

    def _block_forward(self, input, blocks):
        outs = []
        fmaps = []
        for x, block in zip(input, blocks):
            out, fmap = block(x + self.input_noise * torch.randn_like(x))
            outs.append(out)
            fmaps.append(fmap)
        return outs, fmaps

    def _pqmf_forward(self, ys, ys_hat):
        # preprocess for multi_scale forward
        multi_scale_inputs = []
        multi_scale_inputs_hat = []
        for pqmf in self.pqmf:
            multi_scale_inputs.append(
                pqmf.to(ys[-1]).analysis(ys[-1])[:, :1, :]
            )
            multi_scale_inputs_hat.append(
                pqmf.to(ys[-1]).analysis(ys_hat[-1])[:, :1, :]
            )

        outs_real= {}
        outs_fake = {}
        f_maps_real = {}
        f_maps_fake = {}

        # for hierarchical forward : intermediate reconstructions / downsampled originals
        if self.use_hierarchical:
            outs_real_h, f_maps_real_h = self._block_forward(ys, self.blocks)
            # for each intermediate layer
            for i, (outs_r, fmaps_r) in enumerate(zip(outs_real_h, f_maps_real_h)):
                outs_real.update({f"combd_hierarchical_{i}": outs_r})
                f_maps_real.update({f"combd_hierarchical_{i}": fmaps_r})

            outs_fake_h, f_maps_fake_h = self._block_forward(ys_hat, self.blocks)
            for i, (outs_f, fmaps_f) in enumerate(zip(outs_fake_h, f_maps_fake_h)):
                outs_fake.update({f"combd_hierarchical_{i}": outs_f})
                f_maps_fake.update({f"combd_hierarchical_{i}": fmaps_f})

        if self.use_multi_scale:

            # for multi_scale forward : downsampled reconstructions / downsampled originals
            outs_real_m, f_maps_real_m = self._block_forward(multi_scale_inputs, self.blocks[:-1])
            for i, (outs_r, fmaps_r) in enumerate(zip(outs_real_m, f_maps_real_m)):
                outs_real.update({f"combd_multi_scale_{i}": outs_r})
                f_maps_real.update({f"combd_multi_scale_{i}": fmaps_r})

            outs_fake_m, f_maps_fake_m = self._block_forward(multi_scale_inputs_hat, self.blocks[:-1])
            for i, (outs_f, fmaps_f) in enumerate(zip(outs_fake_m, f_maps_fake_m)):
                outs_fake.update({f"combd_multi_scale_{i}": outs_f})
                f_maps_fake.update({f"combd_multi_scale_{i}": fmaps_f})

        return outs_real, outs_fake, f_maps_real, f_maps_fake

    def forward(self, ys, ys_hat):
        outs_real, outs_fake, f_maps_real, f_maps_fake = self._pqmf_forward(
            ys, ys_hat)
        return outs_real, outs_fake, f_maps_real, f_maps_fake
