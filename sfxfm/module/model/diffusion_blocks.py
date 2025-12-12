import torch

from abc import ABC, abstractmethod
from torch import nn
from torch.nn.utils import weight_norm
import math
import numpy as np
from torch.nn import functional as F


class FourierFeaturesTime(nn.Module):
    """
    FourierFeatures from stable audio
    for time embedding only
    (batch, in_features)
    """

    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0, trainable=True):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(
            torch.randn([out_features // 2, in_features]) * std, requires_grad=trainable
        )

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class FourierTimeEmbedding(nn.Module):
    """
    Behaves like an embedding layer
    """

    def __init__(
        self,
        num_fourier_features,
        time_embed_dim,
        trainable_fourier_features=True,
        freq_std=1.0,
        time_embed_hidden_size=1024,
        activation=nn.LeakyReLU(),
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            FourierFeatures(
                1,
                num_fourier_features,
                trainable=trainable_fourier_features,
                std=freq_std,
            ),
            nn.Linear(num_fourier_features, time_embed_hidden_size),
            activation,
            nn.Linear(time_embed_hidden_size, time_embed_dim),
            activation,
        )

    def forward(self, x):
        """
        x: (batch_size,) -> (batch_size, out_features)

        """
        assert len(x.size()) == 1
        return self.net(x.unsqueeze(1))


def get_padding(kernel_size, dilation=1):
    """
    Only works when stride = 1
    """
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


# def init_weights(m, mean=0.0, std=0.01):
#     pass
_kernels = {
    "linear": [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    "cubic": [
        -0.01171875,
        -0.03515625,
        0.11328125,
        0.43359375,
        0.43359375,
        0.11328125,
        -0.03515625,
        -0.01171875,
    ],
    "lanczos3": [
        0.003689131001010537,
        0.015056144446134567,
        -0.03399861603975296,
        -0.066637322306633,
        0.13550527393817902,
        0.44638532400131226,
        0.44638532400131226,
        0.13550527393817902,
        -0.066637322306633,
        -0.03399861603975296,
        0.015056144446134567,
        0.003689131001010537,
    ],
}


# TODO upsample and divide only by 2
class Downsample1d(nn.Module):
    def __init__(self, kernel="linear", pad_mode="reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor(_kernels[kernel])
        self.pad = kernel_1d.shape[0] // 2 - 1
        self.register_buffer("kernel", kernel_1d)

    def forward(self, x):
        x = F.pad(x, (self.pad,) * 2, self.pad_mode)
        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0]])
        indices = torch.arange(x.shape[1], device=x.device)
        weight[indices, indices] = self.kernel.to(weight)
        return F.conv1d(x, weight, stride=2)


class Upsample1d(nn.Module):
    def __init__(self, kernel="linear", pad_mode="reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor(_kernels[kernel]) * 2
        self.pad = kernel_1d.shape[0] // 2 - 1
        self.register_buffer("kernel", kernel_1d)

    def forward(self, x):
        x = F.pad(x, ((self.pad + 1) // 2,) * 2, self.pad_mode)
        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0]])
        indices = torch.arange(x.shape[1], device=x.device)
        weight[indices, indices] = self.kernel.to(weight)
        return F.conv_transpose1d(x, weight, stride=2, padding=self.pad * 2 + 1)


class NormedConv1d(nn.Module):
    """
    Weight_normed convolution
    where padding defaults to "same size"

    Padding is computed automatically
    so that length_in == length_out
    if padding is None
    (only works when stride == 1)

    Weight norm is initialized in this way
    so that it is amenable to deepcopy
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        padding=None,
        norm_kwargs={},
        wn=True,
        gn=False,
        num_groups=1,
        padding_mode="reflect",
    ) -> None:
        """
        If wn=False: behaves as a normal 1d conv
        padding is same-size by default
        """
        super().__init__()
        if padding is None:
            assert stride == 1
            padding = get_padding(kernel_size, dilation)

        self.conv = nn.Conv1d(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
            padding_mode=padding_mode,
        )
        if wn:
            self.conv = weight_norm(self.conv)
            self.conv.weight = self.conv.weight_v.detach()

            init_weights_conv = lambda c: init_weights(c, **norm_kwargs)
            self.conv.apply(init_weights_conv)

        self.gn = gn
        if self.gn:
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=dim_in)

    def forward(self, x):
        if self.gn:
            x = self.norm(x)
        return self.conv(x)


class NormedConvTranspose1d(nn.Module):
    """
    Padding is computed automatically
    so that length_in == length_out

    Weight norm is initialized in this way
    so that it is amenable to deepcopy
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        padding=None,
        norm_kwargs={},
        wn=True,
        gn=False,
    ) -> None:
        super().__init__()
        if padding is None:
            assert stride == 1
            padding = get_padding(kernel_size, dilation)

        self.conv = nn.ConvTranspose1d(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
        )
        if wn:
            self.conv = weight_norm(self.conv)
            self.conv.weight = self.conv.weight_v.detach()

            init_weights_conv = lambda c: init_weights(c, **norm_kwargs)
            self.conv.apply(init_weights_conv)

        self.gn = gn
        if self.gn:
            self.norm = nn.GroupNorm(1, dim_in)

    def forward(self, x):
        if self.gn:
            x = self.norm(x)
        return self.conv(x)


# -------- Upsampling / Downsampling Blocks --------------
# ++ Interface ++
class UpsampleLayer(nn.Module, ABC):
    """
    (b c_in t) - > (b c_out, t*upsampling)
    """

    def __init__(self, dim_in, dim_out, upsampling, **kwargs) -> None:
        super().__init__()


class DownsampleLayer(nn.Module, ABC):
    """
    (b c_in t) - > (b c_out, t / downsampling)
    """

    def __init__(self, dim_in, dim_out, downsampling, **kwargs) -> None:
        super().__init__()


# ++ Implementations ++
class FixedKernelDownsample(DownsampleLayer):
    def __init__(
        self,
        dim_in,
        dim_out,
        downsampling,
        bias=True,
        kernel="linear",
        pad_mode="reflect",
    ) -> None:
        super().__init__(dim_in, dim_out, downsampling)
        self.downsampling = downsampling

        if self.downsampling > 1:
            assert (
                math.log2(self.downsampling).is_integer()
            ), f"downsampling argument = {self.downsampling} must be a power of two"
            # Downsample1d only downsample by 2
            self.downsample_twice = Downsample1d(kernel=kernel, pad_mode=pad_mode)
            self.num_downsampling = int(math.log2(self.downsampling))
        else:
            self.num_downsampling = 0

        if dim_in != dim_out:
            self.conv = NormedConv1d(
                dim_in=dim_in,
                dim_out=dim_out,
                kernel_size=1,
                stride=1,
                dilation=1,
                bias=bias,
                padding_mode=pad_mode,
            )
        else:
            self.conv = None

    def forward(self, x):
        y = x
        if self.conv is not None:
            y = self.conv(y)

        for _ in range(self.num_downsampling):
            y = self.downsample_twice(y)
        return y


class FixedKernelUpsample(UpsampleLayer):
    def __init__(
        self,
        dim_in,
        dim_out,
        upsampling,
        bias=True,
        kernel="linear",
        pad_mode="reflect",
    ) -> None:
        super().__init__(dim_in, dim_out, upsampling)
        self.upsampling = upsampling

        if self.upsampling > 1:
            assert (
                math.log2(self.upsampling).is_integer()
            ), f"upsampling argument = {self.upsampling} must be a power of two"
            # Downsample1d only downsample by 2
            self.upsample_twice = Upsample1d(kernel=kernel, pad_mode=pad_mode)
            self.num_upsampling = int(math.log2(self.upsampling))
        else:
            self.num_upsampling = 0

        if dim_in != dim_out:
            self.conv = NormedConv1d(
                dim_in=dim_in,
                dim_out=dim_out,
                kernel_size=1,
                stride=1,
                dilation=1,
                bias=bias,
            )
        else:
            self.conv = None

    def forward(self, x):
        y = x
        if self.conv is not None:
            y = self.conv(y)

        for _ in range(self.num_upsampling):
            y = self.upsample_twice(y)
        return y


class ConvDownsample(DownsampleLayer):
    def __init__(
        self,
        dim_in,
        dim_out,
        downsampling,
        factor=1,
        bias=True,
        activation=nn.Identity(),
        wn=False,
        gn=True,
        padding_mode="reflect",
    ) -> None:
        super().__init__(dim_in, dim_out, downsampling)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.downsampling = downsampling
        self.activation = activation
        kernel_size = self.downsampling * factor

        if self.downsampling > 1:
            assert self.downsampling % 2 == 0
            self.conv = NormedConv1d(
                dim_in,
                dim_out,
                stride=self.downsampling,
                kernel_size=kernel_size,
                padding=(self.downsampling // 2) * (factor // 2),
                bias=bias,
                wn=wn,
                gn=gn,
                padding_mode=padding_mode,
            )
        else:
            # assert kernel_size % 2 == 1
            if kernel_size % 2 == 0:
                kernel_size += 1
                print("ConvDownsample is using wrong-sized kernel. Adjusting")
            self.conv = NormedConv1d(
                dim_in,
                dim_out,
                stride=self.downsampling,
                kernel_size=kernel_size,
                padding=get_padding(
                    kernel_size,
                    dilation=1,
                ),
                bias=bias,
                wn=wn,
                gn=gn,
                padding_mode=padding_mode,
            )

    def forward(self, x: torch.Tensor):
        """
        x:
        """
        y = self.activation(x)
        y = self.conv(y)
        return y


class ConvUpsample(UpsampleLayer):
    def __init__(
        self,
        dim_in,
        dim_out,
        upsampling,
        factor=1,
        bias=True,
        activation=nn.Identity(),
        wn=False,
        gn=True,
    ) -> None:
        super().__init__(dim_in, dim_out, upsampling)

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.upsampling = upsampling
        self.activation = activation
        kernel_size = self.upsampling * factor

        if self.upsampling > 1:
            assert self.upsampling % 2 == 0
            self.conv = NormedConvTranspose1d(
                dim_in,
                dim_out,
                kernel_size=kernel_size,
                stride=self.upsampling,
                padding=(self.upsampling // 2) * (factor // 2),
                bias=bias,
                wn=wn,
                gn=gn,
            )
        else:
            # assert kernel_size % 2 == 1
            if kernel_size % 2 == 0:
                kernel_size += 1
                print("ConvUsample is using wrong-sized kernel. Adjusting")
            self.conv = NormedConvTranspose1d(
                dim_in,
                dim_out,
                stride=self.upsampling,
                kernel_size=kernel_size,
                padding=get_padding(
                    kernel_size,
                    dilation=1,
                ),
                bias=bias,
                wn=wn,
                gn=gn,
            )

    def forward(self, x: torch.Tensor):
        """
        x:
        """
        # TODO warning
        # normalization applied after activation
        # (before is better)
        # normalization is included in conv
        y = self.activation(x)
        y = self.conv(y)
        return y


# -------- Residual Blocks -------------
# ++ Interface ++
class ResnetBlock(nn.Module, ABC):
    """
    (b c t) - > (b c t)
    """

    def __init__(self, dim, time_embed_dim=0, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x, time_embed=None):
        raise NotImplementedError


# ++ Implementations ++
class DilatedResnetBlock(ResnetBlock):
    def __init__(
        self,
        dim,
        time_embed_dim=0,
        kernel_size=3,
        kernel_size_2=1,
        dilations=(1, 3, 9),
        activation=nn.SiLU(),
        bias_1=True,
        bias_2=True,
    ):
        """
        dim_in: used to change num_channels of the input
        """
        super().__init__(dim, time_embed_dim=time_embed_dim)
        assert len(dilations) == 3
        self.swish = activation

        self.conv_1 = nn.ModuleList(
            [
                NormedConv1d(
                    dim, dim, kernel_size, 1, dilation=dilations[0], bias=bias_1
                ),
                NormedConv1d(
                    dim, dim, kernel_size, 1, dilation=dilations[1], bias=bias_1
                ),
                NormedConv1d(
                    dim, dim, kernel_size, 1, dilation=dilations[2], bias=bias_1
                ),
            ]
        )

        self.conv_2 = nn.ModuleList(
            [
                NormedConv1d(dim, dim, kernel_size_2, 1, dilation=1, bias=bias_2),
                NormedConv1d(dim, dim, kernel_size_2, 1, dilation=1, bias=bias_2),
                NormedConv1d(dim, dim, kernel_size_2, 1, dilation=1, bias=bias_2),
            ]
        )

        if time_embed_dim > 0:
            # bias only
            self.bias = nn.Linear(in_features=time_embed_dim, out_features=dim)

    def forward(self, x, time_embed=None):
        # TODO noise2music rescale BEFORE norm?!
        # not in imagen-pytorch
        # time_embed is (b d)

        if time_embed is not None:
            bias = self.bias(time_embed)
            bias = bias.unsqueeze(2)
        else:
            bias = 0

        y = x
        for (
            c_1,
            c_2,
        ) in zip(self.conv_1, self.conv_2):
            res = y

            y = y + bias
            # swish
            y = self.swish(y)
            y = c_1(y)

            y = self.swish(y)
            y = c_2(y)

            y = y + res

            bias = 0

        return y


class GNDilatedResnetBlock(ResnetBlock):
    def __init__(
        self,
        dim,
        time_embed_dim=0,
        kernel_size=3,
        kernel_size_2=1,
        dilations=(1, 3, 5),
        activation=nn.SiLU(),
        bias_1=True,
        bias_2=True,
        affine_cond=False,
        dropout=0.0,
        num_groups=1,
        padding_mode="reflect",
    ):
        """
        dim_in: used to change num_channels of the input
        """
        super().__init__(dim, time_embed_dim=time_embed_dim)
        assert len(dilations) == 3
        self.affine_cond = affine_cond
        self.swish = activation
        self.conv_1 = nn.ModuleList(
            [
                NormedConv1d(
                    dim,
                    dim,
                    kernel_size,
                    1,
                    dilation=dilations[0],
                    bias=bias_1,
                    wn=False,
                    gn=False,
                    padding_mode=padding_mode,
                ),
                NormedConv1d(
                    dim,
                    dim,
                    kernel_size,
                    1,
                    dilation=dilations[1],
                    bias=bias_1,
                    gn=False,
                    padding_mode=padding_mode,
                    wn=False,
                ),
                NormedConv1d(
                    dim,
                    dim,
                    kernel_size,
                    1,
                    dilation=dilations[2],
                    bias=bias_1,
                    gn=False,
                    padding_mode=padding_mode,
                    wn=False,
                ),
            ]
        )

        self.conv_2 = nn.ModuleList(
            [
                NormedConv1d(
                    dim,
                    dim,
                    kernel_size_2,
                    1,
                    dilation=1,
                    bias=bias_2,
                    wn=False,
                    gn=False,
                    padding_mode=padding_mode,
                ),
                NormedConv1d(
                    dim,
                    dim,
                    kernel_size_2,
                    1,
                    dilation=1,
                    bias=bias_2,
                    wn=False,
                    gn=False,
                    padding_mode=padding_mode,
                ),
                NormedConv1d(
                    dim,
                    dim,
                    kernel_size_2,
                    1,
                    dilation=1,
                    bias=bias_2,
                    wn=False,
                    gn=False,
                    padding_mode=padding_mode,
                ),
            ]
        )

        if time_embed_dim > 0:
            # bias only
            self.bias = nn.Linear(
                in_features=time_embed_dim, out_features=dim, bias=True
            )
            if self.affine_cond:
                self.scalebias = nn.Linear(
                    in_features=time_embed_dim, out_features=dim * 2, bias=True
                )
                self.scalebias.weight.data.normal_(0.0, 0.01)

        # TODO num_groups = 1 or dim?!
        # TODO affine?!
        # self.gn = nn.GroupNorm(num_groups=1, num_channels=dim, affine=False)
        # Meta uses num_groups = 4
        self.gns_1 = nn.ModuleList(
            [
                nn.GroupNorm(num_groups=num_groups, num_channels=dim),
                nn.GroupNorm(num_groups=num_groups, num_channels=dim),
                nn.GroupNorm(num_groups=num_groups, num_channels=dim),
            ]
        )
        self.gns_2 = nn.ModuleList(
            [
                nn.GroupNorm(num_groups=num_groups, num_channels=dim),
                nn.GroupNorm(num_groups=num_groups, num_channels=dim),
                nn.GroupNorm(num_groups=num_groups, num_channels=dim),
            ]
        )

        self.dropout = nn.Dropout1d(p=dropout)

    def forward(self, x, time_embed=None):
        # TODO noise2music rescale BEFORE norm?!
        # not in imagen-pytorch
        # time_embed is (b d)

        if time_embed is not None:
            if self.affine_cond:
                scale, bias = self.scalebias(time_embed).unsqueeze(2).chunk(2, dim=1)
            else:
                bias = self.bias(time_embed)
                bias = bias.unsqueeze(2)
                scale = 0
        else:
            scale, bias = 0, 0

        y = x
        for c_1, c_2, gn_1, gn_2 in zip(
            self.conv_1, self.conv_2, self.gns_1, self.gns_2
        ):
            # TODO where to bias?
            # bias before residual in noise2music

            res = y

            # Before norm
            # y = (scale + 1) * y + bias
            # # swish
            # y = self.swish(gn_1(y))

            # After norm, before non linearity
            y = self.swish((scale + 1) * gn_1(y) + bias)

            y = c_1(y)
            # simple diffusion condition after gn_2
            y = self.swish(gn_2(y))

            y = self.dropout(y)
            y = c_2(y)

            # TODO try dropout here:
            # from "how to use dropout correctly on Residual networks
            # with batch normalization"
            # dropout should be after last normalization
            # and before last linear layer
            # ... but assume ReLU is used
            # location of dropout for pretty-jazz-636
            # y = self.dropout(y)

            # TODO in multiband, put dropout after each conv
            # Use dropout1d!

            y = y + res

            scale, bias = 0, 0

        return y


class BVGanResnetBlock(ResnetBlock):
    """
    Adapted from BigVGAN
    Allow for different kernel sizes for each GNDilatedResnetBlock
    Same dilations for each
    """

    def __init__(
        self,
        dim,
        time_embed_dim=0,
        kernel_sizes=[3, 7, 11],
        dilations=(1, 3, 5),
        activation=nn.GELU(),
        bias_1=True,
        bias_2=True,
        affine_cond=False,
        dropout=0.0,
        num_groups=1,
        padding_mode="reflect",
    ):
        """
        dim_in: used to change num_channels of the input
        """
        # variable kernel sizes
        super().__init__(dim, time_embed_dim=time_embed_dim)
        assert len(dilations) == 3
        self.nets = nn.ModuleList([])
        for kernel_size in kernel_sizes:
            self.nets.append(
                GNDilatedResnetBlock(
                    dim,
                    time_embed_dim=time_embed_dim,
                    kernel_size=kernel_size,
                    kernel_size_2=1,
                    dilations=dilations,
                    activation=activation,
                    bias_1=bias_1,
                    bias_2=bias_2,
                    affine_cond=affine_cond,
                    dropout=dropout,
                    num_groups=num_groups,
                    padding_mode=padding_mode,
                )
            )

    def forward(self, x, time_embed=None):
        y = x
        for net in self.nets:
            y = net(y, time_embed)
        return y


class DBlock(nn.Module):
    def __init__(
        self,
        downsample,  # partial
        dim_in,
        dim_out,
        downsampling,
        resblock,
        num_resblocks,  # partial
        time_embed_dim,
        self_attention: bool = False,
    ) -> None:
        super().__init__()

        self.downsampling = downsampling
        self.num_resblocks = num_resblocks

        self.downsample = downsample(
            dim_in=dim_in, dim_out=dim_out, downsampling=downsampling
        )

        resblocks = []
        for _ in range(num_resblocks):
            resblocks.append(resblock(dim=dim_out, time_embed_dim=time_embed_dim))

        self.resblocks = nn.ModuleList(resblocks)

        if self_attention:
            self.attention = FastSelfAttention1d(dim_out, n_head=8, dropout_rate=0.0)
        else:
            self.attention = nn.Identity()

    def forward(self, x, time_embed, cross_cond=None):
        y = self.downsample(x)

        y = self.attention(y)
        for resblock in self.resblocks:
            y = resblock(y, time_embed=time_embed)

        # TODO cross attention here
        return y


class DBlockL(nn.Module):
    """
    L for large:
    downsampling is done after resblocks
    """

    def __init__(
        self,
        downsample,  # partial
        dim_in,
        dim_out,
        downsampling,
        resblock,
        num_resblocks,  # partial
        time_embed_dim,
        self_attention: bool = False,
        dropout: float = 0.0,
        skip_rescale: bool = True,
        padding_mode="reflect",
    ) -> None:
        super().__init__()

        self.downsampling = downsampling
        self.num_resblocks = num_resblocks
        self.no_downsample = dim_in == dim_out and downsampling == 1
        self.skip_rescale = skip_rescale
        if not self.no_downsample:
            self.downsample = downsample(
                dim_in=dim_in,
                dim_out=dim_out,
                downsampling=downsampling,
                padding_mode=padding_mode,
            )

        resblocks = []
        for _ in range(num_resblocks):
            resblocks.append(
                resblock(
                    dim=dim_in,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    padding_mode=padding_mode,
                )
            )

        self.resblocks = nn.ModuleList(resblocks)

        if self_attention:
            self.attention = FastSelfAttention1d(dim_in, n_head=8, dropout_rate=dropout)
        else:
            self.attention = nn.Identity()

    def forward(self, x, time_embed, skip=None, cross_cond=None):
        y = x
        if skip is not None:
            y = y + skip
            if self.skip_rescale:
                y = y / np.sqrt(2.0)

        for resblock in self.resblocks:
            y = resblock(y, time_embed=time_embed)

        y = self.attention(y)

        skip_out = y
        if not self.no_downsample:
            y = self.downsample(y)

        return y, skip_out


class UBlock(nn.Module):
    def __init__(
        self,
        upsample,
        dim_in,
        dim_out,
        upsampling,
        resblock,
        num_resblocks,
        time_embed_dim,
        self_attention,
    ) -> None:
        """
        dim_in: if dim_in != dim_mid, first project input to dim_mid
        dim_mid: all calculations done with this dim
        dim_out: used during upsampling layer
        """
        super().__init__()
        self.upsampling = upsampling
        self.num_resblocks = num_resblocks

        if self_attention:
            self.attention = FastSelfAttention1d(dim_in, n_head=8, dropout_rate=0.0)
        else:
            self.attention = nn.Identity()

        resblocks = []
        for _ in range(num_resblocks):
            block: ResnetBlock = resblock(dim=dim_in, time_embed_dim=time_embed_dim)
            resblocks.append(block)

        self.resblocks = nn.ModuleList(resblocks)

        self.upsample: UpsampleLayer = upsample(
            dim_in,
            dim_out,
            upsampling=upsampling,
        )

    def forward(self, x, time_embed, cross_cond=None):
        y = x

        y = self.attention(y)
        for resblock in self.resblocks:
            y = resblock(y, time_embed=time_embed)

        y = self.upsample(y)
        return y


class UBlockL(nn.Module):
    """
    upsampling is done BEFORE resblocks
    """

    def __init__(
        self,
        upsample,
        dim_in,
        dim_out,
        upsampling,
        resblock,
        num_resblocks,
        time_embed_dim,
        self_attention,
        dropout,
        skip_rescale,
        padding_mode,
    ) -> None:
        """
        dim_in: if dim_in != dim_mid, first project input to dim_mid
        dim_mid: all calculations done with this dim
        dim_out: used during upsampling layer
        """
        super().__init__()
        self.upsampling = upsampling
        self.num_resblocks = num_resblocks
        self.no_upsampling = dim_in == dim_out and upsampling == 1

        self.upsample: UpsampleLayer = upsample(
            dim_in,
            dim_out,
            upsampling=upsampling,
        )

        if self_attention:
            self.attention = FastSelfAttention1d(
                dim_out, n_head=8, dropout_rate=dropout
            )
        else:
            self.attention = nn.Identity()

        resblocks = []
        for _ in range(num_resblocks):
            block: ResnetBlock = resblock(
                dim=dim_out,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
                padding_mode=padding_mode,
            )
            resblocks.append(block)

        self.resblocks = nn.ModuleList(resblocks)
        self.skip_rescale = skip_rescale

    def forward(self, x, time_embed, skip=None, cross_cond=None):
        if x is None:
            y = skip
        else:
            y = x
            y = self.upsample(y)

            if skip is not None:
                y = y + skip
                if self.skip_rescale:
                    y = y / np.sqrt(2.0)

        for resblock in self.resblocks:
            y = resblock(y, time_embed=time_embed)
        y = self.attention(y)

        return y


class VganResblock(ResnetBlock):
    """
    Take multiple kernels in parallel
    """

    def __init__(
        self,
        dim,
        time_embed_dim=0,
        kernel_sizes=[3, 7, 11],
        dilations=(1, 3, 9),
        activation=nn.SiLU(),
        bias_1=True,
        bias_2=True,
    ):
        super().__init__(dim, time_embed_dim=time_embed_dim)
        assert len(dilations) == 3
        self.swish = activation

        self.resblocks = nn.ModuleList([])
        for kernel_size in kernel_sizes:
            self.resblocks.append(
                DilatedResnetBlock(
                    dim=dim,
                    time_embed_dim=time_embed_dim,
                    kernel_size=kernel_size,
                    kernel_size_2=kernel_size,
                    dilations=dilations,
                    activation=activation,
                    bias_1=bias_1,
                    bias_2=bias_2,
                )
            )

    def forward(self, x, time_embed=None):
        # TODO noise2music rescale BEFORE norm?!
        # not in imagen-pytorch
        # time_embed is (b d)

        y = 0
        for resblock in self.resblocks:
            y = y + resblock(x, time_embed)
        y = y / len(self.resblocks)

        return y


class FILM(nn.Module):
    def __init__(self, time_embed_dim, dim, bias=True) -> None:
        super().__init__()
        self.scalebias = nn.Linear(time_embed_dim, dim * 2, bias=bias)

    def forward(self, x, time_embed):
        scale, bias = self.scalebias(time_embed).unsqueeze(2).chunk(2, dim=1)
        return (scale + 1) * x + bias


# class SnakeBeta(nn.Module):
#     '''
#     A modified Snake function which uses separate parameters for the magnitude of the periodic components
#     Shape:
#         - Input: (B, C, T)
#         - Output: (B, C, T), same shape as the input
#     Parameters:
#         - alpha - trainable parameter that controls frequency
#         - beta - trainable parameter that controls magnitude
#     References:
#         - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
#         https://arxiv.org/abs/2006.08195
#     Examples:
#         >>> a1 = snakebeta(256)
#         >>> x = torch.randn(256)
#         >>> x = a1(x)
#     '''
#     def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
#         '''
#         Initialization.
#         INPUT:
#             - in_features: shape of the input
#             - alpha - trainable parameter that controls frequency
#             - beta - trainable parameter that controls magnitude
#             alpha is initialized to 1 by default, higher values = higher-frequency.
#             beta is initialized to 1 by default, higher values = higher-magnitude.
#             alpha will be trained along with the rest of your model.
#         '''
#         super(SnakeBeta, self).__init__()
#         self.in_features = in_features

#         # initialize alpha
#         self.alpha_logscale = alpha_logscale
#         if self.alpha_logscale: # log scale alphas initialized to zeros
#             self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
#             self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
#         else: # linear scale alphas initialized to ones
#             self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
#             self.beta = nn.Parameter(torch.ones(in_features) * alpha)

#         self.alpha.requires_grad = alpha_trainable
#         self.beta.requires_grad = alpha_trainable

#         self.no_div_by_zero = 0.000000001

#     def forward(self, x):
#         '''
#         Forward pass of the function.
#         Applies the function to the input elementwise.
#         SnakeBeta ∶= x + 1/b * sin^2 (xa)
#         '''
#         alpha = self.alpha.unsqueeze(0).unsqueeze(-1) # line up with x to [B, C, T]
#         beta = self.beta.unsqueeze(0).unsqueeze(-1)
#         if self.alpha_logscale:
#             alpha = torch.exp(alpha)
#             beta = torch.exp(beta)
#         x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

#         return x


# TODO lazy version
class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(
        self, in_features_max, alpha=1.0, alpha_trainable=True, alpha_logscale=False
    ):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super(SnakeBeta, self).__init__()
        self.in_features = in_features_max

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(self.in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(self.in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(self.in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        num_channels = x.size(1)
        alpha = (
            self.alpha[:num_channels].unsqueeze(0).unsqueeze(-1)
        )  # line up with x to [B, C, T]
        beta = self.beta[:num_channels].unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(
            torch.sin(x * alpha), 2
        )

        return x
