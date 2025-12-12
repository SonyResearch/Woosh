from typing import List, Optional
from einops import rearrange
import torch
from torch import nn

from sfxfm.module.model.blocks import FourierFeatures
from sfxfm.module.model.diffusion_blocks import FourierTimeEmbedding
from sfxfm.module.model.stylevocos import StyleVocosBackbone
from sfxfm.module.model.vocos_blocks import ContinuousAdaLayerNorm, IdentityAdaLayerNorm


class DiffusionConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        gamma_noise_init_value: float,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = ContinuousAdaLayerNorm(dim, eps=1e-6)

        self.pwconv1 = nn.Linear(
            dim, intermediate_dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

        # self.gamma_noise = nn.Parameter(
        #     gamma_noise_init_value * torch.ones(dim), requires_grad=True
        # )

    def forward(self, x: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)

        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)

        # add intermediate noise?
        # x = x + torch.randn_like(x) * self.gamma_noise

        x = self.norm(x, modulation=modulation)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class DiffusionVocosBackbone(nn.Module):
    """
    Vocos backbone module built with ConvNeXt blocks. S
    upports additional conditioning with Adaptive Layer Normalization
    Mimics StyleGAN 2
    expect input to be of dimension dim

    Args:
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock and dimension of x!
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        time_embed_dim: int,
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        gamma_noise_init_value: Optional[float] = None,
        input_layer_norm: bool = False,
        final_layer_norm: bool = False,
        proj_kernel_size: int = 1,
    ):
        super().__init__()
        self.dim = dim

        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        gamma_noise_init_value = gamma_noise_init_value or 0.0

        padding_proj = (proj_kernel_size - 1) // 2

        self.convnext = nn.ModuleList(
            [
                DiffusionConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale_init_value,
                    gamma_noise_init_value=gamma_noise_init_value,
                )
                for _ in range(num_layers)
            ]
        )

        self.mod_projs = nn.ModuleList(
            [
                nn.Conv1d(
                    time_embed_dim,
                    dim,
                    kernel_size=proj_kernel_size,
                    padding=padding_proj,
                )
                for _ in range(num_layers)
            ]
        )

        self.apply(self._init_weights)

        # input layer norm with proj or noop
        self.input_norm = (
            ContinuousAdaLayerNorm(dim, eps=1e-6)
            if input_layer_norm
            else IdentityAdaLayerNorm()
        )
        self.input_proj = (
            nn.Conv1d(
                time_embed_dim,
                dim,
                kernel_size=proj_kernel_size,
                padding=padding_proj,
            )
            if input_layer_norm
            else nn.Identity()
        )

        # final layer norm with proj or noop
        self.final_layer_norm = (
            ContinuousAdaLayerNorm(dim, eps=1e-6)
            if final_layer_norm
            else IdentityAdaLayerNorm()
        )
        self.final_proj = (
            nn.Conv1d(
                time_embed_dim,
                dim,
                kernel_size=proj_kernel_size,
                padding=padding_proj,
            )
            if final_layer_norm
            else nn.Identity()
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size, intermediate_dim, length = x.size()

        # input norm
        mod = self.input_proj(t).transpose(1, 2)
        x = self.input_norm(x.transpose(1, 2), modulation=mod).transpose(1, 2)

        for conv_block, mod_proj in zip(self.convnext, self.mod_projs):
            mod = mod_proj(t).transpose(1, 2)
            x = conv_block(x, mod)

        # final norm
        mod = self.final_proj(t).transpose(1, 2)
        x = self.final_layer_norm(x.transpose(1, 2), modulation=mod).transpose(1, 2)

        return x


class DiffusionVocos(nn.Module):
    """
    Similar as StyleVocos
    """

    def __init__(
        self,
        channels_in,
        channels_out,
        time_embed_dim,
        num_fourier_features=64,
        d_model=512,
        intermediate_dim=1536,
        num_layers=8,
        num_layers_embedding=2,
        kernel_size_embedding=3,
        gamma_noise_init_value: Optional[float] = None,
        input_layer_norm: bool = True,
        final_layer_norm: bool = True,
    ):
        super().__init__()
        # TODO add those parameters
        # TODO add num_layers
        self.time_embedding = FourierTimeEmbedding(
            num_fourier_features=num_fourier_features, time_embed_dim=time_embed_dim
        )
        # self.embed = FourierFeatures(channels_in, d_model, trainable=True, std=16)
        self.embed = FourierFeatures(1, d_model // channels_in, trainable=True, std=1)
        # self.embed = FourierFeatures(1, d_model // channels_in, trainable=True, std=16)

        # self.embed = nn.Conv1d(
        #     channels_in,
        #     d_model,
        #     kernel_size=7,
        #     padding=3,
        # )

        self.backbone = DiffusionVocosBackbone(
            dim=d_model,
            intermediate_dim=intermediate_dim,
            time_embed_dim=time_embed_dim,
            num_layers=num_layers,
            gamma_noise_init_value=gamma_noise_init_value,
            input_layer_norm=input_layer_norm,
            final_layer_norm=final_layer_norm,
            proj_kernel_size=1,
        )

        self.head = nn.Conv1d(d_model, out_channels=channels_out, kernel_size=1)

        self.epsilon = 1e-8

    def forward(self, x, t, cond=None):
        # x = self.embed(x)

        x = rearrange(x, "b c t -> (b c) 1 t")
        x = self.embed(x)
        x = rearrange(x, "(b c) C t -> b (c C) t", c=32)

        t = self.time_embedding(t)
        # add time dimension
        t = t.unsqueeze(2).expand(-1, -1, x.size(2))
        x = self.backbone(x=x, t=t)
        x = self.head(x)
        return x
