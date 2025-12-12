from typing import Optional
import torch
from torch import nn
from sfxfm.module.model.autoencoder import AutoEncoder
from sfxfm.module.model.vocos import (
    Backbone,
    FourierHead,
    ISTFTCircleHead,
    VocosEncoder,
)
from sfxfm.module.model.vocos_blocks import ContinuousAdaLayerNorm, IdentityAdaLayerNorm
from sfxfm.utils.dist import rank

import logging

# get logger
log = logging.getLogger(__name__)


class StyleConvNeXtBlock(nn.Module):
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

        self.gamma_noise = nn.Parameter(
            gamma_noise_init_value * torch.ones(dim), requires_grad=True
        )

    def forward(self, x: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)

        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)

        # add intermediate noise?
        x = x + torch.randn_like(x) * self.gamma_noise
        x = self.norm(x, modulation=modulation)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class StyleVocosBackbone(Backbone):
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
        num_layers: int,
        layer_scale_init_value: Optional[float] = None,
        gamma_noise_init_value: Optional[float] = None,
        input_layer_norm: bool = False,
        final_layer_norm: bool = False,
        proj_kernel_size: int = 3,
    ):
        super().__init__()
        self.dim = dim

        layer_scale_init_value = layer_scale_init_value or 1 / num_layers
        gamma_noise_init_value = gamma_noise_init_value or 0.0

        padding_proj = (proj_kernel_size - 1) // 2

        self.convnext = nn.ModuleList(
            [
                StyleConvNeXtBlock(
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
                    intermediate_dim,
                    dim,
                    kernel_size=proj_kernel_size,
                    padding=padding_proj,
                )
                for _ in range(num_layers)
            ]
        )

        self.const_init = nn.Parameter(torch.randn(1, self.dim, 1))
        self.apply(self._init_weights)

        # input layer norm with proj or noop
        self.input_norm = (
            ContinuousAdaLayerNorm(dim, eps=1e-6)
            if input_layer_norm
            else IdentityAdaLayerNorm()
        )
        self.input_proj = (
            nn.Conv1d(
                intermediate_dim,
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
                intermediate_dim,
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

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size, intermediate_dim, length = x.size()

        # x is the conditioning signal
        modulation = x

        # initial noise
        x = self.const_init.expand(batch_size, self.dim, length)

        # input norm
        mod = self.input_proj(modulation).transpose(1, 2)
        x = self.input_norm(x.transpose(1, 2), modulation=mod).transpose(1, 2)

        for conv_block, mod_proj in zip(self.convnext, self.mod_projs):
            mod = mod_proj(modulation).transpose(1, 2)
            x = conv_block(x, mod)

        # final norm
        mod = self.final_proj(modulation).transpose(1, 2)
        x = self.final_layer_norm(x.transpose(1, 2), modulation=mod).transpose(1, 2)

        return x


class StyleVocosEmbedding(nn.Module):
    def __init__(self, input_dim, intermediate_dim, num_layers, kernel_size) -> None:
        """
        channels_out = intermediate dim
        """
        super().__init__()

        self.embed = nn.Sequential()
        channels_in = input_dim
        channels_out = intermediate_dim
        for i in range(num_layers):
            if i > 0:
                channels_in = intermediate_dim
            self.embed.append(
                nn.Conv1d(
                    channels_in,
                    channels_out,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                )
            )
            self.embed.append(nn.GELU())

    def forward(self, x: torch.Tensor):
        return self.embed(x)


class StyleVocosDecoder(nn.Module):
    def __init__(
        self,
        input_channels,
        d_model=512,
        intermediate_dim=1536,
        num_layers=8,
        num_layers_embedding=2,
        kernel_size_embedding=3,
        gamma_noise_init_value: Optional[float] = None,
        n_fft=1024,
        hop_length=256,
        input_layer_norm: bool = True,
        final_layer_norm: bool = True,
        istft_head: Optional[FourierHead] = None,
    ):
        super().__init__()

        self.embed = StyleVocosEmbedding(
            input_dim=input_channels,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers_embedding,
            kernel_size=kernel_size_embedding,
        )

        self.backbone = StyleVocosBackbone(
            dim=d_model,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            gamma_noise_init_value=gamma_noise_init_value,
            input_layer_norm=input_layer_norm,
            final_layer_norm=final_layer_norm,
        )

        if istft_head is None:
            self.head = ISTFTCircleHead(
                d_model, n_fft=n_fft, hop_length=hop_length, padding="center"
            )
        else:
            # istft is a _partial_
            self.head = istft_head(d_model, n_fft=n_fft, hop_length=hop_length)

        self.n_fft = n_fft
        self.epsilon = 1e-8

    def forward(self, x):
        y = self.embed(x)
        y = self.backbone(y)
        y = self.head(y)
        return y.unsqueeze(1)


class StyleVocosAutoEncoder(AutoEncoder):
    def __init__(
        self,
        channels: int = 1,
        spec_embed: str = "stft-complex",
        z_dim: int = 64,
        d_model: int = 1024,
        intermediate_dim: int = 1536,
        n_fft: int = 1024,
        hop_length: int = 256,
        num_layers: int = 8,
        num_layers_embedding: int = 2,
        kernel_size_embedding: int = 1,
        istft_head: Optional[FourierHead] = None,
        input_layer_norm_encoder: bool = True,
        final_layer_norm_encoder: bool = True,
        input_layer_norm_decoder: bool = True,
        final_layer_norm_decoder: bool = True,
        gamma_noise_init_value: Optional[float] = None,
    ) -> None:
        """AutoEncoder using Vocos_like encoders and decoders

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
        self.hop_length = hop_length

        encoder = VocosEncoder(
            d_model=d_model,
            output_channels=z_dim,
            num_layers=num_layers,
            intermediate_dim=intermediate_dim,
            n_fft=n_fft,
            hop_length=hop_length,
            spec_embed=spec_embed,
            input_layer_norm=input_layer_norm_encoder,
            final_layer_norm=final_layer_norm_encoder,
        )

        decoder = StyleVocosDecoder(
            input_channels=z_dim,
            d_model=d_model,
            num_layers=num_layers,
            intermediate_dim=intermediate_dim,
            n_fft=n_fft,
            hop_length=hop_length,
            istft_head=istft_head,
            num_layers_embedding=num_layers_embedding,
            kernel_size_embedding=kernel_size_embedding,
            gamma_noise_init_value=gamma_noise_init_value,
            input_layer_norm=input_layer_norm_decoder,
            final_layer_norm=final_layer_norm_decoder,
        )

        super().__init__(encoder=encoder, decoder=decoder)

        if rank() == 0:
            log.info(
                f"""Using StyleVocosAutoEncoder:
                     time_downscaling: {hop_length}
                     num_channels_in: {channels}
                     z_dim: {z_dim}
                     Total compression factor {hop_length / z_dim}
                     """
            )

    def fix_input_length(self, x):
        """
        in vocos, the input samples should be a multiple of hopsize
        """
        assert len(x.shape) == 3, "VocosAutoEncoder expect input of the shape B,C,T"
        x = x[
            :, :, : x.shape[2] - (x.shape[2] % self.hop_length)
        ]  # make sure we can divide by hopsize, for the waveform loss
        return x.contiguous()
