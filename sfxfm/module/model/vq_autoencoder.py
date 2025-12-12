"""latent quantizer base class"""

from einops import rearrange
from sfxfm.module.quantizer.quantizer import Quantizer
from sfxfm.utils.dist import rank
from functools import partial
import logging
import torch


log = logging.getLogger(__name__)
rank = rank()


class VQAutoEncoder(torch.nn.Module):
    def __init__(
        self,
        encoder,  # import pretrained model or new model
        decoder,
        quantizer,  # learnable quantizer
        normalize: bool = False,
        seed: int = 0,
        infer_max_quantizers: int = None,  # type: ignore
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.seed = seed
        self.quantizer = quantizer
        self.hop_length = encoder.hop_length

    @property
    def codebook_size(self):
        return self.quantizer.codebooks.size(1)

    @property
    def num_quantizers(self):
        return self.quantizer.codebooks.size(0)

    def encode(self, x, return_indices=False, stochastic_latent=False, **kwargs):
        z = self.encoder(x)
        # if stochastic_latent:
        #    z = z + self.latent_noise * torch.randn_like(z)
        # quantizer needs the quantized dim to be last
        if self.quantizer is None:
            raise ValueError("Quantizer has not been initialized")
        z = rearrange(z, "b c t -> b t c")
        z_q, idx, quantization_loss = self.quantizer(z)
        z_q = rearrange(z_q, "b t c -> b c t")
        if return_indices:
            return z_q, idx, quantization_loss
        else:
            return z_q

    def forward(self, x, stochastic_latent=False, return_indices=False):
        if return_indices:
            z_q, idx, quantization_loss = self.encode(
                x, stochastic_latent=stochastic_latent, return_indices=return_indices
            )
            dec = self.decode(z_q)
            return dec, z_q, idx, quantization_loss

        else:
            z_q = self.encode(x, stochastic_latent=stochastic_latent)
            dec = self.decode(z_q)
            return dec, z_q

    def decode(self, z):
        dec = self.decoder(z)
        return dec

    def fix_input_length(self, x):
        assert len(x.shape) == 3, "Expect input of the shape B,C,T"
        x = x[
            :, :, : x.shape[2] - (x.shape[2] % self.hop_length)
        ]
        return x.contiguous()


class PostVQAutoEncoder(VQAutoEncoder):
    def __init__(
        self,
        pretrained_autoencoder,
        quantizer: Quantizer,  # partial
        normalize: bool = False,
        seed: int = 0,
        infer_max_quantizers: int = None,  # type: ignore
    ):
        super().__init__(
            encoder=pretrained_autoencoder.encoder,
            decoder=pretrained_autoencoder.decoder,
            quantizer=quantizer,
        )

        self.seed = seed

        self.seed = seed
        self.pretrained_autoencoder = pretrained_autoencoder
        if isinstance(quantizer, partial):
            self.quantizer = quantizer(
                pretrained_autoencoder=self.pretrained_autoencoder
            )
        else:
            self.quantizer = quantizer

    @property
    def codebook_size(self):
        return self.quantizer.codebook_size

    @property
    def num_quantizers(self):
        return self.quantizer.num_quantizers
