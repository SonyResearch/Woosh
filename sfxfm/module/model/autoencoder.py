import torch

from .blocks import DiagonalGaussianDistribution


class AutoEncoder(torch.nn.Module):
    def __init__(self, encoder, decoder, latent_noise: float = 0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_noise = max(latent_noise, 0)

    def forward(self, x, stochastic_latent=True):
        z = self.encode(x, stochastic_latent=stochastic_latent)
        dec = self.decode(z)

        return dec, z

    def encode(self, x, stochastic_latent=True):
        h = self.encoder(x)
        if stochastic_latent:
            h = h + self.latent_noise * torch.randn_like(h)
        return h

    def decode(self, z):
        dec = self.decoder(z)
        return dec

    def fix_input_length(self, x):
        """
        If the the auto encoder model, requires input of specific size,
        for example, multiple of the hop_length in vocos, this method
        crop the input to match the expected input shape
        """
        return x


class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, stochastic_latent=True):
        posterior = self.encode(x)
        if stochastic_latent:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)

        return dec, posterior

    def encode(self, x):
        h = self.encoder(x)
        posterior = DiagonalGaussianDistribution(h)
        return posterior

    def decode(self, z):
        # z = self.post_quant_conv(z)
        dec = self.decoder(z)
        # bs, ch, shuffled_timesteps, fbins = dec.size()
        # dec = self.time_unshuffle_operation(dec, bs, int(ch*shuffled_timesteps), fbins)
        # dec = self.freq_merge_subband(dec)
        return dec

    def fix_input_length(self, x):
        """
        If the the auto encoder model, requires input
        of specific size, for example, multiple of
        the hop_length in vocos, this method crop
        the input to match the expected input shape
        """
        return x
