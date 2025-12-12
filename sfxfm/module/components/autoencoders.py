import logging
from copy import deepcopy

from pydantic import ConfigDict
import torch
from hydra.utils import instantiate
from tqdm import tqdm


from .base import BaseComponent, ComponentConfig

rank = 0
# get logger
log = logging.getLogger(__name__)


class AudioAutoEncoderConfig(ComponentConfig):
    # allow extra args
    # this config behaves like a dict
    model_config = ConfigDict(
        extra="allow",
    )


class AudioAutoEncoder(torch.nn.Module, BaseComponent):
    r"""y
    Audio AutoEncoder, supports encoding and decoding audio into latent space.


    Class attributes (overridden by derived classes):

        - **config_class** ([`Any`]) -- the configuration class for the config that defines this model architecture.

    """

    config_class = AudioAutoEncoderConfig

    def __init__(self, config):
        super().__init__()
        self.init_from_config(config)
        # self.config is now a Pydantic model
        # cast back to dict to rely on the legacy code
        config = self.config.dict()

        zdim = config["z_dim"]
        self.register_buffer("z_mean", torch.zeros(zdim))
        self.register_buffer("z_std", torch.zeros(zdim))
        aeconfig = deepcopy(config)
        self.normalize = aeconfig.pop("normalize", True)
        # remove components config that are not used by the autoencoder
        aeconfig.pop("exclude_from_checkpoint", None)  # remove z_dim from config
        aeconfig.pop("trainable", None)  # remove z_dim from config
        self.autoencoder = instantiate(aeconfig)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.autoencoder.encode(x)

        if self.normalize:
            z = (z - self.z_mean[None, :, None]) / self.z_std[None, :, None]

        return z

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        z ->  x
        """
        if self.normalize:
            x = x * self.z_std[None, :, None] + self.z_mean[None, :, None]

        x = self.autoencoder.decode(x)
        return x

    @classmethod
    def from_autoencoder_module(
        cls,
        autoencoder,
        datamodule=None,
        stats_batches=None,
        **kwargs,
    ) -> "AudioAutoEncoder":
        """
        Create an AudioAutoEncoder component using external module configured to load from pl.module.
        autoencoder: AutoEncoder, loaded by external module

        This method is used to export plmodule see `configs/export/autoencoder.yaml`

        Example Usage:
        Args:
            autoencoder (AutoEncoder): AutoEncoder model
            datamodule (DataModule): DataModule used to compute stats
            stats_batches (int): number of batches to use to compute stats
            **kwargs: Additional arguments to the AudioAutoEncoder constructor

        Returns:
            AudioAutoEncoder: AudioAutoEncoder component

        """
        ae_config = autoencoder._hydra_external_config
        ae_config["exclude_from_checkpoint"] = True  # exclude from checkpointing
        ae = AudioAutoEncoder(ae_config)
        ae.autoencoder.load_state_dict(autoencoder.state_dict())
        if datamodule is None:
            log.warning("No datamodule provided, stats will not be computed")
            ae.z_mean.zero_()
            ae.z_std.fill_(1.0)
            return ae

        # compute stats

        datamodule.setup()
        dataloader = datamodule.train_dataloader()
        len_dataloader = len(dataloader)
        if stats_batches is not None:
            len_dataloader = min(len_dataloader, max(0, stats_batches))

        device = next(ae.parameters()).device
        ae.to("cuda")
        ae.z_mean.zero_()
        ae.z_std.zero_()
        ae.eval()
        with torch.no_grad():
            # compute mean and stdev of raw forward latents

            for n, d in tqdm(
                enumerate(dataloader),
                desc=f"Computing AutoEncoder stats",
                total=len_dataloader,
                leave=False,
            ):
                # add batch dim
                audio = d["audio"].to("cuda")
                z = ae.autoencoder.encode(audio)
                ae.z_mean += z.mean((0, 2))
                ae.z_std += z.std((0, 2))

                if stats_batches is not None and (n + 1) >= stats_batches:
                    break

            ae.z_mean /= n + 1
            ae.z_std /= n + 1
        assert not torch.isnan(ae.z_mean).any()
        assert not torch.isnan(ae.z_std).any()
        ae.to(device)
        print(f"Computed mean: {ae.z_mean.cpu().numpy()}")
        print(f"Computed std: {ae.z_std.cpu().numpy()}")
        return ae
