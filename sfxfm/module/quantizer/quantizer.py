"""latent quantizer base class"""

from einops import rearrange
from torch import nn
import torch
from sfxfm.module.model.autoencoder import AutoEncoder
from .embedding import Embedding
from sfxfm.utils.dist import rank
from pytorch_lightning import LightningDataModule
import logging
import os
from tqdm import tqdm
import typing as tp
from sfxfm.utils.cache import Cache
from torch.distributed import barrier

log = logging.getLogger(__name__)
rank = rank()


class Quantizer(nn.Module):
    """
    Latent quantizer base class
    acts as an Identity layer on z

    Returns code index -1

    Initialize its codebook on the latent space of a pretrained_autoencoder
    over the provided datamodule

    Results may be cached

    only the methods
    - quantize
    - initialize_codebook
    need to be reimplemented for child classes


    """

    def __init__(
        self,
        datamodule: LightningDataModule,
        pretrained_autoencoder: AutoEncoder,
        embedding: Embedding,
        cache: Cache,
        seed: int = 0,
        max_samples=10000,
        batch_size=16,
        dim=32,
        normalize=True,
        *kwargs,
    ):
        """
        datamodule is used to initialize the codebook

        attributed is_initialized can be used as a boolean that's stored in the state_dict
        """
        super().__init__()
        self.datamodule = datamodule
        if pretrained_autoencoder is not None:
            self.pretrained_autoencoder = pretrained_autoencoder.cuda()
        self.embedding = embedding
        self.cache = cache
        self.seed = seed
        if self.datamodule is not None:
            self.datamodule.setup("fit")
        if not hasattr(self, "is_initialized"):
            self.register_buffer("_is_initialized", torch.tensor([False]))
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.dim = dim
        self.normalize = normalize

    @property
    def is_initialized(self):
        return self._is_initialized.item()

    @is_initialized.setter
    def is_initialized(self, value: bool):
        self._is_initialized[:] = True

    @torch.no_grad()
    def initialize_codebook(self, z_table) -> None:
        """
        This function is called to initialize the codebooks
        and must be implemented

        Only the creation of the codebook from a large collection
        of z z_table must be implemented

        z_table is (...,  z_dim)
        """

        raise NotImplementedError

    def broadcast_buffers(self, src=0):
        raise NotImplementedError

    def forward(
        self, z: torch.Tensor, return_residuals: bool = False
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward returns a tuple
        z_q, indices, quantization_loss

        input: z (batch_size, ..., dim)
        -------------
        output:
        z_q: same as z

        """
        if not self.is_initialized:
            with torch.no_grad():
                if rank == 0:
                    self._initialize_and_evaluate_codebook()
                self.broadcast_buffers(src=0)
                if rank != 0:
                    log.info(f"rank {rank} waiting for rank 0")
                barrier()
                log.info(f"{rank} passed the barrier")
            # set flag to True
            self.is_initialized = True

        z_q, idx, residual, quantization_loss = self.batch_quantize(z)
        # if self.normalize:
        #    z_q = z_q * self.z_std[None, :, None] + self.z_mean[None, :, None]
        if return_residuals:
            return z_q, idx, residual, quantization_loss
        else:
            return z_q, idx, quantization_loss

    def batch_quantize(self, z: torch.Tensor):
        z_batches = torch.split(z, self.batch_size, dim=0)
        device = z.device

        z_q = []
        idx = []
        residual = []
        quantization_loss = []

        multiple_quantizers = False
        for batch in z_batches:
            batch_z_q, batch_idx, batch_residual, batch_quantization_loss = (
                self.quantize(batch)
            )
            if isinstance(batch_idx, tp.List):
                if len(idx) == 0:
                    idx = [torch.empty(0, device=device, dtype=torch.int)] * len(
                        batch_idx
                    )
                    multiple_quantizers = True
                for i in range(len(batch_idx)):
                    idx[i] = torch.cat([idx[i], batch_idx[i]], dim=0)
            else:
                idx.append(batch_idx)

            z_q.append(batch_z_q)
            residual.append(batch_residual)
            quantization_loss.append(batch_quantization_loss)

        z_q = torch.cat(z_q, dim=0)
        residual = torch.cat(residual, dim=0)
        quantization_loss = torch.cat(quantization_loss, dim=0)
        if not multiple_quantizers:
            idx = torch.cat(idx, dim=0)

        return z_q, idx, residual, quantization_loss

    def quantize(self, z: torch.Tensor):
        """
        Main method that needs to be reimplemented by child classes
        """
        z_q = z
        idx = -torch.ones((z.size(0)), device=z.device)
        quantization_loss = torch.zeros((z.size(0)), device=z.device)
        residual = z - z_q
        return z_q, idx, residual, quantization_loss

    def _load_latents(self):
        """
        Returns a dataset of latents as a large tensor

        """
        assert self.datamodule is not None
        assert self.pretrained_autoencoder is not None
        dataloader = self.datamodule.train_dataloader()
        n_samples = len(dataloader)
        log.info(f"Loading dataset, {n_samples} samples")

        datasets = "_".join(self.datamodule.train_datasets.keys())
        model_name = self.pretrained_autoencoder.experiment_name
        norm_flag = "_norm" if self.normalize else ""
        expname = f"{datasets}_{model_name}{norm_flag}"
        filename = expname + ".pt"

        cache_dir = self.cache.cache_dir / "quantization_datasets"
        filepath = cache_dir / expname

        ckpt_hash = getattr(self.pretrained_autoencoder, "checkpoint_hash", "")
        exp_name = getattr(self.pretrained_autoencoder, "experiment_name", "")

        if self.normalize and len(exp_name) > 0:
            norm_path = self.cache.cache_dir / "preprocessor" / exp_name
            if ckpt_hash is not None:
                norm_path = norm_path / ckpt_hash

            assert not self.cache.enter(norm_path), "Normalizing data not found"

            self.register_buffer("z_mean", torch.load(norm_path / "z_mean.pt"))
            self.register_buffer("z_std", torch.load(norm_path / "z_std.pt"))

        if self.cache.enter(filepath, force=False):
            if rank == 0:
                log.info(f"{filepath} not found, computing data from scratch")
                filepath.mkdir(parents=True, exist_ok=True)
                latents = []
                audios = []
                with torch.no_grad():
                    for n, d in tqdm(
                        enumerate(dataloader),
                        desc=f"Computing Latent Dataset",
                        total=len(dataloader),
                        leave=False,
                    ):
                        audio = d["audio"].to("cuda")
                        if len(audio.shape) == 2:
                            audio = audio.unsqueeze(0)
                        z = self.pretrained_autoencoder.encode(
                            audio, stochastic_latent=False
                        )
                        if self.normalize:
                            z = (z - self.z_mean[None, :, None]) / self.z_std[
                                None, :, None
                            ]
                        assert z.size(1) == self.dim
                        latents.append(z)
                        audios.append(audio)
                    z_dataset = torch.cat(latents, dim=0)
                    audio_dataset = torch.cat(audios, dim=0)
                    dataset = {"audio": audio_dataset, "latent": z_dataset}
                torch.save(dataset, str(filepath / filename))
                self.cache.signal_done(filepath)
        else:
            self.cache.wait_done(filepath)
            log.info("cache found, loading data from cache")
            dataset = torch.load(filepath / filename)

        del self.datamodule
        return dataset, n_samples

    @torch.no_grad()
    def evaluate(self, z_table):
        """
        z_table: (..., z_dim)
        """
        pass

    @torch.no_grad()
    def _initialize_and_evaluate_codebook(self):
        log.info("Initializing post-training latent quantization")

        dataset, n_samples = self._load_latents()
        latents = dataset["latent"]
        audios = dataset["audio"]

        _, z_dim, t = latents.size()

        z_table = latents.cuda()
        audio_table = audios.cuda()

        n_samples = z_table.size(0)
        n_samples_train = min(self.max_samples, int(0.5 * n_samples))  # max capacity
        n_samples_test = int(0.2 * n_samples_train)

        log.info(f"quantization z set size {n_samples_train}")
        log.info(f"evaluation z set size {n_samples_test}")
        log.info(f"latent dimension : {z_dim}")

        # put channels last
        z_table = rearrange(z_table, "b c t -> b t c")
        z_table_train = torch.narrow(
            z_table,
            dim=0,
            start=0,
            length=n_samples_train,
        )

        z_table_test = torch.narrow(
            z_table,
            dim=0,
            start=n_samples_train,
            length=n_samples_test,
        )

        self.z_table = z_table
        self.audio_table = audio_table
        log.info("Computing quantizers")

        self.initialize_codebook(z_table=z_table_train)
        self.is_initialized = True

        log.info("Evaluating quantization")
        self.evaluate(z_table=z_table_test)

    def on_load_checkpoint(self, checkpoint):
        """Can change checkpoint keys to be loaded here.
        We remove the saved metrics if any exist in the checkpoint.
        """
        state_dict = checkpoint["state_dict"]
        checkpoint["state_dict"] = {k: state_dict[k] for k in self.state_dict().keys()}
