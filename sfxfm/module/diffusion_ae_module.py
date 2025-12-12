from copy import deepcopy
import torch

from .utils import ema_update
import numpy as np
from .base import BaseLightningModule


class DiffusionAutoEncoderModule(BaseLightningModule):
    """LightningModule to train a diffusion attention Unet


    Args:
    """

    def __init__(
        self,
        diffusion_model,
        encoder,
        preprocessor,
        ema_decay,
        optim,
        quantizer,
        t_schedule,
        condition_dropout=0.0,
        fixed_encoder=False,
        normalize_audio=True,
        **kwargs,
    ):
        super().__init__()
        self.diffusion = diffusion_model
        self.diffusion_ema = deepcopy(self.diffusion)
        self.diffusion_ema.requires_grad_(False)
        self.ema_decay = ema_decay
        self.encoder = encoder
        self.preprocessor = preprocessor
        self.preprocessor.requires_grad_(False)
        self.preprocessor.eval()
        self.t_schedule = t_schedule

        self.quantizer = quantizer

        # self.n_q = self.quantizer.n_q
        # self.main_rq_levels = self.quantizer.main_rq_levels
        # self.p_rq_levels = self.quantizer.p_rq_levels

        # this rng is only used to select the number of layers in RQ during training
        # + if droping conditional info or not
        # It is sync between the threads
        self.rng = np.random.default_rng(12345)

        # this rng is NOT sync between threads
        self.rng_sobol = torch.quasirandom.SobolEngine(1, scramble=True, seed=None)

        self.condition_dropout = condition_dropout

        self.optim_config = optim

        # for fixing the encoder
        self.fixed_encoder = fixed_encoder
        if self.fixed_encoder:
            self.encoder.requires_grad_(False)
            self.quantizer.requires_grad_(False)

        self._normalize_audio = normalize_audio

    def on_fit_start(self):
        # set up validation set names
        super().on_fit_start()

    def configure_optimizers(self):
        optimizer = self.optim_config.optimizer(params=self.parameters())
        # scheduler = self.optim_config.scheduler(optimizer=optimizer)
        scheduler = self._build_scheduler(
            scheduler_config=self.optim_config.scheduler, optimizer=optimizer
        )

        return ([optimizer], [scheduler])

    def _build_scheduler(self, scheduler_config, optimizer):
        # All scheduler_configs contain SequentialLR Scheduler-like kwargs
        # milestones: list[int]
        # schedulers: dict[partial_schedulers]
        # note that the dict must be ordered correctly
        schedulers = [
            partial_sch(optimizer)
            for partial_sch in scheduler_config.schedulers.values()
        ]
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=schedulers,
            milestones=scheduler_config["milestones"],
        )

    # base sampling function associated to DiffusionModule
    # used in callbacks for instance
    def sample(self, noise, cond, n_q=None, **kwargs):
        raise NotImplementedError

    def fix_encoder(self):
        if self.fixed_encoder:
            # EMA in RVQ is influenced by this
            self.encoder.eval()
            self.quantizer.eval()

    def denoise(self, x_t, t, cond=None, n_q=None, ema=False):
        """
        Wrapper around network diffusion or diffusion_ema
        Selects ema or non-ema weights
        Can be reimplemented for preconditioning
        """
        # if type(n_q) == int:
        #     batch_size = x_t.size(0)
        #     n_q = torch.Tensor([n_q] * batch_size).long().to(x_t.device)

        if ema:
            self.diffusion_ema.eval()
            return self.diffusion_ema(x_t, t, cond, n_q)
        else:
            return self.diffusion(x_t, t, cond, n_q)

    def extract_cond(self, x, n_q=None):
        self.fix_encoder()
        # no conditioning
        # if n_q == 0:
        #     return x, None
        # # TODO: remove warnings
        # if n_q is None:
        #     print("WARNING: extract_cond receives n_q is None")

        # if type(n_q) == int:
        #     batch_size = x.size(0)
        #     n_q = torch.Tensor([n_q] * batch_size).long().to(x.device)

        z = self.encoder(x)

        zq, codes, quantization_loss = self.quantizer(z)

        reals = x
        cond = zq
        return reals, cond

    def codes(self, x, n_q=None):
        # no conditioning

        # if n_q is None:
        #     print("WARNING: codes receives n_q is None")

        # if type(n_q) == int:
        #     batch_size = x.size(0)
        #     n_q = torch.Tensor([n_q] * batch_size).long().to(x.device)
        # if batch_size > 1:
        #     print(
        #         'WARNING: using codes with batch_size > 1 and non batched n_q'
        #     )

        z = self.encoder(x)
        zq, codes, quantization_loss = self.quantizer(z)

        return codes

    def sample_t(self, x):
        return self.t_schedule.sample_t(x)

    # def fun(self, t, x, cond, n_q=None):
    #     self.diffusion_ema.eval()

    #     if n_q is None:
    #         raise NotImplementedError

    #     if type(n_q) == int:
    #         batch_size = x.size(0)
    #         n_q = torch.Tensor([n_q] * batch_size).long().to(x.device)

    #     return self.diffusion_ema(x, t, cond=cond, n_q=n_q)

    def normalize_audio(self, x):
        if self._normalize_audio:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
        return x

    def loss_fn(self, reals, cond, n_q, noise, t, quantization_loss, ema):
        """
        From reals and noise, computes objective function
        ema (bool): choose between ema or non ema weights
        """
        raise NotImplementedError

    def autoencode(self, input, **kwargs):
        # Encode:
        z = self.encoder(input)
        zq, codes, quantization_loss = self.quantizer(z)
        cond = zq

        noise = torch.randn_like(input)
        return self.sample(noise, cond)

    def training_step(self, batch, batch_idx):

        self.fix_encoder()
        x = batch["audio"]
        with torch.no_grad():
            x = self.normalize_audio(x)
            x = self.preprocessor(x)

        # Draw well distributed continuous timesteps
        t = self.sample_t(x=x)

        # Compute condition
        if self.rng.uniform() < self.condition_dropout:
            print("DEPRECATED condition_dropout")
            n_q = 0
            cond = None
            quantization_loss = torch.zeros_like(t)
            all_losses = None
        else:
            n_q = None
            # TODO different distribution?
            # n_q = self.rng.choice(self.main_rq_levels, 1)[0]

            batch_size = x.size(0)
            # n_q = torch.IntTensor(
            #     self.rng.choice(
            #         self.main_rq_levels,
            #         batch_size,
            #         p=self.p_rq_levels,
            #     )
            # ).to(device=x.device)

            # TODO noise x?
            # how to noise x
            # range of noise values
            # x_encoder = x + 1e-2 * torch.randn_like(x)
            x_encoder = x

            # Encode:
            z = self.encoder(x_encoder)
            zq, codes, quantization_loss = self.quantizer(z)
            cond = zq

        # get inputs
        reals = x
        noise = torch.randn_like(reals)

        loss, reco_loss = self.loss_fn(
            reals, cond, n_q, noise, t, quantization_loss, ema=False
        )

        with torch.no_grad():

            self.log_everything(
                reco_loss=reco_loss,
                loss=loss,
                quantization_loss=quantization_loss,
                n_q=n_q,
                phase="train",
            )

            # prog bar:
            self.log("loss", loss.mean().item(), prog_bar=True)

        return loss

    def log_everything(self, reco_loss, loss, quantization_loss, n_q, phase):
        """
        reco_loss: unreduced reconstruction loss
        """
        log_dict = {}

        # MSE per t-bin
        mse_loss_per_band = reco_loss
        loss_per_band = {
            f"{phase}/mse_band#{k}": l.item()
            for k, l in enumerate(mse_loss_per_band.mean(2).mean(0))
        }
        log_dict.update(loss_per_band)

        # loss_per_ex = mse_loss_per_band.mean(2).mean(1)

        # for rq_level in self.main_rq_levels:
        #     if (n_q == rq_level).any().item():
        #         l = loss_per_ex[n_q == rq_level].mean().detach().item()

        #         # MSE per quantization level
        #         loss_per_nq = {f"{phase}/mse_nq#{rq_level}": l}
        #         log_dict.update(loss_per_nq)

        # if we quantized and conditioned
        # if len(codebook_error.size()) > 0:
        #     quantization_loss_per_quantizer = {
        #         f"{phase}/quantizer#{k}_loss": l.mean().item()
        #         for k, l in enumerate(codebook_error)
        #     }
        #     log_dict.update(quantization_loss_per_quantizer)

        # quantization loss is the one used during training (so depends on n_q)

        log_dict.update(
            {
                f"{phase}/loss": loss.detach().item(),
                # f"{phase}/quantization_loss": quantization_loss.mean().detach().item(),
            }
        )
        # TODO loss per t-bin?

        # self.log_dict(log_dict, prog_bar=True, on_step=True)
        self.log_dict(log_dict, sync_dist=phase == "val")
        del mse_loss_per_band

    def validation_step(self, batch, batch_idx):
        # print(f'=====\n batch_id: {batch_idx} \t rank={torch.distributed.get_rank()} \n {self.diffusion.dbranch.dblocks[0].downsample.conv.norm.weight.detach().cpu().numpy()}')

        x = batch["audio"]

        # CUT AUDIO TO REMOVE
        x = x[:, :, :16384]
        x = self.normalize_audio(x)

        x = self.preprocessor(x)

        # Draw uniformly  distributed continuous timesteps
        # TODO better split
        # t = torch.rand_like(x[:, 0, 0])
        t = self.sample_t(x)

        # Encode:
        z = self.encoder(x)

        # z = rearrange(z, "b c l -> b l c")
        # max quantization
        # n_q = torch.Tensor([self.n_q] * z.size(0)).int().to(z.device)
        n_q = None
        zq, codes, quantization_loss = self.quantizer(z)
        # get inputs
        reals = x
        noise = torch.randn_like(reals)
        cond = zq

        loss, reco_loss = self.loss_fn(
            reals, cond, n_q, noise, t, quantization_loss, ema=True
        )

        self.log_everything(
            reco_loss=reco_loss,
            loss=loss,
            quantization_loss=quantization_loss,
            n_q=n_q,
            phase="valid",
        )

        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        decay = 0.95 if self.current_epoch < 25 else self.ema_decay
        ema_update(self.diffusion, self.diffusion_ema, decay)
