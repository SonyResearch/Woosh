"""
Module to train encoders
"""

import lightning.pytorch as pl
import torch

from sfxfm.module.loss import AELoss
from sfxfm.module.model.autoencoder import (
    VariationalAutoEncoder,
)
from sfxfm.module.model.vq_autoencoder import VQAutoEncoder
from .base import BaseLightningModule
from sfxfm.utils.dist import rank
import logging

rank = rank()

# get logger
log = logging.getLogger(__name__)


class VQAutoEncoderModule(BaseLightningModule):
    """
    LightningModule to train an autoencoder

    """

    def __init__(
        self,
        model: VQAutoEncoder,
        loss: AELoss,
        optim_enc,
        optim_dec,
        optim_disc,
        metric_collection_train,
        metric_collection_val,
        disc_training_start=-1,
        train_disc_every=1,
        ae_clip_grad_value=None,
        ae_clip_grad_algorithm="value",
        disc_clip_grad_value=None,
        disc_clip_grad_algorithm="value",
        encoder_freeze_start=None,
        **kwargs,
    ):
        super().__init__()
        self.autoencoder = model.cuda()
        self.optim_config_enc = optim_enc
        self.optim_config_dec = optim_dec
        self.optim_config_disc = optim_disc
        self.loss = loss
        self.automatic_optimization = False
        self.metric_collection_train = metric_collection_train
        self.metric_collection_val = metric_collection_val

        # Discriminator training
        self.disc_training_start = disc_training_start
        assert self.disc_training_start <= self.loss.adv_loss_start
        self.train_disc_every = train_disc_every
        self.ae_clip_grad_value = ae_clip_grad_value
        self.ae_clip_grad_algorithm = ae_clip_grad_algorithm
        self.disc_clip_grad_value = disc_clip_grad_value
        self.disc_clip_grad_algorithm = disc_clip_grad_algorithm
        self.encoder_freeze_start = encoder_freeze_start
        self.encoder_freeze_state = False

    def configure_optimizers(self):
        dec_parameters = list(self.autoencoder.decoder.parameters())
        if hasattr(self.loss, "logvar"):
            dec_parameters += [self.loss.logvar]
            assert isinstance(self.autoencoder, VariationalAutoEncoder)
        optimizer_enc = self.optim_config_enc.optimizer(
            params=self.autoencoder.encoder.parameters()
        )
        optimizer_dec = self.optim_config_dec.optimizer(params=dec_parameters)
        optimizer_disc = self.optim_config_disc.optimizer(
            params=self.loss.discriminator.parameters()
        )

        scheduler_enc = {
            "scheduler": self._build_scheduler(
                self.optim_config_enc.scheduler, optimizer=optimizer_enc
            ),
            "name": "enc_lr",
        }
        scheduler_dec = {
            "scheduler": self._build_scheduler(
                self.optim_config_dec.scheduler, optimizer=optimizer_dec
            ),
            "name": "dec_lr",
        }
        scheduler_disc = {
            "scheduler": self._build_scheduler(
                self.optim_config_disc.scheduler, optimizer=optimizer_disc
            ),
            "name": "disc_lr",
        }

        return (
            [optimizer_enc, optimizer_dec, optimizer_disc],
            [scheduler_enc, scheduler_dec, scheduler_disc],
        )

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

    def freeze(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze(self, module):
        module.train()
        for param in module.parameters():
            param.requires_grad = True

    def do_nothing_closure(self):
        return

    def training_step(self, batch, batch_idx):
        enc_opt, dec_opt, d_opt = self.optimizers()  # type: ignore
        enc_lr_sch, dec_lr_sch, d_lr_sch = self.lr_schedulers()  # type: ignore
        d_opt._on_before_step = self.do_nothing_closure
        d_opt._on_after_step = self.do_nothing_closure
        enc_opt._on_before_step = self.do_nothing_closure
        enc_opt._on_after_step = self.do_nothing_closure

        inputs = batch["audio"]

        inputs = self.autoencoder.fix_input_length(inputs)
        reconstructions, posterior, indices, commitment_loss = self.autoencoder(
            inputs, return_indices=True
        )
        commitment_loss = self.autoencoder.commitment_loss * commitment_loss.sum()

        train_disc = (
            self.global_step > self.disc_training_start
            and self.global_step % self.train_disc_every == 0
        )

        if train_disc:
            self.unfreeze(self.loss.discriminator)
            disc_loss, disc_losses, log_dict_disc = self.loss.discriminator_loss(
                inputs=inputs,
                reconstructions=reconstructions.detach(),
                global_step=self.global_step,
            )

            d_opt.zero_grad()
            self.manual_backward(disc_loss)
            if self.disc_clip_grad_value is not None:
                self.clip_gradients(
                    d_opt,
                    gradient_clip_val=self.disc_clip_grad_value,
                    gradient_clip_algorithm=self.disc_clip_grad_algorithm,
                )
            with torch.no_grad():
                disc_param_norm = self.get_aggregate_param_norm(self.loss.discriminator)
                disc_grad_norm = self.get_aggregate_grad_norm(self.loss.discriminator)
            log_dict_disc["disc_param_norm"] = disc_param_norm
            log_dict_disc["disc_grad_norm"] = disc_grad_norm
            d_opt.step()
            d_lr_sch.step()  # type: ignore
        else:
            log_dict_disc = {}

        self.freeze(self.loss.discriminator)

        if self.encoder_freeze_start is not None:
            if (
                self.global_step > self.encoder_freeze_start
                and not self.encoder_freeze_state
            ):
                self.freeze(self.autoencoder.encoder)
                self.encoder_freeze_state = True

        ae_loss, log_dict_ae = self.loss.generator_loss(
            inputs=inputs,
            reconstructions=reconstructions,
            posteriors=posterior,
            global_step=self.global_step,
            disc_loss=disc_loss if train_disc else None,
        )

        enc_opt.zero_grad()
        dec_opt.zero_grad()
        self.manual_backward(ae_loss + commitment_loss)
        if self.ae_clip_grad_value is not None:
            if not self.encoder_freeze_state:
                self.clip_gradients(
                    enc_opt,
                    gradient_clip_val=self.ae_clip_grad_value,
                    gradient_clip_algorithm=self.ae_clip_grad_algorithm,
                )
            self.clip_gradients(
                dec_opt,
                gradient_clip_val=self.ae_clip_grad_value,
                gradient_clip_algorithm=self.ae_clip_grad_algorithm,
            )
        ae_param_norm = self.get_aggregate_param_norm(self.autoencoder)
        ae_grad_norm = self.get_aggregate_grad_norm(self.autoencoder)
        log_dict_ae["ae_param_norm"] = ae_param_norm
        log_dict_ae["ae_grad_norm"] = ae_grad_norm
        log_dict_ae["commitment_loss"] = commitment_loss.item()
        enc_opt.step()
        dec_opt.step()
        enc_lr_sch.step()  # type: ignore
        dec_lr_sch.step()  # type: ignore

        with torch.no_grad():
            self.log_everything(
                phase="train",
                ae_loss=ae_loss,
                log_dict={**log_dict_ae, **log_dict_disc},
                batch_size=len(inputs),
            )

        return {
            "ae_loss": ae_loss,
            "ae_param_norm": ae_param_norm,
            "ae_grad_norm": ae_grad_norm,
            "preds": reconstructions,
            "target": inputs,
        }

    def log_everything(
        self, phase="", ae_loss=None, log_dict={}, batch_size=None, sync_dist=True
    ):
        """
        reco_loss: unreduced reconstruction loss
        """
        if phase != "" and not phase.endswith("/"):
            phase += "/"
        # only needed for monitoring:
        if ae_loss is not None:
            self.log(
                f"{phase}loss",
                (ae_loss).mean().item(),
                prog_bar=True,
                sync_dist=sync_dist,
                add_dataloader_idx=False,
                batch_size=batch_size,
            )
        # dict
        if len(log_dict) > 0:
            # add phase in keys
            log_dict = {f"{phase}{key}": value for key, value in log_dict.items()}
            # Log dict
            self.log_dict(
                log_dict,
                sync_dist=sync_dist,
                batch_size=batch_size,
                add_dataloader_idx=False,
            )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # work with a list of batches that covers single-batch and multi-batch cases
        batches = batch.values() if "batch0" in batch else [batch]

        for batch in batches:
            inputs = batch["audio"]

            inputs = self.autoencoder.fix_input_length(inputs)

            reconstructions, posterior, indices, commitment_loss = self.autoencoder(
                inputs, return_indices=True
            )

            if self.global_step > self.disc_training_start:
                disc_loss, disc_losses, log_dict_disc = self.loss.discriminator_loss(
                    inputs=inputs,
                    reconstructions=reconstructions.detach(),
                    global_step=self.global_step,
                )
            else:
                disc_loss, log_dict_disc = torch.zeros((1,), device=inputs.device), {}

            ae_loss, log_dict_ae = self.loss.generator_loss(
                inputs=inputs,
                reconstructions=reconstructions,
                posteriors=posterior,
                global_step=self.global_step,
            )

            # no sync_dist since other ranks may have a different number of
            # batches per sample in the most general case
            # ranks might go out of sync with sync_dist
            self.log_everything(
                phase=self.val_names[dataloader_idx],
                ae_loss=ae_loss,
                log_dict={**log_dict_ae, **log_dict_disc},
                batch_size=len(inputs),
            )
        return {
            "disc_loss": disc_loss,
            "ae_loss": ae_loss,
            "preds": reconstructions,
            "target": inputs,
            "indices": indices,
        }

    def autoencode(self, input, stochastic_latent=True):
        reconstruction, _, _, _ = self.autoencoder.forward(
            input, stochastic_latent=stochastic_latent
        )
        return reconstruction

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # fix time axis before updating any metric. target and pred may have different
        # lengths if chunks_size is not a power of 2
        with torch.no_grad():
            if outputs["target"].size(-1) != outputs["preds"].size(-1):
                raise RuntimeError(
                    f"Inputs {outputs['target'].shape} and reconstructions {outputs['preds'].shape} have different shapes. Make sure to use the method `fix_input_length` "
                )
            self.metric_collection_train.update(outputs["preds"], outputs["target"])  # type: ignore

    def on_train_epoch_end(self):
        log_dict = {**self.metric_collection_train.compute()}
        self.log_everything(phase="train", log_dict=log_dict, batch_size=1)
        self.metric_collection_train.reset()

    def on_validation_start(self):
        # one metric_collection_val per validation dataloader
        # not possible to create it at init time, as datamodule
        # is not yet ready
        # internal states must be separate for each dataloader
        if not hasattr(self, "valid_is_initialized"):
            self.val_names = [name for name in self.trainer.datamodule.val_dataset()]
            if self.metric_collection_val is not None:
                metric_collection_val = self.metric_collection_val
                self.metric_collection_val = torch.nn.ModuleList(
                    [metric_collection_val.clone() for _ in self.val_names]
                )
            self.valid_is_initialized = True

        # if currently in a sanity validation loop, log at epoch -1
        if self.trainer._evaluation_loop.inference_mode:
            self.trainer.fit_loop.epoch_progress.current.completed = -1

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # fix time axis before updating any metric. target and pred may have different
        # lengths if chunks_size is not a power of 2
        if outputs["target"].size(-1) != outputs["preds"].size(-1):
            raise RuntimeError(
                f"Inputs {outputs['target'].shape} and reconstructions {outputs['preds'].shape} have different shapes. Make sure to use the method `fix_input_length` "
            )
        self.metric_collection_val[dataloader_idx].update(
            outputs["preds"], outputs["target"], batch
        )

    def on_validation_epoch_end(self):
        for i in range(len(self.metric_collection_val)):
            log_dict = {**self.metric_collection_val[i].compute()}
            self.log_everything(
                phase=self.val_names[i], log_dict=log_dict, batch_size=1
            )
            self.metric_collection_val[i].reset()

    def on_load_checkpoint(self, checkpoint):
        """Can change checkpoint kcheckpoint['state_dict'].eys to be loaded here.
        We remove the saved metrics if any exist in the checkpoint.
        """
        checkpoint = checkpoint["state_dict"]
        # fix loading old checkpoints, after PR #612
        if "autoencoder.encoder.stft_embed.window" in checkpoint.keys():
            checkpoint["autoencoder.encoder.spec_embed.window"] = checkpoint[
                "autoencoder.encoder.stft_embed.window"
            ]
            del checkpoint["autoencoder.encoder.stft_embed.window"]

        # don't load metrics if they exist
        for k in list(checkpoint.keys()):
            if "metric_collection_" in k:
                del checkpoint[k]
        # Put the current metrics
        for k, v in self.state_dict().items():
            if "metric_collection_" in k:
                checkpoint[k] = v

    def on_save_checkpoint(self, checkpoint):
        """Can change checkpoint keys to be saved here"""
        checkpoint = checkpoint["state_dict"]
        # don't save metrics if they exist
        for k in list(checkpoint.keys()):
            if "metric_collection_" in k:
                del checkpoint[k]

    def get_aggregate_param_norm(self, model, p=1.0) -> float:
        """
        Compute the aggregated p-norm of the model parameters. This is typically
        called after the backward pass.
        """
        acc_norm = 0.0
        for param in model.parameters():
            if param.requires_grad:
                acc_norm += torch.norm(param, p).item()
        return acc_norm

    def get_aggregate_grad_norm(self, model, p=1.0) -> float:
        """
        Compute the aggregated p-norm of the model partial gradients. This is typically
        called after the backward pass.
        """
        acc_norm = 0.0
        for param in model.parameters():
            if param.requires_grad:
                if param.grad is None:
                    continue
                acc_norm += torch.norm(param.grad, p).item()
        return acc_norm
