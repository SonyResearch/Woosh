"""
Module to train Avocodo encoder
"""

import logging
import torch

from sfxfm.module.model.autoencoder import VariationalAutoEncoder
from sfxfm.module.encoder_module import AutoEncoderModule
from sfxfm.module.discriminator.combd import CoMBD
from sfxfm.module.discriminator.sbd import SBD
from sfxfm.module.model.avocodo import init_weights

from sfxfm.module.discriminator.pqmf import PQMF
from sfxfm.utils.dist import rank

rank = rank()

# get logger
log = logging.getLogger(__name__)


class AvocodoModule(AutoEncoderModule):
    """
    LightningModule to train an autoencoder

    """

    def __init__(
        self,
        pqmf_config,
        combd_config,
        sbd_config,
        use_combd=True,
        use_sbd=True,
        use_hierarchical=False,
        use_multi_scale=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pqmf_lv2 = PQMF(*pqmf_config["lv2"])
        self.pqmf_lv1 = PQMF(*pqmf_config["lv1"])
        self.use_combd = use_combd
        self.use_sbd = use_sbd

        self.discs = {}

        if self.use_combd:
            self.combd = CoMBD(
                combd_config=combd_config,
                pqmf_list=[self.pqmf_lv2, self.pqmf_lv1],
                use_hierarchical=use_hierarchical,
                use_multi_scale=use_multi_scale,
            )
            init_weights(self.combd)
            self.discs.update({"combd": self.combd})
        if self.use_sbd:
            self.sbd = SBD(sbd_config=sbd_config)
            init_weights(self.sbd)
            self.discs.update({"sbd": self.sbd})

    def configure_optimizers(self):
        dec_parameters = list(self.autoencoder.decoder.parameters())
        if hasattr(self.loss, "logvar"):
            dec_parameters += [self.loss.logvar]
            assert isinstance(self.autoencoder, VariationalAutoEncoder)
        optimizer_enc = self.optim_config_enc.optimizer(
            params=self.autoencoder.encoder.parameters()
        )

        optimizer_dec = self.optim_config_dec.optimizer(params=dec_parameters)

        disc_params = list(())
        if self.use_combd:
            disc_params += list(self.combd.parameters())
        if self.use_sbd:
            disc_params += list(self.sbd.parameters())

        optimizer_disc = self.optim_config_disc.optimizer(disc_params)

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

    def training_step(self, batch, batch_idx):
        enc_opt, dec_opt, d_opt = self.optimizers()  # type: ignore
        enc_lr_sch, dec_lr_sch, d_lr_sch = self.lr_schedulers()  # type: ignore
        d_opt._on_before_step = self.do_nothing_closure
        d_opt._on_after_step = self.do_nothing_closure
        enc_opt._on_before_step = self.do_nothing_closure
        enc_opt._on_after_step = self.do_nothing_closure

        inputs = batch["audio"]
        # down sampling in temporal domain
        inputs_multi_scale = [
            self.pqmf_lv2.analysis(inputs)[
                :, : self.autoencoder.decoder.projection_filters[1]
            ],
            self.pqmf_lv1.analysis(inputs)[
                :, : self.autoencoder.decoder.projection_filters[2]
            ],
            inputs,
        ]

        outs_multi_scale, posterior = self.autoencoder(inputs)

        train_disc = (
            self.global_step > self.disc_training_start
            and self.global_step % self.train_disc_every == 0
        )

        if train_disc:
            for disc in self.discs.values():
                self.unfreeze(disc)

            detached_outs_multi_scale = [x.detach() for x in outs_multi_scale]

            outs_real = {}
            outs_fake = {}

            if self.use_combd:
                outs_real_co, outs_fake_co, _, _ = self.combd(
                    inputs_multi_scale, detached_outs_multi_scale
                )
                outs_real.update(outs_real_co)
                outs_fake.update(outs_fake_co)
            if self.use_sbd:
                outs_real_sbd, outs_fake_sbd, _, _ = self.sbd(
                    inputs, detached_outs_multi_scale[-1]
                )
                outs_real.update(outs_real_sbd)
                outs_fake.update(outs_fake_sbd)

            disc_loss, _, log_dict_disc = self.loss.discriminator_loss(
                outs_real=outs_real,
                outs_fake=outs_fake,
            )

            d_opt.zero_grad()
            self.manual_backward(disc_loss)
            if self.disc_clip_grad_value is not None:
                self.clip_gradients(
                    d_opt,
                    gradient_clip_val=self.disc_clip_grad_value,
                    gradient_clip_algorithm=self.disc_clip_grad_algorithm,
                )

            for name_disc, disc in self.discs.items():
                disc_param_norm = self.get_aggregate_param_norm(disc)
                disc_grad_norm = self.get_aggregate_grad_norm(disc)
                log_dict_disc[f"disc_param_norm_{name_disc}"] = disc_param_norm
                log_dict_disc[f"disc_grad_norm_{name_disc}"] = disc_grad_norm

            d_opt.step()
            d_lr_sch.step()  # type: ignore
        else:
            log_dict_disc = {}

        for disc in self.discs.values():
            self.freeze(disc)

        if self.encoder_freeze_start is not None:
            if (
                self.global_step > self.encoder_freeze_start
                and not self.encoder_freeze_state
            ):
                self.freeze(self.autoencoder.encoder)
                self.encoder_freeze_state = True

        detached_outs_multi_scale = [x.detach() for x in outs_multi_scale]

        outs_real = {}
        outs_fake = {}
        fmap_real = {}
        fmap_fake = {}

        if self.use_combd:
            outs_real_co, outs_fake_co, fmap_real_co, fmap_fake_co = self.combd(
                inputs_multi_scale, detached_outs_multi_scale
            )
            outs_real.update(outs_real_co)
            outs_fake.update(outs_fake_co)
            fmap_real.update(fmap_real_co)
            fmap_fake.update(fmap_fake_co)
        if self.use_sbd:
            outs_real_sbd, outs_fake_sbd, fmap_real_sbd, fmap_fake_sbd = self.sbd(
                inputs, detached_outs_multi_scale[-1]
            )
            outs_real.update(outs_real_sbd)
            outs_fake.update(outs_fake_sbd)
            fmap_real.update(fmap_real_sbd)
            fmap_fake.update(fmap_fake_sbd)

        ae_loss, log_dict_ae = self.loss.generator_loss(
            inputs=inputs,
            reconstructions=outs_multi_scale[-1],
            posteriors=posterior,
            global_step=self.global_step,
            disc_loss=disc_loss if train_disc else None,
            outs_real=outs_real,
            outs_fake=outs_fake,
            fmap_fake=fmap_fake,
            fmap_real=fmap_real,
        )

        enc_opt.zero_grad()
        dec_opt.zero_grad()
        self.manual_backward(ae_loss)
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

        e_param_norm = self.get_aggregate_param_norm(self.autoencoder.encoder)
        e_grad_norm = self.get_aggregate_grad_norm(self.autoencoder.encoder)
        d_param_norm = self.get_aggregate_param_norm(self.autoencoder.decoder)
        d_grad_norm = self.get_aggregate_grad_norm(self.autoencoder.decoder)
        log_dict_ae["encoder_param_norm"] = e_param_norm
        log_dict_ae["encoder_grad_norm"] = e_grad_norm
        log_dict_ae["decoder_param_norm"] = d_param_norm
        log_dict_ae["decoder_grad_norm"] = d_grad_norm

        enc_opt.step()
        dec_opt.step()
        enc_lr_sch.step()  # type: ignore
        dec_lr_sch.step()  # type: ignore

        with torch.no_grad():
            self.log_everything(
                ae_loss=ae_loss,
                log_dict={**log_dict_ae, **log_dict_disc},
                phase="train",
                batch_size=len(inputs),
            )

        return {
            "ae_loss": ae_loss,
            "preds": outs_multi_scale[-1],
            "target": inputs,
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batches = batch.values() if "batch0" in batch else [batch]

        for batch in batches:
            inputs = batch["audio"]
            inputs_multi_scale = [
                self.pqmf_lv2.analysis(inputs)[
                    :, : self.autoencoder.decoder.projection_filters[1]
                ],
                self.pqmf_lv1.analysis(inputs)[
                    :, : self.autoencoder.decoder.projection_filters[2]
                ],
                inputs,
            ]

            outs_multi_scale, posterior = self.autoencoder(inputs)

            detached_outs_multi_scale = [x.detach() for x in outs_multi_scale]

            # input et reconstruction ait la même taille
            for i in range(len(inputs_multi_scale)):
                if inputs_multi_scale[i].size(-1) != detached_outs_multi_scale[i].size(
                    -1
                ):
                    inputs_multi_scale[i] = inputs_multi_scale[i][
                        ..., : detached_outs_multi_scale[i].size(-1)
                    ]

            outs_real = {}
            outs_fake = {}
            fmap_real = {}
            fmap_fake = {}

            if self.use_combd:
                outs_real_co, outs_fake_co, fmap_real_co, fmap_fake_co = self.combd(
                    inputs_multi_scale, detached_outs_multi_scale
                )
                outs_real.update(outs_real_co)
                outs_fake.update(outs_fake_co)
                fmap_real.update(fmap_real_co)
                fmap_fake.update(fmap_fake_co)
            if self.use_sbd:
                outs_real_sbd, outs_fake_sbd, fmap_real_sbd, fmap_fake_sbd = self.sbd(
                    inputs_multi_scale[-1], detached_outs_multi_scale[-1]
                )
                outs_real.update(outs_real_sbd)
                outs_fake.update(outs_fake_sbd)
                fmap_real.update(fmap_real_sbd)
                fmap_fake.update(fmap_fake_sbd)

            if (self.global_step > self.disc_training_start) and len(self.discs) != 0:
                detached_outs_multi_scale = [x.detach() for x in outs_multi_scale]

                disc_loss, _, log_dict_disc = self.loss.discriminator_loss(
                    outs_real=outs_real,
                    outs_fake=outs_fake,
                )
            else:
                disc_loss, log_dict_disc = torch.zeros((1,), device=inputs.device), {}

            ae_loss, log_dict_ae = self.loss.generator_loss(
                inputs=inputs,
                reconstructions=outs_multi_scale[-1],
                posteriors=posterior,
                global_step=self.global_step,
                disc_loss=disc_loss,
                outs_real=outs_real,
                outs_fake=outs_fake,
                fmap_fake=fmap_fake,
                fmap_real=fmap_real,
            )

            self.log_everything(
                phase=self.val_names[dataloader_idx],
                ae_loss=ae_loss,
                log_dict={**log_dict_ae, **log_dict_disc},
                batch_size=len(inputs),
            )

        return {
            "disc_loss": disc_loss,
            "ae_loss": ae_loss,
            "preds": outs_multi_scale[-1],
            "target": inputs,
        }

    def autoencode(self, input, stochastic_latent=True):
        reconstruction, posterior = self.autoencoder.forward(
            input, stochastic_latent=stochastic_latent
        )
        return reconstruction[-1]
