from copy import deepcopy
import sys
import lightning.pytorch as pl
import torch
import torchmetrics

from sfxfm.module.model.transformer import register_backward_nan_check_hook
from sfxfm.utils.loading import catchtime


from .utils import ema_update
from .base import BaseLightningModule
from .metrics.multimodal_metric_collection import MultiModalMetricCollection
import os
import logging
from sfxfm.utils.dist import rank


rank = rank()
# get logger
log = logging.getLogger(__name__)

# flag to perform extra checks on the model
DEBUG_NAN = os.environ.get("DEBUG_NAN", False)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DiffusionModule(BaseLightningModule):
    """LightningModule to train a unconditional diffusion attention
    Args:
    """

    def __init__(
        self,
        diffusion_model,
        preprocessor,
        ema_decay,
        optim,
        quantizer,
        t_schedule,
        mask_sampler=None,
        condition_dropout=0.0,
        fixed_encoder=False,
        normalize_audio=True,
        metric_collection_train=torchmetrics.MetricCollection([]),
        metric_collection_val=MultiModalMetricCollection([]),
        gen_edm_kwargs={},
        finetune_checkpoint=None,
    ):
        super().__init__()
        self.diffusion = diffusion_model
        if hasattr(diffusion_model, "condition_processor"):
            self.condition_processor = diffusion_model.condition_processor
            # condition processors  is not used in DiT, we remove it to save space in the checkpoit see issue #1019
            # we kept it in diffusion_model in the config to keep, backward compatibility
            del diffusion_model.condition_processor

        self.diffusion_ema = deepcopy(self.diffusion)
        self.diffusion_ema.requires_grad_(False)
        self.ema_decay = ema_decay
        self.preprocessor = preprocessor
        self.preprocessor.requires_grad_(False)
        self.preprocessor.eval()
        self.t_schedule = t_schedule

        # TODO remove quantizer
        self.quantizer = quantizer

        # this rng is only used to select the number of layers in RQ during training
        # + if droping conditional info or not
        # It is sync between the threads
        # self.rng = np.random.default_rng(12345)

        # this rng is NOT sync between threads
        self.rng_sobol = torch.quasirandom.SobolEngine(1, scramble=True, seed=None)

        # conditional if a condition processor is provided in the diffusion_model
        self.condition_dropout = condition_dropout

        self.optim_config = optim

        self._normalize_audio = normalize_audio
        if metric_collection_val is not None:
            metric_collection_val.eval()
        if metric_collection_train is not None:
            metric_collection_train.eval()
        self.metric_collection_train = metric_collection_train
        self.metric_collection_val = (
            [metric_collection_val] if metric_collection_val is not None else []
        )
        self.gen_edm_kwargs = gen_edm_kwargs
        if finetune_checkpoint is not None:
            self.init_finetune(finetune_checkpoint)
        self.finetune_checkpoint = finetune_checkpoint
        # debuging variable to store the last loss weights
        self.last_weight = None
        self.mask_sampler = mask_sampler

    def init_finetune(self, finetune_checkpoint):
        from ..utils.config import get_checkpoint_filename

        ckpt_path = get_checkpoint_filename(from_checkpoint=finetune_checkpoint)
        log.info(f"Loading FINTUNING checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        # apply fixes to checkpoint
        self.on_load_checkpoint(checkpoint)
        try:
            epochs = checkpoint["loops"]["fit_loop"]["epoch_progress"]["total"][
                "completed"
            ]
            log.info(f"Loaded {finetune_checkpoint}, on Epoch={epochs}")
        except:
            pass

        state_dict = checkpoint["state_dict"]
        diffusion_ema_state_dict = {
            k[len("diffusion_ema.") :]: v
            for k, v in state_dict.items()
            if k.startswith("diffusion_ema")
        }
        self.diffusion.load_state_dict(diffusion_ema_state_dict, strict=False)
        load_res = self.diffusion_ema.load_state_dict(
            diffusion_ema_state_dict, strict=False
        )
        log.info(
            f"Loaded diffusion and diffusion_ema state_dict from {finetune_checkpoint}, missing keys: {load_res.missing_keys} \n "
        )
        if len(load_res.unexpected_keys) > 0:
            log.warning(
                f"Unexpected keys, can cause the pre-trained module to behave unexpectedly: {load_res.unexpected_keys}"
            )

    def configure_optimizers(self):  # type: ignore
        params = [
            (k, v)
            for k, v in self.named_parameters()
            if v.requires_grad and "metric_collection_" not in k
        ]

        pnames = set([k.split(".")[0] for k, v in params])

        log.info(f"Optimizing {len(params)} parameters, of submodules {pnames}")

        params = [v for _, v in params]

        optimizer = self.optim_config.optimizer(params=params)
        scheduler = self._build_scheduler(
            scheduler_config=self.optim_config.scheduler, optimizer=optimizer
        )

        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "name": "lr",
                }
            ],
        )

    def on_fit_start(self):
        super().on_fit_start()
        # one metric_collection_val per validation dataloader
        # not possible to create it at init time, as datamodule
        # is not yet ready
        # internal states must be separate for each dataloader
        self.val_names = [name for name in self.trainer.datamodule.val_dataset()]
        if self.metric_collection_val is not None and len(self.metric_collection_val):
            metric_collection_val = self.metric_collection_val[0]
            # torch.nn.ModuleList(
            self.metric_collection_val = [
                metric_collection_val.clone().cpu() for _ in self.val_names
            ]

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
    def sample(self, noise, cond, **kwargs):
        raise NotImplementedError

    def denoise(self, x_t, t, cond=None, mask=None, ema=False):
        """
        Wrapper around network diffusion or diffusion_ema
        Selects ema or non-ema weights
        Can be reimplemented for preconditioning
        """
        # if type(n_q) == int:
        #     batch_size = x_t.size(0)
        #     n_q = torch.Tensor([n_q] * batch_size).long().to(x_t.device)

        if ema and hasattr(self, "diffusion_ema"):
            self.diffusion_ema.eval()
            return self.diffusion_ema(x_t, t, mask, cond)
        else:
            return self.diffusion(x_t, t, mask, cond)

    def sample_t(self, x, t_schedule=None):
        if t_schedule is None:
            return self.t_schedule.sample_t(x)
        else:
            return t_schedule.sample_t(x)

    def sample_mask(self, x, no_mask=False):
        if self.mask_sampler is None or no_mask:
            return torch.ones_like(x[:, 0, :])
        else:
            return self.mask_sampler.sample(x)

    def normalize_audio(self, x):
        if self._normalize_audio:
            mono = x.mean(dim=1, keepdim=True)
            volume = mono.pow(2).mean(dim=2, keepdim=True).sqrt()
            scale = 1e-8 + volume
            x = x / scale
        return x

    def loss_fn(self, reals, cond, noise, t, mask, ema):
        """
        From reals and noise, computes objective function
        ema (bool): choose between ema or non ema weights
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        x = batch["audio"]
        # print("allow_tf32 no comp no prp ")
        # torch.backends.cudnn.allow_tf32 = True
        # torch.backends.cuda.matmul.allow_tf32 = True
        # torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        register_backward_nan_check_hook(x, "audio")

        with torch.autocast(device_type="cuda", enabled=False):
            with torch.no_grad():
                x = self.normalize_audio(x.float())
                # latent variable
                x = self.preprocessor(x)
                register_backward_nan_check_hook(x, "DM_preprocessor")
            # Draw well distributed continuous timesteps
            t = self.sample_t(x=x)
            mask = self.sample_mask(x=x)

        # Compute condition
        cond = self.get_cond(batch)

        with torch.no_grad():
            # get inputs
            reals = x
            cond["original_x"] = x
            noise = torch.randn_like(reals.float())

        # torch.backends.cudnn.allow_tf32 = False
        # torch.backends.cuda.matmul.allow_tf32 = False
        # torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        loss, reco_loss = self.loss_fn(reals, cond, noise, t, mask, ema=False)

        # torch.backends.cudnn.allow_tf32 = True
        # torch.backends.cuda.matmul.allow_tf32 = True
        # torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

        with torch.no_grad():
            self.log_everything(
                reco_loss=reco_loss,
                loss=loss,
                phase="train",
            )
            if getattr(self.trainer.precision_plugin, "scaler", None) is not None:
                sscale = self.trainer.precision_plugin.scaler.get_scale()
                growthtraker = (
                    self.trainer.precision_plugin.scaler._get_growth_tracker()
                )
                # if rank == 0:
                #     print(
                #         f"scale={sscale}, growth={growthtraker}",
                #         file=sys.stderr,
                #         flush=True,
                #     )

                self.log("scaler/scale", sscale, prog_bar=True)
                self.log("scaler/growth", growthtraker, prog_bar=False)
            # prog bar:
            self.log("loss", loss, prog_bar=True)

        #    except Exception as e:
        #         print("Error in loss_fn", e, file=sys.stderr, flush=True)
        #         torch.save(batch, f"batch{rank}.pt")
        #         torch.save(cond, f"cond{rank}.pt")
        #         torch.save(reals, f"reals{rank}.pt")
        #         torch.save(noise, f"noise{rank}.pt")
        #         torch.save(t, f"t{rank}.pt")
        #         torch.save(self.state_dict(), f"state_dict{rank}.pt")
        #         raise e

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        pass

    # Returns dict a (possibly empty) dict
    def get_cond(self, batch, no_dropout=False, no_cond=False, **kwargs):
        """
        no_dropout=True for validation
        if drop=True return the unconditional
        """
        return self.condition_processor.forward(
            batch,
            condition_dropout=0 if no_dropout else self.condition_dropout,
            no_cond=no_cond,
            **kwargs,
        )

    def no_cond(self, x):
        """return the no cond tokens"""
        return self.get_cond(
            {
                "audio": x,
                "description": [
                    None,
                ]
                * x.size(0),
            },
            no_cond=True,
        )

    def log_everything(self, reco_loss, loss, phase):
        """
        reco_loss: unreduced reconstruction loss
        """
        log_dict = {}

        # MSE per t-bin
        mse_loss_per_band = reco_loss
        # loss_per_band = {
        #     f"{phase}/mse_band#{k}": l.item()
        #     for k, l in enumerate(mse_loss_per_band.mean(2).mean(0))
        # }
        # log_dict.update(loss_per_band)

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
        x = batch["audio"]
        with torch.autocast(device_type="cuda", enabled=False):
            with torch.no_grad():
                x = self.normalize_audio(x.float())
                # latent variable
                x = self.preprocessor(x)

            t = self.sample_t(x)
            mask = self.sample_mask(x, no_mask=True)

        reals = x

        cond = self.get_cond(batch=batch, no_dropout=True)
        cond["original_x"] = x

        noise = torch.randn_like(reals)

        loss, reco_loss = self.loss_fn(reals, cond, noise, t, mask, ema=True)

        # @TODO REMOVE
        self.log_everything(
            reco_loss=reco_loss,
            loss=loss,
            phase="valid",
        )
        fakes = self.generate_for_validation(cond, noise)
        return {
            "preds": fakes,
            "target": self.preprocessor.inverse(reals),
            "batch": batch,
        }

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # fix time axis before updating any metric. target and pred may have different
        # lengths if chunks_size is not a power of 2
        if outputs["target"].size(-1) != outputs["preds"].size(-1):
            raise RuntimeError(
                f"Inputs {outputs['target'].shape} and reconstructions {outputs['preds'].shape} have different shapes. "
            )
        # ensure there's at least one metric in the metric collection
        if len(self.metric_collection_val) > dataloader_idx:
            self.metric_collection_val[dataloader_idx].update(
                outputs["preds"], outputs["target"], batch
            )

    def on_validation_epoch_start(self):
        for i in range(len(self.metric_collection_val)):
            self.metric_collection_val[i].cuda()

    def on_validation_epoch_end(self):
        for i in range(len(self.metric_collection_val)):
            log_dict = {**self.metric_collection_val[i].compute()}
            self.log_metrics(
                phase=self.val_names[i] + "_metrics", log_dict=log_dict, batch_size=1
            )
            self.metric_collection_val[i].reset()
            self.metric_collection_val[i].cpu()
        torch.cuda.empty_cache()

    def log_metrics(self, phase="", log_dict={}, batch_size=None, sync_dist=True):
        """
        reco_loss: unreduced reconstruction loss
        """
        if phase != "" and not phase.endswith("/"):
            phase += "/"

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

    def on_load_checkpoint(self, checkpoint):
        """
        Can change checkpoint kcheckpoint['state_dict'].eys to be loaded here.
        We remove the saved metrics if any exist in the checkpoint.
        """
        checkpoint = checkpoint["state_dict"]

        # don't load metrics if they exist
        for k in list(checkpoint.keys()):
            if "metric_collection_" in k:
                del checkpoint[k]
        # Put the current metrics to avoid crashes with missing keys
        for k, v in self.state_dict().items():
            if "metric_collection_" in k:
                checkpoint[k] = v

        # fix loading old checkpoits before old clap tau changes
        for k, v in list(checkpoint.items()):
            if "our_baseclap.model.tau" in k:
                new_k = k.replace(
                    "our_baseclap.model.tau", "our_baseclap.model.loss.tau"
                )
                checkpoint[new_k] = v
                del checkpoint[k]

        # Remove duplicates condition processors in checkpoint see issue #1019
        for k in list(checkpoint.keys()):
            if k.startswith("diffusion.condition_processor.") or k.startswith(
                "diffusion_ema.condition_processor."
            ):
                del checkpoint[k]

        # add preprocessor to checkpoint if it is not there
        for k, v in self.state_dict().items():
            if "preprocessor." in k and k not in checkpoint:
                checkpoint[k] = v

    def on_save_checkpoint(self, checkpoint):
        """
        Can change checkpoint keys to be saved here
        """
        checkpoint = checkpoint["state_dict"]
        # check for nans in the model
        for k, v in checkpoint.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaNs in model at {k}, checkpointing aborted")
        # don't save metrics if they exist
        for k in list(checkpoint.keys()):
            if "metric_collection_" in k or k.startswith("preprocessor.autoencoder"):
                del checkpoint[k]

    @torch.no_grad()
    def generate_for_validation(self, cond, noise, gen_edm_kwargs=None):
        """
        Returns dict
        x is reshaped like fakes
        fakes is b r c t
        where r is num_variations
        output is c (b t)  interleaving x and fakes
        """
        if gen_edm_kwargs is None:
            gen_edm_kwargs = self.gen_edm_kwargs
        fakes = self.sample(noise, cond=cond, **gen_edm_kwargs)

        with torch.autocast(device_type="cuda", enabled=False):
            # Data preprocessing
            fakes = self.preprocessor.inverse(fakes.float())

        return fakes

    def on_before_zero_grad(self, *args, **kwargs):
        pass

    def on_before_backward(self, loss) -> None:
        """Called before ``loss.backward()``.

        Args:
            loss: Loss divided by number of batches for gradient accumulation and scaled if using AMP.

        """
        pass

    def on_after_backward(self) -> None:
        """Called after ``loss.backward()`` and before optimizers are stepped.

        Note:
            If using native AMP, the gradients will not be unscaled at this point.
            Use the ``on_before_optimizer_step`` if you need the unscaled gradients.

        """
        if not DEBUG_NAN:
            return
        errors = []
        for name, param in self.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                errors.append(name)

        if len(errors) > 0:
            if rank == 0:
                print(
                    f"on_after_backward NaNs in gradients training aborted the following params",
                    ", ".join(errors),
                    file=sys.stderr,
                    flush=True,
                )

            # raise ValueError(
            #     f"NaNs in gradients training aborted the following params",
            #     ", ".join(errors),
            # )

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_closure=None,
    ):
        """
        Skipping updates in case of unstable gradients
        https://github.com/Lightning-AI/lightning/issues/4956
        """
        valid_gradients = True
        if DEBUG_NAN:
            for name, param in self.named_parameters():
                if param.grad is not None:
                    # valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                    valid_gradients = not (torch.isnan(param.grad).any())
                    if not valid_gradients:
                        break
        if not valid_gradients:
            if rank == 0:
                print(
                    f"\ndetected inf or nan values in gradients. not updating model parameters, step={batch_idx}",
                    flush=True,
                )
                print(
                    f"\ndetected inf or nan values in gradients. not updating model parameters, step={batch_idx}",
                    file=sys.stderr,
                    flush=True,
                )
            print(
                f"rank{rank}, step={batch_idx}, last_weight={self.last_weight}",
                file=sys.stderr,
                flush=True,
            )
            self.zero_grad()
            min_scale = 2.0**14
            # prevent the scale if going too small later on in training
            if (
                getattr(self.trainer.precision_plugin, "scaler", None) is not None
                and self.current_epoch >= 200
                and (sscale := self.trainer.precision_plugin.scaler.get_scale())
                < min_scale
            ):
                self.trainer.precision_plugin.scaler.set_backoff_factor(1.0)
                self.trainer.precision_plugin.scaler.update(min_scale)

                print(
                    f"\nForced scaler = {min_scale}, current={sscale}, step={batch_idx}",
                    file=sys.stderr,
                    flush=True,
                )
                print(
                    f"\nForced scaler = {min_scale}, current={sscale}, step={batch_idx}",
                    flush=True,
                )
            elif (
                getattr(self.trainer.precision_plugin, "scaler", None) is not None
                and self.current_epoch >= 200
            ):
                self.trainer.precision_plugin.scaler.set_backoff_factor(0.5)
                # self.trainer.precision_plugin.scaler.set_backoff_factor(1.0)
            if optimizer_closure is not None:
                optimizer_closure()
            return

        pl.LightningModule.optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_closure
        )

        # update EMA in case of no nans
        decay = (
            0.95
            if (self.current_epoch < 25 and self.finetune_checkpoint is None)
            else self.ema_decay
        )
        if hasattr(self, "diffusion_ema"):
            ema_update(self.diffusion, self.diffusion_ema, decay)

    def autoencode(self, x: torch.Tensor, stochastic_latent=False):
        """
        x is an audio!
        """
        batch_size = x.size(0)
        # noise = torch.randn((batch_size, 32, 321), device=x.device)
        noise = torch.randn((batch_size, 32, 160), device=x.device)
        cond = self.get_cond(
            {"audio": x, "description": [None] * batch_size}, no_dropout=True
        )
        z = self.sample(noise, cond=cond)
        x = self.preprocessor.inverse(z)
        return x

    def _get_audio_embeddings(self, preprocessed_audio):
        r"""Load preprocessed audio and return a audio embeddings"""
        with torch.no_grad():
            preprocessed_audio = preprocessed_audio.reshape(
                preprocessed_audio.shape[0], preprocessed_audio.shape[2]
            )
            # Append [0] the audio emebdding, [1] has output class probabilities
            return self.clap.audio_encoder(preprocessed_audio)[0]

    def on_train_start(self):
        super().on_train_start()
        log.info("Starting training")

        # log.info(
        #     f"torch.backends.cudnn.allow_tf32 = {torch.backends.cudnn.allow_tf32}, {torch.backends.cuda.matmul.allow_tf32} , {torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}"
        # )
        # torch.backends.cudnn.allow_tf32 = False
        # torch.backends.cuda.matmul.allow_tf32 = False
        # torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        # log.info(
        #     f"torch.backends.cudnn.allow_tf32 = {torch.backends.cudnn.allow_tf32}, {torch.backends.cuda.matmul.allow_tf32} , {torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}"
        # )
