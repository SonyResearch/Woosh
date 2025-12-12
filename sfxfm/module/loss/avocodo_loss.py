from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from sfxfm.module.loss.balancer import Balancer
from sfxfm.module.loss.ae_loss import get_adv_loss_fns

class AvocodoLoss(nn.Module):
    def __init__(
        self,
        spectral_loss: nn.Module,
        balancer: Balancer,
        disc_loss: str = "hinge",
        adv_loss_start: int = -1,
        fm_params: Dict[str, Any] = {
            "reduction": "sum",
            "norm": "l1",
            "relative": True,
            "aggregated_features": False,
            "aggregate_frequencies": False,
        },
    ):

        super().__init__()
        assert disc_loss in ["hinge", "lsgan", "vanilla", "lsgan_avocodo"]

        # ------------ Adversarial Loss
        if disc_loss == "lsgan_avocodo":
            self.disc_loss_fn = lsgan_avocodo_d_loss
            _, self.g_loss_fn = get_adv_loss_fns("lsgan")
        else:
            self.disc_loss_fn, self.g_loss_fn = get_adv_loss_fns(
                disc_loss_name=disc_loss
            )

        self.balancer = balancer
        self.fm_params = fm_params
        # controls different learning phases
        # no adversarial loss (g_loss & fm_loss) in generator_loss before:
        self.adv_loss_start = adv_loss_start
        self.spectral_loss = spectral_loss

    def generator_loss(
        self,
        inputs,
        reconstructions,
        posteriors,
        global_step,
        outs_fake,
        outs_real,
        fmap_fake,
        fmap_real,
        disc_loss=None,
    ):

        # ------------ negative log likelihood
        rec_loss = torch.abs(inputs - reconstructions).mean()

        # spectral loss:
        spectral_loss = self.spectral_loss(inputs, reconstructions)

        # No adversarial loss in the beginning
        adv_loss = global_step > self.adv_loss_start
        if adv_loss:
            # ------------ Generator update
            g_losses = {}
            for n_disc in fmap_real.keys():
                g_loss = 0
                for out_fake in outs_fake[n_disc]:
                    g_loss += self.g_loss_fn(out_fake)
                g_losses[n_disc] = g_loss
            g_loss = torch.sum(torch.stack(list(g_losses.values())))

            # ------------ Feature Matching loss
            fm_losses = {}
            for n_disc in fmap_real.keys():
                for fmap_r, fmap_f in zip(fmap_real[n_disc], fmap_fake[n_disc]):
                    fm_losses[n_disc] = self.feature_matching_loss(
                        fmap_r, fmap_f, **self.fm_params
                    )

            fm_loss = torch.mean(torch.stack(list(fm_losses.values())))

        else:
            # Only autoencoder loss
            g_loss = torch.zeros_like(rec_loss)
            fm_loss = torch.zeros_like(rec_loss)

        loss_dict: Dict[str, torch.Tensor] = {
            "rec_loss": rec_loss,
            "g_loss": g_loss,
            "fm_loss": fm_loss,
            "spectral_loss": spectral_loss,
        }

        # Do not reblance in validation mode
        if self.training:
            loss = self.balancer.reweight_losses(
                loss_dict,
                batch=reconstructions,
                disc_loss=disc_loss if adv_loss else None,
            )
        else:
            loss = torch.stack(list(loss_dict.values())).sum()

        log = {
            "ae_loss": loss.detach().mean().item(),
            "rec_loss": rec_loss.detach().mean().item(),
            "spectral_loss": spectral_loss.detach().mean().item(),
            "g_loss": g_loss.detach().mean().item(),
            "fm_loss": fm_loss.detach().mean().item(),
            **self.balancer._metrics,
        }
        return loss, log

    def discriminator_loss(self, outs_real, outs_fake):
        # logits_real/fake is (batch_size, 1, d1, d2)

        d_losses = {}
        for n_disc in outs_real.keys():
            d_loss = 0
            for out_real, out_fake in zip(outs_real[n_disc], outs_fake[n_disc]):
                d_loss += self.disc_loss_fn(
                    out_real,
                    out_fake,
                )
            d_losses[n_disc] = d_loss
        d_loss = torch.sum(torch.stack(list(d_losses.values())))

        log = {
            "disc_loss": d_loss.detach().mean().item(),
            **d_losses,
            **{
                f"logits_real_minus_fake_#{n_disc}": (
                    outs_real[n_disc] - outs_fake[n_disc]
                )
                .detach()
                .mean()
                for n_disc in outs_real.keys()
            },
        }

        return d_loss, d_losses, log

    def feature_matching_loss(
        self,
        fmap_real,
        fmap_fake,
        norm="l1",
        reduction="sum",
        relative=True,
        aggregated_features=False,
        aggregate_frequencies=False,
    ):
        assert len(fmap_real) == len(fmap_fake)
        assert norm in ["l1", "l2"]
        assert reduction in ["sum", "mean"]
        fm_loss = []
        for layer_real, layer_fake in zip(fmap_real, fmap_fake):
            if norm == "l1":
                loss = lambda x, y: F.l1_loss(x, y, reduction=reduction)
            elif norm == "l2":
                loss = lambda x, y: F.mse_loss(x, y, reduction=reduction)
            else:
                raise NotImplementedError

            if aggregated_features:
                # we aggregate features over batch and time
                # Original formulation of feature matching
                reduced_dims = (
                    (0, -2, -1)
                    if (len(layer_real.size()) == 4 and aggregate_frequencies)
                    else (0, -1)
                )
                layer_real = layer_real.mean(reduced_dims)
                layer_fake = layer_fake.mean(reduced_dims)

            num = loss(layer_real, layer_fake)
            denom = (
                (loss(layer_real, torch.zeros_like(layer_real)) + 1e-6)
                if relative
                else 1
            )
            fm_loss.append(num / denom)
        return torch.mean(torch.stack(fm_loss))


def lsgan_avocodo_d_loss(logits_real, logits_fake):
    loss_real = torch.mean((1.0 - logits_real) ** 2)
    loss_fake = torch.mean((logits_fake) ** 2)
    d_loss = loss_real + loss_fake
    return d_loss
