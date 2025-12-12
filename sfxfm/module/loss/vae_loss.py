from typing import Dict, Any
import torch
import torch.nn as nn
from sfxfm.module.discriminator.disc_collection import DiscriminatorCollection
from sfxfm.module.loss.ae_loss import AELoss
from sfxfm.module.loss.balancer import Balancer
from sfxfm.module.model.blocks import DiagonalGaussianDistribution


class VAELoss(AELoss):
    def __init__(
        self,
        disc_collection: DiscriminatorCollection,
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
        super().__init__(
            disc_collection=disc_collection,
            spectral_loss=spectral_loss,
            disc_loss=disc_loss,
            balancer=balancer,
            adv_loss_start=adv_loss_start,
            fm_params=fm_params,
        )

    def generator_loss(
        self,
        inputs,
        reconstructions,
        posteriors,
        global_step,
        disc_loss=None,
        **kwargs,
    ):
        assert isinstance(posteriors, DiagonalGaussianDistribution)
        # reshape real/fake tensor to have the same time length
        # this only happens if chunk size is not a power of 2
        if inputs.size(-1) != reconstructions.size(-1):
            inputs = inputs[..., : reconstructions.size(-1)]

        # ------------ negative log likelihood
        rec_loss = torch.abs(inputs - reconstructions).mean()

        kl_loss = posteriors.kl().mean()

        # spectral loss:
        spectral_loss = self.spectral_loss(inputs, reconstructions)

        # No adversarial loss in the beginning
        adv_loss = global_step > self.adv_loss_start
        if adv_loss:
            # ------------ Generator update
            logits_real, fmap_real = self.discriminator(inputs)
            logits_fake, fmap_fake = self.discriminator(reconstructions)

            g_losses = {
                n_disc: self.g_loss_fn(l_fake) for n_disc, l_fake in logits_fake.items()
            }
            g_loss = torch.mean(torch.stack(list(g_losses.values())))

            # ------------ Feature Matching loss
            fm_losses = {}
            for n_disc in self.discriminator.n_discs:
                fm_losses[n_disc] = self.feature_matching_loss(
                    fmap_real[n_disc], fmap_fake[n_disc], **self.fm_params
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
            "kl_loss": kl_loss,
            "spectral_loss": spectral_loss,
        }

        # Do not reblance in validation mode
        if self.training:
            loss = self.balancer.reweight_losses(
                loss_dict,
                batch=reconstructions,
                disc_loss=disc_loss if adv_loss else None,
            )
            with torch.no_grad():
                grads = self.get_norm_grads(
                    {**loss_dict}, reconstructions, weights=self.balancer.weights
                )
        else:
            loss = torch.stack(list(loss_dict.values())).sum()
            grads = {}

        log = {
            "ae_loss": loss.detach().mean().item(),
            "rec_loss": rec_loss.detach().mean().item(),
            "spectral_loss": spectral_loss.detach().mean().item(),
            "g_loss": g_loss.detach().mean().item(),
            "fm_loss": fm_loss.detach().mean().item(),
            "kl_loss": kl_loss.detach().mean().item(),
            **self.balancer._metrics,
            **grads,
        }
        return loss, log
