from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from sfxfm.module.discriminator.disc_collection import DiscriminatorCollection
from sfxfm.module.loss.balancer import Balancer


def get_adv_loss_fns(disc_loss_name):
    if disc_loss_name == "hinge":
        return hinge_d_loss, hinge_g_loss
    elif disc_loss_name == "lsgan":
        return lsgan_d_loss, lsgan_g_loss
    else:
        return vanilla_d_loss, vanilla_g_loss


class AELoss(nn.Module):
    def __init__(
        self,
        disc_collection: DiscriminatorCollection,
        spectral_loss: nn.Module,
        balancer: Balancer,
        disc_loss: str = "hinge",
        adv_loss_start: int = -1,
        fm_params: Dict[str, Any] = {
            "reduction": "sum",  # Loss function reduction, see L1loss
            "norm": "l1",
            "relative": True,
            "aggregated_features": False,
            "aggregate_frequencies": False,
            "features_mean": True,  # reduction across the different features mean/sum
        },
    ):
        """Main class for AutoencoderGAN training

        Args:
            disc_collection (DiscriminatorCollection): Discriminator as a DiscriminatorCollection
            spectral_loss (nn.Module): Module which computes a spectral loss
            balancer (Balancer): loss balancer used in the generator loss
            disc_loss (str): gan training objective. Defaults to "hinge". Can be hinge, lsgan, vanilla
            adv_loss_start (int): number of batches after which the adversarial loss is taken into account by the generator. Defaults to -1.
            fm_params (dict, optional): Feature matching parameters. Defaults to {"reduction": "sum", "norm": "l1", "relative": True}.
        """
        super().__init__()
        assert disc_loss in ["hinge", "lsgan", "vanilla"]

        # ------------ Adversarial Loss
        self.discriminator = disc_collection
        self.disc_loss_fn, self.g_loss_fn = get_adv_loss_fns(disc_loss_name=disc_loss)
        self.balancer = balancer

        self.fm_params = fm_params

        # controls different learning phases
        # no adversarial loss (g_loss & fm_loss) in generator_loss before:
        self.adv_loss_start = adv_loss_start

        self.spectral_loss = spectral_loss

    def get_norm_grads(
        self,
        losses: Dict[str, torch.Tensor],
        batch: torch.Tensor,
        weights: Dict[str, float],
    ):
        norm_grads = {}
        total_norm = 0

        for name, loss in losses.items():
            if loss.requires_grad:
                (grad,) = autograd.grad(loss, [batch], retain_graph=True)
                norm_grad = grad.norm().item()
            else:
                norm_grad = 0

            norm_grads[f"norm_grad_{name}"] = norm_grad
            total_norm += norm_grad * weights[name]

        ratios_scaled_grads = {
            f"ratio_scaled_norm_grad_{name}": weights[name]
            * norm_grads[f"norm_grad_{name}"]
            / total_norm
            if total_norm != 0
            else 0
            for name in losses.keys()
        }

        return {**norm_grads, **ratios_scaled_grads}

    def generator_loss(
        self,
        inputs,
        reconstructions,
        posteriors,
        global_step,
        disc_loss=None,
        **kwargs,
    ):
        # reshape real/fake tensor to have the same time length
        # this only happens if chunk size is not a power of 2
        if inputs.size(-1) != reconstructions.size(-1):
            inputs = inputs[..., : reconstructions.size(-1)]

        # ------------ negative log likelihood
        rec_loss = torch.abs(inputs - reconstructions).mean()

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
            **self.balancer._metrics,
            **grads,
            **{
                f"gloss_{n_disc}": l_value.detach()
                for n_disc, l_value in g_losses.items()
            },
        }
        return loss, log

    def discriminator_loss(self, inputs, reconstructions, global_step):
        # reshape real/fake tensor to have the same time length
        # this only happens if chunk size is not a power of 2
        if inputs.size(-1) != reconstructions.size(-1):
            inputs = inputs[..., : reconstructions.size(-1)]

        logits_real, _ = self.discriminator(inputs.detach())
        logits_fake, _ = self.discriminator(reconstructions.detach())

        # # reshape real/fake tensor to have the same time length
        # # this only happens if chunk size is not a power of 2
        # for n_disc in self.discriminator.n_discs:
        #     if logits_real[n_disc].size(-1) != logits_fake[n_disc].size(-1):
        #         dim = min(logits_real[n_disc].size(-1), logits_fake[n_disc].size(-1))
        #         logits_real[n_disc] = logits_real[n_disc][...,:dim]
        #         logits_fake[n_disc] = logits_fake[n_disc][...,:dim]

        # logits_real/fake is (batch_size, 1, d1, d2)
        d_losses = {}
        for n_disc in self.discriminator.n_discs:
            d_losses[n_disc] = self.disc_loss_fn(
                logits_real[n_disc], logits_fake[n_disc]
            )
        d_loss = torch.mean(torch.stack(list(d_losses.values())))

        log = {
            "disc_loss": d_loss.detach().mean().item(),
            **{
                f"logits_real_#{n_disc}": l_real.detach().mean().item()
                for n_disc, l_real in logits_real.items()
            },
            **{
                f"logits_fake_#{n_disc}": l_fake.detach().mean().item()
                for n_disc, l_fake in logits_fake.items()
            },
            **{
                f"logits_real_minus_fake_#{n_disc}": (
                    logits_real[n_disc] - logits_fake[n_disc]
                )
                .detach()
                .mean()
                for n_disc in self.discriminator.n_discs
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
        features_mean=True,
        detach_real=False,
        loss_clip=False,
        clip_value=1.0,
    ):
        assert len(fmap_real) == len(fmap_fake)
        assert norm in ["l1", "l2"]
        assert reduction in ["sum", "mean"]
        if norm == "l1":
            loss = lambda x, y: F.l1_loss(x, y, reduction=reduction)
        elif norm == "l2":
            loss = lambda x, y: F.mse_loss(x, y, reduction=reduction)
        else:
            raise NotImplementedError

        fm_loss = []
        for layer_real, layer_fake in zip(fmap_real, fmap_fake):
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

            denom = (
                (loss(layer_real, torch.zeros_like(layer_real)) + 1e-6)
                if relative
                else 1
            )
            if detach_real:
                layer_real = layer_real.detach()

            num = loss(layer_fake, layer_real)
            if loss_clip and num.requires_grad:
                num.register_hook(
                    lambda grad: torch.clamp(grad, -clip_value, clip_value)
                )
            fm_loss.append(num / denom)
        if features_mean:
            return torch.mean(torch.stack(fm_loss))
        else:
            return torch.sum(torch.stack(fm_loss))


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def hinge_g_loss(logits_fake):
    return torch.mean(torch.relu(1 - logits_fake))


def lsgan_d_loss(logits_real, logits_fake):
    loss_real = torch.mean((1.0 - logits_real) ** 2)
    loss_fake = torch.mean((-1.0 - logits_fake) ** 2)
    d_loss = loss_real + loss_fake
    return d_loss


def lsgan_g_loss(logits_fake):
    return torch.mean((1.0 - logits_fake) ** 2)


def vanilla_d_loss(logits_real, logits_fake):
    m = nn.Softplus()
    d_loss = 0.5 * (torch.mean(m(-logits_real)) + torch.mean(m(logits_fake)))
    return d_loss


def vanilla_g_loss(logits_fake):
    m = nn.Softplus()
    d_loss = torch.mean(m(-logits_fake))
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight
