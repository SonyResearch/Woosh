from collections import defaultdict
import typing as tp
import torch
import math
from torch import autograd
from sfxfm.utils.dist.distrib import average_metrics, rank
import logging
from omegaconf import DictConfig

rank = rank()

log = logging.getLogger()


class PIDController(torch.nn.Module):

    def __init__(
        self,
        monitor: bool = True,
        target: tp.Dict[str, float] = {"disc_loss": 2.0},
        target_eps: tp.Dict[str, float] = {"disc_loss": 0.001},
        pid_gains: tp.Dict[str, float] = {"g_loss": [40, 0, 0]},
        initial_output: tp.Dict[str, float] = {"g_loss": 1.0},
        mode: tp.Dict[str, float] = {"g_loss": None},
        clip: tp.Dict[str, float] = {"g_loss": [0.2, 20]},
        ema_decay: float = None,
        integral_decay: float = 0.99,
        expand_power: float = 1.0,
    ) -> None:

        super().__init__()
        self._metrics: tp.Dict[str, tp.Any] = {}
        self.monitor = monitor

        # set target values to track, also for disc loss types
        self.target = {}
        for k, v in target.items():
            if isinstance(v, str):
                if v == "lsgan":
                    self.target[k] = 2.0
                elif v == "hinge":
                    self.target[k] = 1.0
                else:
                    self.target[k] = 0.5
            else:
                self.target[k] = v

        self.target_eps = target_eps
        self.pid_gains = pid_gains
        self.initial_output = initial_output
        self.mode = mode
        self.clip = clip
        self.ema_decay = ema_decay
        self.integral_decay = integral_decay
        self.expand_power = expand_power
        if self.expand_power <= 0:
            raise ValueError(f"expand_power must be a positive number: aborting")

        if isinstance(self.initial_output, (dict, DictConfig)):
            self.pid_out = {name: self.initial_output[name] for name in self.pid_gains}
        else:
            self.pid_out = {name: self.initial_output for name in self.pid_gains}

        # PID error states
        self.p_error = {}
        self.p_error_ema = {}
        self.p_error_ema_num = {}
        self.p_error_ema_den = {}
        self.i_error = {}
        self.d_error = {}
        self.d_last = {}

    def error_update(self, name: str, error: float):

        if self.expand_power > 1.0:
            # error signal expander
            error = (
                math.pow(error, 1.0 / self.expand_power)
                if error >= 0
                else -math.pow(-error, 1.0 / self.expand_power)
            )

        # proportional
        self.p_error[name] = error

        # proportional EMA, integral and derivative error takes this
        if self.ema_decay is not None:
            if name not in self.p_error_ema:
                self.p_error_ema_num[name] = 0.0
                self.p_error_ema_den[name] = 0.0
            self.p_error_ema_num[name] = (
                self.p_error_ema_num[name] * self.ema_decay + self.p_error[name]
            )
            self.p_error_ema_den[name] = self.p_error_ema_den[name] * self.ema_decay + 1
            self.p_error_ema[name] = (
                self.p_error_ema_num[name] / self.p_error_ema_den[name]
            )
        else:
            self.p_error_ema[name] = self.p_error[name]

        # apply tolerance
        if abs(self.p_error_ema[name]) < self.target_eps[name]:
            self.p_error_ema[name] = 0.0

        # integral part
        if name not in self.i_error:
            self.i_error[name] = 0.0
        # leaky integrator
        self.i_error[name] = (
            self.i_error[name] * self.integral_decay + self.p_error_ema[name]
        )
        # ideal integrator
        # self.i_error[name] += self.p_error_ema[name]

        # derivative
        if name not in self.d_error:
            self.d_last[name] = 0.0
        self.d_error[name] = self.p_error_ema[name] - self.d_last[name]
        self.d_last[name] = self.d_error[name]

    def update(
        self,
        name: str,
        value: float,
    ) -> tp.Dict[str, float]:

        if self.target is not None:

            if value is not None:
                # work with scalar floats not tensors
                value = (
                    value.item() if isinstance(value, torch.Tensor) else float(value)
                )
                # average error across GPUs
                # log.info(f"rank={rank}, balancer: 1")
                error = average_metrics({name: self.target[name] - value})
                # log.info(f"rank={rank}, balancer: 2, errorsync[{name}]={error}")
                self.error_update(name, error[name])
                # log.info(f"rank={rank}, balancer: 3")
                # self.error_update(name, self.target[name] - value)

                for oname in self.pid_gains:
                    p_gain, i_gain, d_gain = self.pid_gains[oname]
                    pid_out = (
                        p_gain * self.p_error_ema[name]
                        + i_gain * self.i_error[name]
                        + d_gain * self.d_error[name]
                    )

                    if self.mode[oname] is None:
                        pid_out_final = pid_out
                    elif self.mode[oname] == "incremental":
                        pid_out_final = self.pid_out[oname] + pid_out
                    elif self.mode[oname] == "one_plus":
                        pid_out_final = 1.0 + pid_out

                    if pid_out_final > self.clip[oname][1]:
                        pid_out_final = self.clip[oname][1]
                    if pid_out_final < self.clip[oname][0]:
                        pid_out_final = self.clip[oname][0]

                    self.pid_out[oname] = pid_out_final
                    # log.info(f"rank={rank}, balancer: pid_ou[{oname}]={pid_out_final}")

                    # monitor ctrl metrics
                    if self.monitor:
                        self._metrics[f"pid_out_{oname}"] = pid_out
                        self._metrics[f"pid_out_final_{oname}"] = self.pid_out[oname]
                        self._metrics[f"pid_p_error"] = self.p_error[name]
                        self._metrics[f"pid_p_error_ema"] = self.p_error_ema[name]
                        self._metrics[f"pid_i_error"] = self.i_error[name]
                        self._metrics[f"pid_d_error"] = self.d_error[name]

            return self.pid_out

        return None

    def get_metrics(self):
        if self.monitor:
            return self._metrics
        return {}


class Balancer(torch.nn.Module):
    def __init__(self, monitor: bool) -> None:
        super().__init__()
        self._metrics: tp.Dict[str, tp.Any] = {}
        self.monitor = monitor

    @property
    def metrics(self):
        return self._metrics

    def reweight_losses(
        self, losses: tp.Dict[str, torch.Tensor], **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError


class WeightBalancer(Balancer):
    def __init__(
        self,
        weights: tp.Dict[str, float],
        monitor: bool = True,
        pid_ctrl: PIDController = None,
    ) -> None:

        super().__init__(monitor=monitor)
        self.weights = weights
        self.pid_ctrl = pid_ctrl

    def reweight_losses(
        self,
        losses: tp.Dict[str, torch.Tensor],
        disc_loss: float = None,
        **kwargs,
    ) -> torch.Tensor:

        scale_correct = (
            self.pid_ctrl.update("disc_loss", disc_loss)
            if self.pid_ctrl is not None
            else None
        )

        total_loss = 0
        for name, loss in losses.items():

            scale = self.weights[name]

            if scale_correct is not None and name in scale_correct:
                if self.monitor:
                    self._metrics[f"scale_{name}-pid"] = scale
                scale *= scale_correct[name]
                if self.monitor:
                    self._metrics[f"scale_{name}+pid"] = scale
                    self._metrics = {
                        **self._metrics,
                        **self.pid_ctrl.get_metrics(),
                    }

            if self.monitor:
                self._metrics[f"scale_{name}"] = scale

            if scale == 0:
                continue

            total_loss = total_loss + losses[name] * scale

        return total_loss


class UncertaintyBalancer(Balancer):
    def __init__(
        self,
        weights: tp.Dict[str, float],
        monitor: bool = True,
    ) -> None:

        super().__init__(monitor=monitor)

        self.weights = weights

        self.alphas = torch.nn.ParameterDict(
            {
                k: torch.zeros(
                    1,
                )
                for k, v in self.weights.items()
                if v != 0
            }
        )

    def reweight_losses(
        self,
        losses: tp.Dict[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:

        total_loss = 0

        for name, loss in losses.items():

            scale = self.weights[name]
            if scale == 0:
                continue

            alpha = self.alphas[name]

            if self.monitor:
                self._metrics[f"scale_{name}"] = torch.exp(-alpha).item()

            total_loss = total_loss + losses[name] * torch.exp(-alpha) + alpha

        return total_loss


def averager(beta: float = 1):
    """
    Exponential Moving Average callback.
    Returns a single function that can be called to repeatidly update the EMA
    with a dict of metrics. The callback will return
    the new averaged dict of metrics.

    Note that for `beta=1`, this is just plain averaging.
    """
    fix: tp.Dict[str, float] = defaultdict(float)
    total: tp.Dict[str, float] = defaultdict(float)

    def _update(
        metrics: tp.Dict[str, tp.Any],
        weight: float = 1,
    ) -> tp.Dict[str, float]:
        nonlocal total, fix
        for key, value in metrics.items():
            total[key] = total[key] * beta + weight * float(value)
            fix[key] = fix[key] * beta + weight
        return {key: tot / fix[key] for key, tot in total.items()}

    return _update


class GradBalancer(Balancer):
    """Loss balancer.

    The loss balancer combines losses together to compute gradients for the backward.
    A call to the balancer will weight the losses according the specified weight coefficients.
    A call to the backward method of the balancer will compute the gradients, combining all the losses and
    potentially rescaling the gradients, which can help stabilize the training and reasonate
    about multiple losses with varying scales.

    Expected usage:
        weights = {'loss_a': 1, 'loss_b': 4}
        balancer = Balancer(weights, ...)
        losses: dict = {}
        losses['loss_a'] = compute_loss_a(x, y)
        losses['loss_b'] = compute_loss_b(x, y)
        if model.training():
            balancer.backward(losses, x)

    ..Warning:: It is unclear how this will interact with DistributedDataParallel,
        in particular if you have some losses not handled by the balancer. In that case
        you can use `encodec.distrib.sync_grad(model.parameters())` and
        `encodec.distrib.sync_buffwers(model.buffers())` as a safe alternative.

    Args:
        weights (Dict[str, float]): Weight coefficient for each loss. The balancer expect the losses keys
            from the backward method to match the weights keys to assign weight to each of the provided loss.
        rescale_grads (bool): Whether to rescale gradients or not, without. If False, this is just
            a regular weighted sum of losses.
        total_norm (float): Reference norm when rescaling gradients, ignored otherwise.
        emay_decay (float): EMA decay for averaging the norms when `rescale_grads` is True.
        per_batch_item (bool): Whether to compute the averaged norm per batch item or not. This only holds
            when rescaling the gradients.
        epsilon (float): Epsilon value for numerical stability.
        monitor (bool): Whether to store additional ratio for each loss key in metrics.
    """

    def __init__(
        self,
        weights: tp.Dict[str, float],
        rescale_grads: bool = True,
        total_norm: float = 1.0,
        ema_decay: float = 0.999,
        per_batch_item: bool = True,
        epsilon: float = 1e-12,
        monitor: bool = True,
        power: float = 1.0,
        clip: float = 1e5,
        pid_ctrl: PIDController = None,
    ):
        super().__init__(monitor=monitor)
        self.weights = weights
        self.per_batch_item = per_batch_item
        self.total_norm = total_norm
        self.averager = averager(ema_decay)
        self.epsilon = epsilon
        self.monitor = monitor
        self.rescale_grads = rescale_grads
        self.clip = clip
        self.power = power
        self.pid_ctrl = pid_ctrl
        torch.autograd.set_detect_anomaly(True)

    def reweight_losses(
        self,
        losses: tp.Dict[str, torch.Tensor],
        batch: torch.Tensor,
        disc_loss: float = None,
    ) -> torch.Tensor:

        norms = {}
        # grads = {}
        no_grad = {}

        for name, loss in losses.items():
            if loss.requires_grad:
                (grad,) = autograd.grad(loss, [batch], retain_graph=True)
                no_grad[name] = False
            else:
                grad = torch.zeros_like(batch)
                no_grad[name] = True

            if self.per_batch_item:
                dims = tuple(range(1, grad.dim()))
                norm = grad.norm(dim=dims).mean().item()
            else:
                norm = grad.norm().item()
            norms[name] = norm

            # grads[name] = grad

            count = 1
            if self.per_batch_item:
                count = len(grad)
            del grad

        avg_norms = average_metrics(self.averager(norms), count)
        # compute inv_norms
        avg_inv_norms = {}
        for name, no_grad in no_grad.items():
            if no_grad:
                avg_inv_norms[name] = 0.0
            else:
                inv_norm = 1 / (self.epsilon + avg_norms[name])

                if self.clip is not None:
                    inv_norm = min(inv_norm, self.clip)
                avg_inv_norms[name] = (
                    torch.pow(torch.tensor(inv_norm), self.power).item()
                    if inv_norm > 0.0
                    else 0.0
                )

        total_inv_norm = sum(
            [avg_inv_norms[k] for k in avg_inv_norms if self.weights[k] != 0.0]
        )

        self._metrics = {}
        if self.monitor:
            for k, v in avg_inv_norms.items():
                self._metrics[f"ratio_{k}"] = v / total_inv_norm

        if self.pid_ctrl is not None:
            scale_correct = self.pid_ctrl.update("disc_loss", disc_loss)
            total_weights = sum(
                [
                    w * scale_correct[k] if k in scale_correct else w
                    for k, w in self.weights.items()
                ]
            )
            ratios_weights = {
                k: (
                    w * scale_correct[k] / total_weights
                    if k in scale_correct
                    else w / total_weights
                )
                for k, w in self.weights.items()
            }
            if self.monitor:
                self._metrics = {
                    **self._metrics,
                    **self.pid_ctrl.get_metrics(),
                }
        else:
            scale_correct = None
            total_weights = sum([w for k, w in self.weights.items()])
            ratios_weights = {k: w / total_weights for k, w in self.weights.items()}

        final_loss = 0
        for name, avg_inv_norm in avg_inv_norms.items():

            if self.rescale_grads:
                if self.pid_ctrl is not None and name in scale_correct:
                    if self.monitor:
                        self._metrics[f"scale_{name}-pid"] = (
                            self.total_norm * avg_inv_norm
                        )
                scale = ratios_weights[name] * self.total_norm * avg_inv_norm
                if self.pid_ctrl is not None and name in scale_correct:
                    if self.monitor:
                        self._metrics[f"scale_{name}+pid"] = scale
            else:
                if self.pid_ctrl is not None and name in scale_correct:
                    if self.monitor:
                        self._metrics[f"scale_{name}-pid"] = 1.0
                scale = ratios_weights[name]
                if self.pid_ctrl is not None and name in scale_correct:
                    if self.monitor:
                        self._metrics[f"scale_{name}+pid"] = scale

            # monitor scale
            if self.monitor:
                self._metrics[f"scale_{name}"] = scale

            # scale current loss
            loss = losses[name] * scale

            # accumulate to final loss
            final_loss = final_loss + loss

        return final_loss
