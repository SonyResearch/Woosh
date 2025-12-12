from contextlib import nullcontext
from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sfxfm.utils.dist import rank, world_size
from sfxfm.utils.dist.distrib import is_distributed

from torch.distributed.nn.functional import all_gather


class SimpleContrastiveLoss(nn.Module):
    def __init__(
        self,
        dual_temp: bool = False,
        initial_tau_audio: Optional[Union[int, float]] = 0.02,
        initial_tau_text: Optional[Union[int, float]] = 0.02,
        optimize_tau: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.name = "simple-contrastive-loss"
        self.dual_temp = dual_temp
        self.optimize_tau = optimize_tau

        if self.dual_temp:
            self.tau = [torch.tensor(initial_tau_audio), torch.tensor(initial_tau_text)]
            if self.optimize_tau:
                self.tau = nn.ParameterList(
                    [nn.Parameter(self.tau[0]), nn.Parameter(self.tau[1])]
                )
        else:
            self.tau = torch.tensor(initial_tau_audio)
            if self.optimize_tau:
                self.tau = nn.Parameter(self.tau)

        self.logit_bias = None
        self.is_distributed = is_distributed()

    def all_gather(self, x, group=None, sync_grads=True):
        x = x.contiguous()  # https://github.com/pytorch/pytorch/issues/73515
        with nullcontext() if sync_grads else torch.no_grad():
            gathered_x = all_gather(x, group)
        return torch.concat(gathered_x)

    @torch.compiler.disable
    def compute_target_matrix(self, target):
        return target.unsqueeze(0) == target.unsqueeze(1)

    def get_ground_truth(self, labels, device, num_logits):
        if labels is None:
            # assume 1-to-1 mapping of audio and text
            labels = torch.eye(
                num_logits,
                num_logits,
                device=device,
                dtype=torch.int64,
            )
        else:
            # constuct matrix for matching audio samples
            labels = self.compute_target_matrix(labels)
        return labels

    def get_logits(self, audio_features, text_features):
        return torch.matmul(audio_features, text_features.T)

    def _loss(self, audio_features, text_features, labels, return_scores=False):
        labels = self.get_ground_truth(
            labels,
            audio_features.device,
            audio_features.shape[0],
        )

        assert (
            len(audio_features) == len(text_features)
        )  # , f"Captions: {len(batch['caption'])}, Audios: {len(batch['audio'])}, Audio Features Shape: {audio_features.shape} Sentence Features Shape: {sentence_features.shape}"

        C = self.get_logits(audio_features, text_features)

        if self.dual_temp:
            C_audio = torch.log_softmax(C / torch.abs(self.tau[0]), dim=0)
            C_text = torch.log_softmax(C / torch.abs(self.tau[1]), dim=1)
        else:
            C_audio = torch.log_softmax(C / torch.abs(self.tau), dim=0)
            C_text = C_audio
        assert C_audio.shape[0] == C_audio.shape[1], (
            f"Audio Features Shape: {audio_features.shape} Sentence Features Shape: {text_features.shape}"
        )
        assert C_text.shape[0] == C_text.shape[1]
        loss = -0.5 * (
            C_audio[torch.where(labels)].mean() + C_text[torch.where(labels)].mean()
        )

        if return_scores:
            return loss, C
        return loss

    def forward(self, audio_features, text_features, id_hash=None):
        if self.is_distributed:
            audio_features = self.all_gather(audio_features, sync_grads=True)
            text_features = self.all_gather(text_features, sync_grads=True)

        if id_hash is not None:
            if self.is_distributed:
                id_hash = self.all_gather(id_hash)
            labels = id_hash
        else:
            labels = None

        loss = self._loss(audio_features, text_features, labels)

        return loss


class DecoupledContrastiveLoss(SimpleContrastiveLoss):
    """DHEL: Decoupled variant of InfoNCE Loss for Contrastive Learning

    By removing positive pairs from the denominator, DHEL 'decouples' the alignment objective from the uniformity objective.
    This leads to better uniformity of the embedding space.
    See: https://arxiv.org/abs/2405.18045
    Bridging Mini-Batch and Asymptotic Analysis in Contrastive Learning: From InfoNCE to Kernel-Based Losses
    Panagiotis Koromilas et al., ICML 2024
    """

    def __init__(
        self,
        dual_temp: bool = False,
        initial_tau_audio: Optional[Union[int, float]] = 0.07,
        initial_tau_text: Optional[Union[int, float]] = 0.07,
        optimize_tau: bool = False,
        **kwargs,
    ):
        super().__init__(
            dual_temp=dual_temp,
            initial_tau_audio=initial_tau_audio,
            initial_tau_text=initial_tau_text,
            optimize_tau=optimize_tau,
            **kwargs,
        )
        self.name = "decoupled-contrastive-loss"

    def _loss(self, audio_features, text_features, labels, return_scores=False):
        labels = self.get_ground_truth(
            labels,
            audio_features.device,
            audio_features.shape[0],
        )

        assert (
            len(audio_features) == len(text_features)
        )  # , f"Captions: {len(batch['caption'])}, Audios: {len(batch['audio'])}, Audio Features Shape: {audio_features.shape} Sentence Features Shape: {sentence_features.shape}"

        C = self.get_logits(audio_features, text_features)
        # dtype = audio_features.dtype
        # eps = torch.finfo(dtype).eps
        max_val = torch.max(C)
        # TODO: Check if I should add an eps here
        C = C - max_val.detach()  # for numerical stability
        if self.dual_temp:
            numerator_audio = torch.exp(torch.diagonal(C) / torch.abs(self.tau[0]))
            numerator_text = torch.exp(torch.diagonal(C) / torch.abs(self.tau[1]))
            # we only use single positive pairs [diagonal only] here for simplicity
            # other matching labels are used to mask negative pairs
            C_exp_a = torch.exp(C / torch.abs(self.tau[0]))
            C_exp_a = C_exp_a * ~labels # mask positive pairs
            C_exp_t = torch.exp(C / torch.abs(self.tau[1]))
            C_exp_t = C_exp_t * ~labels # mask positive pairs
            # audio is dim 0, text is dim 1
            denominator_text = C_exp_t.sum(dim=1)
            denominator_audio = C_exp_a.sum(dim=0)
            loss_audio = torch.log(numerator_audio) - torch.log(denominator_audio)
            loss_text = torch.log(numerator_text) - torch.log(denominator_text)
        else:
            numerator = torch.exp(torch.diagonal(C) / torch.abs(self.tau))
            # we only use single positive pairs [diagonal only] here for simplicity
            # other matching labels are used to mask negative pairs
            C_exp = torch.exp(C / torch.abs(self.tau))
            C_exp = C_exp * ~labels # mask positive pairs
            # audio is dim 0, text is dim 1
            denominator_text = C_exp.sum(dim=1)
            denominator_audio = C_exp.sum(dim=0)
            loss_audio = torch.log(numerator) - torch.log(denominator_audio)
            loss_text = loss_audio
        assert loss_audio.shape[0] == loss_audio.shape[0], (
            f"Audio Features Shape: {audio_features.shape} Sentence Features Shape: {text_features.shape}"
        )

        loss = -0.5 * (loss_audio.mean() + loss_text.mean())

        if return_scores:
            return loss, C
        return loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(
    left_rank, right_rank, tensor_to_left, tensor_to_right, group=None
):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv(
        [send_op_right, send_op_left, recv_op_right, recv_op_left]
    )
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (
            NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),
        )


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(
            left_rank, right_rank, tensor_to_left, tensor_to_right, group=group
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + NeighbourExchangeBidir.apply(
            ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs
        )


def neighbour_exchange_bidir_with_grad(
    left_rank, right_rank, tensor_to_left, tensor_to_right, group=None
):
    return NeighbourExchangeBidir.apply(
        left_rank, right_rank, group, tensor_to_left, tensor_to_right
    )


class SigLipLoss(nn.Module):
    """Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343
    Adapted from: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/loss.py

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """

    def __init__(
        self,
        initial_tau: float,
        initial_logit_bias: Union[int, float],
        cache_labels: Optional[bool] = False,
        bidir: Optional[bool] = True,
        use_horovod: Optional[bool] = False,
        rank: int = rank(),
        world_size: int = world_size(),
    ):
        super().__init__()

        self.tau = nn.Parameter(torch.ones([]) * np.log(initial_tau))
        self.logit_bias = nn.Parameter(torch.ones([]) * initial_logit_bias)
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    @torch.compiler.disable
    def compute_target_matrix(self, target):
        return target.reshape(-1).unsqueeze(0) == target.reshape(-1).unsqueeze(1)

    def get_ground_truth(
        self, labels, device, num_logits, negative_only=False
    ) -> torch.Tensor:
        negatives = -torch.ones(
            (num_logits, num_logits), device=device, dtype=torch.int64
        )
        if not negative_only:
            if labels is None:
                # diagonal to 1, rest -1
                labels = (
                    2 * torch.eye(num_logits, device=device, dtype=torch.int64)
                    + negatives
                )
            else:
                # positives to 1, rest -1
                labels = (2 * self.compute_target_matrix(labels)) + negatives
        else:
            labels = negatives

        return labels

    def get_logits(self, image_features, text_features):
        logits = self.tau.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            logits += self.logit_bias
        return logits

    def _loss(
        self,
        image_features,
        text_features,
        labels,
        negative_only=False,
        return_scores=False,
    ):
        logits = self.get_logits(image_features, text_features)
        # FIXME: always uses this way of labels (no multiple gt possible)
        labels = self.get_ground_truth(
            labels,
            image_features.device,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]

        if return_scores:
            return loss, logits
        return loss

    def forward(self, image_features, text_features, id_hash=None, output_dict=False):
        loss = self._loss(image_features, text_features, id_hash)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            id_hash,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right
                    )

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        id_hash,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right
                    )

                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        id_hash,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left

        return {"contrastive_loss": loss} if output_dict else loss
