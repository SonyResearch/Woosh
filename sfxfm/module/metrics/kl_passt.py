from typing import Optional

import torch
import torch.nn.functional as F

from .base_aggregatepool import AggregatePoolMetric


class KLScoreMetric(AggregatePoolMetric):
    def __init__(
        self,
        frontend: torch.nn.Module,
        patchify: Optional[
            int
        ] = None,  # Number of consecutive vectors to form a patch.
        temporal_pooling: str = "flatten",  # (flatten, average, sampleXXX, firstXXX, linspaceXXX)
        store_feats_in_cpu: bool = False,  # Note: for multiGPU + storing in CPU, aggregation wil be
        #       the average across GPUs; proper aggregation across
        #       multiple GPUs will only happen when =False.
        eps=1e-6,
    ):
        super().__init__(
            frontend=frontend,
            patchify=patchify,
            temporal_pooling=temporal_pooling,
            store_feats_in_cpu=store_feats_in_cpu,
        )
        self.eps = eps

    def compute(self):
        if self._h_empty():
            return float("nan")
        real_probs, fake_probs = self._get_collected_features()
        scores = self.compute_KL_multiple(real_probs, fake_probs)
        if scores.size(0) == 0:
            return self._return_value(0)
        score = scores.mean()
        return self._return_value(score)

    def compute_KL_multiple(self, real_probs, fake_probs) -> torch.Tensor:
        passt_kl_list = []

        for i in range(real_probs.size(0)):  # Iterate over audios
            real_sample = real_probs[i, :]
            fake_sample = fake_probs[i, :]
            passt_kl_list.append(
                F.kl_div(
                    (real_sample + 1e-6).log(),
                    fake_sample,
                    reduction="sum",
                    log_target=False,
                )
            )
        return torch.tensor(passt_kl_list).to(self.device)
