import torch
from typing import Optional

from .base_aggregatepool import BatchBasedAggregatePoolMetric


class CLAPScoreMetric(BatchBasedAggregatePoolMetric):
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
        self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def compute(self, sgradio: bool = False):
        if self._h_empty():
            return float("nan")
        h_real_embeds, h_fake_embeds = self._get_collected_features()
        sims = self.cos_sim(h_real_embeds, h_fake_embeds)
        if sgradio:
            return sims
        return self._return_value(sims.mean())
