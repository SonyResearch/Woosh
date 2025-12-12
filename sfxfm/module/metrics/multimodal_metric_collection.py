import logging
import torch
import torchmetrics
from typing import (
    Any,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from torchmetrics.metric import Metric

from sfxfm.module.metrics.clap_score import CLAPScoreMetric
from sfxfm.module.metrics.kl_passt import KLScoreMetric
from sfxfm.module.metrics.quantization_metrics import BitrateEfficiency, Perplexity, LatentMSE

LOGGER = logging.getLogger(__name__)

AUDIO_TEXT_METRICS = set([CLAPScoreMetric])  # Metric directly uses desc as parameters
NEED_DESCRIPTION = {KLScoreMetric} | AUDIO_TEXT_METRICS
QUANTIZATION_METRICS = set([BitrateEfficiency, Perplexity, LatentMSE])
DEFAULT_DESCRIPTION_KEYS = [
    "description",
    "descriptions",
    "caption",
    "captions",
    "category",
]


def process_batch(batch, indices_of_not_None):
    filtered_batch = {}
    for key in list(batch.keys()):
        if isinstance(batch[key], list):
            filtered_batch[key] = [
                batch[key][i]
                for i in range(len(indices_of_not_None))
                if indices_of_not_None[i]
            ]
        elif isinstance(batch[key], torch.Tensor):
            filtered_batch[key] = batch[key][indices_of_not_None]
        elif isinstance(batch[key], dict):  # externals
            filtered_batch[key] = process_batch(batch[key], indices_of_not_None)
        else:
            filtered_batch[key] = batch[key]
    return filtered_batch


def remove_none_in_descriptions(audio, target, batch, desc_key=None):
    """Finds how samples have an empty description"""
    if desc_key is None:
        for key in DEFAULT_DESCRIPTION_KEYS:
            if key in batch:
                desc_key = key
                break
        if not desc_key:
            raise ValueError("No description found in the dataset")

    if desc_key in DEFAULT_DESCRIPTION_KEYS:
        indices_of_not_None = [
            (element is not None) and (element != "") for element in batch[desc_key]
        ]
    else:
        indices_of_not_None = [
            (element is not None) and (element != "")
            for element in batch["external"][desc_key]
        ]

    if all(not elem for elem in indices_of_not_None):
        return [], [], {}
    filtered_batch = process_batch(batch, indices_of_not_None)
    filtered_audio = audio[indices_of_not_None]
    filtered_target = target[indices_of_not_None]
    return filtered_audio, filtered_target, filtered_batch


class MultiModalMetricCollection(torchmetrics.MetricCollection):
    def __init__(
        self,
        metrics: Union[Metric, Sequence[Metric], Dict[str, Metric]],
        *additional_metrics: Metric,
        prefix: Optional[str] = None,
        postfix: Optional[str] = None,
        compute_groups: Union[bool, List[List[str]]] = True,
    ) -> None:
        super().__init__(
            metrics,
            *additional_metrics,
            prefix=prefix,
            postfix=postfix,
            compute_groups=compute_groups,
        )

    def update(self, preds, target, batch, *args: Any, **kwargs: Any) -> None:
        for metric in self.values():
            if type(metric) in NEED_DESCRIPTION:
                if type(metric) in (AUDIO_TEXT_METRICS | QUANTIZATION_METRICS):
                    desc_key = metric.frontend.desc_key
                else:
                    desc_key = None
                filtered_preds, filtered_target, filtered_batch = (
                    remove_none_in_descriptions(preds, target, batch, desc_key)
                )
                # print(
                #     f"For metric {type(metric)} we used desc_key: {desc_key} and hence removed {len(preds) - len(filtered_preds)} samples"
                # )
            else:
                filtered_preds = preds
                filtered_target = target
                filtered_batch = batch

            if type(metric) in (AUDIO_TEXT_METRICS | QUANTIZATION_METRICS):
                if len(filtered_preds) != 0:
                    metric.update(filtered_preds, filtered_batch, *args, **kwargs)
            else:
                if type(metric) in NEED_DESCRIPTION:
                    if len(filtered_preds) != 0:
                        metric.update(filtered_preds, filtered_target, *args, **kwargs)
                else:
                    metric.update(preds, target, *args, **kwargs)
