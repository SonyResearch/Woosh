import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Mapping
import torch.nn as nn


rank = 0
# get logger
log = logging.getLogger(__name__)


@dataclass
class ConditionConfig:
    id: str
    shape: List[int]
    type: str


class DiffusionConditioner(ABC):
    r"""
    Base Conditioner for Diffusion models.
    """

    def __init__(self) -> None:
        assert isinstance(self, nn.Module)

    @property
    @abstractmethod
    def output(self) -> Mapping[str, ConditionConfig]:
        r"""
        The description of the output of the conditioner.
        Can be use to configure the diffusion model.
        """
        pass

    # @property
    # @abstractmethod
    # def trainable(self) -> bool:
    #     r"""
    #     deteremin if the conditioner is trainable and should be saved in the training checkpoint.
    #     """
    #     pass

    @abstractmethod
    def forward(
        self, batch, condition_dropout=0.0, no_cond=False, device=None, **kwargs
    ) -> Mapping:
        pass
