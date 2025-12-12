""" DDP strategy using the file system """

import os
from typing import Any, Callable, List, Optional
from datetime import timedelta
import logging
import torch
import lightning.pytorch as pl
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.plugins.precision import PrecisionPlugin
from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning.fabric.utilities.seed import reset_seed

# from lightning.fabric.strategies.ddp import DDPStrategy
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.fabric.utilities.distributed import _init_dist_connection
from sfxfm.utils.dist import rank

rank = rank()

# get logger
log = logging.getLogger(__name__)


class DDPSocketInitStrategy(DDPStrategy):
    """DDP File Init Stragegy"""

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        ddp_comm_state: Optional[object] = None,
        ddp_comm_hook: Optional[Callable] = None,
        ddp_comm_wrapper: Optional[Callable] = None,
        model_averaging_period: Optional[int] = None,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = ...,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            accelerator,
            parallel_devices,
            cluster_environment,
            checkpoint_io,
            precision_plugin,
            ddp_comm_state,
            ddp_comm_hook,
            ddp_comm_wrapper,
            model_averaging_period,
            process_group_backend,
            timeout,
            **kwargs,
        )

    def setup_distributed(self) -> None:
        """Setup DDP process group using a socket TCP strategy"""

        if rank == 0:
            log.info(
                f"{self.__class__.__name__}: setting up distributed, "
                f"MASTER_ADDR={os.getenv('MASTER_ADDR')}, "
                f"MASTER_PORT={os.getenv('MASTER_PORT')}"
            )
        reset_seed()
        self.set_world_ranks()
        rank_zero_only.rank = self.global_rank
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        _init_dist_connection(
            self.cluster_environment,
            torch_distributed_backend="nccl",
        )
