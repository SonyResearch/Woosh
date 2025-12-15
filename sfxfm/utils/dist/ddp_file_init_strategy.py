""" DDP strategy using the file system """

from typing import Any, Callable, List, Optional
from datetime import timedelta
import os
import logging
import torch
import lightning.pytorch as pl
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.plugins.precision import PrecisionPlugin
from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning.fabric.utilities.seed import reset_seed

from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.fabric.utilities.distributed import _init_dist_connection
from sfxfm.utils.dist import rank

rank = rank()

# get logger
log = logging.getLogger(__name__)


class DDPFileInitStrategy(DDPStrategy):
    """DDP File Init Stragegy"""

    def __init__(
        self,
        shared_file,
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
        restart_count = int(os.environ.get("SLURM_RESTART_COUNT", 0))
        # when restarting with/ without the same job id, the shared file should be different
        # the bug is the old job id is retrieved from the resolved config, we get a fresh job id here
        job_id = os.environ.get("SLURM_JOB_ID", "0")    

        shared_file = f"{shared_file}.j{job_id}.r{restart_count}"
        
        self._shared_file = shared_file

    def setup_distributed(self) -> None:
        """Setup DDP process group using a shared file strategy"""
        if rank == 0:
            log.info(
                f"{self.__class__.__name__}: setting up distributed, "
                f"shared file {self._shared_file}"
            )
        reset_seed()
        self.set_world_ranks()
        rank_zero_only.rank = self.global_rank
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        # add timeout?
        #   timeout=self._timeout,
        _init_dist_connection(
            self.cluster_environment,
            #   torch_distributed_backend=self._process_group_backend,
            #   torch_distributed_backend='mpi',
            torch_distributed_backend="nccl",
            init_method=f"file://{self._shared_file}",
        )
