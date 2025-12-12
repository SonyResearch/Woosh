"""Torch distributed utilities. These functions are compatible with MPI """

import os
import typing as tp
import logging
import torch
import psutil

# from mpi4py import MPI

# Gets or creates a logger
log = logging.getLogger(__name__)


def rank():
    """
    Get distributed rank of the current process.
    MPI rank takes precedence over torch.distributed
    as initialization only happens in Trainer
    """
    if torch.distributed.is_initialized():
        # print(f'Rank in distrib {torch.distributed.get_rank()}')
        return torch.distributed.get_rank()
    elif "OMPI_COMM_WORLD_RANK" in os.environ:
        return int(os.environ["OMPI_COMM_WORLD_RANK"])
    elif "SLURM_PROCID" in os.environ:
        return int(os.environ["SLURM_PROCID"])
    return 0


def world_size():
    """
    Get world_size, i.e. total number processes across compute nodes.
    In a multi-node distributed application:
        world_size = n_nodes * n_gpu/node * n_process/gpu
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    elif "OMPI_COMM_WORLD_SIZE" in os.environ:
        return int(os.environ["OMPI_COMM_WORLD_SIZE"])
    elif "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_NTASKS"])
    return 1


def world_local_size():
    """
    Get world_local_size, i.e. total number processes in this compute node.
    In a multi-node distributed application:
        world_local_size = n_gpu/node * n_process/gpu
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    elif "OMPI_COMM_WORLD_LOCAL_SIZE" in os.environ:
        return int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"])
    elif "SLURM_LOCALID" in os.environ:
        return int(os.environ["SLURM_LOCALID"])
    return 1


def local_rank():
    if "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        return int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    elif "SLURM_LOCALID" in os.environ:
        return int(os.environ["SLURM_LOCALID"])
    return 1


def world_local_size_or_num_cores():
    """
    Get world_local_size, i.e. total number processes in this compute node.
    In a multi-node distributed application:
        world_local_size = n_gpu/node * n_process/gpu
    """
    # no MPI, return number of cores in the machine
    # we limit the number of cpus to 16 = 32 // 2
    return min(psutil.cpu_count(logical=False), 16)


def rank_zero_custom(fn):
    """
    This decorator will only execute the decorated function if the process rank is 0.
    Works BEFORE torch.distributed is initialized with OMPI_COMM_WORLD_RANK
    """

    def wrapper(*args, **kwargs):
        if rank() == 0:
            return fn(*args, **kwargs)
        return None

    return wrapper


def is_distributed():
    """Distributed Torch training/eval"""
    return world_size() > 1


def all_reduce(tensor: torch.Tensor, op=torch.distributed.ReduceOp.SUM):
    """all_reduce to work on distributed environment only"""
    if is_distributed():
        return torch.distributed.all_reduce(tensor, op)
    return tensor


def _is_complex_or_float(tensor):
    """Check tensor is either complex or float type"""
    return torch.is_floating_point(tensor) or torch.is_complex(tensor)


def _check_number_of_params(params: tp.List[torch.Tensor]):
    """
    Utility function to check that the number of params in all workers is the same,
    and thus avoid a deadlock with distributed all reduce.
    """
    if not is_distributed() or not params:
        return
    tensor = torch.tensor([len(params)], device=params[0].device, dtype=torch.long)
    all_reduce(tensor)
    if tensor.item() != len(params) * world_size():
        # If not all the workers have the same number, for at least one of them,
        # this inequality will be verified.
        raise RuntimeError(
            f"Mismatch in number of params: ours is {len(params)}, "
            "at least one worker has a different one."
        )


def broadcast_tensors(tensors: tp.Iterable[torch.Tensor], src: int = 0):
    """Broadcast the tensors from the given parameters to all workers.
    This can be used to ensure that all workers have the same model to start with.
    """
    if not is_distributed():
        return
    tensors = [tensor for tensor in tensors if _is_complex_or_float(tensor)]
    _check_number_of_params(tensors)
    handles = []
    for tensor in tensors:
        handle = torch.distributed.broadcast(tensor.data, src=src, async_op=True)
        handles.append(handle)
    for handle in handles:
        handle.wait()


def all_reduce_tensors(
    tensors: tp.Iterable[torch.Tensor], op=torch.distributed.ReduceOp.SUM
):
    """Broadcast the tensors from the given parameters to all workers.
    This can be used to ensure that all workers have the same model to start with.
    """
    # print(f'all_reduce_tensors #{rank()}')

    if not is_distributed():
        return
    tensors = [tensor for tensor in tensors if _is_complex_or_float(tensor)]
    _check_number_of_params(tensors)
    handles = []
    for tensor in tensors:
        handle = torch.distributed.all_reduce(tensor.data, op=op, async_op=True)
        handles.append(handle)
    for handle in handles:
        handle.wait()


def mpi():
    """check if mpi4py.MPI has been imported"""
    try:
        MPI  # pylint: disable=E0602,W0104
        return True
    except:
        return False


def mpi_comm():
    """get mpi communications object"""
    if mpi():
        if is_distributed():
            return MPI.COMM_WORLD  # pylint: disable=I1101,E0602
        return None
    log.warning("mpi_comm always returns None")
    return None


def mpi_barrier():
    """wait for all MPI processes to sync"""
    if mpi():
        comm = mpi_comm()
        if comm is not None:
            comm.Barrier()
    log.warning("mpi_barrier is a no-op")


def nop():
    pass


def average_metrics(metrics: tp.Dict[str, float], count=1.0):
    """Average a dictionary of metrics across all workers, using the optional
    `count` as unnormalized weight.
    """
    if not is_distributed():
        return metrics
    keys, values = zip(*metrics.items())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(values) > 0:
        tensor = torch.tensor(list(values) + [1], device=device, dtype=torch.float32)
    else:
        tensor = torch.tensor([0] + [0], device=device, dtype=torch.float32)
    tensor *= count
    all_reduce(tensor)
    averaged = (tensor[:-1] / tensor[-1]).cpu().tolist()
    return dict(zip(keys, averaged))


def get_cuda_devices():
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        devices = os.environ["CUDA_VISIBLE_DEVICES"]
        return devices.split(",")
    else:
        return None


def get_cuda_device():
    """ get cuda:index device"""
    # use rank as index
    devices = get_cuda_devices()
    if devices is not None:
        rnk = local_rank()
        return f"cuda:{rnk}"
    return None
