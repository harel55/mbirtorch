"""Sums across processes for a fit whose rays are split over several GPUs (one process per GPU, torch.distributed).
Each process holds a share of the rays and a full copy of the volumes and global parameters; every sum over rays is
completed by these helpers. Without an initialized process group they do nothing."""
import torch
import torch.distributed as dist


def active():
    """True when the rays are split over several processes."""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def rank():
    return dist.get_rank() if active() else 0


def all_sum(t):
    """Sum a tensor over the processes, in place; returns it."""
    if active():
        dist.all_reduce(t)
    return t


def all_sum_float(x, device):
    """Sum a Python number over the processes."""
    if not active():
        return x
    t = torch.tensor([float(x)], dtype=torch.float64, device=device)
    dist.all_reduce(t)
    return float(t)


def broadcast_objects(objs, src=0):
    """Give every process the Python objects of process src (a list, changed in place); returns it."""
    if active():
        dist.broadcast_object_list(objs, src=src)
    return objs
