"""Tests for the memory of the hsnt solves: the plan that chooses a full or a streamed solve from the device's budget.

The devices are faked: the plan reads its budget through torch.cuda's memory queries, which are patched here, so the
tests run on the CPU and describe a GPU of any size.
"""
from types import SimpleNamespace

import pytest
import torch

import mbirtorch.kernel_availability as kernel_availability
from mbirtorch.hsnt._fit import _plan

GiB = 2 ** 30


@pytest.fixture(autouse=True, scope="module")
def _one_torch_thread():
    """One intra-op thread, so parallel test workers do not each start one per core."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def _fake_cuda(monkeypatch, total, free, allocated=0.0, unused=0.0, releases=True, triton=True):
    """A CUDA device of `total` GiB as the plan reads it: `free` GiB free, `allocated` GiB allocated by this process
    and `unused` GiB more reserved by its caching allocator. The driver counts the unused reserve as free as well, as
    the WSL2 driver was seen to; empty_cache returns the reserve to the driver unless releases=False (segments that
    stay partly in use). Returns the state, with the number of empty_cache calls."""
    dev = dict(free=free * GiB, allocated=allocated * GiB, unused=unused * GiB, emptied=0)

    def empty_cache():
        dev["emptied"] += 1
        if releases:
            dev["free"] += dev["unused"]
            dev["unused"] = 0

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (int(dev["free"] + dev["unused"]),
                                                                          int(total * GiB)))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: int(dev["allocated"] + dev["unused"]))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: int(dev["allocated"]))
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device=None: SimpleNamespace(total_memory=int(total * GiB)))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "test GPU")
    monkeypatch.setattr(kernel_availability, "triton_available", lambda: (triton, "faked"))
    return dev


def test_the_plan_solves_whole_on_a_large_gpu_and_streams_on_a_small_one(monkeypatch):
    """1,048,576 pixels x 1200 bins with 78.6 GiB free (an H100) plans a full solve when the solve compiles
    (its peak was 43.6 GiB); eager, whose peak was 62.4 GiB, it streams in two chunks
    with the whole warm-up."""
    P, K = 2 ** 20, 1200
    _fake_cuda(monkeypatch, total=79.2, free=78.6)
    plan = _plan(P, K, "cuda", "mle")
    assert plan[0] == "full" and "compiled" in plan[-1] and "78.6 GiB available of 79.2" in plan[-1]
    _fake_cuda(monkeypatch, total=79.2, free=78.6, triton=False)
    mode, chunk, warmup, note = _plan(P, K, "cuda", "mle")
    assert (mode, chunk, warmup) == ("stream", 585_728, 16384) and "eager" in note and "(2 chunks)" in note
    assert _plan(P, K, "cuda", "mle", compile_mode="off")[0] == "stream"                 # compile_mode is heeded
    assert _plan(P, K, "cuda", "mle", mode="full")[0] == "full"                            # and a mode given

    # A laptop GPU: 262,144 pixels x 2500 bins with 3.2 GiB free streams in chunks sized to 0.6 of the budget at 64
    # bytes per entry, and fits its warm-up on fewer than 16,384 pixels, which would need more than that share.
    _fake_cuda(monkeypatch, total=4.0, free=3.2)
    mode, chunk, warmup, note = _plan(262_144, 2500, "cuda", "mle")
    assert (mode, chunk, warmup) == ("stream", 11_264, 13_312) and "(24 chunks)" in note
    assert "warm-up fits 13,312 pixels, not 16,384" in note
    assert _plan(262_144, 2500, "cuda", "basis")[2] is None                   # a given basis has no warm-up
    _fake_cuda(monkeypatch, total=4.0, free=0.1)
    assert _plan(262_144, 2500, "cuda", "mle")[2] == 4096                    # at least a few thousand pixels
