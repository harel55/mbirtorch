"""Tests of the ``sample_mask`` option of ``dehydrate``."""

import numpy as np
import pytest

import mbirtorch.hsnt as hsnt
from test_hsnt_solvers import _sphere_problem

N = 48
CENTERS = ((0.38, 0.38), (0.38, 0.62), (0.6, 0.5))      # the sphere centers of _sphere_problem
RADIUS2 = (0.22 * N) ** 2 * 1.1                         # a little larger than the spheres


@pytest.fixture(scope="module")
def spheres():
    """The three-sphere transmission image and a mask that covers the spheres."""
    T = _sphere_problem("cpu", n=N, dose=100.0).numpy().reshape(N, N, -1)
    yy, xx = np.mgrid[:N, :N] + 0.5
    inside = np.zeros((N, N), dtype=bool)
    for cy, cx in CENTERS:
        inside |= (yy - cy * N) ** 2 + (xx - cx * N) ** 2 < RADIUS2
    return T, inside


def _dehydrate(T, mask):
    return hsnt.dehydrate(T, "transmission", 3, sample_mask=mask, verbose=0)


def test_the_maps_are_zero_outside_the_mask_and_fit_inside_it(spheres):
    T, inside = spheres
    W, H = _dehydrate(T, inside)
    assert W.shape == (N, N, 3) and np.isfinite(W).all()
    assert not W[~inside].any()
    W0, H0 = _dehydrate(T, None)
    err_masked = np.abs(np.exp(-(W @ H)) - T)[inside].mean()
    err_full = np.abs(np.exp(-(W0 @ H0)) - T)[inside].mean()
    assert err_masked < 1.1 * err_full


def test_a_mask_per_view_broadcasts_to_the_views(spheres):
    T, inside = spheres
    data = np.stack([T, np.roll(T, 3, axis=1)])
    mask = np.stack([inside, np.roll(inside, 3, axis=1)])
    W, _ = _dehydrate(data, mask)
    assert W.shape == (2, N, N, 3)
    assert not W[~mask].any()
    assert W[mask].any()


def test_bad_masks_are_rejected(spheres):
    T, inside = spheres
    with pytest.raises(ValueError, match="broadcast"):
        hsnt.dehydrate(T, "transmission", 3, sample_mask=inside[:-1], verbose=0)
    with pytest.raises(ValueError, match="False everywhere"):
        hsnt.dehydrate(T, "transmission", 3, sample_mask=np.zeros((N, N), dtype=bool), verbose=0)
    with pytest.raises(ValueError, match="boolean"):
        hsnt.dehydrate(T, "transmission", 3, sample_mask=inside.astype(np.float32) * 2, verbose=0)
