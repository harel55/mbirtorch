"""Tests of the experimental strain-tomography projectors in ``mbirtorch.hsnt.strain``."""
import numpy as np
import pytest
import torch

from mbirtorch.hsnt.strain.tomo import MultiProjector

N = 12
ANGLES = np.linspace(0, np.pi, 5, endpoint=False)
MOUNTS = [('x', ANGLES),
          (((1, 2, 0), (1, 1, 1), (0.4, -2.3, 1.7), (-0.35, 1.93, -0.63)), ANGLES),
          (((2, 0, 1), (-1, 1, -1), (3.0, -0.5, 0.25)), ANGLES)]


@pytest.fixture(scope="module")
def proj():
    """Three series: one plain, one with a fractional shift and a small rotation, one with a fractional shift only."""
    return MultiProjector(N, MOUNTS, 2, channel_offsets=[0.0, 0.3, -0.2], ror_mask=False)


def test_adjoint(proj):
    """back is the adjoint of fwd, including fractional shifts and small rotations."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(N, N, N, 2, generator=g)
    y = torch.randn(sum(proj.counts), 2, generator=g)
    a = float((proj.fwd(x) * y).sum())
    b = float((x * proj.back(y)).sum())
    assert abs(a - b) <= 1e-4 * abs(a)


def test_set_mount_round_trip(proj):
    """Setting a series' mount to new values and back restores its projection."""
    x = torch.rand(N, N, N, 2, generator=torch.Generator().manual_seed(1))
    before = proj.fwd(x)
    (perm, signs, shift, rot), part = proj.parts[1]
    off = part.channel_offset
    proj.set_mount(1, shift=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0), offset=off + 0.5)
    assert not torch.allclose(proj.fwd(x), before)
    proj.set_mount(1, shift=shift, rot=rot, offset=off)
    assert torch.allclose(proj.fwd(x), before)


@pytest.mark.parametrize("prior", ["quadratic", "qggmrf"])
def test_density_prior_gradient(prior):
    """The density prior's gradient (dens_lap at V) matches central differences of its value."""
    from mbirtorch.hsnt.strain.tomo import TomoFit
    tf = TomoFit.__new__(TomoFit)
    tf.R, tf.voxel_scale, tf.density_prior = 1, 0.7, prior
    tf.sigma_V = torch.tensor([0.05], dtype=torch.float64)
    tf.smooth_V = tf.voxel_scale / tf.sigma_V ** 2
    V = torch.rand(6, 6, 6, 1, dtype=torch.float64, generator=torch.Generator().manual_seed(2)) * 0.1
    V[3:] += 0.5                                          # a sharp jump
    tf.V = V
    g = tf.dens_lap(V, tf._dens_wts(V))
    h = 1e-6
    for idx in [(0, 0, 0, 0), (2, 3, 1, 0), (3, 3, 3, 0)]:
        Vp, Vm = V.clone(), V.clone(); Vp[idx] += h; Vm[idx] -= h
        num = (tf._dens_prior_value(Vp) - tf._dens_prior_value(Vm)) / (2 * h)
        assert abs(float(g[idx]) - num) <= 1e-5 * max(1.0, abs(num))
