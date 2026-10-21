"""Tests of the additive background term of the NNAL fit: the model (1 - b) exp(-W H) + b, with b per bin."""
import numpy as np
import torch

from mbirtorch.hsnt._loss import _nnal_elementwise, _nnal_prep, stable_nnal, stable_nnal_derivatives
from mbirtorch.hsnt.factorization import _nnal_factorization_background, _spectra_background_step
from hsnt_basis import material_basis

N, K = 32, 150
DOSE = 1000.0                                           # open-beam counts per pixel and bin
B_TRUE = 0.05                                           # the background share of every bin
B_NOISE = 0.02                                          # the mean fitted b allowed when there is no background


def _sphere_attenuation(n=N, bins=K):
    """The attenuation of the three overlapping spheres of test_hsnt_solvers._sphere_problem, with shape (n n, bins)."""
    W, H = _sphere_factors(n, bins)
    return W @ H


def _sphere_factors(n=N, bins=K):
    """The maps (n n, 3) and spectra (3, bins) of the three spheres, float64."""
    basis = material_basis()
    yy, xx = np.mgrid[:n, :n] + 0.5
    maps = []
    for (cy, cx), density in zip(((0.38, 0.38), (0.38, 0.62), (0.6, 0.5)), (0.25, 0.25, 0.75)):
        d2 = ((yy - cy * n) ** 2 + (xx - cx * n) ** 2) / (0.22 * n) ** 2
        maps.append(10.0 * density * np.sqrt(np.clip(1.0 - d2, 0.0, None)))
    return (torch.tensor(np.stack(maps, -1).reshape(-1, 3), dtype=torch.float64),
            torch.tensor(basis[:, ::basis.shape[1] // bins].copy(), dtype=torch.float64))


def _with_background(X, b, dose=None, seed=3):
    m = (1 - b) * torch.exp(-X) + b
    if dose is None:
        return m
    return torch.poisson(dose * m, generator=torch.Generator().manual_seed(seed)) / dose


def test_a_zero_background_gives_the_loss_and_derivatives_without_one():
    X = _sphere_attenuation()
    T = _with_background(X, 0.0, dose=10.0)
    plain, zero = _nnal_prep(T), _nnal_prep(T, torch.zeros(K, dtype=T.dtype))
    assert torch.equal(_nnal_elementwise(X, T, plain), _nnal_elementwise(X, T, zero))
    for a, b in zip(stable_nnal_derivatives(X, T, plain), stable_nnal_derivatives(X, T, zero)):
        assert torch.equal(a, b)


def test_the_derivatives_with_a_background_match_autograd():
    """G is the exact derivative.  Where the data equal the model, Z is the exact second derivative."""
    X = _sphere_attenuation()[:200]
    b = torch.linspace(0.01, 0.2, K, dtype=torch.float64)
    for T in (_with_background(X, b, dose=10.0), _with_background(X, b)):
        prep = _nnal_prep(T, b)
        Xg = X.clone().requires_grad_(True)
        g, = torch.autograd.grad(_nnal_elementwise(Xg, T, prep).sum(), Xg, create_graph=True)
        G, Z = stable_nnal_derivatives(X, T, prep)
        np.testing.assert_allclose(G, g.detach(), rtol=1e-9, atol=1e-12)
    h, = torch.autograd.grad(g.sum(), Xg)
    np.testing.assert_allclose(Z, h, rtol=1e-9, atol=1e-12)


def test_the_spectra_and_background_step_recovers_them_for_the_true_maps():
    """Noise-free data: from the true maps, spectra 20% off and no background, the step finds both."""
    W, H = _sphere_factors()
    b = torch.linspace(0.0, 0.2, K, dtype=torch.float64)
    H_fit, b_fit = _spectra_background_step(W, 0.8 * H, torch.zeros(K, dtype=torch.float64), _with_background(W @ H, b))
    np.testing.assert_allclose(b_fit, b, atol=1e-6)
    np.testing.assert_allclose(H_fit, H, rtol=1e-5, atol=1e-6)


def test_the_fit_finds_a_background_and_none_where_there_is_none():
    """b >= 0 is weakly determined in this small problem, so its estimate is biased upward where b = 0."""
    X = _sphere_attenuation().float()
    for b_true, tol in ((B_TRUE, 0.01), (0.0, B_NOISE)):
        T = _with_background(X, b_true, dose=DOSE)
        W, H, b, rounds, _ = _nnal_factorization_background(T, 3, max_steps=300, compile_mode="off")
        assert abs(float(b.mean()) - b_true) < tol, (b_true, float(b.mean()))
        loss = stable_nnal(W.double() @ H.double(), T.double(), _nnal_prep(T.double(), b.double())).item()
        assert np.isfinite(loss) and rounds >= 1
