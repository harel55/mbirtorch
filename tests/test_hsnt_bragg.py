"""Tests for the hybrid Bragg spectral model in mbirtorch.hsnt.bragg.

Self-contained: the true spectra are drawn from the model itself (cubic lattices with random edge heights and a
smooth part) on a 400-bin wavelength grid, so no external basis file is needed. Nothing about the materials is
handed to the solver: lattice types and parameters are discovered from the data. The factorization tests need
CUDA and take ~10-20 s each.
"""
import itertools

import numpy as np
import pytest
import torch

hsnt = pytest.importorskip("mbirtorch.hsnt")
from mbirtorch.hsnt import bragg  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
LAM = np.linspace(1.5, 4.5, 400)                                                    # Angstrom
TRUE = [("fcc", 3.52), ("fcc", 3.61), ("bcc", 2.87)]                                # Ni-, Cu- and Fe-like lattices


def _spectrum(lattice, a, seed):
    """A model spectrum with random nonnegative edge heights and a smooth part, as a numpy row."""
    rng = np.random.default_rng(seed)
    m = bragg.BraggSpectrum(LAM, lattice, a_range=(a - 0.02, a + 0.02), n_spline=8)
    lin = np.concatenate([rng.uniform(0.05, 0.3, m.n_edges), [0.3, 0.1], np.zeros(m.n_spline)])
    return m(m.pack(a, lin)).numpy()


def _basis():
    return np.stack([_spectrum(l, a, i) for i, (l, a) in enumerate(TRUE)])


def _problem(H, P=2048, dose=30.0, seed=0):
    """T = counts / dose for X = W_true @ H with a sparse nonnegative W (one material per pixel, some background)."""
    rng = np.random.default_rng(seed); R = H.shape[0]
    W = np.zeros((P, R)); m = rng.integers(0, R, P); W[np.arange(P), m] = rng.uniform(0.5, 2.0, P)
    W[: P // 8] = 0
    T = rng.poisson(dose * np.exp(-W @ H)) / dose
    return torch.tensor(T, dtype=torch.float32, device="cuda"), W


def _snr(x, y):
    """dB after a scalar fit of y to x (the scale of a component is never identifiable)."""
    c = float(np.dot(x, y) / max(np.dot(y, y), 1e-300))
    return 20 * np.log10(np.linalg.norm(x) / max(np.linalg.norm(x - c * y), 1e-300))


def _natural(H_fit, H_true):
    """Best permutation of fitted rows against the truth, spectra SNR per true row (natural order, no gauge fit)."""
    perms = itertools.permutations(range(H_fit.shape[0]), H_true.shape[0])
    best = max(perms, key=lambda p: sum(_snr(H_true[i], H_fit[p[i]]) for i in range(H_true.shape[0])))
    return [_snr(H_true[i], H_fit[best[i]]) for i in range(H_true.shape[0])], best


def test_reflections_of_the_cubic_lattices():
    ns, edges = bragg.reflections("fcc", 3.5231, 1.51, 4.53)
    assert np.allclose(edges, 2 * 3.5231 / np.sqrt(ns)) and np.all(np.diff(edges) < 0)   # ascending N, descending wavelength
    assert ns.tolist()[:4] == [3.0, 4.0, 8.0, 11.0] and len(ns) == 8                 # 111, 200, 220, 311, ... in range
    assert np.isclose(edges[0], 4.0685, atol=1e-3) and np.isclose(edges[1], 3.5231, atol=1e-4)
    fcc = set(bragg.reflections("fcc", 4.0, 1.0, 9.0)[0]); dia = set(bragg.reflections("diamond", 4.0, 1.0, 9.0)[0])
    bcc = set(bragg.reflections("bcc", 4.0, 1.0, 9.0)[0])
    assert dia < fcc and all(n % 2 == 0 for n in bcc) and 3.0 in fcc and 4.0 not in dia


def test_fit_spectrum_recovers_the_lattice_parameter_and_the_shape():
    truth = _spectrum("fcc", 3.60, seed=5); rng = np.random.default_rng(1)
    target = truth * (1 + 0.003 * rng.standard_normal(truth.size))
    m = bragg.BraggSpectrum(LAM, "fcc", a_range=(3.3, 3.9), n_spline=8)
    theta, fit, a, rms = bragg.fit_spectrum(m, target)
    assert abs(a - 3.60) < 2e-3 and _snr(truth, fit) > 35 and rms < 0.01
    assert m.split(theta)[1].min() >= 0                                              # heights nonnegative


def test_warm_start_discovers_types_and_parameters_from_mixed_rows():
    """The rows of a bilinear fit are mixtures of the pure spectra; the discovery works on their span."""
    H = _basis(); rng = np.random.default_rng(3)
    rows = rng.uniform(0.2, 1.0, (3, 3)) @ H                                          # an arbitrary gauge
    thetas, lattices, residuals, free = bragg.warm_start(rows, LAM, device="cpu")
    assert free == 0 and len(lattices) == 3
    found = sorted((lt, round(float(th[0]), 3)) for (lt, _), th in zip(lattices, thetas))
    for (lt, a), (lt_true, a_true) in zip(found, sorted(TRUE)):
        assert lt == lt_true and abs(a - a_true) < 0.002 * a_true
    fitted = np.stack([bragg.BraggSpectrum(LAM, lt, a_range=rng_, n_spline=8)(th).numpy() for (lt, rng_), th in zip(lattices, thetas)])
    assert min(_natural(fitted, H)[0]) > 30                                          # pure spectra, in natural order


@cuda
def test_hybrid_factorization_identifies_the_spectra_blind():
    H = _basis(); T, W_true = _problem(H)
    Wb, Hb, _ = hsnt.nnal_factorization(T, method="joint_newton", num_materials=3, max_steps=300, rel_tol=1e-8)
    W, Hh, thetas, info = bragg.hybrid_factorization(T, LAM, init_rows=Hb.double().cpu().numpy(), max_outer=20, rel_tol=1e-7)
    assert W.min() >= 0 and Hh.min() >= 0 and W.shape == (T.shape[0], 3) and Hh.shape == (3, LAM.size)
    snr_h, perm = _natural(Hh.double().cpu().numpy(), H); snr_b, _ = _natural(Hb.double().cpu().numpy(), H)
    assert min(snr_h) > 25 and sum(snr_h) > sum(snr_b) + 20                          # identified without any gauge fit
    for (lt, a), (lt_true, a_true) in zip([info["lattices"][p] for p in perm], TRUE):
        assert lt == lt_true and abs(a - a_true) < 1e-3
    maps = [_snr(W_true[:, i], W[:, perm[i]].double().cpu().numpy()) for i in range(3)]
    _, perm_b = _natural(Hb.double().cpu().numpy(), H)
    maps_b = [_snr(W_true[:, i], Wb[:, perm_b[i]].double().cpu().numpy()) for i in range(3)]
    assert min(maps) > 10 and sum(maps) > sum(maps_b) + 15                           # maps in natural order too
    lb = hsnt.stable_nnal(Wb.double() @ Hb.double(), T.double()).item(); lh = hsnt.stable_nnal(W.double() @ Hh.double(), T.double()).item()
    assert lh < lb * 1.01                                                            # the constraint costs little likelihood
    assert np.all(np.diff(info["loss"]) <= 1e-9 * abs(lh))                           # monotone


@cuda
def test_free_component_absorbs_a_spectrum_the_bragg_model_cannot_represent():
    H3 = _basis()
    line = 0.35 + 0.1 * LAM / 3 + 0.5 * 0.05 ** 2 / ((LAM - 2.8) ** 2 + 0.05 ** 2)    # smooth + a resonance line
    H = np.concatenate([H3, line[None]], 0); T, W_true = _problem(H, dose=100.0)
    Wb, Hb, _ = hsnt.nnal_factorization(T, method="joint_newton", num_materials=4, max_steps=300, rel_tol=1e-8)
    W, Hh, thetas, info = bragg.hybrid_factorization(T, LAM, n_bragg=3, n_free=1, init_rows=Hb.double().cpu().numpy(), max_outer=15)
    assert Hh.shape == (4, LAM.size) and len(info["lattices"]) == 3
    snr_h, perm = _natural(Hh[:3].double().cpu().numpy(), H3)
    assert min(snr_h) > 25                                                           # the Bragg rows stay identified
    free_row = Hh[3].double().cpu().numpy()
    peak = slice(np.argmin(abs(LAM - 2.7)), np.argmin(abs(LAM - 2.9)))
    assert free_row[peak].max() > 1.5 * np.median(free_row)                          # the line lives in the free row
    lb = hsnt.stable_nnal(Wb.double() @ Hb.double(), T.double()).item(); lh = hsnt.stable_nnal(W.double() @ Hh.double(), T.double()).item()
    assert lh < lb * 1.01
