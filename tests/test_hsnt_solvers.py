"""Smoke tests for the NNAL solvers in mbirtorch.hsnt.

Self-contained: a small random nonnegative factorization stands in for the material phantom, so
no external basis file is needed. Each test runs in a few seconds on a GPU and is skipped without
one (the streaming path pins host memory for CUDA transfers).
"""
import numpy as np
import pytest
import torch

hsnt = pytest.importorskip("mbirtorch.hsnt")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _problem(P=2048, K=200, R=3, dose=10.0, seed=0, noisy=True, dtype=torch.float32):
    """T = counts / dose for X = W_true @ H_true with a sparse nonnegative W (one material per pixel)."""
    rng = np.random.default_rng(seed)
    H = rng.uniform(0.05, 1.0, size=(R, K)); H[:, K // 3:] *= 0.5                # rough edge structure
    W = np.zeros((P, R)); m = rng.integers(0, R, P); W[np.arange(P), m] = rng.uniform(0.2, 2.0, P)
    W[: P // 8] = 0                                                                 # background pixels
    X = W @ H
    T = rng.poisson(dose * np.exp(-X)) / dose if noisy else np.exp(-X)
    return (torch.tensor(T, dtype=dtype, device="cuda"), torch.tensor(W, dtype=torch.float64, device="cuda"),
            torch.tensor(H, dtype=torch.float64, device="cuda"))


def _loss(W, H, T):
    return hsnt.stable_nnal(W.double() @ H.double(), T.double()).item()


@cuda
@pytest.mark.parametrize("method", ["joint_newton", "block_newton", "mann_multiplicative"])
def test_every_method_decreases_the_loss_and_stays_nonnegative(method):
    T, _, _ = _problem()
    W, H, steps = hsnt.nnal_factorization(T, method=method, num_materials=3, max_steps=300, rel_tol=1e-6)
    assert steps > 0 and W.min() >= 0 and H.min() >= 0
    # the initialization alone, for reference
    real = T > 1e-12; floor = 0.5 * T[real].min()
    W0, H0 = hsnt.nndsvda(-torch.log(torch.where(real, T, floor)), n_components=3)
    assert _loss(W, H, T) < _loss(W0, H0, T)


@cuda
def test_joint_newton_reaches_machine_precision_on_exact_data():
    T, _, _ = _problem(noisy=False, dtype=torch.float64)
    W, H, _ = hsnt.nnal_factorization(T, method="joint_newton", num_materials=3, max_steps=300, rel_tol=1e-6)
    assert _loss(W, H, T) < 1e-8 * T.numel()


@cuda
def test_default_method_is_joint_newton():
    assert hsnt.nnal_factorization.__defaults__[0] == "joint_newton"


@cuda
def test_compiled_matches_eager():
    T, _, _ = _problem()
    W1, H1, _ = hsnt.nnal_factorization(T, method="joint_newton", num_materials=3, max_steps=200, rel_tol=1e-8)
    W2, H2, _ = hsnt.nnal_factorization(T, method="joint_newton", num_materials=3, max_steps=200, rel_tol=1e-8,
                                        compile_mode="default")
    assert abs(_loss(W1, H1, T) - _loss(W2, H2, T)) <= 1e-6 * _loss(W1, H1, T)


@cuda
def test_spectra_estimators_run_and_keep_w_nonnegative():
    T, _, _ = _problem()
    W, H, _ = hsnt.nnal_factorization(T, method="joint_newton", num_materials=3, max_steps=200, rel_tol=1e-8)
    Wu, Hu, _ = hsnt.unconstrained_spectra(T, W, H)
    assert Wu.min() >= 0 and Hu.shape == H.shape
    Ws, Hs, support, _ = hsnt.support_selected_spectra(T, W, H, dose=10.0)
    assert Ws.min() >= 0 and support.dtype == torch.bool and support.shape == W.shape
    assert bool((Ws[~support] == 0).all())                                          # off-support coefficients stay zero


@cuda
def test_streaming_matches_monolithic_within_a_decibel_of_loss():
    T, _, _ = _problem(P=4096)
    tiles = [T[i:i + 1024].cpu() for i in range(0, 4096, 1024)]
    W_chunks, H, passes = hsnt.stream_factorization(tiles, 3, max_passes=3, rel_tol=1e-8, warmup_pixels=1024)
    assert passes >= 1 and all(w.min() >= 0 for w in W_chunks)
    W = torch.cat([w.cuda() for w in W_chunks])
    Wm, Hm, _ = hsnt.nnal_factorization(T, method="joint_newton", num_materials=3, max_steps=200, rel_tol=1e-8)
    assert _loss(W, H, T) <= 1.01 * _loss(Wm, Hm, T)
