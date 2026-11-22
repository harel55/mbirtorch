"""Tests for the open_beam argument of mbirtorch.hsnt.dehydrate: a per-bin open beam in full and streamed solves,
with the spectra given, in the rank estimate, and its checks.

Self-contained: a small random two-material factorization whose counts are drawn with an open beam that rises a
hundredfold across the bins.  The Poisson negative log-likelihood of those counts is the measure of a better fit.
"""
import numpy as np
import pytest
import torch

import mbirtorch.hsnt as hsnt
from mbirtorch.hsnt._fit import _fit
from mbirtorch.hsnt.rank import _estimate_rank

N, K, R = 24, 60, 2
OPEN_BEAM = np.geomspace(2.0, 200.0, K)                  # The open beam count of each bin.
STREAM_CHUNK = 200                                       # Pixels per chunk in the streamed tests, so there are several.


@pytest.fixture(autouse=True, scope="module")
def _one_torch_thread():
    """One intra-op thread, so parallel test workers do not each start one per core."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@pytest.fixture(scope="module")
def data():
    """The transmission (N, N, K) of counts drawn with OPEN_BEAM, and the true spectra."""
    rng = np.random.default_rng(0)
    H = rng.uniform(0.05, 0.8, (R, K))
    H[:, K // 2:] *= 0.5
    W = rng.dirichlet(np.full(R, 0.5), N * N) * rng.uniform(0.3, 1.5, (N * N, 1))
    W[: N * N // 5] = 0
    T = rng.poisson(OPEN_BEAM * np.exp(-(W @ H))) / OPEN_BEAM
    return T.reshape(N, N, K).astype(np.float32), H


def _nll(T, W, H):
    """The Poisson negative log-likelihood of the counts, up to a constant, at the factors W and H."""
    X = W.reshape(-1, R).astype(np.float64) @ H
    return float((OPEN_BEAM * (np.exp(-X) + T.reshape(-1, K) * X)).sum())


@pytest.mark.parametrize("mode", ["full", "stream"])
def test_an_equal_open_beam_is_the_unweighted_fit(data, mode):
    """A scalar open beam and an array with the same count in every bin give equal bin weights and the same fit, bit
    for bit.  Without open_beam the weights are also equal, but the counts that set the start are estimated."""
    from mbirtorch.hsnt.denoise import _bin_weights
    T, _ = data
    kw = dict(mode=mode, verbose=0)
    assert _bin_weights(None, K) is None and _bin_weights(50.0, K) is None and _bin_weights(np.full(K, 50.0), K) is None
    scalar = hsnt.dehydrate(T, "transmission", R, open_beam=50.0, **kw)
    out = hsnt.dehydrate(T, "transmission", R, open_beam=np.full(K, 50.0), **kw)
    assert np.array_equal(out[0], scalar[0]) and np.array_equal(out[1], scalar[1])


@pytest.mark.parametrize("mode", ["full", "stream"])
def test_a_per_bin_open_beam_fits_the_counts_better(data, mode):
    """With the open beam of each bin, the fit changes, and it has a lower Poisson negative log-likelihood than the
    unweighted fit."""
    T, _ = data
    plain = hsnt.dehydrate(T, "transmission", R, mode=mode, verbose=0)
    weighted = hsnt.dehydrate(T, "transmission", R, mode=mode, open_beam=OPEN_BEAM, verbose=0)
    assert not np.array_equal(weighted[1], plain[1])
    assert weighted[0].min() >= 0 and weighted[1].min() >= 0
    assert _nll(T, *weighted) < _nll(T, *plain)


def test_the_streamed_fit_in_several_chunks_uses_the_weights(data):
    """A streamed fit in several chunks with the weights fits the counts better than without them."""
    T, _ = data
    Tf = T.reshape(-1, K)
    w = OPEN_BEAM / OPEN_BEAM.mean()
    kw = dict(mode="stream", chunk_pixels=STREAM_CHUNK, max_passes=10)
    W0, H0, _ = _fit(Tf, R, **kw)
    W1, H1, rep = _fit(Tf, R, weights=w, **kw)
    assert rep["chunk_pixels"] == STREAM_CHUNK and _nll(T, W1, H1) < _nll(T, W0, H0)


def test_the_rank_estimate_uses_the_open_beam(data):
    """The rank estimate weights the bins as the fit does.  Its effective dose is then near the mean open beam, and it
    finds the two materials."""
    T, _ = data
    rank, _, detail = _estimate_rank(T.reshape(-1, K), (1, N, N), "cpu", 4, weights=OPEN_BEAM / OPEN_BEAM.mean())
    assert rank == R and detail["full"]["effective_dose"] == pytest.approx(OPEN_BEAM.mean(), rel=0.2)
    assert hsnt.dehydrate(T, "transmission", open_beam=OPEN_BEAM, max_num_est_materials=4,
                          verbose=0)[1].shape == (R, K)


@pytest.mark.parametrize("mode", ["full", "stream"])
def test_given_spectra_with_a_per_bin_open_beam(data, mode):
    """With the spectra given, a per-bin open beam gives maps of a lower negative log-likelihood."""
    T, H = data
    plain = hsnt.dehydrate(T, "transmission", spectra=H, mode=mode, verbose=0)
    weighted = hsnt.dehydrate(T, "transmission", spectra=H, mode=mode, open_beam=OPEN_BEAM, verbose=0)
    assert np.array_equal(weighted[1], H.astype(np.float32)) and _nll(T, *weighted) < _nll(T, *plain)


def test_a_bad_open_beam_is_refused(data):
    T, _ = data
    for bad, match in ((np.full(K - 1, 50.0), "spectral bins on its last axis"), (np.full((3, K - 1), 50.0), "shape"),
                       (np.r_[OPEN_BEAM[:-1], 0.0], "positive"), (np.r_[OPEN_BEAM[:-1], np.nan], "finite"),
                       (-1.0, "positive")):
        with pytest.raises(ValueError, match=match):
            hsnt.dehydrate(T, "transmission", R, open_beam=bad, verbose=0)


def test_a_full_open_beam_scan_is_averaged_to_one_count_per_bin(data):
    """An open beam scan of shape (scans, rows, cols, bins), as an array or a tensor, gives the fit of its per-bin
    mean.  Single pixels may hold zero counts, as long as every bin's mean is positive."""
    from mbirtorch.hsnt.denoise import _bin_weights
    T, _ = data
    rng = np.random.default_rng(1)
    scan = rng.poisson(OPEN_BEAM, (2, N, N, K)).astype(np.float32)
    assert (scan == 0).any()
    per_bin = scan.reshape(-1, K).mean(axis=0, dtype=np.float64)
    np.testing.assert_allclose(_bin_weights(scan, K), per_bin / per_bin.mean(), rtol=1e-12)
    np.testing.assert_allclose(_bin_weights(torch.from_numpy(scan), K), per_bin / per_bin.mean(), rtol=1e-12)
    kw = dict(mode="full", max_steps=50, verbose=0)
    ref = hsnt.dehydrate(T, "transmission", R, open_beam=per_bin, **kw)
    out = hsnt.dehydrate(T, "transmission", R, open_beam=scan, **kw)
    for a, b in zip(ref, out):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="positive"):
        hsnt.dehydrate(T, "transmission", R, open_beam=np.zeros((N, N, K)), verbose=0)
