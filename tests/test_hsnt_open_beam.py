"""Tests for a per-bin open beam in mbirtorch.hsnt: dehydrate with an array dose, in full and streamed solves, with
the spectra held fixed and with support selection.

Self-contained: a small random two-material factorization whose counts are drawn with an open beam that rises a
hundredfold across the bins.  The Poisson negative log-likelihood of those counts is the measure of a better fit.
"""
import numpy as np
import pytest
import torch

import mbirtorch.hsnt as hsnt
from mbirtorch.hsnt.rank import _estimate_rank

N, K, R = 24, 60, 2
OPEN_BEAM = np.geomspace(2.0, 200.0, K)                  # The open beam count of each bin.
STREAM = dict(mode="stream", chunk_pixels=200, max_passes=10)


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


def _nll(T, out):
    """The Poisson negative log-likelihood of the counts, up to a constant, at the fit ``out`` of dehydrate."""
    X = out[0].reshape(-1, R).astype(np.float64) @ out[1]
    return float((OPEN_BEAM * (np.exp(-X) + T.reshape(-1, K) * X)).sum())


@pytest.mark.parametrize("spectra", ["mle", "support"])
@pytest.mark.parametrize("mode", ["full", "stream"])
def test_a_constant_array_dose_is_the_scalar_dose(data, spectra, mode):
    """A constant per-bin dose gives every bin the weight 1, so the fit equals the fit with the scalar dose."""
    T, _ = data
    kw = dict(spectra=spectra, verbose=0, **(STREAM if mode == "stream" else {}))
    scalar = hsnt.dehydrate(T, "transmission", R, dose=50.0, **kw)
    array = hsnt.dehydrate(T, "transmission", R, dose=np.full(K, 50.0), **kw)
    assert np.allclose(array[0], scalar[0], rtol=1e-6, atol=1e-7) and np.allclose(array[1], scalar[1], rtol=1e-6,
                                                                                  atol=1e-7)


@pytest.mark.parametrize("spectra", ["mle", "support"])
@pytest.mark.parametrize("mode", ["full", "stream"])
def test_a_per_bin_dose_fits_the_counts_better(data, spectra, mode):
    """With the open beam of each bin as the dose, the fit has a lower Poisson negative log-likelihood than the fit
    with the mean open beam as a scalar dose."""
    T, _ = data
    kw = dict(spectra=spectra, verbose=0, **(STREAM if mode == "stream" else {}))
    plain = hsnt.dehydrate(T, "transmission", R, dose=float(OPEN_BEAM.mean()), **kw)
    weighted = hsnt.dehydrate(T, "transmission", R, dose=OPEN_BEAM, **kw)
    assert weighted[0].min() >= 0 and weighted[1].min() >= 0
    assert _nll(T, weighted) < _nll(T, plain)


def test_the_rank_estimate_uses_the_per_bin_dose(data):
    """The rank estimate weights the bins as the fit does.  Its effective dose is then near the mean open beam, and it
    finds the two materials."""
    T, _ = data
    rank, _, detail = _estimate_rank(T.reshape(-1, K), (1, N, N), "cpu", 4, weights=OPEN_BEAM / OPEN_BEAM.mean())
    assert rank == R and detail["effective_dose"] == pytest.approx(OPEN_BEAM.mean(), rel=0.2)
    assert hsnt.dehydrate(T, "transmission", dose=OPEN_BEAM, max_rank=4, verbose=0)[1].shape == (R, K)


def test_fixed_spectra_with_a_per_bin_dose(data):
    """With the spectra held fixed, a per-bin dose gives maps of a lower negative log-likelihood, and the streamed
    maps match the full ones."""
    T, H = data
    kw = dict(init=(None, H), fix_spectra=True, verbose=0)
    plain = hsnt.dehydrate(T, "transmission", **kw)
    weighted = hsnt.dehydrate(T, "transmission", dose=OPEN_BEAM, **kw)
    streamed = hsnt.dehydrate(T, "transmission", dose=OPEN_BEAM, mode="stream", chunk_pixels=200, **kw)
    assert np.array_equal(weighted[1], H.astype(np.float32)) and _nll(T, weighted) < _nll(T, plain)
    assert np.abs(streamed[0] - weighted[0]).max() < 1e-4 * weighted[0].max()


def test_the_mask_check_with_a_per_bin_dose(data, capsys):
    """The mask check fits the pixels outside the mask with the per-bin dose.  Outside pixels that hold no sample are
    not flagged."""
    T, _ = data
    mask = np.ones((N, N), dtype=bool)
    mask[:4] = False                                    # These rows hold only background pixels.
    hsnt.dehydrate(T, "transmission", R, dose=OPEN_BEAM, sample_mask=mask, check_mask=True, verbose=1)
    line = [s for s in capsys.readouterr().out.splitlines() if "Outside the sample_mask" in s][0]
    assert f"{4 * N} pixels.  0% hold" in line


def test_a_bad_array_dose_is_refused(data):
    T, _ = data
    for bad, match in ((np.full(K - 1, 50.0), "one open beam count per bin"), (np.full((1, K), 50.0), "shape"),
                       (np.r_[OPEN_BEAM[:-1], 0.0], "positive"), (np.r_[OPEN_BEAM[:-1], np.nan], "finite")):
        with pytest.raises(ValueError, match=match):
            hsnt.dehydrate(T, "transmission", R, dose=bad, verbose=0)
