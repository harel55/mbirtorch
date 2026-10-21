"""Tests that ``dehydrate`` gives the same result for a numpy array and a torch tensor, and that a tensor is not
copied to the host."""
import warnings

import numpy as np
import pytest
import torch

import mbirtorch.hsnt as hsnt
import mbirtorch.hsnt.denoise as denoise
from test_hsnt_solvers import _sphere_problem

N = 32
RANK = 3
STREAM_CHUNK = 300                                      # pixels per chunk in the streamed solve
ROUNDING_RTOL = 1e-4                                    # a GPU solve or a different exp rounds differently


@pytest.fixture(scope="module")
def spheres():
    """The three-sphere transmission image, with shape (N, N, bins)."""
    return _sphere_problem("cpu", n=N, dose=100.0).numpy().reshape(N, N, -1)


def _dehydrate(data, *args, **kwargs):
    """Runs dehydrate on the CPU and returns W, H, and the package's warnings.  It leaves out torch's warnings, which
    torch gives only once per process."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        W, H, _ = hsnt.dehydrate(data, *args, verbose=0, device="cpu", **kwargs)
    return W, H, sorted(str(w.message) for w in caught if "torch" not in str(w.message).lower())


def _assert_same(a, b):
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("kwargs", [dict(), dict(mode="stream", chunk_pixels=STREAM_CHUNK), dict(spectra="unconstrained")],
                         ids=["full", "stream", "unconstrained"])
def test_a_tensor_and_an_array_give_the_same_fit(spheres, kwargs):
    _assert_same(_dehydrate(spheres, "transmission", RANK, **kwargs),
                 _dehydrate(torch.from_numpy(spheres), "transmission", RANK, **kwargs))


def test_a_tensor_and_an_array_give_the_same_fit_with_a_mask(spheres):
    yy, xx = np.mgrid[:N, :N]
    mask = (yy - N / 2) ** 2 + (xx - N / 2) ** 2 < (0.45 * N) ** 2
    _assert_same(_dehydrate(spheres, "transmission", RANK, sample_mask=mask, check_mask=True),
                 _dehydrate(torch.from_numpy(spheres), "transmission", RANK, sample_mask=mask, check_mask=True))


def test_a_tensor_and_an_array_give_the_same_fit_for_a_given_basis(spheres):
    H = _dehydrate(spheres, "transmission", RANK)[1]
    _assert_same(_dehydrate(spheres, "transmission", subspace_basis=H),
                 _dehydrate(torch.from_numpy(spheres), "transmission", subspace_basis=H))


def test_a_tensor_and_an_array_give_the_same_rank(spheres):
    _assert_same(_dehydrate(spheres, "transmission"), _dehydrate(torch.from_numpy(spheres), "transmission"))


def test_a_tensor_is_cleaned_like_an_array_and_left_unchanged(spheres):
    dirty = spheres.copy()
    dirty[0, 0, :3] = (np.nan, np.inf, -0.5)
    tensor = torch.from_numpy(dirty.copy())
    _assert_same(_dehydrate(dirty, "transmission", RANK), _dehydrate(tensor, "transmission", RANK))
    np.testing.assert_array_equal(tensor.numpy(), dirty)


def test_attenuation_from_a_tensor_matches_an_array(spheres):
    """numpy and torch may round exp differently in the last bit, so the fits agree to rounding."""
    with np.errstate(divide="ignore"):                  # zero counts give an infinite attenuation
        A = -np.log(spheres)
    W0, H0, _ = _dehydrate(A, "attenuation", RANK)
    W1, H1, _ = _dehydrate(torch.from_numpy(A), "attenuation", RANK)
    np.testing.assert_allclose(W1 @ H1, W0 @ H0, rtol=ROUNDING_RTOL, atol=ROUNDING_RTOL)


def test_a_tensor_is_not_copied_to_the_host(spheres, monkeypatch):
    def no_tensor(array):
        assert not torch.is_tensor(array), "dehydrate copied the tensor data to the host"
        return np.asarray(array)
    monkeypatch.setattr(denoise, "_to_host", no_tensor)
    _dehydrate(torch.from_numpy(spheres), "transmission", RANK)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_gpu_tensor_matches_an_array_on_the_gpu(spheres):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        W0, H0, _ = hsnt.dehydrate(spheres, "transmission", RANK, verbose=0, device="cuda")
        W1, H1, _ = hsnt.dehydrate(torch.from_numpy(spheres).cuda(), "transmission", RANK, verbose=0, device="cuda")
    np.testing.assert_allclose(W1 @ H1, W0 @ H0, rtol=ROUNDING_RTOL, atol=ROUNDING_RTOL)
