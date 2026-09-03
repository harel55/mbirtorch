"""Hyperspectral neutron tomography: NNAL factorization, spectral denoising and the hsnt HDF5 format.

Submodules, one line each:

    _loss               the non-negative attenuation loss (stable_nnal), its derivatives and its per-row sums
    _linalg             randomized SVD, the NNDSVDa initialization and the small batched SPD / Cholesky solves
    _multiplicative     the shifted multiplicative, quadratic and scalar-step Newton updates
    _newton             projected block-Newton and joint Newton-PCG solvers, the compiled kernels and their tuning constants
    _streaming          stream_factorization: the out-of-core factorization over chunks
    factorization       the `optimize` driver and the `nnal_factorization` front end
    spectra             unconstrained_spectra and support_selected_spectra, the re-estimates that remove the truncation bias
    _bias_experimental  bias_corrected_spectra: the modified profile likelihood and the bootstrap (experimental; kept for reference)
    denoise             hyper_denoise / dehydrate / rehydrate and the subspace-dimension estimate (scikit-learn)
    io                  the hsnt HDF5 format: import_hsnt_data_hdf5, create_hsnt_metadata, export_hsnt_data_hdf5 (h5py)
    simulate            generate_hyper_data, the Ni/Cu/Al phantom
    plots               compare_spectra

Every name the former single-file module ``mbirtorch/hsnt.py`` defined is
re-exported here, underscore-prefixed ones included, so
``from mbirtorch.hsnt import X`` keeps working for every X it used to accept;
``__all__`` lists the public ones.

The tuning constants ``_ARMIJO_FLOOR``, ``_TRUST_FLOOR`` and ``_ACTIVE_TOL``
are owned by ``_newton`` and every solver reads them through that module at
call time. The copies bound here are plain floats and are for reading only:
to monkeypatch one, target the owner, ``mbirtorch.hsnt._newton._ARMIJO_FLOOR``,
not ``mbirtorch.hsnt._ARMIJO_FLOOR``. (``_COMPILED_KERNELS`` is a dict, so the
name here is the same object as the owner's.)

matplotlib is imported lazily, inside the three functions that plot
(compare_spectra, and the verbose plotting blocks of generate_hyper_data and
_estimate_subspace_dimension), so importing this package does not import it.
"""
from ._loss import _nnal_prep, stable_nnal, stable_nnal_derivatives, _nnal_rowwise
from ._linalg import _randomized_svd, nndsvda, _batched_spd_solve, _joint_blocks, _joint_dot
from ._multiplicative import _shifted, _rebalance, _reseed_dead, quadratic_update, newton_update, multiplicative_update
from ._newton import (_COMPILED_KERNELS, _ARMIJO_FLOOR, _TRUST_FLOOR, _ACTIVE_TOL, _kernels, block_newton_step,
                      block_newton_optimize, _joint_newton_pcg, joint_newton_optimize)
from ._streaming import _h_stats_accumulate, _h_direction, stream_factorization
from .factorization import optimize, nnal_factorization
from .spectra import unconstrained_spectra, support_selected_spectra
from ._bias_experimental import (_CORRECTIONS, _kr_matrices, _adjustment_value_and_grad, _bootstrap_score_bias,
                                 bias_corrected_spectra)
from .denoise import hyper_denoise, dehydrate, rehydrate, _estimate_subspace_dimension
from .io import (KEY_DESCRIPTIONS, VALIDATION_RULES, ALLOWED_KEYS, _validate_key, _with_key_docstring,
                 import_hsnt_data_hdf5, create_hsnt_metadata, export_hsnt_data_hdf5)
from .simulate import generate_hyper_data
from .plots import compare_spectra

__all__ = [
    "hyper_denoise", "dehydrate", "rehydrate",
    "import_hsnt_data_hdf5", "create_hsnt_metadata", "export_hsnt_data_hdf5",
    "generate_hyper_data",
    "nnal_factorization", "stable_nnal", "stable_nnal_derivatives",
    "compare_spectra",
    "stream_factorization",
    "unconstrained_spectra", "support_selected_spectra", "bias_corrected_spectra",
    "nndsvda", "optimize", "block_newton_optimize", "joint_newton_optimize", "block_newton_step",
]
