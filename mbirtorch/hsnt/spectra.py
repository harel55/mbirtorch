import math

import torch

from ._loss import _nnal_prep
from ._newton import _joint_newton_pcg, _kernels, block_newton_optimize


def unconstrained_spectra(T, W, H, max_steps=300, cg_max=10, rel_tol=1e-10, w_max_steps=100, compile_mode=None):
    """Re-estimate the spectra with the bound on the pixel coefficients dropped, then re-solve W >= 0.

    The maximum-likelihood spectra are biased by the truncation of pixel
    coefficients at zero: a coefficient whose true value is zero is estimated
    positive half the time and clipped the other half, an O(1/sqrt m) effect per
    pixel (m = information per pixel) that does not average out over pixels.
    Dropping the bound removes it -- the pixel problem stays strictly convex for
    any real w -- at the price of the variance reduction the constraint provides.
    Measured on the phantom at dose 3, rank 3 (spectral SNR, MLE -> this):
    4k px -0.8 dB, 16k -0.7, 65k +0.3, 262k 40.4 -> 43.2, 524k 41.1 -> 45.8,
    1M 41.7 -> 49.0, restoring the sqrt(N) rate the MLE had lost; maps +0.15 dB.
    Cost: a continuation of the joint solve from the ML point, about 0.6x the
    ML solve's time. Use it when pixels are plentiful (above ~10^5 at dose 3;
    the crossover moves to larger P at lower dose). Not for coefficients that are
    physically nonnegative and mostly zero, such as fractions over a dictionary
    of many similar atoms: there the unconstrained fit is ill-conditioned and
    the bound carries real information; see support_selected_spectra.

    Returns (W, H, steps) with W >= 0 re-solved for the returned H.
    """
    nnal_fn, deriv, _, _ = _kernels(compile_mode)
    prep = _nnal_prep(T)
    _, Hu, steps, _ = _joint_newton_pcg(T, W, H, max_steps=max_steps, cg_max=cg_max, rel_tol=rel_tol,
                                        prep=prep, nnal=nnal_fn, deriv=deriv, nonneg_W=False)
    Wc, _, _ = block_newton_optimize(T, H.shape[0], w_max_steps, 1e-12, update_H=False, W_init=W, H_init=Hu,
                                     compile_mode=compile_mode)
    return Wc, Hu, steps


def support_selected_spectra(T, W, H, dose, penalty=None, max_steps=300, cg_max=10, rel_tol=1e-10,
                             w_max_steps=100, compile_mode=None, verbose=False):
    """Choose each pixel's material subset by penalised likelihood, then refit with the supports fixed.

    The truncation bias of the ML spectra (see unconstrained_spectra) comes from
    coefficients whose true value is zero. If those are identified and held at
    zero, the remaining coefficients sit in the interior and only the much
    smaller curvature bias is left, while W >= 0 -- and the variance reduction it
    brings -- is kept. Every nonempty subset of the R materials is solved for
    every pixel (2^R - 1 grouped convex solves, so R <= 6), each pixel takes the
    subset minimising  dose * loss + penalty * |subset|  (the empty subset is
    allowed: a pixel with no material), and (W on the selected supports, H) is
    then refit jointly. Measured at 65k px, dose 3, rank 3 against the MLE:
    penalty 1 (AIC) +0.79 dB spectra / +0.08 maps; 0.5 log K (BIC) +0.95 / +0.23;
    2 log K +1.04 / +0.44, with the exact support recovered in 59% of pixels --
    the best of the estimators tried at that size, and the only one that also
    lifts the maps appreciably. The selection is a model-selection step and
    carries its own errors; a stronger penalty helped monotonically here.

    Args:
        dose: open-beam counts per pixel and bin, which converts the loss to
            log-likelihood units for the penalty.
        penalty: per selected coefficient, in log-likelihood units. Default 2 log K.

    Returns (W, H, support, steps) with support a bool mask of W's shape.
    """
    R, K = H.shape
    if R > 6:
        raise ValueError("support_selected_spectra enumerates all 2^R - 1 subsets; for R > 6 use "
                         "unconstrained_spectra (signed coefficients do not truncate).")
    import itertools
    penalty = 2.0 * math.log(K) if penalty is None else penalty
    nnal_fn, deriv, rowwise, _ = _kernels(compile_mode)
    prep = _nnal_prep(T)
    subsets = [list(c) for r in range(1, R + 1) for c in itertools.combinations(range(R), r)]
    crit = [rowwise(torch.zeros_like(T), T, prep, 1, dtype=torch.float64) * dose]      # the empty subset
    W_sub = []
    for S in subsets:
        idx = torch.tensor(S, device=T.device)
        Ws, _, _ = block_newton_optimize(T, len(S), w_max_steps, 1e-12, update_H=False,
                                         W_init=W[:, idx].contiguous(), H_init=H[idx].contiguous(),
                                         compile_mode=compile_mode)
        crit.append(rowwise(Ws @ H[idx], T, prep, 1, dtype=torch.float64) * dose + penalty * len(S))
        W_sub.append(Ws)
    best = torch.stack(crit, 1).argmin(1)                    # 0 = empty, j + 1 = subsets[j]
    W0 = torch.zeros_like(W)
    for j, S in enumerate(subsets):
        m = best == j + 1
        if m.any():
            W0[m.nonzero().squeeze(1)[:, None], torch.tensor(S, device=T.device)[None, :]] = W_sub[j][m]
    support = W0 > 0
    Wn, Hn, steps, _ = _joint_newton_pcg(T, W0, H, max_steps=max_steps, cg_max=cg_max, rel_tol=rel_tol,
                                         prep=prep, nnal=nnal_fn, deriv=deriv, w_mask=support)
    if verbose:
        print(f'  supports: mean size {support.sum(1).double().mean().item():.2f}; joint refit {steps} steps', flush=True)
    return Wn, Hn, support, steps
