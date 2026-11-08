import numpy as np
import torch


def _weights_tensor(weights, device, dtype=torch.float32):
    """This function converts the weight of each bin to a (1, bins) tensor on the device.  It
    returns None when ``weights`` is None."""
    return None if weights is None else torch.as_tensor(np.asarray(weights), dtype=dtype, device=device).reshape(1, -1)


def _nnal_prep(T, weights=None):
    """This function computes the quantities in the loss that depend only on the transmission T.

    Args:
        T: (torch.Tensor) The transmission ratio.
        weights: (torch.Tensor, optional) Nonnegative weights that broadcast against T, such as the
            open beam count of each bin as a (1, bins) tensor.  Each entry's loss and derivatives are
            multiplied by its weight.  With the open beam counts as weights, the loss is the Poisson
            negative log-likelihood of the counts when the open beam differs between bins.  The
            default is None, which gives every entry the weight 1.

    Returns:
        tuple: (log_T, positive, all_positive, weights).  ``log_T`` is log(T), with 0 where T is not
        positive.  ``positive`` is the boolean mask T > 0.  ``all_positive`` is True if every entry
        of T is positive.  ``weights`` is the argument.
    """
    positive = T > 0
    Tsafe = torch.where(positive, T, torch.ones((), dtype=T.dtype, device=T.device))
    log_T = torch.log(Tsafe)
    all_positive = bool(positive.all())
    return log_T, positive, all_positive, weights


def _nnal_elementwise(X, T, prep):
    """This function computes each term of the shifted non-negative attenuation loss (NNAL).

    Where T > 0, the term is T * phi(X + log T), with phi(u) = exp(-u) - 1 + u.  Where T = 0, the
    term is exp(-X).
    """
    log_T, positive, all_positive, weights = prep
    Xp = X + log_T
    loss = T * (torch.expm1(-Xp) + Xp)
    if not all_positive:
        # Where T == 0, the term is exp(-Xp).  The form expm1(-Xp) + 1 underflows for large Xp.
        loss = torch.where(positive, loss, torch.exp(-Xp))
    return loss if weights is None else loss * weights


def stable_nnal(X, T, prep=None, dtype=None):
    """This function computes the non-negative attenuation loss sum[exp(-X) + T X], shifted by a
    term that depends only on T.

    Each term of the shifted loss is nonnegative, and it is zero where X = -log T.  This form
    keeps the sum accurate in float32.  The shift does not change the minimizer.

    Args:
        X: (torch.Tensor) The attenuation estimate.  X must broadcast against T.
        T: (torch.Tensor) The measured transmission ratio, counts divided by open beam.
        prep: (tuple, optional) The output of ``_nnal_prep(T)``.  Passing it avoids recomputing
            log(T).
        dtype: (torch.dtype, optional) The dtype used to accumulate the sum.  The default is the
            dtype of X.  Use float64 when the value is used in a convergence test.

    Returns:
        torch.Tensor: The loss summed over the last two axes.
    """
    loss = _nnal_elementwise(X, T, _nnal_prep(T) if prep is None else prep)
    return torch.sum(loss, dim=(-2, -1), dtype=dtype)


def stable_nnal_derivatives(X: torch.Tensor, T: torch.Tensor, prep=None):
    """This function computes the first and second derivatives of the non-negative attenuation
    loss L with respect to X = W @ H.

    The first derivative is G = dL/dX = T - exp(-X).  The second derivative is
    Z = d^2L/dX^2 = exp(-X).  The function returns correct values where T = 0.
    """
    log_T, positive, all_positive, weights = _nnal_prep(T) if prep is None else prep

    Xp = X + log_T

    # expm1 is accurate near Xp = 0.  exp is used where T = 0, because expm1 + 1 underflows at large Xp.
    E = torch.expm1(-Xp)
    eXp = torch.exp(-Xp)

    G = -T * E
    Z = T * eXp

    if not all_positive:
        G = torch.where(positive, G, -eXp)
        Z = torch.where(positive, Z, eXp)

    if weights is not None:
        G = G * weights
        Z = Z * weights
    return G, Z


def _nnal_rowwise(X, T, prep, dim, dtype=None):
    """This function computes the non-negative attenuation loss summed over the axis ``dim``.

    Use dim=1 for the loss of each pixel and dim=0 for the loss of each wavelength.  The argument
    ``dtype`` sets the dtype used to accumulate the sum.
    """
    return _nnal_elementwise(X, T, prep).sum(dim=dim, dtype=dtype)
