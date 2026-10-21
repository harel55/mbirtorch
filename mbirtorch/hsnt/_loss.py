import torch

# The largest background share of a bin.  The neutrons keep the rest of the open beam.
BACKGROUND_MAX = 0.99


def _nnal_prep(T, background=None):
    """This function computes the quantities in the loss that depend only on the transmission T
    and on the background.

    The argument ``background`` is the additive background b of each bin, as a fraction of the
    open beam, with shape (bins,) and 0 <= b < 1.  With a background, the model of the
    transmission is (1 - b) exp(-X) + b instead of exp(-X).  The background passes the sample
    without attenuation.

    Returns:
        tuple: (log_T, positive, all_positive), and also (log(1 - b), log b) when a background is
        given.  ``log_T`` is log(T), with 0 where T is not positive.  ``positive`` is the boolean
        mask T > 0.  ``all_positive`` is True if every entry of T is positive.
    """
    positive = T > 0
    Tsafe = torch.where(positive, T, torch.ones((), dtype=T.dtype, device=T.device))
    log_T = torch.log(Tsafe)
    all_positive = bool(positive.all())
    if background is None:
        return log_T, positive, all_positive
    return log_T, positive, all_positive, _background_logs(torch.as_tensor(background, dtype=T.dtype, device=T.device))


def _background_logs(b):
    """This function returns the pair (log(1 - b), log b) that ``_nnal_prep`` stores for the
    background b."""
    return torch.log1p(-b), torch.log(b)


def _background_terms(X, T, b):
    """This function returns the quantities that a joint Newton step on the background b needs:
    the gradient of the loss with respect to b, with shape (bins,), the Fisher information
    between X and b in each entry, with the shape of X, and the Fisher information of b, with
    shape (bins,)."""
    e = torch.exp(-X)
    m = (1 - b) * e + b
    ratio = torch.where(T > 0, T / m, torch.zeros_like(m))
    grad = ((1 - ratio) * (1 - e)).sum(0)
    cross = -(1 - b) * e * (1 - e) / m
    info = ((1 - e) ** 2 / m).sum(0)
    return grad, cross, info


def _effective_attenuation(X, background):
    """This function returns the attenuation Xe = -log((1 - b) exp(-X) + b) of the model with a
    background, and its derivative s = dXe/dX.  The argument ``background`` is the pair
    (log(1 - b), log b) from ``_nnal_prep``."""
    log_1mb, log_b = background
    a = log_1mb - X
    log_m = torch.logaddexp(a, log_b)
    return -log_m, torch.exp(a - log_m)


def _nnal_elementwise(X, T, prep):
    """This function computes each term of the shifted non-negative attenuation loss (NNAL).

    Where T > 0, the term is T * phi(X + log T), with phi(u) = exp(-u) - 1 + u.  Where T = 0, the
    term is exp(-X).  With a background in ``prep``, X is replaced by the effective attenuation.
    """
    log_T, positive, all_positive = prep[:3]
    if len(prep) > 3:
        X = _effective_attenuation(X, prep[3])[0]
    Xp = X + log_T
    loss = T * (torch.expm1(-Xp) + Xp)
    if not all_positive:
        # Where T == 0, the term is exp(-Xp).  The form expm1(-Xp) + 1 underflows for large Xp.
        loss = torch.where(positive, loss, torch.exp(-Xp))
    return loss


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

    With a background in ``prep``, G is the derivative of the loss with respect to X, and Z is
    its expected (Fisher) second derivative, which is positive.  The exact second derivative can
    be negative.
    """
    prep = _nnal_prep(T) if prep is None else prep
    log_T, positive, all_positive = prep[:3]
    if len(prep) > 3:
        X, s = _effective_attenuation(X, prep[3])

    Xp = X + log_T

    # expm1 is accurate near Xp = 0.  exp is used where T = 0, because expm1 + 1 underflows at large Xp.
    E = torch.expm1(-Xp)
    eXp = torch.exp(-Xp)

    G = -T * E
    Z = T * eXp

    if not all_positive:
        G = torch.where(positive, G, -eXp)
        Z = torch.where(positive, Z, eXp)

    if len(prep) > 3:
        G, Z = G * s, Z * s * s
    return G, Z


def _nnal_rowwise(X, T, prep, dim, dtype=None):
    """This function computes the non-negative attenuation loss summed over the axis ``dim``.

    Use dim=1 for the loss of each pixel and dim=0 for the loss of each wavelength.  The argument
    ``dtype`` sets the dtype used to accumulate the sum.
    """
    return _nnal_elementwise(X, T, prep).sum(dim=dim, dtype=dtype)
