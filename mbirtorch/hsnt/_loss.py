import torch


def _nnal_prep(T, weights=None):
    """
    Precompute the quantities that depend only on T.

    These are constant for a whole solve, so hoisting them out of the iteration
    removes a full log over T from every loss and derivative evaluation.

    weights (optional, broadcastable to T, >= 0) multiplies every entry's loss and derivatives: the likelihood of
    counts recorded with a per-entry efficiency, e.g. 1 - P for MCP/Timepix data corrected for overlap (Tremsin et
    al., JINST 9 C05026, 2014), whose recorded counts are Poisson with mean (1 - P) times the corrected mean.

    Returns:
        (log_T, positive, all_positive, weights)
    """
    positive = T > 0
    Tsafe = torch.where(positive, T, torch.ones((), dtype=T.dtype, device=T.device))
    log_T = torch.log(Tsafe)
    all_positive = bool(positive.all())
    return log_T, positive, all_positive, weights


def _nnal_elementwise(X, T, prep):
    """The shifted NNAL term by term: T * phi(X + log T) with phi(u) = exp(-u) - 1 + u, and exp(-X) where
    T == 0. stable_nnal and _nnal_rowwise are reductions of this one tensor."""
    log_T, positive, all_positive = prep[:3]
    weights = prep[3] if len(prep) > 3 else None
    Xp = X + log_T
    loss = T * (torch.expm1(-Xp) + Xp)
    if not all_positive:
        # T == 0 means Xp == X, so the zero-count term is exp(-Xp). It must be a
        # real exp: expm1(-Xp) saturates at exactly -1 for Xp above ~37 in
        # float64, so reconstructing it as expm1(-Xp) + 1 underflows to zero.
        loss = torch.where(positive, loss, torch.exp(-Xp))
    return loss if weights is None else loss * weights


def stable_nnal(X, T, prep=None, dtype=None):
    """The non-negative attenuation loss sum[exp(-X) + T X], shifted by a term that depends only on T.

    Each term is written T phi(X + log T) with phi(u) = exp(-u) - 1 + u, so it is nonnegative and zero where
    X = -log T, which keeps the sum accurate in float32; the shift does not change the minimizer.

    Args:
        X: Attenuation estimate, broadcastable against T.
        T: Measured transmission ratio (counts / open beam).
        prep: Optional tuple from _nnal_prep(T). Pass it inside an iteration to
            avoid recomputing log(T) on every call.
        dtype: Accumulation dtype for the final sum. Defaults to X's dtype. Pass
            torch.float64 when the value drives a convergence test: in float32 a
            loss near 3e6 has a resolution of 0.25, so two consecutive losses that
            differ by less than that compare equal and a relative-change test
            fires spuriously.

    Returns:
        torch.Tensor: The loss summed over the last two axes.
    """
    loss = _nnal_elementwise(X, T, _nnal_prep(T) if prep is None else prep)
    return torch.sum(loss, dim=(-2, -1), dtype=dtype)


def stable_nnal_derivatives(X: torch.Tensor, T: torch.Tensor, prep=None):
    """
    Given X = W @ H, compute

        G = dL/dX = T - exp(-X)
        Z = d^2L/dX^2 = exp(-X)

    where L is the non-negative attenuation loss in a
    numerically stable way that handles T = 0 appropriately.
    """
    prep = _nnal_prep(T) if prep is None else prep
    log_T, positive, all_positive = prep[:3]
    weights = prep[3] if len(prep) > 3 else None

    Xp = X + log_T

    # Two transcendentals, not four: torch.where evaluates both of its branches,
    # so the original form paid for four exponentials per call. expm1 is the
    # accurate one near Xp = 0, where T - exp(-X) cancels; exp is the accurate
    # one at large Xp, where expm1 saturates at -1 and expm1 + 1 underflows.
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
    """NNAL summed over `dim` only: per-pixel (dim=1) or per-wavelength (dim=0).

    dtype is the accumulation dtype of the sum: pass float64 whenever the value
    drives a line search or a stopping test, because the float32 ulp of a sum over
    many pixels hides the improvement of a single H step (the float32 truncation
    of the elementwise terms remains, and is what _ARMIJO_FLOOR accounts for).
    """
    return _nnal_elementwise(X, T, prep).sum(dim=dim, dtype=dtype)
