import torch


def _nnal_prep(T):
    """
    Precompute the quantities that depend only on T.

    These are constant for a whole solve, so hoisting them out of the iteration
    removes a full log over T from every loss and derivative evaluation.

    Returns:
        (log_T, positive, all_positive, taylor_cutoff)
    """
    positive = T > 0
    Tsafe = torch.where(positive, T, torch.ones((), dtype=T.dtype, device=T.device))
    log_T = torch.log(Tsafe)
    all_positive = bool(positive.all())
    # Crossover for the phi series: truncation ~ |Xp|^3/24 against cancellation
    # ~ eps/|Xp|^2 (a fixed 1e-3 would be ~40x too small in float32).
    taylor_cutoff = (24.0 * torch.finfo(T.dtype).eps) ** 0.25
    return log_T, positive, all_positive, taylor_cutoff


def _nnal_elementwise(X, T, prep):
    """The shifted NNAL term by term: T * phi(X + log T) with phi(u) = exp(-u) - 1 + u,
    a Taylor branch below the dtype-aware cutoff, and exp(-X) where T == 0.
    stable_nnal and _nnal_rowwise are reductions of this one tensor."""
    log_T, positive, all_positive, taylor_cutoff = prep
    Xp = X + log_T
    phi = torch.where(
        torch.abs(Xp) < taylor_cutoff,
        Xp * Xp * (0.5 + Xp * (-1.0 / 6.0 + Xp / 24.0)),
        torch.expm1(-Xp) + Xp,
    )
    loss = T * phi
    if not all_positive:
        # T == 0 means Xp == X, so the zero-count term is exp(-Xp). It must be a
        # real exp: expm1(-Xp) saturates at exactly -1 for Xp above ~37 in
        # float64, so reconstructing it as expm1(-Xp) + 1 underflows to zero.
        loss = torch.where(positive, loss, torch.exp(-Xp))
    return loss


def stable_nnal(X, T, prep=None, dtype=None):
    """
    Compute a shifted form of the non-negative attenuation loss
    that is much more numerically stable

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
    log_T, positive, all_positive, _ = _nnal_prep(T) if prep is None else prep

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

    return G, Z


def _nnal_rowwise(X, T, prep, dim, dtype=None):
    """NNAL summed over `dim` only: per-pixel (dim=1) or per-wavelength (dim=0).

    dtype is the accumulation dtype of the sum: pass float64 whenever the value
    drives a line search or a stopping test, because the float32 ulp of a sum over
    many pixels hides the improvement of a single H step (the float32 truncation
    of the elementwise terms remains, and is what _ARMIJO_FLOOR accounts for).
    See docs/hsnt_solver_notes.md, section 1.
    """
    return _nnal_elementwise(X, T, prep).sum(dim=dim, dtype=dtype)
