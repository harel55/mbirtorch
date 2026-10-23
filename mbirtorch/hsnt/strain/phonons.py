"""The inelastic (phonon) neutron cross-section of a crystalline material, for the smooth part of the hybrid spectrum.

The model is a Debye solid in the incoherent Gaussian approximation, summed over phonon orders. Its parameters are the
Debye temperature (fitted), the mean atomic mass and the temperature (given). Against NCrystal's non-Bragg cross-sections
of 37 elements from 1 to 6 A, absorption + constant + this term leaves a median residual of 0.1% (studies/non_bragg_physical).
"""
import math

import numpy as np
import torch
from scipy.signal import fftconvolve
from scipy.special import gammainc

KB_MEV = 8.617333e-2                 # Boltzmann constant, meV / K
E_OF_WL = 81.80421                   # neutron energy (meV) = E_OF_WL / wavelength (A)^2
NEUTRON_MASS_AMU = 1.00866           # neutron mass in atomic mass units
N_PHONON = 12                        # phonon orders in the expansion
BETA_POINTS = 200                    # grid points of the energy transfer per Debye cutoff
THETA_MIN, THETA_MAX = 50.0, 2500.0  # Debye temperatures the table covers (K)
N_THETA = 96                         # table points, evenly spaced in log(theta)
WL_RANGE = (0.3, 12.0)               # wavelengths (A) of the cached table, log-spaced; models interpolate to their grid
N_WL = 200                           # its number of wavelengths


def debye_inelastic(wl, theta, temperature=293.15, mass_amu=55.845):
    """The inelastic cross-section per unit bound scattering cross-section, at wavelengths wl (A), for a Debye solid with
    Debye temperature theta (K) at the given temperature (K) and mean atomic mass (amu)."""
    kT = KB_MEV * temperature; bd = theta / temperature; A = mass_amu / NEUTRON_MASS_AMU
    DBETA = bd / BETA_POINTS
    nb = (N_PHONON + 2) * BETA_POINTS
    beta = np.arange(-nb, nb + 1) * DBETA                       # symmetric and of odd length, so the convolutions stay centred
    ab = np.abs(beta)
    rho = np.where(ab < bd, 3 * ab ** 2 / bd ** 3, 0.0)                              # Debye density of states
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.where(ab > 0, rho / (2 * ab * np.sinh(ab / 2)), 3 / bd ** 3)
    T1 = P * np.exp(-beta / 2)
    lam = T1.sum() * DBETA                                                           # Debye-Waller: 2W = alpha * lam
    E = E_OF_WL / np.asarray(wl, dtype=np.float64) ** 2
    ep = E[:, None] + beta[None, :] * kT
    ok = np.clip(ep / (kT * DBETA) + 0.5, 0.0, 1.0)              # the share of each grid cell above E' = 0
    sq = np.sqrt(np.clip(ep, 0, None))
    am = (np.sqrt(E)[:, None] - sq) ** 2 / (A * kT); ap = (np.sqrt(E)[:, None] + sq) ** 2 / (A * kT)
    out = np.zeros_like(E)
    Tn = T1.copy()
    for n in range(1, N_PHONON + 1):
        w = (gammainc(n + 1, lam * ap) - gammainc(n + 1, lam * am)) * ok
        out += (w * Tn[None, :]).sum(1) * DBETA / lam ** (n + 1)
        Tn = fftconvolve(Tn, T1, mode='same') * DBETA
    return A * kT / (4 * E) * out


_TABLES = {}                         # (mass, temperature) -> the table on the WL_RANGE grid


def debye_table(mass_amu, temperature):
    """The term on the log-spaced WL_RANGE grid for each Debye temperature of the table: (wavelengths, log thetas, values
    (N_THETA, N_WL)). Computed once per mass and temperature in a process."""
    key = (round(float(mass_amu), 4), round(float(temperature), 3))
    if key not in _TABLES:
        wl = np.geomspace(*WL_RANGE, N_WL)
        lt = np.linspace(math.log(THETA_MIN), math.log(THETA_MAX), N_THETA)
        _TABLES[key] = (wl, lt, np.stack([debye_inelastic(wl, math.exp(t), temperature, mass_amu) for t in lt]))
    return _TABLES[key]


class DebyeTable:
    """The Debye term of one material on a fine wavelength grid, tabulated in log(theta), with a smooth (cubic
    Hermite) interpolation that torch can differentiate. Use table(log_theta) in a spectrum model."""

    def __init__(self, lamf, mass_amu, temperature=293.15, device='cpu'):
        lamf = np.asarray(lamf, dtype=np.float64)
        wl, self.log_theta, tab = debye_table(mass_amu, temperature)
        assert WL_RANGE[0] <= lamf[0] and lamf[-1] <= WL_RANGE[1], 'the fine grid lies outside the tabulated wavelengths'
        D = np.stack([np.interp(np.log(lamf), np.log(wl), row) for row in tab])
        dD = np.gradient(D, self.log_theta, axis=0)
        self.step = float(self.log_theta[1] - self.log_theta[0])
        self.D = torch.as_tensor(D, device=device); self.dD = torch.as_tensor(dD, device=device)

    def __call__(self, log_theta):
        """The term (Kf,) at log_theta (a 0-d tensor, clamped to the table)."""
        u = (log_theta.clamp(self.log_theta[0], self.log_theta[-1]) - self.log_theta[0]) / self.step
        i = torch.clamp(torch.floor(u).long(), 0, len(self.log_theta) - 2)
        t = u - i
        h00 = 2 * t ** 3 - 3 * t ** 2 + 1; h10 = t ** 3 - 2 * t ** 2 + t; h01 = -2 * t ** 3 + 3 * t ** 2; h11 = t ** 3 - t ** 2
        return (h00 * self.D[i] + h10 * self.step * self.dD[i] + h01 * self.D[i + 1] + h11 * self.step * self.dD[i + 1])
