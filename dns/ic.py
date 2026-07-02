"""Initial conditions: random solenoidal field with a prescribed spectrum."""

import numpy as np

from .spectral import SpectralGrid


def model_spectrum(k, E0, kp):
    """E(k) ~ k^4 exp(-2 (k/kp)^2), normalised so sum E(k) = E0."""
    Ek = k**4 * np.exp(-2.0 * (k / kp) ** 2)
    s = Ek.sum()
    return Ek * (E0 / s) if s > 0 else Ek


def isotropic_field(grid: SpectralGrid, E0=1.5, kp=3.0, seed=12345):
    """Random solenoidal velocity coefficients with shell energies matching
    the model spectrum. Built by transforming white noise (which guarantees
    the Hermitian symmetry of a real field), projecting, then rescaling
    each spherical shell."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((3, grid.N, grid.N, grid.N))
    c = grid.fwd3(noise)
    c *= grid.dealias
    grid.project(c)
    c[:, 0, 0, 0] = 0.0

    Ek_cur = grid.spectrum(c)
    k = np.arange(len(Ek_cur), dtype=np.float64)
    Ek_tar = model_spectrum(k, E0, kp)
    Ek_tar[grid.kc + 1 :] = 0.0  # nothing beyond the dealias cutoff
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(Ek_cur > 0, np.sqrt(Ek_tar / np.maximum(Ek_cur, 1e-300)), 0.0)
    c *= scale[grid.shell]
    return c
