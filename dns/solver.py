"""Pseudo-spectral incompressible Navier-Stokes solver.

du/dt = P[u x omega] - nu k^2 u + f        (rotational form; the pressure
                                            + u^2/2 gradient is removed by
                                            the Leray projection P)

- Nonlinear term evaluated pseudo-spectrally with 2/3-rule dealiasing.
  In rotational form the dealiased (Galerkin-truncated) nonlinear term
  conserves energy exactly, which is used as a correctness check.
- Time integration: classical explicit RK4 with adaptive CFL time step.
  The viscous term is treated explicitly; stability requires
  nu * kc^2 * dt << 2.79, which is checked at runtime.
- Forcing (optional): deterministic fixed-power forcing
      f_hat = (P_inj / (2 E_f)) * u_hat   for 0 < |k| <= k_f,
  which injects energy at exactly the rate P_inj, so statistical
  stationarity implies <eps> = P_inj.
"""

import numpy as np

from .spectral import SpectralGrid


class Solver:
    def __init__(self, grid: SpectralGrid, nu, forcing_power=None,
                 k_force=2.5, cfl=0.8):
        self.g = grid
        self.nu = float(nu)
        self.forcing_power = forcing_power  # None => unforced (decay)
        self.cfl = cfl
        g = grid
        kmag2 = g.k2
        self.force_mask = (kmag2 > 0.0) & (kmag2 <= k_force**2 + 1e-12)

        self.c = np.zeros((3, g.N, g.N, g.Nk), dtype=np.complex128)
        # work buffers
        self._what = np.empty_like(self.c)
        self._nhat = np.empty_like(self.c)
        self._u = np.empty((3, g.N, g.N, g.N))
        self._w = np.empty((3, g.N, g.N, g.N))
        self._cross = np.empty((3, g.N, g.N, g.N))
        self._c0 = np.empty_like(self.c)
        self._acc = np.empty_like(self.c)
        self._k1 = np.empty_like(self.c)

        self.t = 0.0
        self.step_count = 0
        self._umax = None  # cached for CFL, refreshed in rhs

    # ------------------------------------------------------------------ RHS
    def rhs(self, c, out, measure_umax=False):
        """out <- P[u x omega]_dealiased - nu k^2 c + f(c)."""
        g = self.g
        g.curl(c, out=self._what)
        g.bwd3(c, out=self._u)
        g.bwd3(self._what, out=self._w)
        u, w, x = self._u, self._w, self._cross
        np.multiply(u[1], w[2], out=x[0])
        x[0] -= u[2] * w[1]
        np.multiply(u[2], w[0], out=x[1])
        x[1] -= u[0] * w[2]
        np.multiply(u[0], w[1], out=x[2])
        x[2] -= u[1] * w[0]
        if measure_umax:
            self._umax = float(
                np.max(np.abs(u[0]) + np.abs(u[1]) + np.abs(u[2]))
            )
        g.fwd3(x, out=out)
        out *= g.dealias
        g.project(out)
        # viscous
        out -= self.nu * g.k2 * c
        # forcing
        if self.forcing_power is not None:
            m = self.force_mask
            ef = self._band_energy(c)
            if ef > 1e-14:
                alpha = self.forcing_power / (2.0 * ef)
                for i in range(3):
                    out[i][m] += alpha * c[i][m]
        return out

    def _band_energy(self, c):
        m = self.force_mask
        w = np.broadcast_to(self.g.w, c.shape[1:])[m]
        e = 0.0
        for i in range(3):
            ci = c[i][m]
            e += float(np.sum(w * (ci.real**2 + ci.imag**2)))
        return 0.5 * e

    # ----------------------------------------------------------- time step
    def compute_dt(self):
        if self._umax is None or self._umax == 0.0:
            return 1e-3
        dx = 2.0 * np.pi / self.g.N
        dt = self.cfl * dx / self._umax
        # explicit viscous stability for RK4 (real-axis bound ~2.79)
        dt_visc = 2.5 / (self.nu * self.g.kc**2 * 3.0)
        return min(dt, dt_visc)

    def step(self, dt):
        """Classical RK4."""
        c, c0, acc, k1 = self.c, self._c0, self._acc, self._k1
        c0[:] = c
        # stage 1 (also refresh umax for the next CFL evaluation)
        self.rhs(c, out=k1, measure_umax=True)
        np.multiply(k1, dt / 6.0, out=acc)
        np.multiply(k1, 0.5 * dt, out=c)
        c += c0
        # stage 2
        self.rhs(c, out=k1)
        acc += (dt / 3.0) * k1
        np.multiply(k1, 0.5 * dt, out=c)
        c += c0
        # stage 3
        self.rhs(c, out=k1)
        acc += (dt / 3.0) * k1
        np.multiply(k1, dt, out=c)
        c += c0
        # stage 4
        self.rhs(c, out=k1)
        acc += (dt / 6.0) * k1
        np.add(c0, acc, out=c)
        c *= self.g.dealias
        self.t += dt
        self.step_count += 1

    # --------------------------------------------------------- diagnostics
    def diagnostics(self):
        g = self.g
        c = self.c
        E = g.energy(c)
        eps = g.dissipation(c, self.nu)
        return {
            "t": self.t,
            "step": self.step_count,
            "E": E,
            "eps": eps,
            "E_force_band": self._band_energy(c),
            "umax": self._umax,
        }

    def integral_quantities(self):
        """Derived turbulence quantities from the current spectrum."""
        g = self.g
        Ek = g.spectrum(self.c)
        k = np.arange(len(Ek), dtype=np.float64)
        E = Ek.sum()
        eps = self.g.dissipation(self.c, self.nu)
        urms = np.sqrt(2.0 * E / 3.0)
        eta = (self.nu**3 / eps) ** 0.25 if eps > 0 else np.inf
        # integral length scale L = (3 pi / 4 E) * int E(k)/k dk
        with np.errstate(divide="ignore", invalid="ignore"):
            L = 3.0 * np.pi / (4.0 * E) * np.nansum(
                np.where(k > 0, Ek / np.maximum(k, 1e-30), 0.0)
            )
        lam = np.sqrt(15.0 * self.nu / eps) * urms if eps > 0 else np.inf
        re_lambda = urms * lam / self.nu
        T_eddy = L / urms if urms > 0 else np.inf
        return {
            "E": E, "eps": eps, "urms": urms, "eta": eta,
            "L_int": L, "lambda_taylor": lam, "Re_lambda": re_lambda,
            "T_eddy": T_eddy, "kmax_eta": g.kc * eta,
        }
