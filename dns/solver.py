"""Pseudo-spectral incompressible Navier-Stokes solver.

du/dt = P[u x omega] - nu k^2 u + f        (rotational form; the pressure
                                            + u^2/2 gradient is removed by
                                            the Leray projection P)

- Nonlinear term evaluated pseudo-spectrally with 2/3-rule dealiasing.
  In rotational form the dealiased (Galerkin-truncated) nonlinear term
  conserves energy exactly, which is used as a correctness check.
- Time integration: classical explicit RK4 with adaptive CFL time step.
  The viscous term is treated explicitly; stability requires
  nu * kc^2 * dt well below the RK4 real-axis bound (~2.79), which is
  enforced by compute_dt().
- Forcing (optional): deterministic fixed-power forcing
      f_hat = (P_inj / (2 E_f)) * u_hat   for 0 < |k| <= k_f,
  which injects energy at exactly the rate P_inj, so statistical
  stationarity implies <eps> = P_inj.

The hot path avoids temporaries: FFT plans are bound to dedicated aligned
buffers (FFTW's multi-dimensional c2r destroys its input, so the state is
copied into a scratch buffer before inverse transforms), all elementwise
updates are in place, and the forcing band is touched only through
precomputed mode indices.
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

        mask = (g.k2 > 0.0) & (g.k2 <= k_force**2 + 1e-12)
        self._fidx = np.nonzero(mask)  # forced-mode indices (small set)
        self._fw = np.broadcast_to(g.w, mask.shape)[self._fidx].astype(
            np.float64
        )

        cshape = (3, g.N, g.N, g.Nk)
        rshape = (3, g.N, g.N, g.N)
        self.c = np.zeros(cshape, dtype=g.cdt)

        # FFT-bound buffers
        self._Cs = g.aligned(cshape, "complex")   # scratch for c (destroyed)
        self._Ws = g.aligned(cshape, "complex")   # vorticity / N-hat buffer
        self._Ur = g.aligned(rshape, "real")
        self._Wr = g.aligned(rshape, "real")
        self._Xr = g.aligned(rshape, "real")
        self._bwd_u = g.make_plan(self._Ur, self._Cs, "backward")
        self._bwd_w = g.make_plan(self._Wr, self._Ws, "backward")
        self._fwd_n = g.make_plan(self._Xr, self._Ws, "forward")

        # RK4 state buffers and elementwise scratch
        self._c0 = np.empty_like(self.c)
        self._acc = np.empty_like(self.c)
        self._k1 = np.empty_like(self.c)
        self._t1 = np.empty(cshape[1:], dtype=g.cdt)
        self._t2 = np.empty(cshape[1:], dtype=g.cdt)

        # precomputed fields (working precision)
        self._dealias_norm = (g.dealias.astype(g.rdt) / g.N**3)
        self._nu_k2 = (self.nu * g.k2).astype(g.rdt)

        self.t = 0.0
        self.step_count = 0
        self._umax = None  # cached for CFL, refreshed in rhs

    # ------------------------------------------------------------------ RHS
    def _curl_into_Ws(self, c):
        g, W, t = self.g, self._Ws, self._t1
        np.multiply(c[2], g.ky, out=W[0])
        np.multiply(c[1], g.kz, out=t)
        W[0] -= t
        W[0] *= 1j
        np.multiply(c[0], g.kz, out=W[1])
        np.multiply(c[2], g.kx, out=t)
        W[1] -= t
        W[1] *= 1j
        np.multiply(c[1], g.kx, out=W[2])
        np.multiply(c[0], g.ky, out=t)
        W[2] -= t
        W[2] *= 1j

    def _project_Ws(self):
        g, W, div, t = self.g, self._Ws, self._t1, self._t2
        np.multiply(W[0], g.kx, out=div)
        np.multiply(W[1], g.ky, out=t)
        div += t
        np.multiply(W[2], g.kz, out=t)
        div += t
        div *= g.inv_k2
        np.multiply(div, g.kx, out=t)
        W[0] -= t
        np.multiply(div, g.ky, out=t)
        W[1] -= t
        np.multiply(div, g.kz, out=t)
        W[2] -= t

    def rhs(self, c, out, measure_umax=False):
        """out <- P[u x omega]_dealiased - nu k^2 c + f(c)."""
        g = self.g
        if g.use_pyfftw:
            self._Cs[:] = c            # c2r destroys input; work on a copy
            self._bwd_u()
            self._curl_into_Ws(c)
            self._bwd_w()
        else:
            g.bwd3(c, out=self._Ur)
            self._curl_into_Ws(c)
            g.bwd3(self._Ws, out=self._Wr)
        u, w, x = self._Ur, self._Wr, self._Xr
        np.multiply(u[1], w[2], out=x[0])
        x[0] -= u[2] * w[1]
        np.multiply(u[2], w[0], out=x[1])
        x[1] -= u[0] * w[2]
        np.multiply(u[0], w[1], out=x[2])
        x[2] -= u[1] * w[0]
        if measure_umax:
            self._umax = self._sum_umax(u)
        if g.use_pyfftw:
            self._fwd_n()              # Xr -> Ws (r2c preserves input)
        else:
            g.fwd3(self._Xr, out=self._Ws)
            self._Ws *= g.N**3         # undo fwd3's normalisation
        W = self._Ws
        for i in range(3):
            W[i] *= self._dealias_norm
        self._project_Ws()
        # out = W - nu k^2 c  (+ forcing)
        t = self._t1
        for i in range(3):
            np.multiply(c[i], self._nu_k2, out=t)
            np.subtract(W[i], t, out=out[i])
        if self.forcing_power is not None:
            idx = self._fidx
            band = [c[i][idx] for i in range(3)]
            ef = 0.5 * float(
                sum(
                    np.sum(self._fw * (b.real.astype(np.float64) ** 2
                                       + b.imag.astype(np.float64) ** 2))
                    for b in band
                )
            )
            if ef > 1e-14:
                alpha = self.forcing_power / (2.0 * ef)
                for i in range(3):
                    out[i][idx] += (alpha * band[i]).astype(out.dtype)
        return out

    @staticmethod
    def _sum_umax(u):
        # max over the grid of |u1| + |u2| + |u3| (single fused pass)
        a = np.abs(u[0])
        a += np.abs(u[1])
        a += np.abs(u[2])
        return float(a.max())

    def band_energy(self, c):
        idx = self._fidx
        return 0.5 * float(
            sum(
                np.sum(self._fw * (c[i][idx].real.astype(np.float64) ** 2
                                   + c[i][idx].imag.astype(np.float64) ** 2))
                for i in range(3)
            )
        )

    # ----------------------------------------------------------- time step
    def compute_dt(self):
        if self._umax is None or self._umax == 0.0:
            return 1e-3
        dx = 2.0 * np.pi / self.g.N
        dt = self.cfl * dx / self._umax
        # explicit viscous stability for RK4 (real-axis bound ~2.79)
        dt_visc = 2.5 / (self.nu * (3.0 * self.g.kc**2))
        return min(dt, dt_visc)

    def step(self, dt):
        """Classical RK4 (k1 is reused as the stage-RHS buffer)."""
        c, c0, acc, k1 = self.c, self._c0, self._acc, self._k1
        c0[:] = c
        # stage 1 (also refresh umax for the next CFL evaluation)
        self.rhs(c, out=k1, measure_umax=True)
        np.multiply(k1, dt / 6.0, out=acc)
        k1 *= 0.5 * dt
        np.add(c0, k1, out=c)
        # stage 2
        self.rhs(c, out=k1)
        k1 *= dt / 3.0
        acc += k1
        k1 *= 1.5                      # (dt/3)*1.5 = dt/2
        np.add(c0, k1, out=c)
        # stage 3
        self.rhs(c, out=k1)
        k1 *= dt / 3.0
        acc += k1
        k1 *= 3.0                      # (dt/3)*3 = dt
        np.add(c0, k1, out=c)
        # stage 4
        self.rhs(c, out=k1)
        k1 *= dt / 6.0
        acc += k1
        np.add(c0, acc, out=c)
        self.t += dt
        self.step_count += 1

    # --------------------------------------------------------- diagnostics
    def diagnostics(self):
        E, eps = self.g.energy_and_dissipation(self.c, self.nu)
        return {
            "t": self.t, "step": self.step_count, "E": E, "eps": eps,
            "E_force_band": self.band_energy(self.c), "umax": self._umax,
        }

    def integral_quantities(self):
        """Derived turbulence quantities from the current spectrum."""
        g = self.g
        Ek = g.spectrum(self.c)
        k = np.arange(len(Ek), dtype=np.float64)
        E = Ek.sum()
        eps = g.dissipation(self.c, self.nu)
        urms = np.sqrt(2.0 * E / 3.0)
        eta = (self.nu**3 / eps) ** 0.25 if eps > 0 else np.inf
        # integral length scale L = (3 pi / 4 E) * int E(k)/k dk
        L = 3.0 * np.pi / (4.0 * E) * float(
            np.sum(Ek[1:] / k[1:])
        )
        lam = np.sqrt(15.0 * self.nu / eps) * urms if eps > 0 else np.inf
        re_lambda = urms * lam / self.nu
        T_eddy = L / urms if urms > 0 else np.inf
        return {
            "E": E, "eps": eps, "urms": urms, "eta": eta,
            "L_int": L, "lambda_taylor": lam, "Re_lambda": re_lambda,
            "T_eddy": T_eddy, "kmax_eta": g.kc * eta,
        }
