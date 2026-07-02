"""Spectral-space machinery for pseudo-spectral DNS on a [0, 2pi)^3 periodic box.

Conventions
-----------
- Physical fields are real, shape (N, N, N); spectral fields use the rfftn
  layout, shape (N, N, N//2 + 1), and store *Fourier-series coefficients*
  c_k such that u(x) = sum_k c_k exp(i k.x)  (forward transform divides by
  N^3; inverse transform applies no scaling).
- Wavenumbers are integers (box size 2pi => dk = 1).
- Dealiasing: 2/3-rule cube truncation, |k_i| <= kc = N // 3.
- Energy is the mean energy density E = (1/V) int 0.5 |u|^2 dV
  = 0.5 * sum_k w_k |c_k|^2, where w_k accounts for the modes not stored
  in the rfft half-space (w = 2 except on the kz = 0 and kz = N/2 planes).
- dtype: float64 (default) or float32; reductions always accumulate in
  float64.
"""

import numpy as np

try:
    import pyfftw

    _HAVE_PYFFTW = True
except ImportError:  # pragma: no cover
    _HAVE_PYFFTW = False

from scipy import fft as sfft


def load_wisdom(path):
    if _HAVE_PYFFTW:
        try:
            import pickle

            with open(path, "rb") as f:
                pyfftw.import_wisdom(pickle.load(f))
        except (OSError, pickle.PickleError):
            pass


def save_wisdom(path):
    if _HAVE_PYFFTW:
        import pickle

        with open(path, "wb") as f:
            pickle.dump(pyfftw.export_wisdom(), f)


class SpectralGrid:
    def __init__(self, N, threads=4, use_pyfftw=True, planner="FFTW_MEASURE",
                 dtype="float64"):
        self.N = N
        self.Nk = N // 2 + 1
        self.threads = threads
        self.kc = N // 3  # 2/3-rule cutoff (max retained integer wavenumber)
        self.rdt = np.dtype(dtype)
        self.cdt = np.dtype("complex64" if dtype == "float32" else "complex128")

        k = np.fft.fftfreq(N, 1.0 / N)  # integers 0..N/2-1, -N/2..-1
        kz = np.arange(self.Nk, dtype=np.float64)
        # broadcastable wavenumber arrays in the working precision
        self.kx = k.reshape(N, 1, 1).astype(self.rdt)
        self.ky = k.reshape(1, N, 1).astype(self.rdt)
        self.kz = kz.reshape(1, 1, self.Nk).astype(self.rdt)
        k2 = (
            k.reshape(N, 1, 1) ** 2
            + k.reshape(1, N, 1) ** 2
            + kz.reshape(1, 1, self.Nk) ** 2
        )
        self.k2 = k2.astype(self.rdt)  # full (N, N, Nk)
        self.k2_nozero = self.k2.copy()
        self.k2_nozero[0, 0, 0] = 1.0
        self.inv_k2 = (1.0 / self.k2_nozero).astype(self.rdt)

        # rfft multiplicity weights (as a 1-D kz vector; float64 for sums)
        wvec = np.full(self.Nk, 2.0)
        wvec[0] = 1.0
        if N % 2 == 0:
            wvec[-1] = 1.0
        self.wvec = wvec
        self.w = wvec.reshape(1, 1, self.Nk)

        # 2/3-rule dealias mask (cube truncation)
        kc = self.kc
        self.dealias = (
            (np.abs(self.kx) <= kc)
            & (np.abs(self.ky) <= kc)
            & (np.abs(self.kz) <= kc)
        )

        # shell index for spectra: shell s contains kmag in [s-0.5, s+0.5)
        kmag = np.sqrt(k2)
        self.shell = np.rint(kmag).astype(np.int64)
        self.nshells = int(self.shell.max()) + 1
        self._shell_flat = self.shell.ravel()
        self._w_full = np.broadcast_to(self.w, (N, N, self.Nk))

        self._setup_fft(use_pyfftw, planner)

    # ------------------------------------------------------------------ FFT
    def _setup_fft(self, use_pyfftw, planner):
        N, Nk = self.N, self.Nk
        self.use_pyfftw = use_pyfftw and _HAVE_PYFFTW
        self.planner = planner
        if self.use_pyfftw:
            self._r3 = pyfftw.empty_aligned((3, N, N, N), dtype=self.rdt)
            self._c3 = pyfftw.empty_aligned((3, N, N, Nk), dtype=self.cdt)
            flags = (planner,)
            self._fwd3 = pyfftw.FFTW(
                self._r3, self._c3, axes=(1, 2, 3),
                direction="FFTW_FORWARD", flags=flags, threads=self.threads,
            )
            self._bwd3 = pyfftw.FFTW(
                self._c3, self._r3, axes=(1, 2, 3),
                direction="FFTW_BACKWARD", flags=flags, threads=self.threads,
                normalise_idft=False,
            )
        self._norm = 1.0 / N**3

    def make_plan(self, rbuf, cbuf, direction):
        """Plan a transform bound to caller-owned aligned buffers."""
        if not self.use_pyfftw:
            return None
        if direction == "forward":
            return pyfftw.FFTW(
                rbuf, cbuf, axes=(1, 2, 3), direction="FFTW_FORWARD",
                flags=(self.planner,), threads=self.threads,
            )
        return pyfftw.FFTW(
            cbuf, rbuf, axes=(1, 2, 3), direction="FFTW_BACKWARD",
            flags=(self.planner,), threads=self.threads,
            normalise_idft=False,
        )

    def aligned(self, shape, kind):
        dt = self.rdt if kind == "real" else self.cdt
        if self.use_pyfftw:
            return pyfftw.zeros_aligned(shape, dtype=dt)
        return np.zeros(shape, dtype=dt)

    def fwd3(self, u, out=None):
        """Forward transform of a (3,N,N,N) real field -> coefficients."""
        if out is None:
            out = np.empty((3, self.N, self.N, self.Nk), dtype=self.cdt)
        if self.use_pyfftw:
            self._r3[:] = u
            self._fwd3()
            np.multiply(self._c3, self._norm, out=out)
        else:
            out[:] = sfft.rfftn(u, axes=(1, 2, 3), workers=self.threads)
            out *= self._norm
        return out

    def bwd3(self, c, out=None):
        """Inverse transform of (3,N,N,Nk) coefficients -> real field."""
        if out is None:
            out = np.empty((3, self.N, self.N, self.N), dtype=self.rdt)
        if self.use_pyfftw:
            self._c3[:] = c
            self._bwd3()
            out[:] = self._r3
        else:
            out[:] = sfft.irfftn(
                c, s=(self.N, self.N, self.N), axes=(1, 2, 3),
                workers=self.threads,
            ) * self.N**3
        return out

    # ----------------------------------------------------------- operators
    def curl(self, c, out=None):
        """Spectral curl: omega_hat = i k x c."""
        if out is None:
            out = np.empty_like(c)
        kx, ky, kz = self.kx, self.ky, self.kz
        out[0] = 1j * (ky * c[2] - kz * c[1])
        out[1] = 1j * (kz * c[0] - kx * c[2])
        out[2] = 1j * (kx * c[1] - ky * c[0])
        return out

    def project(self, c):
        """Leray projection (in place): remove the component parallel to k."""
        div = (self.kx * c[0] + self.ky * c[1] + self.kz * c[2]) * self.inv_k2
        c[0] -= self.kx * div
        c[1] -= self.ky * div
        c[2] -= self.kz * div
        return c

    # --------------------------------------------------------- diagnostics
    def mode_energy(self, c, out=None):
        """Weighted per-mode energy density 0.5 * w * |c|^2, summed over
        components. float64 output for accurate reductions."""
        if out is None:
            out = np.zeros(c.shape[1:], dtype=np.float64)
        else:
            out[:] = 0.0
        for i in range(3):
            re, im = c[i].real, c[i].imag
            out += re.astype(np.float64) ** 2
            out += im.astype(np.float64) ** 2
        out *= 0.5 * self._w_full
        return out

    def energy(self, c):
        return float(self.mode_energy(c).sum())

    def dissipation(self, c, nu):
        e = self.mode_energy(c)
        return 2.0 * nu * float((e * self.k2).sum())

    def energy_and_dissipation(self, c, nu):
        e = self.mode_energy(c)
        return float(e.sum()), 2.0 * nu * float((e * self.k2).sum())

    def spectrum(self, c):
        """Shell-summed energy spectrum E(k), k = 0..nshells-1."""
        e = self.mode_energy(c)
        return np.bincount(
            self._shell_flat, weights=e.ravel(), minlength=self.nshells
        )

    def max_divergence(self, c):
        """Max |div u| in physical space (should be ~roundoff)."""
        d = 1j * (self.kx * c[0] + self.ky * c[1] + self.kz * c[2])
        div = sfft.irfftn(
            d.astype(np.complex128), s=(self.N,) * 3, workers=self.threads
        ) * self.N**3
        return float(np.abs(div).max())
