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
    def __init__(self, N, threads=4, use_pyfftw=True, planner="FFTW_MEASURE"):
        self.N = N
        self.Nk = N // 2 + 1
        self.threads = threads
        self.kc = N // 3  # 2/3-rule cutoff (max retained integer wavenumber)

        k = np.fft.fftfreq(N, 1.0 / N)  # integers 0..N/2-1, -N/2..-1
        kz = np.arange(self.Nk, dtype=np.float64)
        self.kx = k.reshape(N, 1, 1).astype(np.float64)
        self.ky = k.reshape(1, N, 1).astype(np.float64)
        self.kz = kz.reshape(1, 1, self.Nk)
        self.k2 = self.kx**2 + self.ky**2 + self.kz**2  # (N, N, Nk)
        self.k2_nozero = self.k2.copy()
        self.k2_nozero[0, 0, 0] = 1.0  # avoid divide-by-zero at k=0

        # rfft multiplicity weights for sums over the full sphere
        w = np.full((1, 1, self.Nk), 2.0)
        w[..., 0] = 1.0
        if N % 2 == 0:
            w[..., -1] = 1.0
        self.w = w

        # 2/3-rule dealias mask (cube truncation)
        kc = self.kc
        self.dealias = (
            (np.abs(self.kx) <= kc)
            & (np.abs(self.ky) <= kc)
            & (np.abs(self.kz) <= kc)
        )

        # shell index for spectra: shell s contains kmag in [s-0.5, s+0.5)
        kmag = np.sqrt(self.k2)
        self.shell = np.rint(kmag).astype(np.int64)
        self.nshells = int(self.shell.max()) + 1

        self._setup_fft(use_pyfftw, planner)

    # ------------------------------------------------------------------ FFT
    def _setup_fft(self, use_pyfftw, planner):
        N, Nk = self.N, self.Nk
        self.use_pyfftw = use_pyfftw and _HAVE_PYFFTW
        if self.use_pyfftw:
            # batched 3-component plans (axes 1..3 of a (3,N,N,N) array)
            self._r3 = pyfftw.empty_aligned((3, N, N, N), dtype="float64")
            self._c3 = pyfftw.empty_aligned((3, N, N, Nk), dtype="complex128")
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

    def fwd3(self, u, out=None):
        """Forward transform of a (3,N,N,N) real field -> coefficients."""
        if out is None:
            out = np.empty((3, self.N, self.N, self.Nk), dtype=np.complex128)
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
            out = np.empty((3, self.N, self.N, self.N), dtype=np.float64)
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
        div = (self.kx * c[0] + self.ky * c[1] + self.kz * c[2]) / self.k2_nozero
        c[0] -= self.kx * div
        c[1] -= self.ky * div
        c[2] -= self.kz * div
        return c

    # --------------------------------------------------------- diagnostics
    def energy(self, c):
        return 0.5 * float(np.sum(self.w * (c.real**2 + c.imag**2)))

    def dissipation(self, c, nu):
        return 2.0 * nu * 0.5 * float(
            np.sum(self.w * self.k2 * (c.real**2 + c.imag**2))
        )

    def spectrum(self, c):
        """Shell-summed energy spectrum E(k), k = 0..nshells-1."""
        e = 0.5 * np.sum(self.w * (c.real**2 + c.imag**2), axis=0)
        return np.bincount(
            self.shell.ravel(), weights=e.ravel(), minlength=self.nshells
        )

    def max_divergence(self, c):
        """Max |div u| in physical space (should be ~roundoff)."""
        d = 1j * (self.kx * c[0] + self.ky * c[1] + self.kz * c[2])
        if self.use_pyfftw:
            div = sfft.irfftn(
                d, s=(self.N,) * 3, workers=self.threads
            ) * self.N**3
        else:
            div = sfft.irfftn(
                d, s=(self.N,) * 3, workers=self.threads
            ) * self.N**3
        return float(np.abs(div).max())
