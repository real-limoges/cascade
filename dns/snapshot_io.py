"""Snapshot I/O.

Snapshots are stored as *compact dealiased spectral coefficients* in
complex64: for a grid of size N with 2/3-rule cutoff kc = N//3, only the
retained modes are kept, giving an array of shape (3, 2*kc+1, 2*kc+1, kc+1)
ordered so that axis-1/2 indices map to wavenumbers via numpy's fft layout
restricted to |k| <= kc (see pack/unpack below). This is lossless with
respect to the dealiased simulation state up to float32 precision and
~3x smaller than a float32 physical-space dump.

Use `load_velocity(path, N)` to recover the physical velocity field.
"""

import json

import numpy as np
from scipy import fft as sfft


def _sl(kc):
    # rfft layout: index 0..kc are k=0..kc; index N-kc..N-1 are k=-kc..-1
    return kc


def pack(c, kc):
    """(3,N,N,N//2+1) complex128 -> compact (3,2kc+1,2kc+1,kc+1) complex64."""
    lo = kc + 1
    idx = np.r_[0:lo, c.shape[1] - kc : c.shape[1]]
    out = c[:, idx][:, :, idx][:, :, :, : kc + 1]
    return np.ascontiguousarray(out.astype(np.complex64))


def unpack(p, N):
    """Compact array -> full (3,N,N,N//2+1) complex128 rfft layout."""
    kc = (p.shape[1] - 1) // 2
    lo = kc + 1
    c = np.zeros((3, N, N, N // 2 + 1), dtype=np.complex128)
    idx = np.r_[0:lo, N - kc : N]
    c[np.ix_(range(3), idx, idx, range(kc + 1))] = p.astype(np.complex128)
    return c


def save_snapshot(path, c, kc, meta):
    p = pack(c, kc)
    np.savez(path, coeffs=p, meta=json.dumps(meta))


def load_snapshot(path):
    d = np.load(path, allow_pickle=False)
    return d["coeffs"], json.loads(str(d["meta"]))


def load_velocity(path, N, workers=4):
    """Return the physical-space velocity field (3, N, N, N) float64."""
    p, meta = load_snapshot(path)
    c = unpack(p, N)
    u = sfft.irfftn(c, s=(N, N, N), axes=(1, 2, 3), workers=workers) * N**3
    return u, meta
