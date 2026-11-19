"""Pseudo-spectral solver for 2D incompressible Navier-Stokes (Kolmogorov flow), NumPy/SciPy.

    d_t w + u . grad(w) = nu * lap(w) - alpha * w + f,   f = -k_f cos(k_f y),
    lap(psi) = -w,   u = d_y psi,   v = -d_x psi,         (x, y) in [0, 2pi)^2.

* Space: Fourier-Galerkin, `rfft2` layout, fields indexed `[x, y]` (as in jax-cfd).
  Advection is computed pseudo-spectrally and dealiased with the 2/3 rule.
* Time: low-storage IMEX RK3/Crank-Nicolson (Spalart, Moser & Rogers 1991).
  Advection + forcing are explicit (RK3), viscosity + drag implicit (CN). The linear
  operator is diagonal in Fourier space, so the implicit solve is a division; it is folded
  into per-stage coefficients `w <- a_k w + b_k N_k + c_k N_{k-1}` computed once.
* Batching is native: every function accepts arbitrary leading axes, e.g. `w0` of shape
  `(B, n, n)` with `nu` of shape `(B,)` gives a `(B, T, n, n)` trajectory.
* FFTs: `scipy.fft` (pocketfft), multithreaded over the batch with `FFT_WORKERS` threads.
  The working precision is the dtype of the input (float32 or float64).

With unit forcing amplitude and unit length scale, `Re = 1 / nu`.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import scipy.fft

# Spalart-Moser-Rogers coefficients; implicit weights alpha_k = beta_k = (gamma_k + zeta_k) / 2.
_GAMMA = (8 / 15, 5 / 12, 3 / 4)
_ZETA = (0.0, -17 / 60, -5 / 12)
_BETA = (4 / 15, 1 / 15, 1 / 6)

# Threads for scipy.fft (-1 = all cores).
FFT_WORKERS: int = -1


def _rfft2(x: np.ndarray) -> np.ndarray:
    return scipy.fft.rfft2(x, workers=FFT_WORKERS)


def _irfft2(x_hat: np.ndarray, n: int) -> np.ndarray:
    return scipy.fft.irfft2(x_hat, s=(n, n), workers=FFT_WORKERS)


class SpectralGrid(NamedTuple):
    kx: np.ndarray  # (n, n//2+1) integer wavenumbers
    ky: np.ndarray
    k2: np.ndarray  # |k|^2
    inv_k2: np.ndarray  # 1/|k|^2, 0 at k = 0
    dealias: np.ndarray  # 2/3-rule mask (0/1, real dtype)
    ops: np.ndarray  # (4, n, n//2+1) multipliers w_hat -> (u, v, w_x, w_y)_hat

    @property
    def n(self) -> int:
        return self.kx.shape[0]

    @classmethod
    def create(cls, n: int, dtype: np.typing.DTypeLike = np.float64) -> SpectralGrid:
        kx, ky = np.meshgrid(np.fft.fftfreq(n, 1 / n), np.fft.rfftfreq(n, 1 / n), indexing='ij')
        k2 = kx**2 + ky**2
        inv_k2 = np.divide(1, k2, out=np.zeros_like(k2), where=k2 > 0)
        dealias = (np.abs(kx) < n / 3) & (np.abs(ky) < n / 3)
        ops = np.stack([1j * ky * inv_k2, -1j * kx * inv_k2, 1j * kx, 1j * ky])
        real = np.dtype(dtype)
        return cls(
            *(a.astype(real) for a in (kx, ky, k2, inv_k2, dealias)),
            ops.astype(np.result_type(real, np.complex64)),
        )


def coordinates(n: int, dtype: np.typing.DTypeLike = np.float64) -> tuple[np.ndarray, np.ndarray]:
    x = (2 * np.pi * np.arange(n) / n).astype(dtype)
    x, y = np.meshgrid(x, x, indexing='ij')
    return x, y


def forcing(n: int, k_f: int = 4, dtype: np.typing.DTypeLike = np.float64) -> np.ndarray:
    _, y = coordinates(n, dtype)
    return -k_f * np.cos(k_f * y)


# ----------------------------------------------------------------------------- diagnostics


def vorticity_to_velocity(w: np.ndarray) -> np.ndarray:
    """(..., n, n) -> (..., 2, n, n)."""
    g = SpectralGrid.create(w.shape[-1], w.dtype)
    return _irfft2(g.ops[:2] * _rfft2(w)[..., None, :, :], g.n)


def velocity_to_vorticity(uv: np.ndarray) -> np.ndarray:
    """(..., 2, n, n) -> (..., n, n)."""
    g = SpectralGrid.create(uv.shape[-1], uv.dtype)
    uv_hat = _rfft2(uv)
    return _irfft2(1j * g.kx * uv_hat[..., 1, :, :] - 1j * g.ky * uv_hat[..., 0, :, :], g.n)


def divergence(uv: np.ndarray) -> np.ndarray:
    """(..., 2, n, n) -> (..., n, n)."""
    g = SpectralGrid.create(uv.shape[-1], uv.dtype)
    uv_hat = _rfft2(uv)
    return _irfft2(1j * g.kx * uv_hat[..., 0, :, :] + 1j * g.ky * uv_hat[..., 1, :, :], g.n)


# ----------------------------------------------------------------------------- time stepping


def _explicit_term(w_hat: np.ndarray, f_hat: np.ndarray, g: SpectralGrid) -> np.ndarray:
    """N(w) = f - P_dealias[u . grad(w)], in Fourier space."""
    fields = _irfft2(g.ops * w_hat[..., None, :, :], g.n)  # (..., 4, n, n): u, v, w_x, w_y
    adv = fields[..., 0, :, :] * fields[..., 2, :, :]
    adv += fields[..., 1, :, :] * fields[..., 3, :, :]
    adv_hat = _rfft2(adv)
    adv_hat *= g.dealias
    return np.subtract(f_hat, adv_hat, out=adv_hat)


def imex_coefficients(
    g: SpectralGrid, nu: float | np.ndarray, alpha: float, dt: float
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-stage (a, b, c) with w <- a * w + b * N_cur + c * N_prev.

    Equivalent to  w <- (w + dt (beta L w + gamma N_cur + zeta N_prev)) / (1 - dt beta L),
    L = -nu |k|^2 - alpha. `nu` broadcasts against the leading (batch) axes.
    """
    nu = np.asarray(nu, dtype=g.k2.dtype)[..., None, None]
    lin = -nu * g.k2 - alpha
    coeffs = []
    for gamma, zeta, beta in zip(_GAMMA, _ZETA, _BETA):
        inv = 1 / (1 - dt * beta * lin)
        coeffs.append(((1 + dt * beta * lin) * inv, (dt * gamma) * inv, (dt * zeta) * inv))
    return coeffs


def step(
    w_hat: np.ndarray,
    f_hat: np.ndarray,
    g: SpectralGrid,
    coeffs: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    n_prev = None
    for a, b, c in coeffs:
        n_cur = _explicit_term(w_hat, f_hat, g)
        w_new = a * w_hat
        w_new += b * n_cur
        if n_prev is not None:
            w_new += c * n_prev
        w_hat, n_prev = w_new, n_cur
    return w_hat


def simulate(
    w0: np.ndarray,
    nu: float | np.ndarray,
    dt: float,
    n_snapshots: int,
    steps_per_snapshot: int,
    alpha: float = 0.1,
    k_f: int = 4,
) -> np.ndarray:
    """w0: (..., n, n), nu: scalar or broadcastable to the leading axes -> (..., T, n, n)."""
    n, dtype = w0.shape[-1], w0.dtype
    g = SpectralGrid.create(n, dtype)
    f_hat = _rfft2(forcing(n, k_f, dtype))
    coeffs = imex_coefficients(g, nu, alpha, float(dt))

    w_hat = _rfft2(w0) * g.dealias
    traj = np.empty((*w0.shape[:-2], n_snapshots, n, n), dtype=dtype)
    for t in range(n_snapshots):
        for _ in range(steps_per_snapshot):
            w_hat = step(w_hat, f_hat, g, coeffs)
        traj[..., t, :, :] = _irfft2(w_hat, n)
    return traj


# ----------------------------------------------------------------------------- initial conditions


def random_vorticity(
    rng: np.random.Generator,
    n: int,
    k_peak: float = 4.0,
    max_velocity: float = 3.0,
    dtype: np.typing.DTypeLike = np.float64,
) -> np.ndarray:
    """(n, n) band-limited random vorticity, scaled to max |u| = max_velocity."""
    g = SpectralGrid.create(n, dtype)
    k = np.sqrt(g.k2)
    envelope = np.exp(-0.5 * (k - k_peak) ** 2 / (0.5 * k_peak) ** 2) * (k > 0) * g.dealias
    w = _irfft2(_rfft2(rng.standard_normal((n, n)).astype(dtype)) * envelope, n)
    speed = np.sqrt((vorticity_to_velocity(w) ** 2).sum(axis=0))
    return w * (max_velocity / speed.max())


def stable_time_step(max_velocity: float, n: int, cfl: float = 0.5) -> float:
    return cfl * (2 * math.pi / n) / max_velocity
