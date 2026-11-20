"""Pseudo-spectral solver for 2D incompressible Navier-Stokes (Kolmogorov flow), JAX.

    d_t w + u . grad(w) = nu * lap(w) - alpha * w + f,   f = -k_f cos(k_f y),
    lap(psi) = -w,   u = d_y psi,   v = -d_x psi,         (x, y) in [0, 2pi)^2.

* Space: Fourier-Galerkin, `rfft2` layout, fields indexed `[x, y]` (as in jax-cfd).
  Advection is computed pseudo-spectrally and dealiased with the 2/3 rule.
* Time: low-storage IMEX RK3/Crank-Nicolson (Spalart, Moser & Rogers 1991).
  Advection + forcing are explicit (RK3), viscosity + drag implicit (CN). The linear
  operator is diagonal in Fourier space, so the implicit solve is a division; it is folded
  into per-stage coefficients `w <- a_k w + b_k N_k + c_k N_{k-1}` computed once.
* Batching is native: every function accepts arbitrary leading axes, e.g. `w0` of shape
  `(B, n, n)` with `nu` of shape `(B,)` gives a `(B, T, n, n)` trajectory. A whole trajectory
  is one compiled program (`lax.scan` over the snapshots, `lax.fori_loop` over the steps).
* Precision: the dtype of the input (float32 or float64). float64 needs
  `jax.config.update('jax_enable_x64', True)` before any JAX array is created (otherwise JAX
  silently computes in float32); TPUs have no float64 FFTs, so float64 runs on GPU or CPU.
* The wavenumber grids are NumPy constants computed from the static shapes.

With unit forcing amplitude and unit length scale, `Re = 1 / nu`.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# Spalart-Moser-Rogers coefficients; implicit weights alpha_k = beta_k = (gamma_k + zeta_k) / 2.
_GAMMA = (8 / 15, 5 / 12, 3 / 4)
_ZETA = (0.0, -17 / 60, -5 / 12)
_BETA = (4 / 15, 1 / 15, 1 / 6)


def _rfft2(x: jax.Array) -> jax.Array:
    return jnp.fft.rfft2(x)


def _irfft2(x_hat: jax.Array, n: int) -> jax.Array:
    return jnp.fft.irfft2(x_hat, s=(n, n))


def _check_dtype(x: jax.Array | np.ndarray) -> None:
    if np.dtype(x.dtype) == np.float64 and not jax.config.read('jax_enable_x64'):
        raise RuntimeError(
            "float64 needs jax.config.update('jax_enable_x64', True) (JAX would silently "
            'compute in float32)'
        )


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


def vorticity_to_velocity(w: jax.Array) -> jax.Array:
    """(..., n, n) -> (..., 2, n, n)."""
    g = SpectralGrid.create(w.shape[-1], w.dtype)
    return _irfft2(_rfft2(w)[..., None, :, :] * g.ops[:2], g.n)


def velocity_to_vorticity(uv: jax.Array) -> jax.Array:
    """(..., 2, n, n) -> (..., n, n)."""
    g = SpectralGrid.create(uv.shape[-1], uv.dtype)
    uv_hat = _rfft2(uv)
    return _irfft2(1j * g.kx * uv_hat[..., 1, :, :] - 1j * g.ky * uv_hat[..., 0, :, :], g.n)


def divergence(uv: jax.Array) -> jax.Array:
    """(..., 2, n, n) -> (..., n, n)."""
    g = SpectralGrid.create(uv.shape[-1], uv.dtype)
    uv_hat = _rfft2(uv)
    return _irfft2(1j * g.kx * uv_hat[..., 0, :, :] + 1j * g.ky * uv_hat[..., 1, :, :], g.n)


def truncate(w: jax.Array, n: int) -> jax.Array:
    """Spectral truncation (..., N, N) -> (..., n, n), keeping |k_x|, |k_y| < n / 2."""
    big = w.shape[-1]
    if big == n:
        return w
    k = np.fft.fftfreq(n, 1 / n).astype(np.int64)
    idx = k % big
    keep = np.abs(k) < n // 2  # drop the Nyquist modes of the target grid
    w_hat = jnp.fft.fft2(w)[..., idx[:, None], idx[None, :]] * (keep[:, None] & keep[None, :])
    return jnp.fft.ifft2(w_hat).real * (n / big) ** 2


def enstrophy_tail(w: jax.Array) -> jax.Array:
    """Fraction of enstrophy near the dealiasing cutoff, (..., N, N) -> (...).

    Large values mean the simulation is under-resolved for this Re."""
    g = SpectralGrid.create(w.shape[-1], w.dtype)
    e = jnp.abs(_rfft2(w)) ** 2
    return (e * (np.sqrt(g.k2) > 0.8 * g.n / 3)).sum(axis=(-2, -1)) / e.sum(axis=(-2, -1))


# ----------------------------------------------------------------------------- time stepping


def _explicit_term(w_hat: jax.Array, f_hat: jax.Array, g: SpectralGrid) -> jax.Array:
    """N(w) = f - P_dealias[u . grad(w)], in Fourier space."""
    fields = _irfft2(w_hat[..., None, :, :] * g.ops, g.n)  # (..., 4, n, n): u, v, w_x, w_y
    adv = fields[..., 0, :, :] * fields[..., 2, :, :] + fields[..., 1, :, :] * fields[..., 3, :, :]
    return f_hat - _rfft2(adv) * g.dealias


def imex_coefficients(
    g: SpectralGrid, nu: float | jax.Array, alpha: float, dt: float | jax.Array
) -> list[tuple[jax.Array, jax.Array, jax.Array]]:
    """Per-stage (a, b, c) with w <- a * w + b * N_cur + c * N_prev.

    Equivalent to  w <- (w + dt (beta L w + gamma N_cur + zeta N_prev)) / (1 - dt beta L),
    L = -nu |k|^2 - alpha. `nu` broadcasts against the leading (batch) axes.
    """
    nu = jnp.asarray(nu, dtype=g.k2.dtype)[..., None, None]
    lin = -nu * g.k2 - alpha
    coeffs = []
    for gamma, zeta, beta in zip(_GAMMA, _ZETA, _BETA, strict=True):
        inv = 1 / (1 - dt * beta * lin)
        coeffs.append(((1 + dt * beta * lin) * inv, (dt * gamma) * inv, (dt * zeta) * inv))
    return coeffs


def step(
    w_hat: jax.Array,
    f_hat: jax.Array,
    g: SpectralGrid,
    coeffs: list[tuple[jax.Array, jax.Array, jax.Array]],
) -> jax.Array:
    n_prev = None
    for a, b, c in coeffs:
        n_cur = _explicit_term(w_hat, f_hat, g)
        w_new = a * w_hat + b * n_cur
        if n_prev is not None:
            w_new = w_new + c * n_prev
        w_hat, n_prev = w_new, n_cur
    return w_hat


@jax.jit(static_argnames=('n_snapshots', 'alpha', 'k_f'))
def _simulate(
    w0: jax.Array,
    nu: jax.Array,
    dt: jax.Array,
    steps_per_snapshot: jax.Array,
    n_snapshots: int,
    alpha: float,
    k_f: int,
) -> jax.Array:
    n = w0.shape[-1]
    g = SpectralGrid.create(n, w0.dtype)
    f_hat = _rfft2(jnp.asarray(forcing(n, k_f, w0.dtype)))
    coeffs = imex_coefficients(g, nu, alpha, dt)

    def snapshot(w_hat: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
        w_hat = jax.lax.fori_loop(
            0, steps_per_snapshot, lambda _, w: step(w, f_hat, g, coeffs), w_hat
        )
        return w_hat, _irfft2(w_hat, n)

    _, traj = jax.lax.scan(snapshot, _rfft2(w0) * g.dealias, length=n_snapshots)
    return jnp.moveaxis(traj, 0, -3)  # (T, ..., n, n) -> (..., T, n, n)


def simulate(
    w0: jax.Array | np.ndarray,
    nu: float | jax.Array | np.ndarray,
    dt: float,
    n_snapshots: int,
    steps_per_snapshot: int,
    alpha: float = 0.1,
    k_f: int = 4,
) -> jax.Array:
    """
    w0: (..., n, n), nu: scalar or broadcastable to the leading axes -> (..., T, n, n), in the
    dtype of w0. Returns without waiting for the computation (JAX async dispatch).
    """
    _check_dtype(w0)
    w0 = jnp.asarray(w0)
    return _simulate(
        w0,
        jnp.asarray(nu, dtype=w0.dtype),
        jnp.asarray(dt, dtype=w0.dtype),
        jnp.asarray(steps_per_snapshot, dtype=jnp.int32),
        n_snapshots=n_snapshots,
        alpha=float(alpha),
        k_f=int(k_f),
    )


# ----------------------------------------------------------------------------- initial conditions


def random_vorticity(
    noise: jax.Array | np.ndarray,
    k_peak: float = 4.0,
    max_velocity: float = 3.0,
) -> jax.Array:
    """
    Band-limited random vorticity (..., n, n) from white noise (..., n, n) (e.g. standard
    normal from a NumPy generator, for reproducible data), scaled to max |u| = max_velocity
    per field.
    """
    _check_dtype(noise)
    noise = jnp.asarray(noise)
    n = noise.shape[-1]
    g = SpectralGrid.create(n, noise.dtype)
    k = np.sqrt(g.k2)
    envelope = np.exp(-0.5 * (k - k_peak) ** 2 / (0.5 * k_peak) ** 2) * (k > 0) * g.dealias
    w = _irfft2(_rfft2(noise) * envelope.astype(g.k2.dtype), n)
    speed = jnp.sqrt((vorticity_to_velocity(w) ** 2).sum(axis=-3))
    return w * (max_velocity / speed.max(axis=(-2, -1), keepdims=True))


def stable_time_step(max_velocity: float, n: int, cfl: float = 0.5) -> float:
    return cfl * (2 * math.pi / n) / max_velocity
