"""Spectral operators for periodic channels-last vector fields (..., Nx, Ny, 2).

Domain [0, 2pi)^2, integer wavenumbers. Nyquist modes are dropped by the
derivative and projection operators (they break Hermitian symmetry of a real
field). FFTs run in float32 or float64; half-precision inputs are upcast.

The wavenumber grids are constants (computed from the static shapes), and all
functions accept arbitrary leading axes, so they work on single examples (under vmap)
and on batches alike.
"""

import jax
import jax.numpy as jnp
import numpy as np

_SPATIAL_AXES = (-3, -2)


def _k_grid(nx: int, ny: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Wavenumbers with Nyquist set to 0, and the mask that removes Nyquist modes.

    Computed with NumPy from the static shapes, so they are constants under jit.
    """
    kx = np.fft.fftfreq(nx, d=1.0 / nx)
    ky = np.fft.fftfreq(ny, d=1.0 / ny)
    nyq_x, nyq_y = np.abs(kx) == nx // 2, np.abs(ky) == ny // 2
    keep = ~nyq_x[:, None] & ~nyq_y[None, :]
    kx, ky = np.where(nyq_x, 0, kx)[:, None], np.where(nyq_y, 0, ky)[None, :]
    return jnp.asarray(kx, jnp.float32), jnp.asarray(ky, jnp.float32), jnp.asarray(keep)


def _fft(u: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    if u.dtype in (jnp.float16, jnp.bfloat16):
        u = u.astype(jnp.float32)
    kx, ky, keep = _k_grid(*u.shape[-3:-1])
    return jnp.fft.fft2(u, axes=_SPATIAL_AXES), kx, ky, keep


def _ifft(uh: jax.Array) -> jax.Array:
    return jnp.fft.ifft2(uh, axes=_SPATIAL_AXES).real


def _k_dot(uh: jax.Array, kx: jax.Array, ky: jax.Array) -> jax.Array:
    return kx * uh[..., 0] + ky * uh[..., 1]


def divergence(u: jax.Array) -> jax.Array:
    """Spectral divergence, shape (..., Nx, Ny)."""
    uh, kx, ky, _ = _fft(u)
    return jnp.fft.ifft2(1j * _k_dot(uh, kx, ky), axes=(-2, -1)).real


def vorticity(u: jax.Array) -> jax.Array:
    """Spectral vorticity d(u_y)/dx - d(u_x)/dy, shape (..., Nx, Ny)."""
    uh, kx, ky, _ = _fft(u)
    curl = kx * uh[..., 1] - ky * uh[..., 0]
    return jnp.fft.ifft2(1j * curl, axes=(-2, -1)).real


def leray_project(u: jax.Array) -> jax.Array:
    """P u_hat = u_hat - k (k . u_hat) / |k|^2; mean flow (k = 0) is kept."""
    uh, kx, ky, keep = _fft(u)
    uh = uh * keep[..., None]
    k2 = jnp.maximum(kx**2 + ky**2, 1)
    dot = _k_dot(uh, kx, ky) / k2
    ph = jnp.stack((uh[..., 0] - kx * dot, uh[..., 1] - ky * dot), axis=-1)
    return _ifft(ph).astype(u.dtype)


def divergence_energy(u: jax.Array) -> jax.Array:
    """sum_k |k . u_hat|^2 per sample (Parseval: ~ ||div u||_2^2), shape (...)."""
    uh, kx, ky, _ = _fft(u)
    return jnp.square(jnp.abs(_k_dot(uh, kx, ky))).sum(axis=(-2, -1))


def gradient_energy(u: jax.Array) -> jax.Array:
    """sum_k |k|^2 |u_hat|^2 per sample (Parseval: ~ ||grad u||_2^2), shape (...)."""
    uh, kx, ky, _ = _fft(u)
    return ((kx**2 + ky**2)[..., None] * jnp.square(jnp.abs(uh))).sum(axis=(-3, -2, -1))


def relative_divergence(u: jax.Array, eps: float = 1e-12) -> jax.Array:
    """||div u||_2 / ||grad u||_2 per sample, scale-invariant metric, shape (...)."""
    return jnp.sqrt(divergence_energy(u)) / (jnp.sqrt(gradient_energy(u)) + eps)
