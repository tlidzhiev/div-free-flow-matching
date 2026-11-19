from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import struct
from jax.typing import DTypeLike

from .physics import leray_project

type VelocityFn = Callable[..., jax.Array]


def sample_noise(
    key: jax.Array,
    shape: tuple[int, ...],
    dtype: DTypeLike = jnp.float32,
    hard_div_free: bool = False,
) -> jax.Array:
    noise = jax.random.normal(key, shape, dtype=dtype)
    return leray_project(noise) if hard_div_free else noise


def sample_t(
    key: jax.Array,
    batch_size: int,
    t_dist: str = 'logit_normal',
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    dtype: DTypeLike = jnp.float32,
) -> jax.Array:
    if t_dist == 'uniform':
        return jax.random.uniform(key, (batch_size,), dtype=dtype)
    if t_dist == 'logit_normal':
        z = jax.random.normal(key, (batch_size,), dtype=dtype) * logit_std + logit_mean
        return jax.nn.sigmoid(z)
    raise ValueError(f"t_dist must be 'uniform' or 'logit_normal', got {t_dist!r}")


def _expand(t: jax.Array) -> jax.Array:
    return t.reshape(-1, 1, 1, 1)


def interpolate(x0: jax.Array, x1: jax.Array, t: jax.Array) -> jax.Array:
    t = _expand(t)
    return (1 - t) * x0 + t * x1


def x1_hat(x_t: jax.Array, velocity: jax.Array, t: jax.Array) -> jax.Array:
    return x_t + (1 - _expand(t)) * velocity


@struct.dataclass
class FlowMatchingPath:
    """
    Processing of a data batch into a flow matching problem: the noise, the times, the
    condition dropout and the point of the linear path from the noise x0 to the data x1.

    Attributes
    ----------
    t_dist : str
        Distribution of t: 'uniform' or 'logit_normal'.
    logit_mean, logit_std : float
        Mean and std of logit(t) for 'logit_normal'.
    p_uncond : float
        Probability to drop the condition (classifier-free guidance).
    hard_div_free : bool
        Sample divergence-free (Leray-projected) noise.
    """

    t_dist: str = struct.field(pytree_node=False, default='logit_normal')
    logit_mean: float = struct.field(pytree_node=False, default=0.0)
    logit_std: float = struct.field(pytree_node=False, default=1.0)
    p_uncond: float = struct.field(pytree_node=False, default=0.1)
    hard_div_free: bool = struct.field(pytree_node=False, default=False)

    def __post_init__(self) -> None:
        if self.t_dist not in ('uniform', 'logit_normal'):
            raise ValueError(f"t_dist must be 'uniform' or 'logit_normal', got {self.t_dist!r}")
        if self.logit_std <= 0:
            raise ValueError(f'logit_std must be positive, got {self.logit_std}')
        if not 0.0 <= self.p_uncond <= 1.0:
            raise ValueError(f'p_uncond must be in [0, 1], got {self.p_uncond}')

    def __call__(
        self, rng: jax.Array, x: jax.Array, cond: jax.Array, **kwargs
    ) -> dict[str, jax.Array]:
        """
        Parameters
        ----------
        rng : jax.Array
            PRNG key of the noise, the times and the condition dropout.
        x : jax.Array
            Data fields x1, shape (B, X, Y, 2).
        cond : jax.Array
            Conditions, shape (B,).

        Returns
        -------
        dict[str, jax.Array]
            'x_t' (B, X, Y, 2), 't' (B,), 'cond' (B,), 'cond_mask' (B,) (False = dropped):
            the inputs of the model; 'x1' and the target velocity 'target' = x1 - x0.
        """
        x1 = x.astype(jnp.float32)
        batch_size = x1.shape[0]
        noise_key, t_key, mask_key = jax.random.split(rng, 3)
        x0 = sample_noise(noise_key, x1.shape, x1.dtype, hard_div_free=self.hard_div_free)
        t = sample_t(
            t_key,
            batch_size,
            t_dist=self.t_dist,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            dtype=x1.dtype,
        )
        cond_mask = jax.random.uniform(mask_key, (batch_size,)) >= self.p_uncond
        return {
            'x_t': interpolate(x0, x1, t),
            't': t,
            'cond': cond,
            'cond_mask': cond_mask,
            'x1': x1,
            'target': x1 - x0,
        }


def sample(
    velocity_fn: VelocityFn,
    key: jax.Array,
    cond: jax.Array,
    shape: tuple[int, ...],
    steps: int = 32,
    hard_div_free: bool = False,
) -> jax.Array:
    x = sample_noise(key, shape, jnp.float32, hard_div_free=hard_div_free)
    ts = jnp.linspace(0.0, 1.0, steps + 1, dtype=jnp.float32)

    def euler_step(i: jax.Array, x: jax.Array) -> jax.Array:
        t0, t1 = ts[i], ts[i + 1]
        v = velocity_fn(x, jnp.full((shape[0],), t0), cond)
        return x + (t1 - t0) * v.astype(jnp.float32)

    return jax.lax.fori_loop(0, steps, euler_step, x)
