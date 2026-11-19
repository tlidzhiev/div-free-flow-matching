from typing import ClassVar

import jax
import jax.numpy as jnp
from flax import struct

from lab.model.flow import x1_hat
from lab.model.physics.spectral import divergence_energy, gradient_energy, relative_divergence

from .base import BaseLoss


@struct.dataclass
class FlowMatchingLoss(BaseLoss):
    """
    Conditional flow matching loss with an optional soft divergence penalty.

    Attributes
    ----------
    div_weight : float
        Weight of the soft divergence penalty on x1_hat (0 = disabled).
    div_t_power : float
        The penalty of a sample is weighted by t ** div_t_power.
    """

    div_weight: float = struct.field(pytree_node=False, default=0.0)
    div_t_power: float = struct.field(pytree_node=False, default=2.0)

    loss_names: ClassVar[tuple[str, ...]] = ('loss', 'fm', 'div_penalty', 'rel_div_x1hat_late')

    def __post_init__(self) -> None:
        if self.div_weight < 0:
            raise ValueError(f'div_weight must be >= 0, got {self.div_weight}')

    def __call__(
        self,
        velocity: jax.Array,
        target: jax.Array,
        x_t: jax.Array,
        t: jax.Array,
        x1: jax.Array,
        **kwargs,
    ) -> dict[str, jax.Array]:
        """
        Parameters
        ----------
        velocity : jax.Array
            Predicted velocity at (x_t, t), shape (B, X, Y, 2).
        target : jax.Array
            Target velocity x1 - x0, shape (B, X, Y, 2).
        x_t : jax.Array
            Points of the path the velocity was predicted at, shape (B, X, Y, 2).
        t : jax.Array
            Times, shape (B,).
        x1 : jax.Array
            Data fields, shape (B, X, Y, 2); the scale of the divergence penalty.
        """
        velocity = velocity.astype(jnp.float32)
        fm = jnp.square(velocity - target).mean()
        estimate = x1_hat(x_t, velocity, t)

        if self.div_weight > 0:
            weights = t**self.div_t_power
            ref = jax.lax.stop_gradient(
                jnp.maximum(gradient_energy(x1).mean(), jnp.finfo(x1.dtype).eps)
            )
            div_penalty = (weights * divergence_energy(estimate)).mean() / ref
            loss = fm + self.div_weight * div_penalty
        else:
            div_penalty = jnp.zeros((), fm.dtype)
            loss = fm

        rel_div = relative_divergence(jax.lax.stop_gradient(estimate))
        late = (t > 0.5).astype(rel_div.dtype)
        rel_div_late = (rel_div * late).sum() / jnp.maximum(late.sum(), 1)

        return {
            'loss': loss,
            'fm': fm,
            'div_penalty': div_penalty,
            'rel_div_x1hat_late': rel_div_late,
        }
