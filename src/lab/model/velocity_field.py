from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn

from .base import BaseModel
from .physics import leray_project


class VelocityField(BaseModel):
    """
    Velocity field of the flow: the denoiser output in float32, Leray-projected onto the
    divergence-free fields with 'hard_div_free'.
    """

    denoiser: BaseModel
    hard_div_free: bool = False

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        t: jax.Array,
        cond: jax.Array,
        cond_mask: jax.Array | None = None,
        train: bool = False,
    ) -> jax.Array:
        v = self.denoiser(x, t, cond, cond_mask, train=train).astype(jnp.float32)
        return leray_project(v) if self.hard_div_free else v

    def init_inputs(self, example: dict[str, Any]) -> dict[str, Any]:
        return {'x': example['x'], 't': jnp.zeros((), jnp.float32), 'cond': example['cond']}
