from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict
from flax.training import train_state

from .config import EMAConfig


class TrainState(train_state.TrainState):
    """
    Flax train state with a PRNG key and optional EMA parameters.

    The EMA is updated inside 'apply_gradients' every 'ema_update_every' steps.
    """

    rng: jax.Array
    ema_params: FrozenDict[str, Any] | None = None
    ema_decay: float = struct.field(pytree_node=False, default=0.0)
    ema_update_every: int = struct.field(pytree_node=False, default=1)
    ema_warmup_steps: int = struct.field(pytree_node=False, default=0)

    @classmethod
    def create(
        cls,
        apply_fn: Any,
        params: FrozenDict[str, Any],
        tx: optax.GradientTransformation | optax.GradientTransformationExtraArgs,
        rng: jax.Array | None = None,
        ema: EMAConfig | None = None,
        **kwargs,
    ) -> TrainState:
        """
        Create the state with the optimizer state initialized for 'params'.

        Parameters
        ----------
        apply_fn : Any
            Model apply function.
        params : FrozenDict[str, Any]
            Model parameters.
        tx : optax.GradientTransformation or optax.GradientTransformationExtraArgs
            Optimizer.
        rng : jax.Array or None, optional
            PRNG key for training randomness (a fixed key if None).
        ema : EMAConfig or None, optional
            EMA settings; the EMA is disabled if None.
        """
        ema_params = None
        if ema:
            ema_params = jax.tree.map(lambda p: jnp.array(p, dtype=jnp.float32, copy=True), params)

        state = super().create(
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            rng=jax.random.PRNGKey(0) if rng is None else rng,
            ema_params=ema_params,
            ema_decay=ema.decay if ema else 0.0,
            ema_update_every=ema.update_every if ema else 1,
            ema_warmup_steps=ema.warmup_steps if ema else 0,
            **kwargs,
        )
        return state.replace(step=jnp.asarray(state.step, dtype=jnp.int32))

    def current_ema_decay(self, step: jax.Array) -> jax.Array:
        """
        EMA decay with linear warmup over the first 'ema_warmup_steps' optimizer steps.

        At the first EMA update (step == ema_update_every), decay is 0 so EMA copies the
        current parameters directly.
        """
        num_updates = step - self.ema_update_every
        decay = jnp.asarray(self.ema_decay, dtype=jnp.float32)
        if self.ema_warmup_steps > 0:
            decay = jnp.where(
                num_updates < self.ema_warmup_steps,
                self.ema_decay * num_updates / self.ema_warmup_steps,
                decay,
            )
        return jnp.where(num_updates == 0, 0.0, decay)

    def apply_gradients(self, *, grads: FrozenDict[str, Any], **kwargs) -> TrainState:
        """
        Do an optimizer step and, with EMA enabled, update the EMA parameters.
        """
        new_state = super().apply_gradients(grads=grads, **kwargs)
        if self.ema_params is None:
            return new_state

        current_ema = self.ema_params

        def update_ema() -> FrozenDict[str, Any]:
            decay = self.current_ema_decay(new_state.step)
            return cast(
                FrozenDict[str, Any],
                optax.incremental_update(new_state.params, current_ema, step_size=1.0 - decay),
            )

        if self.ema_update_every == 1:
            ema_params = update_ema()
        else:
            ema_params = jax.lax.cond(
                new_state.step % self.ema_update_every == 0,
                update_ema,
                lambda: current_ema,
            )
        return new_state.replace(ema_params=ema_params)
