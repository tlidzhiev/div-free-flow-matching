from typing import ClassVar

import jax
import jax.numpy as jnp
import optax
from flax import struct

from .base import BaseLoss


@struct.dataclass
class CrossEntropyLoss(BaseLoss):
    """
    Cross-entropy loss of a single example with an integer label.
    """

    loss_names: ClassVar[tuple[str, ...]] = ('loss',)

    def __call__(
        self,
        logits: jax.Array,
        label: jax.Array,
        **kwargs,
    ) -> dict[str, jax.Array]:
        """
        Compute the cross-entropy loss of an example, returned under the 'loss' key.

        Parameters
        ----------
        logits : jax.Array
            Class scores, shape (num_classes,).
        label : jax.Array
            Integer class label, shape ().
        """
        # The logits can be in bfloat16, the loss is always computed in float32
        logits = logits.astype(jnp.float32)
        return {'loss': optax.softmax_cross_entropy_with_integer_labels(logits, label)}
