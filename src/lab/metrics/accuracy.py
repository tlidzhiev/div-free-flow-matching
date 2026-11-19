from typing import ClassVar, Self

import jax
import jax.numpy as jnp
from flax import struct

from .base import BaseMetric


@struct.dataclass
class AccuracyMetric(BaseMetric):
    """
    Multiclass accuracy.

    Configure it with 'AccuracyMetric.configure(num_classes=...)'.

    Attributes
    ----------
    correct : jax.Array
        Number of correct predictions of each class.
    support : jax.Array
        Number of samples of each class.
    """

    correct: jax.Array
    support: jax.Array

    num_classes: ClassVar[int] = 0

    @classmethod
    def configure(cls, num_classes: int) -> type[Self]:
        """
        Parameters
        ----------
        num_classes : int
            Number of classes.
        """
        if num_classes < 1:
            raise ValueError(f'num_classes must be positive, got {num_classes}.')
        return super().configure(num_classes=num_classes)

    @classmethod
    def empty(cls) -> Self:
        """
        Zero counters for all classes.
        """
        return cls(
            correct=jnp.zeros(cls.num_classes, dtype=jnp.int32),
            support=jnp.zeros(cls.num_classes, dtype=jnp.int32),
        )

    @classmethod
    def from_model_output(
        cls,
        logits: jax.Array,
        label: jax.Array,
        mask: jax.Array | None = None,
        **kwargs,
    ) -> Self:
        """
        Count the correct predictions and the samples of each class in a batch.

        Parameters
        ----------
        logits : jax.Array
            Class scores, shape (batch, num_classes).
        label : jax.Array
            Integer class labels, shape (batch,).
        mask : jax.Array or None, optional
            Valid samples (False for batch padding); all are valid by default.
        """
        if cls.num_classes < 1:
            raise ValueError('Configure the metric with AccuracyMetric.configure(num_classes=...).')
        # The examples outside the mask are the padding of a batch
        is_valid = jnp.ones_like(label, dtype=bool) if mask is None else mask
        is_correct = (logits.argmax(axis=-1) == label) & is_valid
        return cls(
            correct=jnp.bincount(
                label, weights=is_correct.astype(jnp.int32), length=cls.num_classes
            ),
            support=jnp.bincount(label, weights=is_valid.astype(jnp.int32), length=cls.num_classes),
        )

    def merge(self, other: Self) -> Self:
        """
        Sum the counters of two batches.
        """
        return type(self)(
            correct=self.correct + other.correct,
            support=self.support + other.support,
        )

    def compute(self) -> jax.Array:
        """
        Accuracy over all valid samples (0 if there are none).
        """
        return self.correct.sum() / jnp.maximum(self.support.sum(), 1)
