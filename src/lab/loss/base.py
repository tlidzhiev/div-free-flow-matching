from typing import ClassVar

import jax
from flax import struct


@struct.dataclass
class BaseLoss:
    """
    Base class for all loss functions.

    A loss takes a batch: the model outputs and the processed batch (see 'batch_processor'
    of the trainer), and returns the scalar losses of the batch. It only computes the losses:
    the randomness of the problem (e.g. noise, times) is sampled by the batch processor.

    Attributes
    ----------
    loss_names : tuple[str, ...]
        Names of all loss components.
    """

    loss_names: ClassVar[tuple[str, ...]]

    def __call__(self, **kwargs) -> dict[str, jax.Array]:
        """
        Compute the scalar losses of an example; the sum of all losses is returned under 'loss'.
        """
        raise NotImplementedError(f'{type(self).__name__} must implement __call__ method.')
