from typing import ClassVar

import jax
from flax import struct


@struct.dataclass
class BaseLoss:
    """
    Base class for all loss functions.

    A loss takes a single example (without the batch axis), like the model: the train steps
    vmap it over the batch and average the losses over the valid examples ('mask').

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
