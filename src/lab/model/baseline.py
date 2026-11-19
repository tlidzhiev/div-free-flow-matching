from functools import partial

import jax
from flax import linen as nn

from .base import BaseModel
from .utils import get_activation, get_kernel_init


class BaselineModel(BaseModel):
    num_classes: int
    hidden_dim: int = 512

    @nn.compact
    def __call__(self, img: jax.Array, train: bool = False, **kwargs) -> dict[str, jax.Array]:
        """
        Return a dict with the 'logits' of a single input image (without the batch dimension).
        Use 'jax.vmap' to apply the model to a batch.
        """
        x = img.reshape(-1)

        kernel_init = get_kernel_init('relu')
        # people say it can approximate any function...
        dense = partial(
            nn.Dense,
            kernel_init=kernel_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
        )
        x = dense(features=self.hidden_dim)(x)
        x = get_activation('relu')(x)
        x = dense(features=self.hidden_dim)(x)
        x = get_activation('relu')(x)
        x = dense(features=self.num_classes)(x)
        return {'logits': x}
