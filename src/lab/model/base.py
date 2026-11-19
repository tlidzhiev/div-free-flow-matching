from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.typing import DTypeLike


class BaseModel(nn.Module, kw_only=True):
    """
    Base class for all models.

    Mixed precision (AMP) is the 'dtype' attribute of a model, passed to its layers as
    'dtype=self.dtype' together with 'param_dtype=self.param_dtype': the layers compute in 'dtype'
    while the parameters stay in 'param_dtype' (float32 by default). All layers, including the
    last one, compute in 'dtype', so the outputs are in 'dtype' too; the loss should upcast them
    to float32 itself.

    A model takes a single example (without the batch axis): the train steps vmap it over the
    batch. Variables other than 'params' (e.g. 'batch_stats' of BatchNorm) are not supported.
    """

    dtype: DTypeLike | None = None  # compute dtype (AMP)
    param_dtype: DTypeLike = jnp.float32  # parameter dtype

    def summary(self, variables: dict[str, Any]) -> str:
        """
        Model description with the number of parameters.
        """

        def format_number(num: int) -> str:
            if abs(num) >= 1_000_000:
                return f'{num / 1_000_000:.2f}M'
            elif abs(num) >= 1_000:
                return f'{num / 1_000:.2f}K'
            else:
                return str(num)

        def count(tree: Any) -> int:
            return sum(x.size for x in jax.tree.leaves(tree))

        all_parameters = count(variables)
        trainable_parameters = count(variables.get('params', {}))

        result_info = repr(self)
        result_info += f'\nAll parameters: {format_number(all_parameters)}'
        result_info += f'\nTrainable parameters: {format_number(trainable_parameters)}'
        return result_info
