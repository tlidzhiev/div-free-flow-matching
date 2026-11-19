from collections.abc import Callable
from functools import partial

import jax
from flax import linen as nn

_SUPPORTED_ACTIVATIONS = ('relu', 'tanh', 'silu', 'gelu')


def get_activation(activation: str) -> Callable[[jax.Array], jax.Array]:
    """
    Get an activation function by name ('relu', 'tanh', 'silu' or 'gelu').

    Raises
    ------
    ValueError
        If the activation is not supported.
    """
    match activation.lower():
        case 'relu':
            return nn.relu
        case 'tanh':
            return nn.tanh
        case 'silu':
            return nn.silu
        case 'gelu':
            # The exact GELU, as in PyTorch (flax approximates it with tanh by default)
            return partial(nn.gelu, approximate=False)
        case _:
            supported = ', '.join(f"'{name}'" for name in _SUPPORTED_ACTIVATIONS)
            raise ValueError(
                f"Unknown activation type: '{activation}'. Supported types: {supported}."
            )


def get_kernel_init(activation: str) -> nn.initializers.Initializer:
    """
    Get the kernel initializer of layers: Xavier for tanh, Kaiming otherwise.

    Biases are initialized with zeros by flax by default.

    Parameters
    ----------
    activation : str
        Activation name: 'relu', 'tanh', 'silu' or 'gelu'.

    Raises
    ------
    ValueError
        If the activation is not supported.
    """
    activation = activation.lower()
    if activation not in _SUPPORTED_ACTIVATIONS:
        supported = ', '.join(f"'{name}'" for name in _SUPPORTED_ACTIVATIONS)
        raise ValueError(
            f"Unknown activation type for initialization: '{activation}'. Supported types: {supported}."
        )

    if activation == 'tanh':
        gain = 5.0 / 3.0  # Recommended gain for tanh
        return nn.initializers.variance_scaling(gain**2, 'fan_avg', 'normal')
    # Kaiming normal for ReLU (fan_in mode)
    return nn.initializers.variance_scaling(2.0, 'fan_in', 'normal')
