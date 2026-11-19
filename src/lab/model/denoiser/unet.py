import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.typing import DTypeLike

from ..base import BaseModel

_KERNEL_INIT = nn.initializers.kaiming_normal()


def conv3(
    features: int,
    dtypes: dict[str, Any],
    stride: int = 1,
    zero: bool = False,
    name: str | None = None,
) -> nn.Conv:
    return nn.Conv(
        features,
        (3, 3),
        strides=stride,
        padding='CIRCULAR',
        kernel_init=nn.initializers.zeros if zero else _KERNEL_INIT,
        name=name,
        **dtypes,
    )


def dense(features: int, dtypes: dict[str, Any], zero: bool = False, **kwargs) -> nn.Dense:
    return nn.Dense(
        features,
        kernel_init=nn.initializers.zeros if zero else _KERNEL_INIT,
        **dtypes,
        **kwargs,
    )


def group_norm(
    channels: int,
    dtypes: dict[str, Any],
    affine: bool = True,
    name: str | None = None,
) -> nn.GroupNorm:
    return nn.GroupNorm(
        num_groups=min(32, channels),
        epsilon=1e-6,
        use_scale=affine,
        use_bias=affine,
        reduction_axes=(-3, -2, -1),
        name=name,
        **dtypes,
    )


class _Layer(nn.Module, kw_only=True):
    """
    Base of the UNet layers: the compute and parameter dtypes of the model.
    """

    dtype: DTypeLike | None = None
    param_dtype: DTypeLike = jnp.float32

    @property
    def dtypes(self) -> dict[str, Any]:
        return {'dtype': self.dtype, 'param_dtype': self.param_dtype}


def sinusoidal(s: jax.Array, dim: int, min_freq: float, max_freq: float) -> jax.Array:
    freqs = np.logspace(math.log10(min_freq), math.log10(max_freq), dim // 2, dtype=np.float32)
    args = jnp.asarray(s, dtype=jnp.float32)[..., None] * freqs
    return jnp.concatenate((jnp.sin(args), jnp.cos(args)), axis=-1)


class Embedding(_Layer):
    dim: int

    @nn.compact
    def __call__(self, t: jax.Array, c: jax.Array, cond_mask: jax.Array | None) -> jax.Array:
        def mlp(x: jax.Array, name: str) -> jax.Array:
            x = dense(self.dim, self.dtypes, name=f'{name}_0')(x)
            return dense(self.dim, self.dtypes, name=f'{name}_1')(nn.silu(x))

        null_cond = self.param(
            'null_cond', nn.initializers.normal(stddev=0.8), (self.dim,), self.param_dtype
        )
        t_emb = mlp(sinusoidal(t, self.dim, 0.1, 1000.0), 't_mlp')
        c = jnp.asarray(c, dtype=jnp.float32)
        c_emb = mlp(
            jnp.concatenate((c[..., None], sinusoidal(c, self.dim, 0.5, 16.0)), axis=-1), 'c_mlp'
        )
        if cond_mask is not None:
            c_emb = jnp.where(cond_mask, c_emb, null_cond.astype(c_emb.dtype))
        return t_emb + c_emb


class ResBlock(_Layer):
    out_channels: int
    dropout: float

    @nn.compact
    def __call__(self, x: jax.Array, emb: jax.Array, train: bool = False) -> jax.Array:
        h = conv3(self.out_channels, self.dtypes, name='conv1')(
            nn.silu(group_norm(x.shape[-1], self.dtypes, name='norm1')(x))
        )
        scale, shift = jnp.split(
            dense(2 * self.out_channels, self.dtypes, zero=True, name='emb')(nn.silu(emb)),
            2,
            axis=-1,
        )
        h = (
            group_norm(self.out_channels, self.dtypes, affine=False, name='norm2')(h) * (1 + scale)
            + shift
        )
        h = nn.Dropout(self.dropout, deterministic=not train)(nn.silu(h))
        h = conv3(self.out_channels, self.dtypes, zero=True, name='conv2')(h)
        if x.shape[-1] != self.out_channels:
            x = nn.Conv(
                self.out_channels, (1, 1), kernel_init=_KERNEL_INIT, name='skip', **self.dtypes
            )(x)
        return x + h


class SelfAttention(_Layer):
    num_heads: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        nx, ny, ch = x.shape[-3:]
        if ch % self.num_heads:
            raise ValueError(f'channels ({ch}) must be divisible by num_heads ({self.num_heads})')
        h = group_norm(ch, self.dtypes, name='norm')(x).reshape(nx * ny, ch)
        qkv = dense(3 * ch, self.dtypes, use_bias=False, name='qkv')(h)
        q, k, v = jnp.moveaxis(qkv.reshape(nx * ny, 3, self.num_heads, -1), 1, 0)
        h = jax.nn.dot_product_attention(q, k, v).reshape(nx * ny, ch)
        return x + dense(ch, self.dtypes, zero=True, name='proj')(h).reshape(nx, ny, ch)


class Block(_Layer):
    out_channels: int
    dropout: float
    attention: bool
    num_heads: int

    @nn.compact
    def __call__(self, x: jax.Array, emb: jax.Array, train: bool = False) -> jax.Array:
        h = ResBlock(
            out_channels=self.out_channels, dropout=self.dropout, name='res', **self.dtypes
        )(x, emb, train)
        if self.attention:
            h = SelfAttention(num_heads=self.num_heads, name='attn', **self.dtypes)(h)
        return h


class Upsample(_Layer):
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.repeat(jnp.repeat(x, 2, axis=-3), 2, axis=-2)
        return conv3(x.shape[-1], self.dtypes, name='conv')(x)


class UNet(BaseModel):
    """
    UNet denoiser of periodic 2D fields, conditioned on the time and a scalar condition.
    """

    in_channels: int = 2
    out_channels: int = 2
    resolution: int = 64
    base_features: int = 64
    channel_mult: tuple[int, ...] = (1, 2, 2, 4)
    num_res_blocks: int = 2
    attn_resolutions: tuple[int, ...] = (16, 8)
    num_heads: int = 4
    embed_dim: int = 256
    dropout: float = 0.0
    # Recompute the activations of the blocks in the backward pass instead of storing them:
    # several times less memory for about 20% more compute
    remat: bool = False

    def __post_init__(self) -> None:
        # Lists from the config are not hashable (jit hashes the model via its apply method)
        object.__setattr__(self, 'channel_mult', tuple(self.channel_mult))
        object.__setattr__(self, 'attn_resolutions', tuple(self.attn_resolutions))
        widths = [self.base_features * m for m in self.channel_mult]
        if self.resolution % 2 ** (len(widths) - 1):
            raise ValueError(
                f'resolution {self.resolution} must be divisible by {2 ** (len(widths) - 1)}'
            )
        super().__post_init__()

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        t: jax.Array,
        c: jax.Array,
        cond_mask: jax.Array | None = None,
        train: bool = False,
    ) -> jax.Array:
        dtypes: dict[str, Any] = {'dtype': self.dtype, 'param_dtype': self.param_dtype}
        widths = [self.base_features * m for m in self.channel_mult]
        # 'train' (argument 3, counting self) is a Python bool: static for the recomputation
        block_cls = nn.remat(Block, static_argnums=(3,)) if self.remat else Block

        def block(width: int, res: int, name: str, attention: bool | None = None) -> Block:
            return block_cls(
                out_channels=width,
                dropout=self.dropout,
                attention=res in self.attn_resolutions if attention is None else attention,
                num_heads=self.num_heads,
                name=name,
                **dtypes,
            )

        emb = Embedding(dim=self.embed_dim, name='embedding', **dtypes)(t, c, cond_mask)
        h = conv3(widths[0], dtypes, name='input')(x)

        skips, res, i = [h], self.resolution, 0
        for level, width in enumerate(widths):
            for _ in range(self.num_res_blocks):
                h = block(width, res, f'down_{i}')(h, emb, train)
                skips.append(h)
                i += 1
            if level < len(widths) - 1:
                h = conv3(h.shape[-1], dtypes, stride=2, name=f'down_{i}')(h)  # downsample
                res //= 2
                skips.append(h)
                i += 1

        h = block(h.shape[-1], res, 'mid_0', attention=True)(h, emb, train)
        h = block(h.shape[-1], res, 'mid_1', attention=False)(h, emb, train)

        i = 0
        for level, width in reversed(list(enumerate(widths))):
            for _ in range(self.num_res_blocks + 1):
                h = jnp.concatenate((h, skips.pop()), axis=-1)
                h = block(width, res, f'up_{i}')(h, emb, train)
                i += 1
            if level > 0:
                h = Upsample(name=f'up_{i}', **dtypes)(h)
                res *= 2
                i += 1

        h = nn.silu(group_norm(widths[0], dtypes, name='out_norm')(h))
        return conv3(self.out_channels, dtypes, zero=True, name='output')(h)
