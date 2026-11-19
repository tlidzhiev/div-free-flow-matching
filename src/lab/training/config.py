from __future__ import annotations

import logging
from dataclasses import dataclass, field

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

MATMUL_PRECISIONS = ('default', 'high', 'highest')


@dataclass(kw_only=True)
class EMAConfig:
    decay: float = 0.999
    update_every: int = 1  # optimizer steps between EMA updates
    warmup_steps: int = 0  # steps with linearly increasing decay (0 = disabled)

    def __post_init__(self) -> None:
        if not 0.0 < self.decay < 1.0:
            raise ValueError(f'trainer.ema.decay must be in (0, 1), got {self.decay}.')
        if self.update_every < 1:
            raise ValueError(f'trainer.ema.update_every must be >= 1, got {self.update_every}.')
        if self.warmup_steps < 0:
            raise ValueError(f'trainer.ema.warmup_steps must be >= 0, got {self.warmup_steps}.')


@dataclass(kw_only=True)
class TrainerConfig:
    """
    The 'trainer' section of the config, with defaults and validation in one place.

    Unknown keys raise a TypeError, so typos in the config are not silently ignored; add a
    field here for every new option.
    """

    # Steps
    total_steps: int
    eval_every_steps: int
    checkpoint_every_steps: int
    grad_accum_steps: int = 1

    # Logging
    log_step: int = 50
    log_images: int = 16
    sample_steps: int = 32  # Euler steps of the ODE sampler for the logged generated fields
    debug: bool = False

    # Checkpoints
    checkpoint_dir: str
    resume_from: str | None = None
    from_pretrained: str | None = None
    from_pretrained_ema: bool = False
    only_best: bool = False
    monitor: str = 'off'  # "off" or "min/max metric_name"
    early_stop: int | None = None

    # Hardware and numerics
    device: str = 'auto'
    device_tensors: list[str] = field(default_factory=list)
    use_bfloat16: bool = False
    # Precision of the float32 matmuls and convolutions: 'default', 'high' or 'highest'
    matmul_precision: str = 'default'
    # The evaluation and the logged samples (their own dtype and precision)
    eval_use_bfloat16: bool = False
    eval_matmul_precision: str = 'highest'
    jit: bool = True
    max_grad_norm: float | None = None

    # Run
    seed: int = 42
    eval_seed: int = 0  # noise and times of the evaluation and the logged samples
    override: bool = False
    ema: EMAConfig | None = None

    def __post_init__(self) -> None:
        for name in ('total_steps', 'eval_every_steps', 'checkpoint_every_steps', 'log_step'):
            if getattr(self, name) < 1:
                raise ValueError(f'trainer.{name} must be positive, got {getattr(self, name)}.')
        if self.checkpoint_every_steps % self.eval_every_steps != 0:
            raise ValueError(
                'trainer.checkpoint_every_steps must be a multiple of '
                'trainer.eval_every_steps, '
                f'got {self.checkpoint_every_steps} and {self.eval_every_steps}.'
            )
        if self.sample_steps < 1:
            raise ValueError(f'trainer.sample_steps must be positive, got {self.sample_steps}.')
        if self.log_images < 0:
            raise ValueError(f'trainer.log_images must be >= 0, got {self.log_images}.')
        if self.grad_accum_steps < 1:
            raise ValueError(
                f'trainer.grad_accum_steps must be positive, got {self.grad_accum_steps}.'
            )
        for name in ('matmul_precision', 'eval_matmul_precision'):
            if getattr(self, name) not in MATMUL_PRECISIONS:
                raise ValueError(
                    f'trainer.{name} must be one of {MATMUL_PRECISIONS}, got {getattr(self, name)!r}.'
                )
        if self.use_bfloat16 and self.matmul_precision != 'default':
            logger.warning(
                f'trainer.matmul_precision={self.matmul_precision!r} has no effect on the '
                'bfloat16 computations of trainer.use_bfloat16=true.'
            )
        if self.eval_use_bfloat16 and self.eval_matmul_precision != 'default':
            logger.warning(
                f'trainer.eval_matmul_precision={self.eval_matmul_precision!r} has no effect on '
                'the bfloat16 computations of trainer.eval_use_bfloat16=true.'
            )
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError(f'trainer.max_grad_norm must be positive, got {self.max_grad_norm}.')
        if not self.checkpoint_dir:
            raise ValueError('trainer.checkpoint_dir must not be empty.')
        if self.monitor != 'off':
            parts = self.monitor.split()
            if len(parts) != 2 or parts[0] not in ('min', 'max'):
                raise ValueError(
                    f"trainer.monitor must be 'off' or '<min|max> <metric_name>', "
                    f'got {self.monitor!r}.'
                )
        if self.early_stop is not None:
            if self.early_stop < 1:
                raise ValueError(f'trainer.early_stop must be positive, got {self.early_stop}.')
            if self.monitor == 'off':
                raise ValueError(
                    "trainer.early_stop requires trainer.monitor to be set (not 'off')."
                )
        if self.only_best and self.monitor == 'off':
            raise ValueError("trainer.only_best requires trainer.monitor to be set (not 'off').")
        if self.resume_from is not None and self.override:
            raise ValueError(
                'trainer.resume_from and trainer.override are mutually exclusive: '
                'resuming keeps the run directory, overriding deletes it.'
            )
        if self.resume_from is not None and self.from_pretrained:
            raise ValueError(
                'trainer.resume_from and trainer.from_pretrained are mutually exclusive.'
            )
        for name in ('eval_every_steps', 'checkpoint_every_steps'):
            if getattr(self, name) > self.total_steps:
                logger.warning(
                    f'trainer.{name} ({getattr(self, name)}) > trainer.total_steps '
                    f'({self.total_steps}): this will never happen during training.'
                )

    @classmethod
    def from_config(cls, cfg: DictConfig) -> TrainerConfig:
        """
        Build from the resolved 'trainer' section of the experiment config.
        """
        data = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(data, dict)
        ema = data.pop('ema', None)
        return cls(**data, ema=EMAConfig(**ema) if ema is not None else None)  # ty:ignore[invalid-argument-type]
