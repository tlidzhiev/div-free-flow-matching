import math
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import jax


class StepTimer:
    """
    Measures the time of waiting for the data and the compute time of train steps.

    The times are accumulated over the steps since the last 'reset' and averaged in
    'result'. When disabled, nothing is recorded and the device is not synchronized.
    """

    def __init__(self, enabled: bool) -> None:
        """
        Parameters
        ----------
        enabled : bool
            Measure the times (the step waits for its outputs to be computed).
        """
        self.enabled = enabled
        self._step_start = 0.0
        self._data_time = 0.0
        self.reset()

    def reset(self) -> None:
        """
        Drop the accumulated times.
        """
        self._total_data_time = 0.0
        self._total_step_time = 0.0
        self._num_steps = 0

    def start_step(self) -> None:
        """
        Start the step.
        """
        self._step_start = time.perf_counter()
        self._data_time = 0.0

    @contextmanager
    def data(self) -> Generator[None]:
        """
        Measure waiting for a batch, adding it to the data time of the step.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self._data_time += time.perf_counter() - start

    def end_step(self, outputs: Any) -> None:
        """
        Finish the step: wait for its (asynchronous) 'outputs' and accumulate the times.

        Parameters
        ----------
        outputs : Any
            Pytree of the step outputs to wait for.
        """
        if not self.enabled:
            return
        jax.block_until_ready(outputs)
        self._total_data_time += self._data_time
        self._total_step_time += time.perf_counter() - self._step_start - self._data_time
        self._num_steps += 1

    def result(self) -> dict[str, float]:
        """
        Average data and step time per step and the share of the data time (empty when
        disabled or without steps).
        """
        if not self.enabled or self._num_steps == 0:
            return {}
        total = self._total_data_time + self._total_step_time
        return {
            'data_time': self._total_data_time / self._num_steps,
            'step_time': self._total_step_time / self._num_steps,
            'data_wait_ratio': self._total_data_time / total if total > 0 else 0.0,
        }


@dataclass
class EarlyStopping:
    monitor: str = 'off'  # 'off' or '<min|max> <metric_name>'
    patience: int | None = None  # evaluations without improvement before stopping
    best: float = field(init=False)
    not_improved_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.mode, self.metric = ('off', '') if self.monitor == 'off' else self.monitor.split()
        if self.mode not in ('off', 'min', 'max'):
            raise ValueError(
                f"monitor must be 'off' or '<min|max> <metric_name>': {self.monitor!r}"
            )
        if self.patience is not None:
            if self.patience < 1:
                raise ValueError(f'patience must be positive, got {self.patience}.')
            if self.mode == 'off':
                raise ValueError("Early stopping requires monitor to be set (not 'off').")
        self.best = -math.inf if self.mode == 'max' else math.inf

    @property
    def should_stop(self) -> bool:
        return self.patience is not None and self.not_improved_count >= self.patience

    def update(self, logs: dict[str, float]) -> bool:
        if self.mode == 'off':
            return False
        # A KeyError on a typo in the metric name
        value = float(logs[self.metric])
        improved = value < self.best if self.mode == 'min' else value > self.best
        if improved:
            self.best = value
            self.not_improved_count = 0
        else:
            self.not_improved_count += 1
        return improved

    def state_dict(self) -> dict[str, Any]:
        # JSON has no infinity: None before the first evaluation
        best = self.best if math.isfinite(self.best) else None
        return {'best': best, 'not_improved_count': self.not_improved_count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self.mode != 'off' and state.get('best') is not None:
            self.best = float(state['best'])
            self.not_improved_count = int(state.get('not_improved_count', 0))
