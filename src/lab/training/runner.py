import logging
from pathlib import Path
from typing import Any, Literal

import clu.metrics as clu_metrics
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from lab.metrics.base import BaseMetric
from lab.model.base import BaseModel
from lab.utils.io import get_root

from .steps import Params

logger = logging.getLogger(__name__)


class BaseRunner:
    """
    Base class of the trainer and the inferencer: a model run on batches on a device.

    Provides the batch preparation, mixed precision, weight loading and metric computation
    they share.
    """

    def __init__(
        self,
        device: jax.Device,
        model: BaseModel,
        metrics: dict[str, dict[str, type[BaseMetric]]],
        device_tensors: list[str],
        use_bfloat16: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        device : jax.Device
            Device for arrays and model.
        model : BaseModel
            Flax model.
        metrics : dict[str, dict[str, type[BaseMetric]]]
            Metrics for training ('train') and inference ('inference').
        device_tensors : list[str]
            Names of the batch entries to move to the device.
        use_bfloat16 : bool, optional
            Enable bfloat16 mixed precision.
        """
        self.device = device
        self.metrics = metrics
        self.device_tensors = device_tensors
        if use_bfloat16:
            # Also for the submodules passed as attributes (the denoiser)
            model = model.with_dtype(jnp.bfloat16)
        self.model = model
        self.amp_dtype = jnp.bfloat16 if use_bfloat16 else None
        if self.amp_dtype is not None:
            logger.info(f'AMP enabled with dtype={self.amp_dtype}')

    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, jax.Array]:
        """
        Keep the 'device_tensors' of the batch (all if empty) and move them to the device.
        """
        if self.device_tensors:
            batch = {key: value for key, value in batch.items() if key in self.device_tensors}
        return jax.device_put(batch, self.device)

    def load_weights(
        self,
        path: Path | str,
        target: Params,
        use_ema: bool = False,
    ) -> Params:
        """
        Load the model (or EMA) parameters from the checkpoint of a step, without the
        optimizer state.

        Parameters
        ----------
        path : Path or str
            Checkpoint directory of a step (e.g. '.../checkpoints/1000'), relative to the
            project root.
        target : Params
            Parameters (or their abstract structure, e.g. from 'jax.eval_shape') to restore
            into.
        use_ema : bool, optional
            Load the EMA parameters instead of the model ones.

        Returns
        -------
        Params
            Restored parameters.
        """
        logger.info(f'Loading {"EMA" if use_ema else "model"} weights from: {path}...')
        checkpoint_path = get_root() / path
        state_path = checkpoint_path / 'state'
        if not state_path.is_dir():
            raise FileNotFoundError(
                f"'{checkpoint_path}' is not a checkpoint directory of a step "
                "(e.g. '.../checkpoints/1000')."
            )
        key = 'ema_params' if use_ema else 'params'

        # The arrays are read straight into their sharding: the one of the target arrays, or
        # 'self.device' for abstract targets (e.g. from 'jax.eval_shape') without it
        default_sharding = jax.sharding.SingleDeviceSharding(self.device)

        def to_abstract(x: Any) -> jax.ShapeDtypeStruct:
            sharding = getattr(x, 'sharding', None) or default_sharding
            return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding)

        abstract = {key: jax.tree.map(to_abstract, target)}

        with ocp.PyTreeCheckpointer() as checkpointer:
            metadata = checkpointer.metadata(state_path).item_metadata
            if metadata is None or metadata.tree.get(key) is None:
                hint = ' The model was likely trained without EMA.' if use_ema else ''
                raise RuntimeError(f"No '{key}' in checkpoint: {checkpoint_path}.{hint}")

            # Partial restore: only the 'key' subtree is read (not the optimizer state)
            restored = checkpointer.restore(
                state_path,
                args=ocp.args.PyTreeRestore(
                    item=abstract,
                    restore_args=ocp.checkpoint_utils.construct_restore_args(abstract),
                    partial_restore=True,
                ),
            )
        return restored[key]

    def make_metrics_cls(
        self,
        loss_names: list[str] | tuple[str, ...],
        part: Literal['train', 'inference'],
        include_grad_norm: bool = False,
    ) -> type[clu_metrics.Collection]:
        """
        Create the metric collection class: the average of each loss, the metrics of the
        partition and, optionally, the average gradient norm.

        Parameters
        ----------
        loss_names : list[str] or tuple[str, ...]
            Names of the losses to average.
        part : {'train', 'inference'}
            Partition whose metrics to include.
        include_grad_norm : bool, optional
            Add the 'grad_norm' average.
        """
        averages = {name: clu_metrics.Average.from_output(name) for name in loss_names}
        grad_norm = (
            {'grad_norm': clu_metrics.Average.from_output('grad_norm')} if include_grad_norm else {}
        )
        return clu_metrics.Collection.create(**averages, **self.metrics[part], **grad_norm)

    def compute_metrics(self, *metrics: clu_metrics.Collection) -> dict[str, float]:
        """
        Compute the values of the metric collections on the host (waits for the device).
        """
        values = jax.device_get([m.compute() for m in metrics])
        return {name: float(value) for v in values for name, value in v.items()}
