import logging
from pathlib import Path
from typing import Any

import grain
import jax
import numpy as np
from omegaconf import DictConfig
from tqdm.auto import tqdm

from lab.dataset.utils import get_dataloaders
from lab.metrics.base import BaseMetric
from lab.model.base import BaseModel
from lab.utils.io import get_root

from .runner import BaseRunner
from .steps import Params, make_inference_step, with_matmul_precision

logger = logging.getLogger(__name__)


class Inferencer(BaseRunner):
    """
    Generates fields for the conditions of the dataset with the trained flow model, evaluates
    them and saves them to disk, without optimizers and writers.
    """

    def __init__(
        self,
        config: DictConfig,
        device: jax.Device,
        model: BaseModel,
        datasets: dict[str, grain.sources.RandomAccessDataSource],
        metrics: dict[str, dict[str, type[BaseMetric]]],
        save_path: Path | str,
        params: Params | None = None,
        skip_model_load: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        config : DictConfig
            Experiment config with the 'inferencer' section.
        device : jax.Device
            Device for computation.
        model : BaseModel
            Flax model.
        datasets : dict[str, grain.sources.RandomAccessDataSource]
            Dataset of each partition.
        metrics : dict[str, dict[str, type[BaseMetric]]]
            Metric classes for inference ('inference').
        save_path : Path or str
            Directory to save the generated fields to ('{save_path}/{partition}/').
        params : Params or None, optional
            Preloaded model parameters (loaded from 'from_pretrained' if None).
        skip_model_load : bool, optional
            Skip loading model weights from the checkpoint (the parameters are randomly
            initialized instead).
        """
        self.config = config
        self.cfg_inferencer = config.inferencer
        if (
            not skip_model_load
            and params is None
            and self.cfg_inferencer.get('from_pretrained') is None
        ):
            raise ValueError('Provide the checkpoint to load: inferencer.from_pretrained.')

        super().__init__(
            device=device,
            model=model,
            metrics=metrics,
            device_tensors=list(self.cfg_inferencer.device_tensors),
            use_bfloat16=self.cfg_inferencer.get('use_bfloat16', False),
        )

        dataloaders = get_dataloaders(config, datasets)
        # The model works with normalized fields: the saved ones are in physical units
        self.u_scales = {part: getattr(ds, 'u_scale', 1.0) for part, ds in datasets.items()}
        self.eval_dataloaders = {k: v for k, v in dataloaders.items() if k != 'train'}
        self.save_path = get_root() / save_path

        seed = self.cfg_inferencer.get('seed', 42)
        init_key, self.key = jax.random.split(jax.random.PRNGKey(seed))

        self.metrics_cls = self.make_metrics_cls(('sample_rel_div',), 'inference')
        self.inference_step_fn = with_matmul_precision(
            fn=make_inference_step(
                apply_fn=self.model.apply,
                metrics_cls=self.metrics_cls,
                steps=self.cfg_inferencer.get('sample_steps', 32),
                hard_div_free=getattr(self.model, 'hard_div_free', False),
                jit=self.cfg_inferencer.get('jit', True),
            ),
            precision=self.cfg_inferencer.get('matmul_precision', 'highest'),
        )

        if params is not None:
            self.params = params
        else:
            example = self.model.init_inputs(next(iter(datasets.values()))[0])
            if skip_model_load:
                self.params = self.model.init(init_key, **example, train=False)['params']
            else:
                # The structure of the parameters, without initializing them
                abstract_variables = jax.eval_shape(
                    lambda key: self.model.init(key, **example, train=False), init_key
                )
                self.params = self.load_weights(
                    self.cfg_inferencer.from_pretrained,
                    target=abstract_variables['params'],
                    use_ema=self.cfg_inferencer.get('use_ema', False),
                )

    def run_inference(self) -> dict[str, dict[str, float]]:
        """
        Run inference on each partition.

        Returns
        -------
        dict[str, dict[str, float]]
            Metrics of each partition.
        """
        part_logs: dict[str, dict[str, float]] = {}
        for index, (part, dataloader) in enumerate(self.eval_dataloaders.items()):
            part_key = jax.random.fold_in(self.key, index)
            part_logs[part] = self._inference_part(part, dataloader, part_key)
        return part_logs

    def _inference_part(
        self, part: str, dataloader: grain.IterDataset, key: jax.Array
    ) -> dict[str, float]:
        """
        Generate the fields of a partition and save them.

        Returns
        -------
        dict[str, float]
            Metrics of the partition.
        """
        save_dir = self.save_path / part
        save_dir.mkdir(exist_ok=True, parents=True)

        part_metrics = self.metrics_cls.empty()
        sample_offset = 0  # index of the next saved sample in the partition
        for batch_index, batch in enumerate(tqdm(dataloader, desc=part)):
            device_batch = self.prepare_batch(batch)
            rng = jax.random.fold_in(key, batch_index)
            part_metrics, outputs = self.inference_step_fn(
                self.params, part_metrics, device_batch, rng
            )
            sample_offset = self.save_predictions(
                save_dir, sample_offset, batch, jax.device_get(outputs), self.u_scales[part]
            )
        # Accumulated over the whole partition, computed once at the end
        return self.compute_metrics(part_metrics)

    def save_predictions(
        self,
        save_dir: Path,
        sample_offset: int,
        batch: dict[str, Any],
        outputs: dict[str, np.ndarray],
        u_scale: float = 1.0,
    ) -> int:
        """
        Save the generated fields of a batch, one file per sample:
        'output_{i}.npz' with 'x' (X, Y, 2), channels-last as in the dataset, and 'cond'.
        The fields are multiplied by 'u_scale' (the normalization of the dataset), so they are
        in the physical units of 'fields.npy'.

        Returns
        -------
        int
            Index of the next sample.
        """
        generated = np.asarray(outputs['generated']) * u_scale
        for i, field in enumerate(generated):
            output = {'x': field, 'cond': np.asarray(batch['cond'][i])}
            np.savez(save_dir / f'output_{sample_offset + i}.npz', allow_pickle=False, **output)
        return sample_offset + len(generated)
