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
from .steps import Params, make_inference_step

logger = logging.getLogger(__name__)


class Inferencer(BaseRunner):
    """
    Evaluates the model on the dataset and saves predictions, without optimizers and writers.
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
            Directory to save the predictions to ('{save_path}/{partition}/').
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
        self.eval_dataloaders = {k: v for k, v in dataloaders.items() if k != 'train'}
        self.save_path = get_root() / save_path

        self.metrics_cls = self.make_metrics_cls((), 'inference')
        self.inference_step_fn = make_inference_step(
            self.model.apply, self.metrics_cls, jit=self.cfg_inferencer.get('jit', True)
        )

        if params is not None:
            self.params = params
        else:
            example = next(iter(datasets.values()))[0]
            seed = self.cfg_inferencer.get('seed', 42)
            if skip_model_load:
                self.params = self.model.init(jax.random.PRNGKey(seed), **example, train=False)[
                    'params'
                ]
            else:
                # The structure of the parameters, without initializing them
                abstract_variables = jax.eval_shape(
                    lambda key: self.model.init(key, **example, train=False),
                    jax.random.PRNGKey(seed),
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
        for part, dataloader in self.eval_dataloaders.items():
            part_logs[part] = self._inference_part(part, dataloader)
        return part_logs

    def _inference_part(self, part: str, dataloader: grain.IterDataset) -> dict[str, float]:
        """
        Run inference on a partition and save predictions.

        Returns
        -------
        dict[str, float]
            Metrics of the partition.
        """
        save_dir = self.save_path / part
        save_dir.mkdir(exist_ok=True, parents=True)

        part_metrics = self.metrics_cls.empty()
        sample_offset = 0  # index of the next saved sample in the partition
        for batch in tqdm(dataloader, desc=part):
            device_batch = self.prepare_batch(batch)
            part_metrics, outputs = self.inference_step_fn(self.params, part_metrics, device_batch)
            sample_offset = self.save_predictions(
                save_dir, sample_offset, batch, jax.device_get(outputs)
            )
        # Accumulated over the whole partition, computed once at the end
        return self.compute_metrics(part_metrics)

    def save_predictions(
        self,
        save_dir: Path,
        sample_offset: int,
        batch: dict[str, Any],
        outputs: dict[str, np.ndarray],
    ) -> int:
        """
        Save the predictions of a batch to disk, one file per sample.

        Some saving logic: this is an example, change it for your task.

        Returns
        -------
        int
            Index of the next sample.
        """
        pred_labels = np.asarray(outputs['logits']).argmax(axis=-1)
        for i, pred_label in enumerate(pred_labels):
            output: dict[str, np.ndarray] = {'pred_label': np.asarray(pred_label)}
            if 'label' in batch:
                output['label'] = np.asarray(batch['label'][i])
            np.savez(save_dir / f'output_{sample_offset + i}.npz', allow_pickle=False, **output)
        return sample_offset + len(pred_labels)
