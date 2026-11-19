import logging
import time
from pathlib import Path
from typing import Any, cast

import clu.metrics as clu_metrics
import grain
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import yaml
from omegaconf import DictConfig, OmegaConf
from orbax.checkpoint.checkpoint_managers import preservation_policy as pp
from orbax.checkpoint.checkpoint_managers import save_decision_policy as sp
from tqdm.auto import tqdm

from lab.dataset.utils import get_dataloaders
from lab.logger.base import BaseWriter
from lab.loss.base import BaseLoss
from lab.metrics.base import BaseMetric
from lab.model.base import BaseModel
from lab.utils.io import get_root

from .config import TrainerConfig
from .runner import BaseRunner
from .state import TrainState
from .steps import (
    Batch,
    BatchProcessor,
    Params,
    make_eval_step,
    make_sample_step,
    make_train_step,
    with_matmul_precision,
)
from .utils import EarlyStopping, StepTimer

logger = logging.getLogger(__name__)


class BaseTrainer(BaseRunner):
    """
    Base class for all trainers.

    Provides step management, metric tracking, checkpointing and evaluation.
    """

    def __init__(
        self,
        config: DictConfig,
        device: jax.Device,
        key: jax.Array,
        datasets: dict[str, grain.sources.RandomAccessDataSource],
        model: BaseModel,
        variables: dict[str, Any],
        criterion: BaseLoss,
        batch_processor: BatchProcessor,
        metrics: dict[str, dict[str, type[BaseMetric]]],
        optimizer: optax.GradientTransformation | optax.GradientTransformationExtraArgs,
        lr_scheduler: optax.Schedule,
        writer: BaseWriter,
    ) -> None:
        """
        Parameters
        ----------
        config : DictConfig
            Experiment config.
        device : jax.Device
            Device for arrays and computation.
        key : jax.Array
            PRNG key for the randomness of training (noise, times, dropout). Evaluation and
            sampling use keys from 'trainer.eval_seed'.
        datasets : dict[str, grain.sources.RandomAccessDataSource]
            Dataset of each partition.
        model : BaseModel
            Flax model.
        variables : dict[str, Any]
            Initialized model variables containing 'params'.
        criterion : BaseLoss
            Loss function.
        batch_processor : BatchProcessor
            Processing of a batch (inside the jitted steps) into the inputs of the model and the
            loss, e.g. the noise and the times of flow matching.
        metrics : dict[str, dict[str, type[BaseMetric]]]
            Metrics for training ('train') and inference ('inference').
        optimizer : optax.GradientTransformation or optax.GradientTransformationExtraArgs
            Optimizer of the model.
        lr_scheduler : optax.Schedule
            Learning rate scheduler (the one the optimizer was built with; used for logging).
        writer : BaseWriter
            Experiment tracker.
        """
        self.config = config
        self.cfg_trainer = TrainerConfig.from_config(config.trainer)
        cfg = self.cfg_trainer

        super().__init__(
            device=device,
            model=model,
            metrics=metrics,
            device_tensors=cfg.device_tensors,
            use_bfloat16=cfg.use_bfloat16,
        )
        self.timer = StepTimer(enabled=cfg.debug)
        self.key = key
        # Evaluation and sampling for the logs use fixed keys from their own seed, independent
        # of the training seed: the same noise and times at every evaluation of every run, so
        # the results are comparable across steps and across runs
        self.eval_key, self.sample_key = jax.random.split(jax.random.PRNGKey(cfg.eval_seed))

        if cfg.max_grad_norm is not None:
            optimizer = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optimizer)
            logger.info(f'Gradient clipping enabled with max_grad_norm={cfg.max_grad_norm}')

        self.criterion = criterion
        self.batch_processor = batch_processor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        if cfg.ema is not None:
            logger.info(f'EMA enabled: {cfg.ema}')

        # Define metrics and step functions
        self.train_metrics_cls = self.make_metrics_cls(
            self.criterion.loss_names, 'train', include_grad_norm=True
        )
        self.eval_metrics_cls = self.make_metrics_cls(self.criterion.loss_names, 'inference')
        self.train_step_fn = with_matmul_precision(
            fn=make_train_step(
                criterion=self.criterion,
                batch_processor=self.batch_processor,
                metrics_cls=self.train_metrics_cls,
                grad_accum_steps=cfg.grad_accum_steps,
                jit=cfg.jit,
            ),
            precision=cfg.matmul_precision,
        )
        self.eval_model = self.model.with_dtype(jnp.bfloat16 if cfg.eval_use_bfloat16 else None)
        self.eval_step_fn = with_matmul_precision(
            fn=make_eval_step(
                criterion=self.criterion,
                batch_processor=self.batch_processor,
                metrics_cls=self.eval_metrics_cls,
                jit=cfg.jit,
                apply_fn=self.eval_model.apply,
            ),
            precision=cfg.eval_matmul_precision,
        )
        self.sample_step_fn = with_matmul_precision(
            fn=make_sample_step(
                apply_fn=self.eval_model.apply,
                steps=cfg.sample_steps,
                hard_div_free=getattr(self.model, 'hard_div_free', False),
                jit=cfg.jit,
            ),
            precision=cfg.eval_matmul_precision,
        )

        # Define dataloaders: a train batch holds the 'grad_accum_steps' micro-batches of a step
        dataloaders = get_dataloaders(config, datasets, cfg.grad_accum_steps)
        # One iterator for the whole run (checkpointed), so evaluation does not restart it
        self.train_iter = iter(dataloaders['train'])
        self.eval_dataloaders = {k: v for k, v in dataloaders.items() if k != 'train'}

        def create_state(params: Params) -> TrainState:
            return TrainState.create(
                apply_fn=self.model.apply,
                params=params,
                tx=self.optimizer,
                rng=self.key,
                ema=self.cfg_trainer.ema,
            )

        self.state = create_state(variables['params'])
        # Optimizer steps done; a host-side copy of 'state.step' to avoid device syncs
        self.global_step = 0
        # Reference point of the 'steps_per_sec' of the train logs
        self._last_log_step = 0
        self._last_log_time = time.perf_counter()

        self.writer = writer
        self.early_stopping = EarlyStopping(cfg.monitor, cfg.early_stop)
        self.checkpoint_dir = get_root() / cfg.checkpoint_dir
        self._ckpt_mngr = self._make_checkpoint_manager()

        if cfg.resume_from is not None:
            self._resume_from(cfg.resume_from)
        if cfg.from_pretrained is not None:
            # Only the weights: optimizer, step and data start anew (EMA starts from them)
            params = self.load_weights(
                cfg.from_pretrained,
                target=self.state.params,
                use_ema=cfg.from_pretrained_ema,
            )
            self.state = create_state(params)

    def train(self) -> TrainState:
        """
        Wrapper around training process to save model on keyboard interrupt and always
        close the writer and the checkpoint manager.

        Returns
        -------
        TrainState
            The final training state.
        """
        try:
            self._train_process()
            # The tracker copies the kept checkpoints, so the last save must be finished
            self._ckpt_mngr.wait_until_finished()
            self.writer.add_checkpoint(str(self.checkpoint_dir), str(self.checkpoint_dir.parent))
        except KeyboardInterrupt:
            if self.global_step > 0 and not self._has_checkpoint(self.global_step):
                logger.info('Saving model on keyboard interrupt.')
                self._save_checkpoint(None, force=True)
            self._ckpt_mngr.wait_until_finished()
            raise
        finally:
            self.writer.close()
            self._ckpt_mngr.close()

        return self.state

    def _train_process(self) -> None:
        """
        Train for a fixed number of optimizer steps, evaluate after each window, and monitor
        performance (early stopping and the best checkpoint).
        """
        cfg = self.cfg_trainer
        while self.global_step < cfg.total_steps:
            # Windows end on multiples of 'eval_every_steps', so after resuming from a step
            # inside a window, the evaluations and checkpoints stay on the same steps
            steps_to_eval = cfg.eval_every_steps - self.global_step % cfg.eval_every_steps
            window_steps = min(steps_to_eval, cfg.total_steps - self.global_step)
            train_logs = self._train_window(window_steps)
            logs = {
                'step': self.global_step,
                **train_logs,
                **self._evaluate(),
            }
            logger.info(f'\nMetrics:\n{yaml.dump(logs)}')

            self.early_stopping.update(logs)
            saved = self._save_checkpoint(logs)
            if self.early_stopping.should_stop:
                logger.info(
                    f"Validation performance didn't improve for "
                    f'{self.early_stopping.patience} '
                    'evaluations. Training stops.'
                )
                break

        # Always keep the final state (unless it has just been saved)
        if not saved:
            self._save_checkpoint(logs, force=True)

    def _train_window(self, steps: int) -> dict[str, float]:
        """
        Train for a window of 'steps' optimizer steps and return the train metrics averaged
        over the last logging interval of the window (about 'log_step' steps, not the whole
        window).
        """
        train_metrics = self.train_metrics_cls.empty()
        self.timer.reset()

        start_step = self.global_step
        self.writer.set_step(start_step)
        self._last_log_step, self._last_log_time = start_step, time.perf_counter()

        last_train_metrics: dict[str, float] = {}
        for _ in tqdm(range(steps), desc=f'Train Steps {start_step}–{start_step + steps}'):
            batch, train_metrics = self._train_step(train_metrics)
            if self.global_step % self.cfg_trainer.log_step == 0:
                last_train_metrics = self._log_train(batch, train_metrics)
                train_metrics = self.train_metrics_cls.empty()

        # Steps after the last logging point have not been averaged yet
        if self.global_step != self._last_log_step:
            last_train_metrics = {**self.compute_metrics(train_metrics), **self.timer.result()}

        return {f'train_{name}': value for name, value in last_train_metrics.items()}

    def _train_step(
        self, train_metrics: clu_metrics.Collection
    ) -> tuple[Batch, clu_metrics.Collection]:
        """
        Do an optimizer step (gradient accumulation and EMA are inside the train step
        function) and the bookkeeping after it.

        Returns
        -------
        tuple[Batch, clu_metrics.Collection]
            The batch of the step (all its micro-batches) and the updated train metrics.
        """
        self.timer.start_step()
        with self.timer.data():
            raw_batch = next(self.train_iter)
        batch = self.prepare_batch(raw_batch)
        self.state, train_metrics = self.train_step_fn(self.state, train_metrics, batch)
        self.global_step += 1
        self.timer.end_step((self.state, train_metrics))
        return batch, train_metrics

    def _log_train(self, batch: Batch, train_metrics: clu_metrics.Collection) -> dict[str, float]:
        """
        Log the train metrics averaged since the previous logging point (and reset them),
        the learning rate, speed and the batch. Returns the logged metrics.
        """
        now = time.perf_counter()
        self.writer.set_step(self.global_step)
        # The learning rate of the last update (optax counts the updates from 0)
        self.writer.add_scalar('lr', float(jax.device_get(self.lr_scheduler(self.global_step - 1))))
        self.writer.add_scalar(
            'steps_per_sec',
            (self.global_step - self._last_log_step) / (now - self._last_log_time),
        )
        self._last_log_step, self._last_log_time = self.global_step, now

        metrics = {**self.compute_metrics(train_metrics), **self.timer.result()}
        self.timer.reset()
        self.writer.add_scalars(metrics)
        self._log_batch(self.state.params, batch, 'train')
        return metrics

    def _eval_params(self) -> Params:
        """
        Parameters to evaluate with: the EMA ones if available, otherwise the current ones.
        """
        if self.state.ema_params is None:
            return self.state.params
        logger.info('Switching to EMA model...')
        return self.state.ema_params

    def _evaluate(self) -> dict[str, float]:
        """
        Evaluate on all eval dataloaders (with EMA weights if available); the logs are
        prefixed with the dataloader name.
        """
        params = self._eval_params()
        logs: dict[str, float] = {}
        for part, dataloader in self.eval_dataloaders.items():
            part_logs = self._evaluate_part(part, dataloader, params)
            logs.update({f'{part}_{name}': value for name, value in part_logs.items()})
        return logs

    def _evaluate_part(
        self,
        part: str,
        dataloader: grain.IterDataset,
        params: Params,
    ) -> dict[str, float]:
        """
        Evaluate the model on a partition and return its logs.

        Parameters
        ----------
        part : str
            Partition name.
        dataloader : grain.IterDataset
            Dataloader of the partition.
        params : Params
            Parameters to evaluate with (see '_eval_params').
        """
        metrics = self.eval_metrics_cls.empty()
        batch = None
        num_batches = 0
        start_time = time.perf_counter()
        for raw_batch in tqdm(dataloader, desc=f'{part.capitalize()} Step {self.global_step}'):
            batch = self.prepare_batch(raw_batch)
            rng = jax.random.fold_in(self.eval_key, num_batches)
            metrics = self.eval_step_fn(self.state, params, metrics, batch, rng)
            num_batches += 1

        # Accumulated over the whole partition, computed once at the end (waits for the device)
        logs = self.compute_metrics(metrics)
        elapsed = time.perf_counter() - start_time

        self.writer.set_step(self.global_step, part)
        self.writer.add_scalars(logs)
        if num_batches > 0:
            self.writer.add_scalar('steps_per_sec', num_batches / elapsed)
        if batch is not None:
            self._log_batch(params, batch, part)
        return logs

    def _log_batch(self, params: Params, batch: Batch, part: str) -> None:
        """
        Log data from a batch to the experiment tracker via 'self.writer.add_*'; the writer
        step is already set to the current step and 'part'. 'params' are the ones the batch
        was evaluated with ('self.sample_step_fn' generates fields with them).
        """
        raise NotImplementedError(f'{type(self).__name__} must implement _log_batch method.')

    def _save_checkpoint(self, logs: dict[str, float] | None, force: bool = False) -> bool:
        """
        Save the training state of the current step: state (params, optimizer, EMA, rng),
        train data iterator, early stopping and config. The save is asynchronous.

        Parameters
        ----------
        logs : dict[str, float] or None
            Metrics of the step, for keeping the best checkpoint by 'monitor' (None: the
            checkpoint is never the best one).
        force : bool, optional
            Save regardless of the save policies.

        Returns
        -------
        bool
            Whether the checkpoint is saved.
        """
        items = {
            'state': ocp.args.StandardSave(self.state),
            'meta': ocp.args.JsonSave({'early_stopping': self.early_stopping.state_dict()}),
            'config': ocp.args.JsonSave(
                cast(dict, OmegaConf.to_container(self.config, resolve=True))
            ),
            # The iterator state is taken synchronously, before 'save' returns
            'data': grain.checkpoint.CheckpointSave(self.train_iter),
        }
        # After resuming from an earlier step, the later checkpoints of the run directory are
        # overwritten when training reaches them again
        if (force or self._ckpt_mngr.should_save(self.global_step)) and self._has_checkpoint(
            self.global_step
        ):
            self._ckpt_mngr.wait_until_finished()
            self._ckpt_mngr.delete(self.global_step)
        saved = self._ckpt_mngr.save(
            self.global_step, args=ocp.args.Composite(**items), metrics=logs, force=force
        )
        if saved:
            path = self._checkpoint_path(self.global_step)
            logger.info(f'Saving checkpoint at step {self.global_step}: {path}...')
        return saved

    def _resume_from(self, resume_from: str) -> None:
        """
        Restore the training state, train data iterator and early stopping state in place.

        Parameters
        ----------
        resume_from : str
            Step to resume from, or 'latest'.

        Raises
        ------
        FileNotFoundError
            If 'resume_from' is 'latest' and there are no checkpoints.
        RuntimeError
            If the model, optimizer or lr_scheduler config differs from the checkpoint one.
        """
        step = self._ckpt_mngr.latest_step() if resume_from == 'latest' else int(resume_from)
        logger.info(f"Resuming from checkpoint '{resume_from}' in {self.checkpoint_dir}...")
        if step is None:
            raise FileNotFoundError(f"No checkpoints to resume from in '{self.checkpoint_dir}'.")

        saved = self._ckpt_mngr.restore(
            step, args=ocp.args.Composite(config=ocp.args.JsonRestore())
        )['config']
        current = cast(dict, OmegaConf.to_container(self.config, resolve=True))
        if saved['model'] != current['model']:
            raise RuntimeError(
                'Architecture configuration given in the config file is different '
                'from that of the checkpoint.'
            )
        if (
            saved['optimizer'] != current['optimizer']
            or saved['lr_scheduler'] != current['lr_scheduler']
        ):
            raise RuntimeError(
                'Optimizer or lr_scheduler given in the config file is different '
                'from that of the checkpoint. Cannot resume optimizer and scheduler states.'
            )

        restored = self._ckpt_mngr.restore(
            step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(self.state),
                meta=ocp.args.JsonRestore(),
                data=grain.checkpoint.CheckpointRestore(self.train_iter),
            ),
        )
        self.state = restored['state']
        self.early_stopping.load_state_dict(restored['meta'].get('early_stopping', {}))
        self.global_step = int(self.state.step)
        logger.info(f'Resuming from optimizer step {self.global_step}')

    def _make_checkpoint_manager(self) -> ocp.CheckpointManager:
        """
        Create the checkpoint manager that decides which steps are saved and kept
        (periodic, best by 'monitor', last).
        """
        cfg = self.cfg_trainer

        # When to save: at every evaluation (the checkpoint gets its metrics), periodically
        # (unless only_best) and at the last step
        save_policies: list[sp.SaveDecisionPolicy] = [
            sp.FixedIntervalPolicy(cfg.eval_every_steps),
            sp.SpecificStepsPolicy([cfg.total_steps]),
        ]
        # What to keep: the latest checkpoint (to resume from), the periodic ones (unless
        # only_best) and the best one by 'monitor'; the others are deleted after the save
        preservation_policies: list[pp.PreservationPolicy] = [pp.LatestN(n=1)]
        if not cfg.only_best:
            save_policies.append(sp.FixedIntervalPolicy(cfg.checkpoint_every_steps))
            preservation_policies.append(pp.EveryNSteps(interval_steps=cfg.checkpoint_every_steps))
        if cfg.monitor != 'off':
            mode, metric = cfg.monitor.split()
            if mode not in ('min', 'max'):
                raise ValueError(
                    f"monitor must be 'off' or '<min|max> <metric_name>': {cfg.monitor!r}"
                )

            def get_metric(metrics: dict[str, float]) -> float:
                return metrics[metric]

            # BestN keeps the largest metric with reverse=False, so 'min' needs reverse=True.
            # Checkpoints saved without metrics are not kept as "best" (only by the others).
            preservation_policies.append(
                pp.BestN(
                    get_metric_fn=get_metric,
                    reverse=mode == 'min',
                    n=1,
                    keep_checkpoints_without_metrics=False,
                )
            )

        return ocp.CheckpointManager(
            directory=self.checkpoint_dir,
            options=ocp.CheckpointManagerOptions(
                save_decision_policy=sp.AnySavePolicy(save_policies),
                preservation_policy=pp.AnyPreservationPolicy(preservation_policies),
            ),
        )

    def _has_checkpoint(self, step: int) -> bool:
        return step in self._ckpt_mngr.all_steps()

    def _checkpoint_path(self, step: int) -> Path:
        return self.checkpoint_dir / str(step)
