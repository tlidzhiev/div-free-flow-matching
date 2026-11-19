from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optax
from clu import metrics as clu_metrics
from flax.core import FrozenDict

from lab.loss.base import BaseLoss

from .state import TrainState

type Params = FrozenDict[str, Any]
type Batch = dict[str, jax.Array]
type TrainStepFn = Callable[
    [TrainState, clu_metrics.Collection, Batch], tuple[TrainState, clu_metrics.Collection]
]
type EvalStepFn = Callable[
    [TrainState, Params, clu_metrics.Collection, Batch], clu_metrics.Collection
]
type PredictStepFn = Callable[[Params, Batch], dict[str, jax.Array]]
type InferenceStepFn = Callable[
    [Params, clu_metrics.Collection, Batch], tuple[clu_metrics.Collection, dict[str, jax.Array]]
]


def apply_model(
    apply_fn: Callable[..., Any],
    params: Params,
    batch: Batch,
    train: bool,
    rng: jax.Array | None = None,
) -> dict[str, jax.Array]:
    """
    Run the model on a batch. The model takes a single example, so it is vmapped over the
    batch (with a separate dropout key per example in train mode).

    Parameters
    ----------
    apply_fn : Callable[..., Any]
        Model apply function.
    params : Params
        Model parameters.
    batch : Batch
        Batch of examples.
    train : bool
        Train mode (dropout is active).
    rng : jax.Array or None, optional
        PRNG key, required when 'train' is True.

    Returns
    -------
    dict[str, jax.Array]
        Model outputs of each example.
    """
    if train:
        if rng is None:
            raise ValueError('A PRNG key is required when forwarding with train=True.')
        example_rngs = jax.random.split(rng, jax.tree.leaves(batch)[0].shape[0])
        return jax.vmap(
            lambda example, example_rng: apply_fn(
                {'params': params},
                **example,
                train=True,
                rngs={'dropout': example_rng},
            )
        )(batch, example_rngs)
    return jax.vmap(lambda example: apply_fn({'params': params}, **example, train=False))(batch)


def forward(
    criterion: BaseLoss,
    state: TrainState,
    params: Params,
    batch: Batch,
    train: bool,
    rng: jax.Array | None = None,
) -> dict[str, jax.Array]:
    """
    Run the model and the loss on a batch. Both take a single example, so the outputs and
    the losses are per example.
    """
    outputs = apply_model(state.apply_fn, params, batch, train, rng)
    losses = jax.vmap(lambda example, output: criterion(**example, **output))(batch, outputs)
    return {**outputs, **losses}


def masked_mean(values: jax.Array, mask: jax.Array | None = None) -> jax.Array:
    """
    Average per-example values over the examples where 'mask' is True (the others are the
    padding of a batch).
    """
    if mask is None:
        return values.mean()
    return jnp.sum(values * mask) / jnp.maximum(mask.sum(), 1)


def make_train_step(
    criterion: BaseLoss,
    metrics_cls: type[clu_metrics.Collection],
    grad_accum_steps: int = 1,
    jit: bool = True,
) -> TrainStepFn:
    """
    Make the train step: gradients accumulated over the micro-batches of a batch, an
    optimizer step (with the EMA update) and the metrics update.

    Parameters
    ----------
    criterion : BaseLoss
        Loss function.
    metrics_cls : type[clu_metrics.Collection]
        Metric collection to update.
    grad_accum_steps : int, optional
        Number of micro-batches the batch is split into.
    jit : bool, optional
        Compile the step (the state and metrics buffers are donated).
    """

    def train_step(
        state: TrainState,
        metrics: clu_metrics.Collection,
        batch: Batch,
    ) -> tuple[TrainState, clu_metrics.Collection]:

        def loss_fn(
            params: Params, batch: Batch, rng: jax.Array
        ) -> tuple[jax.Array, dict[str, jax.Array]]:
            outputs = forward(criterion, state, params, batch, train=True, rng=rng)
            return masked_mean(outputs['loss'], batch.get('mask')), outputs

        grad_fn = jax.grad(loss_fn, has_aux=True)
        keys = jax.random.split(state.rng, grad_accum_steps + 1)
        state = state.replace(rng=keys[0])

        if grad_accum_steps == 1:
            # No scan: XLA on CPU compiles convolutions inside a loop into much slower code
            # (about 40x here), so the batch is differentiated directly
            grads, outputs = grad_fn(state.params, batch, keys[1])
        else:
            # The batch holds 'grad_accum_steps' micro-batches: (grad_accum_steps * micro, ...)
            micro_batches = jax.tree.map(
                lambda x: x.reshape((grad_accum_steps, -1, *x.shape[1:])), batch
            )

            def micro_step(
                grads_sum: Params, inputs: tuple[Batch, jax.Array]
            ) -> tuple[Params, dict[str, jax.Array]]:
                grads, outputs = grad_fn(state.params, *inputs)
                return jax.tree.map(jnp.add, grads_sum, grads), outputs

            grads_sum, outputs = jax.lax.scan(
                micro_step, jax.tree.map(jnp.zeros_like, state.params), (micro_batches, keys[1:])
            )
            grads = jax.tree.map(lambda g: g / grad_accum_steps, grads_sum)
            # (grad_accum_steps, micro, ...) -> (batch, ...)
            outputs = jax.tree.map(lambda x: x.reshape((-1, *x.shape[2:])), outputs)

        grad_norm = optax.global_norm(grads)
        metrics = metrics.merge(
            metrics_cls.single_from_model_output(**batch, **outputs, grad_norm=grad_norm)
        )
        return state.apply_gradients(grads=grads), metrics

    return jax.jit(train_step, donate_argnums=(0, 1)) if jit else train_step


def make_eval_step(
    criterion: BaseLoss,
    metrics_cls: type[clu_metrics.Collection],
    jit: bool = True,
) -> EvalStepFn:
    """
    Make the eval step: the loss and metrics of a batch merged into the running metrics,
    with the given (e.g. EMA) parameters.

    Parameters
    ----------
    criterion : BaseLoss
        Loss function.
    metrics_cls : type[clu_metrics.Collection]
        Metric collection to update.
    jit : bool, optional
        Compile the step (the metrics buffer is donated).
    """

    def eval_step(
        state: TrainState,
        params: Params,
        metrics: clu_metrics.Collection,
        batch: Batch,
    ) -> clu_metrics.Collection:
        outputs = forward(criterion, state, params, batch, train=False)
        return metrics.merge(metrics_cls.single_from_model_output(**batch, **outputs))

    return jax.jit(eval_step, donate_argnums=2) if jit else eval_step


def make_predict_step(apply_fn: Callable[..., Any], jit: bool = True) -> PredictStepFn:
    """
    Make the step returning the model outputs of a batch (e.g. for logging).

    Parameters
    ----------
    apply_fn : Callable[..., Any]
        Model apply function.
    jit : bool, optional
        Compile the step.
    """

    def predict_step(params: Params, batch: Batch) -> dict[str, jax.Array]:
        return apply_model(apply_fn, params, batch, train=False)

    return jax.jit(predict_step) if jit else predict_step


def make_inference_step(
    apply_fn: Callable[..., Any],
    metrics_cls: type[clu_metrics.Collection],
    jit: bool = True,
) -> InferenceStepFn:
    """
    Make the inference step: the model outputs of a batch and the metrics merged into the
    running ones.

    Parameters
    ----------
    apply_fn : Callable[..., Any]
        Model apply function.
    metrics_cls : type[clu_metrics.Collection]
        Metric collection to update.
    jit : bool, optional
        Compile the step (the metrics buffer is donated).
    """

    def inference_step(
        params: Params,
        metrics: clu_metrics.Collection,
        batch: Batch,
    ) -> tuple[clu_metrics.Collection, dict[str, jax.Array]]:
        outputs = apply_model(apply_fn, params, batch, train=False)
        return metrics.merge(metrics_cls.single_from_model_output(**batch, **outputs)), outputs

    return jax.jit(inference_step, donate_argnums=1) if jit else inference_step
