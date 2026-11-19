from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optax
from clu import metrics as clu_metrics
from flax.core import FrozenDict

from lab.loss.base import BaseLoss
from lab.model.flow import VelocityFn, sample
from lab.model.physics.spectral import relative_divergence

from .state import TrainState

type Params = FrozenDict[str, Any]
type Batch = dict[str, jax.Array]
# (rng, **batch) -> processed batch: the inputs of the model and the loss (e.g. FlowMatchingPath)
type BatchProcessor = Callable[..., Batch]
type TrainStepFn = Callable[
    [TrainState, clu_metrics.Collection, Batch], tuple[TrainState, clu_metrics.Collection]
]
type EvalStepFn = Callable[
    [TrainState, Params, clu_metrics.Collection, Batch, jax.Array], clu_metrics.Collection
]
type SampleStepFn = Callable[[Params, Batch, jax.Array], jax.Array]
type InferenceStepFn = Callable[
    [Params, clu_metrics.Collection, Batch, jax.Array],
    tuple[clu_metrics.Collection, dict[str, jax.Array]],
]


def with_matmul_precision[**P, R](fn: Callable[P, R], precision: str | None) -> Callable[P, R]:
    """
    Run 'fn' (and trace it, if it is jitted) with the precision of the float32 matmuls and
    convolutions: 'default', 'high' or 'highest' (None keeps the global setting).

    On TPU 'default' is one bfloat16 pass (about bfloat16 accuracy even for float32 arrays)
    and on recent GPUs it is TF32; 'highest' is full float32. It does not change computations
    in bfloat16 (the operands are already rounded).
    """
    if precision is None:
        return fn

    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        with jax.default_matmul_precision(precision):
            return fn(*args, **kwargs)

    return wrapped


def apply_model(
    apply_fn: Callable[..., Any],
    params: Params,
    inputs: dict[str, jax.Array],
    train: bool,
    rng: jax.Array | None = None,
) -> Any:
    """
    Run the model on a batch of inputs. The model takes a single example, so it is vmapped
    over the batch (with a separate dropout key per example in train mode).

    Parameters
    ----------
    apply_fn : Callable[..., Any]
        Model apply function.
    params : Params
        Model parameters.
    inputs : dict[str, jax.Array]
        Batched keyword inputs of the model.
    train : bool
        Train mode (dropout is active).
    rng : jax.Array or None, optional
        PRNG key, required when 'train' is True.

    Returns
    -------
    Any
        Model outputs of each example.
    """
    if train:
        if rng is None:
            raise ValueError('A PRNG key is required when forwarding with train=True.')
        example_rngs = jax.random.split(rng, jax.tree.leaves(inputs)[0].shape[0])
        return jax.vmap(
            lambda example, example_rng: apply_fn(
                {'params': params},
                **example,
                train=True,
                rngs={'dropout': example_rng},
            )
        )(inputs, example_rngs)
    return jax.vmap(lambda example: apply_fn({'params': params}, **example, train=False))(inputs)


def make_velocity_fn(
    apply_fn: Callable[..., Any],
    params: Params,
    train: bool = False,
    rng: jax.Array | None = None,
) -> VelocityFn:
    """
    Bind the model to its parameters as a batched function
    'velocity_fn(x, t, cond, cond_mask=None)'.
    """

    def velocity_fn(
        x: jax.Array, t: jax.Array, cond: jax.Array, cond_mask: jax.Array | None = None
    ) -> jax.Array:
        inputs = {'x': x, 't': t, 'cond': cond}
        if cond_mask is not None:
            inputs['cond_mask'] = cond_mask
        return apply_model(apply_fn, params, inputs, train, rng)

    return velocity_fn


def forward(
    criterion: BaseLoss,
    batch_processor: BatchProcessor,
    apply_fn: Callable[..., Any],
    params: Params,
    batch: Batch,
    rng: jax.Array,
    train: bool,
) -> dict[str, jax.Array]:
    """
    Compute the losses of a batch: the batch is processed into the flow matching problem
    (noise, times, condition dropout), the model predicts the velocity and the loss compares
    it to the target. 'rng' gives the randomness of the processing and the dropout.
    """
    process_rng, dropout_rng = jax.random.split(rng)
    batch = batch_processor(process_rng, **batch)
    model = make_velocity_fn(apply_fn, params, train=train, rng=dropout_rng if train else None)
    velocity = model(batch['x_t'], batch['t'], batch['cond'], batch['cond_mask'])
    return criterion(velocity=velocity, **batch)


def batch_metrics(
    metrics_cls: type[clu_metrics.Collection],
    loss_names: tuple[str, ...],
    batch: Batch,
    outputs: dict[str, jax.Array],
    **extra: jax.Array,
) -> clu_metrics.Collection:
    batch_size = jax.tree.leaves(batch)[0].shape[0]
    losses = {name: jnp.broadcast_to(outputs[name], (batch_size,)) for name in loss_names}
    return metrics_cls.single_from_model_output(**{**batch, **outputs, **losses, **extra})


def make_train_step(
    criterion: BaseLoss,
    batch_processor: BatchProcessor,
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
    batch_processor : BatchProcessor
        Processing of a batch into the inputs of the model and the loss.
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
            outputs = forward(
                criterion, batch_processor, state.apply_fn, params, batch, rng, train=True
            )
            return outputs['loss'], outputs

        # The batch holds 'grad_accum_steps' micro-batches: (grad_accum_steps * micro, ...)
        micro_batches = jax.tree.map(
            lambda x: x.reshape((grad_accum_steps, x.shape[0] // grad_accum_steps, *x.shape[1:])),
            batch,
        )
        keys = jax.random.split(state.rng, grad_accum_steps + 1)
        state = state.replace(rng=keys[0])

        def micro_step(
            grads_sum: Params, inputs: tuple[Batch, jax.Array]
        ) -> tuple[Params, dict[str, jax.Array]]:
            micro_batch, rng = inputs
            grads, outputs = jax.grad(loss_fn, has_aux=True)(state.params, micro_batch, rng)
            return jax.tree.map(jnp.add, grads_sum, grads), outputs

        if grad_accum_steps == 1:
            grads_sum, outputs = micro_step(
                jax.tree.map(jnp.zeros_like, state.params),
                (jax.tree.map(lambda x: x[0], micro_batches), keys[1]),
            )
            outputs = jax.tree.map(lambda x: x[None], outputs)
        else:
            grads_sum, outputs = jax.lax.scan(
                micro_step, jax.tree.map(jnp.zeros_like, state.params), (micro_batches, keys[1:])
            )
        grads = jax.tree.map(lambda g: g / grad_accum_steps, grads_sum)
        grad_norm = optax.global_norm(grads)

        micro_size = jax.tree.leaves(batch)[0].shape[0] // grad_accum_steps
        outputs = {
            name: (
                jnp.repeat(value, micro_size)
                if name in criterion.loss_names
                else value.reshape((-1, *value.shape[2:]))
            )
            for name, value in outputs.items()
        }
        metrics = metrics.merge(
            batch_metrics(metrics_cls, criterion.loss_names, batch, outputs, grad_norm=grad_norm)
        )
        return state.apply_gradients(grads=grads), metrics

    return jax.jit(train_step, donate_argnums=(0, 1)) if jit else train_step


def make_eval_step(
    criterion: BaseLoss,
    batch_processor: BatchProcessor,
    metrics_cls: type[clu_metrics.Collection],
    jit: bool = True,
    apply_fn: Callable[..., Any] | None = None,
) -> EvalStepFn:
    """
    Make the eval step: the losses and metrics of a batch merged into the running metrics,
    with the given (e.g. EMA) parameters. The key gives the randomness of the batch processing.

    Parameters
    ----------
    criterion : BaseLoss
        Loss function.
    batch_processor : BatchProcessor
        Processing of a batch into the inputs of the model and the loss.
    metrics_cls : type[clu_metrics.Collection]
        Metric collection to update.
    jit : bool, optional
        Compile the step (the metrics buffer is donated).
    apply_fn : Callable[..., Any] or None, optional
        Model apply function to evaluate with (e.g. in another compute dtype than the train
        steps); 'state.apply_fn' if None.
    """

    def eval_step(
        state: TrainState,
        params: Params,
        metrics: clu_metrics.Collection,
        batch: Batch,
        rng: jax.Array,
    ) -> clu_metrics.Collection:
        outputs = forward(
            criterion=criterion,
            batch_processor=batch_processor,
            apply_fn=apply_fn or state.apply_fn,
            params=params,
            batch=batch,
            rng=rng,
            train=False,
        )
        return metrics.merge(batch_metrics(metrics_cls, criterion.loss_names, batch, outputs))

    return jax.jit(eval_step, donate_argnums=2) if jit else eval_step


def make_sample_step(
    apply_fn: Callable[..., Any],
    steps: int,
    hard_div_free: bool = False,
    jit: bool = True,
) -> SampleStepFn:
    """
    Make the step generating fields (B, X, Y, 2) for the conditions 'cond' of a batch (the
    real fields 'x' only give the shape).

    Parameters
    ----------
    apply_fn : Callable[..., Any]
        Model apply function.
    steps : int
        Euler steps of the ODE.
    hard_div_free : bool, optional
        Start from divergence-free noise.
    jit : bool, optional
        Compile the step.
    """

    def sample_step(params: Params, batch: Batch, rng: jax.Array) -> jax.Array:
        return sample(
            velocity_fn=make_velocity_fn(apply_fn, params),
            key=rng,
            cond=batch['cond'],
            shape=batch['x'].shape,
            steps=steps,
            hard_div_free=hard_div_free,
        )

    return jax.jit(sample_step) if jit else sample_step


def make_inference_step(
    apply_fn: Callable[..., Any],
    metrics_cls: type[clu_metrics.Collection],
    steps: int,
    hard_div_free: bool = False,
    jit: bool = True,
) -> InferenceStepFn:
    """
    Make the inference step: generated fields for the conditions of a batch, their relative
    divergence, and the metrics merged into the running ones.

    Returns the metrics and the outputs {'generated': (B, X, Y, 2), 'sample_rel_div': (B,)}.
    """
    sample_step = make_sample_step(apply_fn, steps, hard_div_free, jit=False)

    def inference_step(
        params: Params,
        metrics: clu_metrics.Collection,
        batch: Batch,
        rng: jax.Array,
    ) -> tuple[clu_metrics.Collection, dict[str, jax.Array]]:
        generated = sample_step(params, batch, rng)
        outputs = {'generated': generated, 'sample_rel_div': relative_divergence(generated)}
        return metrics.merge(metrics_cls.single_from_model_output(**batch, **outputs)), outputs

    return jax.jit(inference_step, donate_argnums=1) if jit else inference_step
