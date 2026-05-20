"""Shared JAX training utilities for emulator fitting.

The repository supports two equivalent entry styles:

- prepared feature/target arrays
- dataset objects that yield tiled batches

Both routes are implemented here so batching and optimization logic stay in
one place.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from time import perf_counter

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from nenufar_emulators.core.datasets import SpectrumDataset, TiledBatch
from nenufar_emulators.core.metrics import mse
from nenufar_emulators.models import ActivationName, DenseMLP, init_mlp


@dataclass(frozen=True)
class TrainingHistory:
    """Training curves returned by the shared trainer.

    In practical terms, this is the minimum information a human usually wants
    to inspect after a training run: did the training loss go down, and did the
    validation loss behave sensibly? Using a dataclass keeps that information
    named and extensible.
    """

    train_losses: list[float]
    validation_losses: list[float]
    best_epoch: int | None = None
    best_validation_loss: float | None = None


def clone_model_state(model: DenseMLP) -> nnx.State:
    """Copy the trainable state of a live NNX model for later restoration.

    This is primarily used by early stopping. When validation loss improves we
    snapshot the current model parameters, and if training later degrades we
    can restore the best-known state rather than returning the final epoch.
    """
    return nnx.from_flat_state(
        [(tuple(path), jnp.array(value)) for path, value in nnx.to_flat_state(nnx.state(model))]
    )


def _iter_array_batches(
    features: jnp.ndarray,
    targets: jnp.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    key: jax.Array | None = None,
) -> Iterator[tuple[jnp.ndarray, jnp.ndarray]]:
    """Yield mini-batches from already-prepared in-memory arrays.

    This helper stays private because array batching is only an implementation
    detail of the array-based trainer. The public batching contract for the
    broader codebase lives on :class:`SpectrumDataset`.
    """
    if len(features) != len(targets):
        raise ValueError("features and targets must have the same length.")
    indices = jnp.arange(len(features))
    if shuffle:
        if key is None:
            key = jax.random.PRNGKey(0)
        indices = jax.random.permutation(key, indices)
    for start in range(0, len(indices), batch_size):
        batch_index = indices[start : start + batch_size]
        yield features[batch_index], targets[batch_index]


def train_mlp_regressor(
    train_features: jnp.ndarray,
    train_targets: jnp.ndarray,
    validation_features: jnp.ndarray,
    validation_targets: jnp.ndarray,
    *,
    hidden_features: int = 64,
    hidden_layers: int = 2,
    activation: ActivationName = "relu",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    epochs: int = 50,
    seed: int = 0,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    log_every: int | None = 1,
    log_prefix: str = "train_mlp_regressor",
) -> tuple[DenseMLP, TrainingHistory]:
    """Train the shared dense emulator network on prepared in-memory arrays.

    This trainer is intentionally array-based because the workflow preparation
    step often produces explicit feature/target matrices before fitting.
    It is also useful for:

    - synthetic smoke testing
    - validating model and tiling contracts
    - fitting prepared HERA training rows directly

    Parameters
    ----------
    train_features, validation_features:
        Two-dimensional feature matrices already in the model input space.
        These are usually produced by parameter preparation, tiling, and
        scaling steps earlier in the pipeline.
    train_targets, validation_targets:
        One-dimensional target arrays aligned row-by-row with the feature
        matrices.
    hidden_features, hidden_layers, activation:
        Network architecture choices for the shared MLP.
    learning_rate, weight_decay:
        Optimizer settings passed to Optax AdamW.
    batch_size:
        Number of training examples processed per gradient update.
    epochs:
        Number of full passes over the training set.
    seed:
        Random seed used for model initialization and batch shuffling.
    early_stopping_patience:
        If provided, stop training after this many epochs without a meaningful
        validation-loss improvement and restore the best-seen weights.
    early_stopping_min_delta:
        Minimum decrease in validation loss required to count as an
        improvement for early stopping.

    Returns
    -------
    DenseMLP, TrainingHistory
        The trained live NNX model and the recorded train/validation curves.
    """
    key = jax.random.PRNGKey(seed)
    model = init_mlp(
        key=key,
        in_features=int(train_features.shape[1]),
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        activation=activation,
    )
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
        wrt=nnx.Param,
    )

    @nnx.jit
    def train_step(
        model_instance: DenseMLP,
        optimizer_instance: nnx.Optimizer,
        batch_features: jnp.ndarray,
        batch_targets: jnp.ndarray,
    ) -> jnp.ndarray:
        """Run one optimizer step on a mini-batch.

        In practice this means: make predictions, measure squared-error loss,
        compute gradients with respect to trainable parameters, and update the
        live NNX model in place through the optimizer wrapper.
        """

        def loss_fn(current_model: DenseMLP) -> jnp.ndarray:
            preds = current_model(batch_features).squeeze(-1)
            return mse(preds, batch_targets)

        loss, grads = nnx.value_and_grad(loss_fn)(model_instance)
        optimizer_instance.update(model_instance, grads)
        return loss

    @nnx.jit
    def eval_step(
        model_instance: DenseMLP,
        batch_features: jnp.ndarray,
        batch_targets: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate mini-batch loss without changing model parameters."""
        preds = model_instance(batch_features).squeeze(-1)
        return mse(preds, batch_targets)

    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_loss = float("inf")
    best_epoch: int | None = None
    best_state: nnx.State | None = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        epoch_start = perf_counter()
        key, train_key = jax.random.split(key)
        train_loss = 0.0
        train_batches = 0
        for batch_features, batch_targets in _iter_array_batches(
            train_features,
            train_targets,
            batch_size,
            shuffle=True,
            key=train_key,
        ):
            loss = train_step(model, optimizer, batch_features, batch_targets)
            train_loss += float(loss)
            train_batches += 1
        train_loss /= max(train_batches, 1)

        validation_loss = 0.0
        validation_batches = 0
        for batch_features, batch_targets in _iter_array_batches(
            validation_features,
            validation_targets,
            batch_size,
            shuffle=False,
        ):
            validation_loss += float(eval_step(model, batch_features, batch_targets))
            validation_batches += 1
        validation_loss /= max(validation_batches, 1)

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_validation_loss - early_stopping_min_delta:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = clone_model_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if log_every is not None and ((epoch + 1) % log_every == 0):
            message = (
                f"[{log_prefix}] epoch={epoch + 1}/{epochs} "
                f"train_loss={train_loss:.6e} val_loss={validation_loss:.6e}"
            )
            if best_epoch is not None and best_validation_loss < float("inf"):
                message += (
                    f" best_epoch={best_epoch + 1} "
                    f"best_val_loss={best_validation_loss:.6e}"
                )
            message += f" epoch_seconds={perf_counter() - epoch_start:.2f}"
            print(message, flush=True)

        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            break

    if best_state is not None:
        nnx.update(model, best_state)

    return model, TrainingHistory(
        train_losses=train_losses,
        validation_losses=validation_losses,
        best_epoch=best_epoch,
        best_validation_loss=None if best_epoch is None else best_validation_loss,
    )


def train_mlp_dataset(
    train_dataset: SpectrumDataset,
    validation_dataset: SpectrumDataset,
    *,
    hidden_features: int = 64,
    hidden_layers: int = 2,
    activation: ActivationName = "relu",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    epochs: int = 50,
    seed: int = 0,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    log_every: int | None = 1,
    log_prefix: str = "train_mlp_dataset",
) -> tuple[DenseMLP, TrainingHistory]:
    """Train the shared MLP directly from dataset objects.

    This is the preferred training entrypoint once data has been wrapped in the
    new loader architecture. The datasets are responsible for preprocessing,
    batching, and tiling, while this function only handles optimization.

    Both datasets must be created with ``tiling=True`` so their batch iterators
    yield :class:`~nenufar_emulators.core.datasets.TiledBatch` objects.
    """
    if not train_dataset.tiling or not validation_dataset.tiling:
        raise ValueError("train_mlp_dataset expects train and validation datasets with tiling=True.")

    example_batch = next(train_dataset.get_batch_iterator(batch_size=1, shuffle=False))
    if not isinstance(example_batch, TiledBatch):
        raise ValueError("Expected tiled batches from the training dataset.")

    key = jax.random.PRNGKey(seed)
    model = init_mlp(
        key=key,
        in_features=int(example_batch.features.shape[1]),
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        activation=activation,
    )
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
        wrt=nnx.Param,
    )

    @nnx.jit
    def train_step(
        model_instance: DenseMLP,
        optimizer_instance: nnx.Optimizer,
        batch_features: jnp.ndarray,
        batch_targets: jnp.ndarray,
    ) -> jnp.ndarray:
        """Run one optimizer step on one already-tiled mini-batch."""

        def loss_fn(current_model: DenseMLP) -> jnp.ndarray:
            preds = current_model(batch_features).squeeze(-1)
            return mse(preds, batch_targets)

        loss, grads = nnx.value_and_grad(loss_fn)(model_instance)
        optimizer_instance.update(model_instance, grads)
        return loss

    @nnx.jit
    def eval_step(
        model_instance: DenseMLP,
        batch_features: jnp.ndarray,
        batch_targets: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate loss on a tiled mini-batch without updating the model."""
        preds = model_instance(batch_features).squeeze(-1)
        return mse(preds, batch_targets)

    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_loss = float("inf")
    best_epoch: int | None = None
    best_state: nnx.State | None = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        epoch_start = perf_counter()
        key, train_key = jax.random.split(key)
        train_loss = 0.0
        train_batches = 0
        for batch in train_dataset.get_batch_iterator(
            batch_size,
            shuffle=True,
            key=train_key,
        ):
            if not isinstance(batch, TiledBatch):
                raise ValueError("Training dataset yielded an untiled batch unexpectedly.")
            loss = train_step(model, optimizer, batch.features, batch.targets)
            train_loss += float(loss)
            train_batches += 1
        train_loss /= max(train_batches, 1)

        validation_loss = 0.0
        validation_batches = 0
        for batch in validation_dataset.get_batch_iterator(batch_size, shuffle=False):
            if not isinstance(batch, TiledBatch):
                raise ValueError("Validation dataset yielded an untiled batch unexpectedly.")
            validation_loss += float(eval_step(model, batch.features, batch.targets))
            validation_batches += 1
        validation_loss /= max(validation_batches, 1)

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_validation_loss - early_stopping_min_delta:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = clone_model_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if log_every is not None and ((epoch + 1) % log_every == 0):
            message = (
                f"[{log_prefix}] epoch={epoch + 1}/{epochs} "
                f"train_loss={train_loss:.6e} val_loss={validation_loss:.6e}"
            )
            if best_epoch is not None and best_validation_loss < float("inf"):
                message += (
                    f" best_epoch={best_epoch + 1} "
                    f"best_val_loss={best_validation_loss:.6e}"
                )
            message += f" epoch_seconds={perf_counter() - epoch_start:.2f}"
            print(message, flush=True)

        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            break

    if best_state is not None:
        nnx.update(model, best_state)

    return model, TrainingHistory(
        train_losses=train_losses,
        validation_losses=validation_losses,
        best_epoch=best_epoch,
        best_validation_loss=None if best_epoch is None else best_validation_loss,
    )
