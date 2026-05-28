"""
Training utilities for dense emulator models with JAX and Flax NNX.

This module provides functions to train and evaluate a DenseMLP on prepared
feature and target arrays. It handles:
- mini-batch iteration
- host-to-device batch prefetching
- Optax optimisation
- validation loss tracking
- optional early stopping
"""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from jaxemu_21cmSPACE.utils.metrics import mse
from jaxemu_21cmSPACE.architectures.mlp import DenseMLP


# Training History
# ----------------
# Stores the loss curves and best validation epoch returned by the trainer.

@dataclass(frozen=True)
class TrainingHistory:
    """
    Storage utility for training history.
    """

    train_losses: list[float]
    validation_losses: list[float]
    best_epoch: int | None = None
    best_validation_loss: float | None = None


def clone_model_state(model: DenseMLP) -> nnx.State:
    """
    Copy the current trainable state of a live NNX model in memory.

    This is used for early stopping. When validation loss improves, the trainer
    copies the current weights so they can be restored at the end of training.
    This does not save a model file to disk.
    """
    # Copy the NNX state so later training updates do not change this snapshot.
    return copy.deepcopy(nnx.state(model))


def _preserve_device_array(array: np.ndarray | jax.Array) -> np.ndarray | jax.Array:
    """
    Keep JAX arrays on device while normalising non-JAX inputs to NumPy arrays.
    """
    if isinstance(array, jax.Array):
        return array
    return np.asarray(array)


# Mini-Batch Transfer
# -------------------
# Slices batches from host arrays and keeps a small queue ready on the device.

def _iter_device_batches(
    features: np.ndarray | jax.Array,
    targets: np.ndarray | jax.Array,
    batch_size: int,
    *,
    shuffle: bool,
    rng: np.random.Generator | None = None,
    prefetch_batches: int,
) -> Iterator[tuple[jax.Array, jax.Array]]:
    """
    For a given dataset stored on host memory, slice mini-batches and queue
    a small number of batches onto the JAX device ahead of them being requested
    by the training loop. This lets host-side batch preparation and host-to-device
    transfer overlap with computation already queued on the device.

    Parameters
    ----------
    features:
        Host feature matrix with shape (n_samples, n_features).
    targets:
        Host target array with shape (n_samples,).
    batch_size:
        Number of examples per mini-batch.
    shuffle:
        Whether to shuffle the dataset at the start of each epoch.
    rng:
        Optional random number generator for shuffling.
    prefetch_batches:
        Number of mini-batches to keep queued on the JAX device.

    Yields
    ------
    tuple[jax.Array, jax.Array]
        A tuple of device feature and target batches.
    """
    # Ensure that at least one batch is being prefetched.
    if prefetch_batches < 1:
        raise ValueError("prefetch_batches must be at least 1.")

    # Ensure features and targets have compatible shapes.
    if len(features) != len(targets):
        raise ValueError("features and targets must have the same length.")

    # Build row indices and optionally shuffle them for this epoch.
    indices = np.arange(len(features))
    if shuffle:
        if rng is None:
            rng = np.random.default_rng(0)
        indices = rng.permutation(indices)

    # Create an iterator over the starting row of each host mini-batch.
    batch_starts = iter(range(0, len(indices), batch_size))

    # Initialise an empty queue that will hold device batches.
    queue: deque[tuple[jax.Array, jax.Array]] = deque()

    def enqueue_next_batch() -> bool:
        """
        Slice the next host batch, transfer it to the device, and add it to the queue.
        """
        # Stop when there are no more host batches to provide.
        try:
            start = next(batch_starts)
        except StopIteration:
            return False

        # Select the next batch of indices and slice the host arrays.
        batch_index = indices[start : start + batch_size]
        batch_features = features[batch_index]
        batch_targets = targets[batch_index]

        # device_put queues the host-to-device transfer and returns a device array.
        queue.append((jax.device_put(batch_features), jax.device_put(batch_targets)))
        return True

    # Fill the queue before training starts consuming batches.
    for _ in range(prefetch_batches):
        if not enqueue_next_batch():
            break

    # Yield the oldest queued batch and replace it with the next host batch.
    while queue:
        device_batch = queue.popleft()
        enqueue_next_batch()
        yield device_batch


# Training Loop
# -------------
# Trains a DenseMLP on prepared arrays and returns the trained model plus loss history.

def train_mlp_regressor(
    model: DenseMLP,
    train_features: np.ndarray | jax.Array,
    train_targets: np.ndarray | jax.Array,
    validation_features: np.ndarray | jax.Array,
    validation_targets: np.ndarray | jax.Array,
    *,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    epochs: int = 50,
    seed: int = 0,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    prefetch_batches: int = 2,
    log_every: int | None = 1,
    log_prefix: str = "train_mlp_regressor",
) -> tuple[DenseMLP, TrainingHistory]:
    """
    Train an existing DenseMLP regressor on prepared feature and target arrays.

    Parameters
    ----------
    model:
        DenseMLP model to train.
    train_features, validation_features:
        Host feature matrices with shape (n_samples, n_features).
    train_targets, validation_targets:
        Host target arrays with shape (n_samples,).
    learning_rate, weight_decay:
        Optax AdamW parameters.
    batch_size:
        Number of training examples processed per gradient update.
    epochs:
        Number of passes over the training set.
    seed:
        Random seed for batch shuffling.
    early_stopping_patience:
        Stop after this many epochs without validation improvement.
    early_stopping_min_delta:
        Minimum validation-loss decrease counted as an improvement.
    prefetch_batches:
        Number of mini-batches to keep queued on the JAX device.

    Returns
    -------
    DenseMLP, TrainingHistory
        The trained model and training history.
    """
    # Keep NumPy arrays on the host for prefetching. If the caller passes JAX arrays,
    # leave them on device and slice batches from there instead of copying them back.
    train_features = _preserve_device_array(train_features)
    train_targets = _preserve_device_array(train_targets)
    validation_features = _preserve_device_array(validation_features)
    validation_targets = _preserve_device_array(validation_targets)

    # Initialise the random number generator for host-side batch shuffling.
    rng = np.random.default_rng(seed)

    # Initialise the AdamW optimiser for all trainable NNX parameters.
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
        """
        Run one compiled optimiser step on a device mini-batch.
        """

        def loss_fn(current_model: DenseMLP) -> jnp.ndarray:
            """
            Predict the mini-batch and return the mean squared error.
            """
            preds = current_model(batch_features).squeeze(-1)
            return mse(preds, batch_targets)

        # Compute the loss and gradients for current model parameters on this mini-batch.
        loss, grads = nnx.value_and_grad(loss_fn)(model_instance)

        # Use the optimizer to update the live NNX model in place.
        optimizer_instance.update(model_instance, grads)
        return loss

    @nnx.jit
    def eval_step(
        model_instance: DenseMLP,
        batch_features: jnp.ndarray,
        batch_targets: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Run one compiled validation step on a device mini-batch.
        """
        # Predict the mini-batch and measure the loss without changing weights.
        preds = model_instance(batch_features).squeeze(-1)
        return mse(preds, batch_targets)

    # Epoch-level host synchronization is only needed for Python-side logging or early stopping.
    sync_each_epoch = early_stopping_patience is not None or log_every is not None

    # Initalise lists for loss curves and early-stopping state to be stored.
    train_losses: list[float] = []
    validation_losses: list[float] = []
    train_loss_arrays: list[jax.Array] = []
    validation_loss_arrays: list[jax.Array] = []
    best_validation_loss = float("inf")
    best_epoch: int | None = None
    best_state: nnx.State | None = None
    epochs_without_improvement = 0

    # Main training loop.
    for epoch in range(epochs):
        epoch_start = perf_counter()
        train_loss_sum: jax.Array | None = None
        train_batch_count = 0
        # Within each epoch, the training set is reshuffeled and sliced into mini-batches.
        # Each training step is then performed on each mini-batch
        for batch_features, batch_targets in _iter_device_batches(
            train_features,
            train_targets,
            batch_size,
            shuffle=True,
            rng=rng,
            prefetch_batches=prefetch_batches,
        ):
            # Train (update) the model on the mini-batch
            loss = train_step(model, optimizer, batch_features, batch_targets)
            # Accumulate losses on device instead of storing one scalar per mini-batch.
            train_loss_sum = loss if train_loss_sum is None else train_loss_sum + loss
            train_batch_count += 1
        if train_loss_sum is None:
            raise ValueError("Training data produced no mini-batches.")
        # Average the training mini-batch losses on the device.
        train_loss_array = train_loss_sum / train_batch_count

        # Evaluate on validation dataset after all model updates for this epoch are done.
        validation_loss_sum: jax.Array | None = None
        validation_batch_count = 0
        for batch_features, batch_targets in _iter_device_batches(
            validation_features,
            validation_targets,
            batch_size,
            shuffle=False,
            prefetch_batches=prefetch_batches,
        ):
            # Evaluate the model on the mini-batch without updating weights.
            loss = eval_step(model, batch_features, batch_targets)
            validation_loss_sum = (
                loss if validation_loss_sum is None else validation_loss_sum + loss
            )
            validation_batch_count += 1
        if validation_loss_sum is None:
            raise ValueError("Validation data produced no mini-batches.")
        # Average the validation mini-batch losses on the device.
        validation_loss_array = validation_loss_sum / validation_batch_count

        # Keep the device scalars until host-side control flow or final history construction needs them.
        train_loss_arrays.append(train_loss_array)
        validation_loss_arrays.append(validation_loss_array)

        if sync_each_epoch:
            # Logging and early stopping require concrete Python floats, so this is the epoch sync point.
            train_loss = float(train_loss_array)
            validation_loss = float(validation_loss_array)

            train_losses.append(train_loss)
            validation_losses.append(validation_loss)

            # Track the best validation state for early stopping.
            if validation_loss < best_validation_loss - early_stopping_min_delta:
                best_validation_loss = validation_loss
                best_epoch = epoch
                best_state = clone_model_state(model)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        # Optionally print progress.
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

        # Stop if validation loss has not improved for the requested patience.
        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            break

    # Restore the best validation state if one was recorded.
    if best_state is not None:
        nnx.update(model, best_state)

    # If no Python-side control flow needed losses during training, transfer them once at the end.
    if not sync_each_epoch:
        train_losses = [float(loss) for loss in jax.device_get(jnp.asarray(train_loss_arrays))]
        validation_losses = [
            float(loss)
            for loss in jax.device_get(jnp.asarray(validation_loss_arrays))
        ]

    return model, TrainingHistory(
        train_losses=train_losses,
        validation_losses=validation_losses,
        best_epoch=best_epoch,
        best_validation_loss=None if best_epoch is None else best_validation_loss,
    )


# Evaluation
# ----------
# Computes validation-style loss for a trained model, including final test-set loss.

def evaluate_mlp_regressor(
    model: DenseMLP,
    features: np.ndarray | jax.Array,
    targets: np.ndarray | jax.Array,
    *,
    batch_size: int = 256,
    prefetch_batches: int = 2,
) -> float:
    """
    Evaluate mean squared error for a trained model on prepared arrays.

    This is used after training to score the held-out test dataset. It can also
    be used anywhere the model needs to be evaluated without updating weights.

    Parameters
    ----------
    model:
        Trained DenseMLP model.
    features:
        Host feature matrix, usually from the validation or test split.
    targets:
        Host target array aligned row-by-row with the feature matrix.
    batch_size:
        Number of examples evaluated per mini-batch.
    prefetch_batches:
        Number of mini-batches to keep queued on the JAX device.

    Returns
    -------
    float
        Mean squared error averaged over mini-batches.
    """

    @nnx.jit
    def eval_step(
        model_instance: DenseMLP,
        batch_features: jnp.ndarray,
        batch_targets: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Run one compiled evaluation step on a device mini-batch.
        """
        # Predict the mini-batch and measure the loss without changing weights.
        preds = model_instance(batch_features).squeeze(-1)
        return mse(preds, batch_targets)

    # Keep NumPy arrays on the host for prefetching. Preserve JAX arrays if already on device.
    features = _preserve_device_array(features)
    targets = _preserve_device_array(targets)

    # Evaluate every mini-batch and accumulate losses on device.
    loss_sum: jax.Array | None = None
    batch_count = 0
    for batch_features, batch_targets in _iter_device_batches(
        features,
        targets,
        batch_size,
        shuffle=False,
        prefetch_batches=prefetch_batches,
    ):
        loss = eval_step(model, batch_features, batch_targets)
        loss_sum = loss if loss_sum is None else loss_sum + loss
        batch_count += 1
    if loss_sum is None:
        raise ValueError("Evaluation data produced no mini-batches.")
    return float(loss_sum / batch_count)
