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
from dataclasses import dataclass
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from jax_emu.architectures.mlp import DenseMLP
from jax_emu.training.dataloader import iter_device_batches
from jax_emu.training.scheduler import build_learning_rate_schedule, count_steps_per_epoch
from jax_emu.training.shutdown import GracefulShutdown, time_limit_reached


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
    stopped_reason: str | None = None


def clone_model_state(model: DenseMLP) -> nnx.State:
    """
    Copy the current trainable state of a live NNX model in memory.

    This is used for early stopping. When validation loss improves, the trainer
    copies the current weights so they can be restored at the end of training.
    This does not save a model file to disk.
    """
    # Copy the NNX state so later training updates do not change this snapshot.
    return copy.deepcopy(nnx.state(model))


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
    learning_rate_schedule: str = "constant",
    learning_rate_final_fraction: float = 0.1,
    learning_rate_warmup_epochs: int = 0,
    batch_size: int = 256,
    epochs: int = 50,
    seed: int = 0,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    prefetch_batches: int = 2,
    data_device_mode: str = "auto",
    max_runtime_seconds: float | None = None,
    shutdown_margin_seconds: float = 600.0,
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
    learning_rate_schedule:
        Schedule used for the AdamW learning rate. Supported values are
        `constant`, `cosine`, `warmup_cosine`, and `exponential_decay`.
    learning_rate_final_fraction:
        Final learning-rate fraction for `cosine`, `warmup_cosine`, and
        `exponential_decay`. For example, `0.05` means the schedule ends at
        `learning_rate * 0.05`. Ignored by `constant`.
    learning_rate_warmup_epochs:
        Number of epochs used to ramp from zero to `learning_rate` for
        `warmup_cosine`. Ignored by the other schedules.
    batch_size:
        Number of training examples processed per gradient update.
    epochs:
        Number of passes over the training set. This is also the schedule
        horizon used by the decay schedules.
    seed:
        Random seed for batch shuffling.
    early_stopping_patience:
        Stop after this many epochs without validation improvement.
    early_stopping_min_delta:
        Minimum validation-loss decrease counted as an improvement.
    prefetch_batches:
        Number of mini-batches to keep queued on the JAX device.
    data_device_mode:
        Mini-batch loading mode. `host_prefetch` streams host mini-batches to
        the device. `device_resident` copies full arrays to the device once and
        slices mini-batches there. `auto` uses `device_resident` only when the
        input arrays are already JAX arrays.
    max_runtime_seconds:
        Optional wall-clock training budget. After each epoch, the trainer
        checks whether another epoch is likely to fit inside this budget.
    shutdown_margin_seconds:
        Time reserved at the end of a timed run for test evaluation and model
        saving after the training loop returns.

    Returns
    -------
    DenseMLP, TrainingHistory
        The trained model and training history.
    """
    # Initialise the random number generator for host-side batch shuffling.
    rng = np.random.default_rng(seed)

    # Build the step-wise learning-rate schedule for the optimiser.
    steps_per_epoch = count_steps_per_epoch(len(train_features), batch_size)
    learning_rate_or_schedule = build_learning_rate_schedule(
        learning_rate=learning_rate,
        schedule_name=learning_rate_schedule,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        final_fraction=learning_rate_final_fraction,
        warmup_epochs=learning_rate_warmup_epochs,
    )

    # Initialise the AdamW optimiser for all trainable NNX parameters.
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(learning_rate=learning_rate_or_schedule, weight_decay=weight_decay),
        wrt=nnx.Param,
    )

    @nnx.jit
    def train_step(
        model_instance: DenseMLP,
        optimizer_instance: nnx.Optimizer,
        batch_features: jnp.ndarray,
        batch_targets: jnp.ndarray,
        batch_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Run one compiled optimiser step on a device mini-batch.
        """

        def loss_fn(current_model: DenseMLP) -> jnp.ndarray:
            """
            Predict the mini-batch and return the mean squared error.
            """
            preds = current_model(batch_features).squeeze(-1)
            squared_error = jnp.square(preds - batch_targets) * batch_mask
            return jnp.sum(squared_error) / jnp.maximum(jnp.sum(batch_mask), 1.0)

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
        batch_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Run one compiled validation step on a device mini-batch.
        """
        preds = model_instance(batch_features).squeeze(-1)
        squared_error = jnp.square(preds - batch_targets) * batch_mask
        return jnp.sum(squared_error)

    # Initialise lists for loss curves and early-stopping state to be stored.
    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_loss = float("inf")
    best_epoch: int | None = None
    best_state: nnx.State | None = None
    epochs_without_improvement = 0
    epoch_seconds: list[float] = []
    stopped_reason: str | None = None
    training_start = perf_counter()

    # Main training loop.
    with GracefulShutdown(log_prefix=log_prefix) as shutdown:
        for epoch in range(epochs):
            epoch_start = perf_counter()
            train_squared_error: list[jax.Array] = []
            train_example_count = 0
            # Within each epoch, the training set is reshuffeled and sliced into mini-batches.
            # Each training step is then performed on each mini-batch
            for batch_features, batch_targets, batch_mask, real_examples in iter_device_batches(
                train_features,
                train_targets,
                batch_size,
                shuffle=True,
                rng=rng,
                prefetch_batches=prefetch_batches,
                data_device_mode=data_device_mode,
            ):
                # Train (update) the model on the mini-batch
                loss = train_step(model, optimizer, batch_features, batch_targets, batch_mask)
                train_squared_error.append(loss * real_examples)
                train_example_count += real_examples
            if not train_squared_error:
                raise ValueError("Training data produced no mini-batches.")
            train_loss = float(
                np.asarray(jax.device_get(train_squared_error), dtype=np.float64).sum()
            )
            train_loss /= max(train_example_count, 1)

            # Evaluate on validation dataset after all model updates for this epoch are done.
            validation_squared_error: list[jax.Array] = []
            validation_example_count = 0
            for batch_features, batch_targets, batch_mask, real_examples in iter_device_batches(
                validation_features,
                validation_targets,
                batch_size,
                shuffle=False,
                prefetch_batches=prefetch_batches,
                data_device_mode=data_device_mode,
            ):
                # Evaluate the model on the mini-batch without updating weights.
                loss = eval_step(model, batch_features, batch_targets, batch_mask)
                validation_squared_error.append(loss)
                validation_example_count += real_examples
            if not validation_squared_error:
                raise ValueError("Validation data produced no mini-batches.")
            validation_loss = float(
                np.asarray(jax.device_get(validation_squared_error), dtype=np.float64).sum()
            )
            validation_loss /= max(validation_example_count, 1)

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

            elapsed_epoch_seconds = perf_counter() - epoch_start
            epoch_seconds.append(elapsed_epoch_seconds)

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
                message += f" epoch_seconds={elapsed_epoch_seconds:.2f}"
                print(message, flush=True)

            # Stop if Slurm or the user has requested a clean shutdown.
            if shutdown.stop_requested:
                stopped_reason = shutdown.reason
                break

            # Stop if the next epoch may leave too little time to evaluate and save.
            if time_limit_reached(
                training_start=training_start,
                max_runtime_seconds=max_runtime_seconds,
                shutdown_margin_seconds=shutdown_margin_seconds,
                epoch_seconds=epoch_seconds,
            ):
                stopped_reason = (
                    "wall-time budget reached before another full epoch could be run safely"
                )
                print(f"[{log_prefix}] {stopped_reason}.", flush=True)
                break

            # Stop if validation loss has not improved for the requested patience.
            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                stopped_reason = "early stopping patience reached"
                break

    # Restore the best validation state if one was recorded.
    if best_state is not None:
        nnx.update(model, best_state)

    return model, TrainingHistory(
        train_losses=train_losses,
        validation_losses=validation_losses,
        best_epoch=best_epoch,
        best_validation_loss=None if best_epoch is None else best_validation_loss,
        stopped_reason=stopped_reason,
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
    data_device_mode: str = "auto",
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
    data_device_mode:
        Mini-batch loading mode. `host_prefetch` streams host mini-batches to
        the device. `device_resident` copies full arrays to the device once and
        slices mini-batches there. `auto` uses `device_resident` only when the
        input arrays are already JAX arrays.

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
        batch_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Run one compiled evaluation step on a device mini-batch.
        """
        preds = model_instance(batch_features).squeeze(-1)
        squared_error = jnp.square(preds - batch_targets) * batch_mask
        return jnp.sum(squared_error)

    # Evaluate every mini-batch and accumulate losses on device.
    squared_error: list[jax.Array] = []
    example_count = 0
    for batch_features, batch_targets, batch_mask, real_examples in iter_device_batches(
        features,
        targets,
        batch_size,
        shuffle=False,
        prefetch_batches=prefetch_batches,
        data_device_mode=data_device_mode,
    ):
        loss = eval_step(model, batch_features, batch_targets, batch_mask)
        squared_error.append(loss)
        example_count += real_examples
    if not squared_error:
        raise ValueError("Evaluation data produced no mini-batches.")
    total_squared_error = float(np.asarray(jax.device_get(squared_error), dtype=np.float64).sum())
    return total_squared_error / max(example_count, 1)
