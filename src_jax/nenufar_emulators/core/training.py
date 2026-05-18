"""Minimal JAX training utilities for emulator development.

The goal of this module is still the same as before: keep the optimization
logic small and auditable. The difference is that it now supports both the new
dataset-driven workflow and the older direct-array fallback used in tests and
small synthetic experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from nenufar_emulators.core.batching import iter_batches
from nenufar_emulators.core.datasets import SpectrumDataset, TiledBatch
from nenufar_emulators.core.metrics import mse
from nenufar_emulators.core.network import ActivationName, DenseMLP, init_mlp


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
) -> tuple[DenseMLP, TrainingHistory]:
    """Train the shared dense emulator network on prepared in-memory arrays.

    This trainer is intentionally array-based and now serves as the fallback
    path beneath the newer dataset-driven workflow. It is still useful for:

    - synthetic smoke testing
    - validating model and tiling contracts
    - prototyping legacy-aligned configuration bundles

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

    for epoch in range(epochs):
        key, train_key = jax.random.split(key)
        train_loss = 0.0
        train_batches = 0
        # Training batches are shuffled each epoch, mirroring the behavior we
        # will want once real dataset iterators are connected.
        for batch_features, batch_targets in iter_batches(
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
        # Validation is intentionally deterministic so changes in reported loss
        # reflect model updates rather than batch-order noise.
        for batch_features, batch_targets in iter_batches(
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

    return model, TrainingHistory(train_losses=train_losses, validation_losses=validation_losses)


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

    for _ in range(epochs):
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

    return model, TrainingHistory(train_losses=train_losses, validation_losses=validation_losses)
