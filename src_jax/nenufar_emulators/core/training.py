"""Minimal JAX training utilities for emulator development.

This is deliberately a small, transparent training loop. It is meant to be
easy to audit during the migration phase before we introduce more elaborate
training infrastructure or dataset abstractions.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from nenufar_emulators.core.batching import iter_batches
from nenufar_emulators.core.metrics import mse
from nenufar_emulators.core.network import ActivationName, DenseMLP, init_mlp


@dataclass(frozen=True)
class TrainingHistory:
    """Training curves returned by the shared trainer.

    Using a dataclass rather than a raw tuple makes future extensions
    straightforward if we later want to track learning-rate schedules, best
    checkpoints, or auxiliary metrics.
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
    """Train a small MLP regressor on in-memory arrays.

    This trainer is intentionally array-based because the repository does not
    yet have production dataset loaders. It is still sufficient for:

    - synthetic smoke testing
    - validating model and tiling contracts
    - prototyping legacy-aligned configuration bundles
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
        """Run one gradient-update step on a mini-batch."""

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
        """Evaluate loss on one validation mini-batch without updating weights."""
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
