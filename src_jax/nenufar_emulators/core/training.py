"""Minimal JAX training utilities for emulator development."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from nenufar_emulators.core.batching import iter_batches
from nenufar_emulators.core.metrics import mse
from nenufar_emulators.core.network import forward_mlp, init_mlp


@dataclass(frozen=True)
class TrainingHistory:
    """Training curves returned by the shared trainer."""

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
    activation: str = "relu",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    epochs: int = 50,
    seed: int = 0,
) -> tuple[list[dict[str, jnp.ndarray]], TrainingHistory]:
    """Train a small MLP regressor on in-memory arrays."""
    key = jax.random.PRNGKey(seed)
    params = init_mlp(
        key=key,
        in_features=int(train_features.shape[1]),
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
    )
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(
        model_params: list[dict[str, jnp.ndarray]],
        state: optax.OptState,
        batch_features: jnp.ndarray,
        batch_targets: jnp.ndarray,
    ) -> tuple[list[dict[str, jnp.ndarray]], optax.OptState, jnp.ndarray]:
        def loss_fn(p: list[dict[str, jnp.ndarray]]) -> jnp.ndarray:
            preds = forward_mlp(p, batch_features, activation=activation).squeeze(-1)
            return mse(preds, batch_targets)

        loss, grads = jax.value_and_grad(loss_fn)(model_params)
        updates, new_state = optimizer.update(grads, state, model_params)
        new_params = optax.apply_updates(model_params, updates)
        return new_params, new_state, loss

    @jax.jit
    def eval_step(
        model_params: list[dict[str, jnp.ndarray]],
        batch_features: jnp.ndarray,
        batch_targets: jnp.ndarray,
    ) -> jnp.ndarray:
        preds = forward_mlp(model_params, batch_features, activation=activation).squeeze(-1)
        return mse(preds, batch_targets)

    train_losses: list[float] = []
    validation_losses: list[float] = []

    for epoch in range(epochs):
        key, train_key = jax.random.split(key)
        train_loss = 0.0
        train_batches = 0
        for batch_features, batch_targets in iter_batches(
            train_features,
            train_targets,
            batch_size,
            shuffle=True,
            key=train_key,
        ):
            params, opt_state, loss = train_step(params, opt_state, batch_features, batch_targets)
            train_loss += float(loss)
            train_batches += 1
        train_loss /= max(train_batches, 1)

        validation_loss = 0.0
        validation_batches = 0
        for batch_features, batch_targets in iter_batches(
            validation_features,
            validation_targets,
            batch_size,
            shuffle=False,
        ):
            validation_loss += float(eval_step(params, batch_features, batch_targets))
            validation_batches += 1
        validation_loss /= max(validation_batches, 1)

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        _ = epoch

    return params, TrainingHistory(train_losses=train_losses, validation_losses=validation_losses)
