"""Generic JAX MLP building blocks."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp


ActivationName = Literal["relu", "tanh", "gelu"]


def init_mlp(
    key: jax.Array,
    in_features: int,
    hidden_features: int,
    out_features: int = 1,
    hidden_layers: int = 2,
    scale: float = 1e-1,
) -> list[dict[str, jnp.ndarray]]:
    """Initialize a dense MLP as a JAX pytree."""
    layer_sizes = [in_features]
    layer_sizes.extend([hidden_features] * hidden_layers)
    layer_sizes.append(out_features)
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params: list[dict[str, jnp.ndarray]] = []
    for idx, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        k = keys[idx]
        params.append(
            {
                "weights": scale * jax.random.normal(k, (in_dim, out_dim)),
                "bias": jnp.zeros((out_dim,), dtype=jnp.float32),
            }
        )
    return params


def forward_mlp(
    params: list[dict[str, jnp.ndarray]],
    inputs: jnp.ndarray,
    activation: ActivationName = "relu",
) -> jnp.ndarray:
    """Run a dense MLP forward pass."""
    act_fn = getattr(jax.nn, activation)
    x = inputs
    for layer in params[:-1]:
        x = act_fn(jnp.dot(x, layer["weights"]) + layer["bias"])
    final = params[-1]
    return jnp.dot(x, final["weights"]) + final["bias"]
