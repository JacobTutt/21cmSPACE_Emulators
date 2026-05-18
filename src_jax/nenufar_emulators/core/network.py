"""Generic JAX MLP building blocks.

The goal here is not to lock in a final network API yet. Instead, this module
provides a very small, readable baseline that mirrors the dense feed-forward
structure used in the legacy PyTorch emulators.
"""

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
    """Initialize a dense MLP as a JAX pytree.

    The returned list-of-dicts structure is intentionally plain: it is easy to
    inspect in tests, easy to pass into Optax, and easy to serialize later.
    """
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
    """Run a dense MLP forward pass.

    Hidden layers use the requested non-linearity; the final layer stays linear
    because the emulator target transform, if any, is handled outside the
    network itself.
    """
    act_fn = getattr(jax.nn, activation)
    x = inputs
    # Apply activation only on hidden layers so the caller retains full control
    # over any output-space transform such as log10 or offset handling.
    for layer in params[:-1]:
        x = act_fn(jnp.dot(x, layer["weights"]) + layer["bias"])
    final = params[-1]
    return jnp.dot(x, final["weights"]) + final["bias"]
