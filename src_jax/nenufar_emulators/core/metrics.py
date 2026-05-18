"""Metrics utilities for training and validation code."""

from __future__ import annotations

import jax.numpy as jnp


def mse(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Mean squared error."""
    return jnp.mean(jnp.square(predictions - targets))


def rmse(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Root mean squared error."""
    return jnp.sqrt(mse(predictions, targets))
