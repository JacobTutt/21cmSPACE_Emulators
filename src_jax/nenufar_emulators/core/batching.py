"""Batch iteration utilities."""

from __future__ import annotations

from collections.abc import Iterator

import jax
import jax.numpy as jnp


def iter_batches(
    features: jnp.ndarray,
    targets: jnp.ndarray,
    batch_size: int,
    *,
    shuffle: bool = True,
    key: jax.Array | None = None,
) -> Iterator[tuple[jnp.ndarray, jnp.ndarray]]:
    """Yield mini-batches from in-memory arrays."""
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
