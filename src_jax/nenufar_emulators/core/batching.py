"""Mini-batch helpers for the array-based training fallback.

The repository now has a proper dataset abstraction for the main training
path, but these helpers still matter for low-level tests and simple synthetic
array experiments where constructing a dataset object would add noise rather
than clarity.
"""

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
    """Yield feature/target mini-batches from matching in-memory arrays.

    In practice this is the lightweight alternative to the dataset-owned batch
    iterators. It handles only two responsibilities:

    - verify that features and targets line up sample-by-sample
    - optionally shuffle sample order once per epoch before slicing batches

    Parameters
    ----------
    features:
        Two-dimensional feature matrix with one row per training example.
    targets:
        Target array with the same leading dimension as ``features``.
    batch_size:
        Number of examples yielded in each batch.
    shuffle:
        Whether to randomize sample order before batching. Use ``True`` for
        training and ``False`` for deterministic evaluation.
    key:
        JAX random key used when ``shuffle`` is enabled. If omitted, a fixed
        default key is used so the behavior stays deterministic.
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
