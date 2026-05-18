"""Mini-batch helpers for array-based training.

Right now the repository does not yet have a full dataset abstraction. This
module fills that gap with one small helper that slices in-memory arrays into
mini-batches in the same shape expected by the shared training loop.
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

    In practice this is the bridge between the current "all data already in
    memory" stage of the project and the eventual streamed data-loader stage.
    It handles only two responsibilities:

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
