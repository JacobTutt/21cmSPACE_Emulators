"""
Mini-batch loading utilities for JAX training.

This module controls where prepared arrays live during training. Large datasets
can stay in host memory and stream fixed-size mini-batches to the accelerator.
Small datasets can be copied to the accelerator once and then sliced directly
on device.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np


DataDeviceMode = Literal["auto", "host_prefetch", "device_resident"]


def normalise_data_device_mode(mode: str) -> DataDeviceMode:
    """
    Validate and return the requested mini-batch loading mode.

    Parameters
    ----------
    mode:
        Loading mode. Supported values are `auto`, `host_prefetch`, and
        `device_resident`.

    Returns
    -------
    DataDeviceMode
        The validated loading mode.
    """
    if mode not in {"auto", "host_prefetch", "device_resident"}:
        raise ValueError(
            "data_device_mode must be one of: auto, host_prefetch, device_resident."
        )
    return mode  # type: ignore[return-value]


def resolve_data_device_mode(
    features: np.ndarray | jax.Array,
    targets: np.ndarray | jax.Array,
    mode: str,
) -> Literal["host_prefetch", "device_resident"]:
    """
    Resolve `auto` into a concrete mini-batch loading mode.

    Parameters
    ----------
    features, targets:
        Prepared arrays passed to the dataloader.
    mode:
        Requested loading mode.

    Returns
    -------
    Literal["host_prefetch", "device_resident"]
        Concrete loading mode used by the dataloader.
    """
    resolved_mode = normalise_data_device_mode(mode)
    if resolved_mode == "auto":
        return (
            "device_resident"
            if isinstance(features, jax.Array) and isinstance(targets, jax.Array)
            else "host_prefetch"
        )
    return resolved_mode


def move_arrays_to_device(
    *arrays: np.ndarray | jax.Array,
) -> tuple[jax.Array, ...]:
    """
    Move prepared arrays to the active JAX device once.

    This is used by `device_resident` training before the epoch loop starts.
    The dataloader can then slice mini-batches from arrays that are already on
    the device, instead of repeating device placement inside each epoch.
    """
    # device_put is asynchronous. The returned objects are JAX device arrays.
    return tuple(jax.device_put(array) for array in arrays)


def iter_device_batches(
    features: np.ndarray | jax.Array,
    targets: np.ndarray | jax.Array,
    batch_size: int,
    *,
    shuffle: bool,
    rng: np.random.Generator | None = None,
    prefetch_batches: int = 2,
    data_device_mode: str = "host_prefetch",
) -> Iterator[tuple[jax.Array, jax.Array, jax.Array, int]]:
    """
    Yield fixed-size mini-batches on the JAX device.

    Parameters
    ----------
    features:
        Prepared feature matrix with shape (n_samples, n_features).
    targets:
        Prepared target array with shape (n_samples,).
    batch_size:
        Number of examples per mini-batch.
    shuffle:
        Whether to shuffle the row order before batching.
    rng:
        Optional random number generator for reproducible shuffling.
    prefetch_batches:
        Number of mini-batches to keep queued on the device in
        `host_prefetch` mode.
    data_device_mode:
        `host_prefetch` keeps arrays on the host and queues mini-batches to the
        device. `device_resident` expects arrays that have already been moved
        to the device and slices mini-batches there. `auto` uses
        `device_resident` only when both inputs are already JAX arrays.

    Yields
    ------
    tuple[jax.Array, jax.Array, jax.Array, int]
        Device feature batch, device target batch, device mask batch, and the
        number of real examples before padding.
    """
    mode = resolve_data_device_mode(features, targets, data_device_mode)

    if mode == "device_resident":
        yield from _iter_device_resident_batches(
            features,
            targets,
            batch_size,
            shuffle=shuffle,
            rng=rng,
        )
        return

    yield from _iter_host_prefetched_batches(
        features,
        targets,
        batch_size,
        shuffle=shuffle,
        rng=rng,
        prefetch_batches=prefetch_batches,
    )


def _iter_host_prefetched_batches(
    features: np.ndarray | jax.Array,
    targets: np.ndarray | jax.Array,
    batch_size: int,
    *,
    shuffle: bool,
    rng: np.random.Generator | None,
    prefetch_batches: int,
) -> Iterator[tuple[jax.Array, jax.Array, jax.Array, int]]:
    """
    Slice host mini-batches and keep a small device queue ahead of training.
    """
    # Ensure that at least one batch is being prefetched.
    if prefetch_batches < 1:
        raise ValueError("prefetch_batches must be at least 1.")

    # In this mode, arrays are intentionally treated as host arrays.
    host_features = np.asarray(features)
    host_targets = np.asarray(targets)
    _validate_arrays(host_features, host_targets, batch_size)

    # Build row indices and optionally shuffle them for this epoch.
    indices = np.arange(len(host_features))
    if shuffle:
        if rng is None:
            rng = np.random.default_rng(0)
        indices = rng.permutation(indices)

    # Create an iterator over the starting row of each host mini-batch.
    batch_starts = iter(range(0, len(indices), batch_size))

    # Store batches that have already been queued onto the JAX device.
    queue: deque[tuple[jax.Array, jax.Array, jax.Array, int]] = deque()

    def enqueue_next_batch() -> bool:
        """
        Slice the next host batch, transfer it to the device, and queue it.
        """
        # Stop when there are no more host batches to provide.
        try:
            start = next(batch_starts)
        except StopIteration:
            return False

        # Select the next batch of rows and pad the final batch if needed.
        batch_index = indices[start : start + batch_size]
        real_examples = int(len(batch_index))
        batch_index = _pad_numpy_index(batch_index, batch_size)

        batch_features = host_features[batch_index]
        batch_targets = host_targets[batch_index]
        batch_mask = np.zeros(batch_size, dtype=np.float32)
        batch_mask[:real_examples] = 1.0

        # device_put queues the host-to-device transfer and returns a device array.
        queue.append(
            (
                jax.device_put(batch_features),
                jax.device_put(batch_targets),
                jax.device_put(batch_mask),
                real_examples,
            )
        )
        return True

    # Fill the queue before training starts consuming batches.
    for _ in range(prefetch_batches):
        if not enqueue_next_batch():
            break

    # Yield the oldest queued batch and replace it with the next host batch.
    while queue:
        device_batch = queue.popleft()
        enqueue_next_batch()
        yield device_batch


def _iter_device_resident_batches(
    features: np.ndarray | jax.Array,
    targets: np.ndarray | jax.Array,
    batch_size: int,
    *,
    shuffle: bool,
    rng: np.random.Generator | None,
) -> Iterator[tuple[jax.Array, jax.Array, jax.Array, int]]:
    """
    Slice mini-batches from arrays that already live on the JAX device.
    """
    if not isinstance(features, jax.Array) or not isinstance(targets, jax.Array):
        raise ValueError(
            "device_resident batching expects JAX arrays that have already been "
            "moved to the device."
        )
    _validate_arrays(features, targets, batch_size)

    # Build device-side row indices and optionally shuffle them for this epoch.
    indices = jnp.arange(len(features))
    if shuffle:
        seed = 0 if rng is None else int(rng.integers(0, np.iinfo(np.uint32).max))
        indices = jax.random.permutation(jax.random.PRNGKey(seed), indices)

    # Slice each mini-batch directly from the device-resident arrays.
    for start in range(0, len(features), batch_size):
        real_examples = min(batch_size, len(features) - start)
        batch_index = indices[start : start + batch_size]
        batch_index = _pad_jax_index(batch_index, batch_size)
        batch_mask = (jnp.arange(batch_size) < real_examples).astype(jnp.float32)

        yield (
            features[batch_index],
            targets[batch_index],
            batch_mask,
            real_examples,
        )


def _validate_arrays(
    features: np.ndarray | jax.Array,
    targets: np.ndarray | jax.Array,
    batch_size: int,
) -> None:
    """
    Validate the array shapes required by the mini-batch loaders.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if len(features) != len(targets):
        raise ValueError("features and targets must have the same length.")
    if len(features) == 0:
        raise ValueError("features and targets must contain at least one row.")


def _pad_numpy_index(batch_index: np.ndarray, batch_size: int) -> np.ndarray:
    """
    Pad a NumPy index array so the final mini-batch has a fixed shape.
    """
    real_examples = len(batch_index)
    if real_examples == batch_size:
        return batch_index

    # Repeat the final real row for padding. The mask removes it from the loss.
    pad_value = batch_index[-1]
    pad_index = np.full(batch_size - real_examples, pad_value, dtype=batch_index.dtype)
    return np.concatenate([batch_index, pad_index])


def _pad_jax_index(batch_index: jax.Array, batch_size: int) -> jax.Array:
    """
    Pad a JAX index array so the final mini-batch has a fixed shape.
    """
    real_examples = len(batch_index)
    if real_examples == batch_size:
        return batch_index

    # Repeat the final real row for padding. The mask removes it from the loss.
    pad_value = batch_index[-1]
    pad_index = jnp.full(batch_size - real_examples, pad_value, dtype=batch_index.dtype)
    return jnp.concatenate([batch_index, pad_index])
