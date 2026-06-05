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


DataDeviceMode = Literal["auto", "cpu_memory", "gpu_memory"]
ResolvedDataDeviceMode = Literal["cpu_memory", "gpu_memory"]


def normalise_data_device_mode(mode: str) -> DataDeviceMode:
    """
    Validate and return the requested mini-batch loading mode.

    Parameters
    ----------
    mode:
        Loading mode. Supported values are `auto`, `cpu_memory`, and
        `gpu_memory`.

    Returns
    -------
    DataDeviceMode
        The validated loading mode.
    """
    if mode not in {"auto", "cpu_memory", "gpu_memory"}:
        raise ValueError(
            "data_device_mode must be one of: auto, cpu_memory, gpu_memory."
        )
    return mode  # type: ignore[return-value]


def resolve_data_device_mode(
    features: np.ndarray | jax.Array,
    targets: np.ndarray | jax.Array,
    mode: str,
) -> ResolvedDataDeviceMode:
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
    ResolvedDataDeviceMode
        Concrete loading mode used by the dataloader.
    """
    resolved_mode = normalise_data_device_mode(mode)
    if resolved_mode == "auto":
        return (
            "gpu_memory"
            if isinstance(features, jax.Array) and isinstance(targets, jax.Array)
            else "cpu_memory"
        )
    return resolved_mode


def move_arrays_to_device(
    *arrays: np.ndarray | jax.Array,
) -> tuple[jax.Array, ...]:
    """
    Move prepared arrays to the active JAX device once.

    This is used by `gpu_memory` training before the epoch loop starts.
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
    data_device_mode: str = "cpu_memory",
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
        `cpu_memory` mode.
    data_device_mode:
        `cpu_memory` keeps arrays on the host and queues mini-batches to the
        device. `gpu_memory` expects arrays that have already been moved
        to the device and slices mini-batches there. `auto` uses
        `gpu_memory` only when both inputs are already JAX arrays.

    Yields
    ------
    tuple[jax.Array, jax.Array, jax.Array, int]
        Device feature batch, device target batch, device mask batch, and the
        number of real examples before padding.
    """
    mode = resolve_data_device_mode(features, targets, data_device_mode)

    if mode == "gpu_memory":
        yield from _iter_gpu_memory_batches(
            features,
            targets,
            batch_size,
            shuffle=shuffle,
            rng=rng,
        )
        return

    yield from _iter_cpu_prefetched_batches(
        features,
        targets,
        batch_size,
        shuffle=shuffle,
        rng=rng,
        prefetch_batches=prefetch_batches,
    )


def iter_device_batch_blocks(
    features: np.ndarray | jax.Array,
    targets: np.ndarray | jax.Array,
    batch_size: int,
    *,
    shuffle: bool,
    rng: np.random.Generator | None = None,
    prefetch_batches: int = 2,
    data_device_mode: str = "cpu_memory",
    batches_per_block: int = 1,
) -> Iterator[tuple[jax.Array, jax.Array, jax.Array, int]]:
    """
    Yield blocks of fixed-size mini-batches on the JAX device.

    Parameters
    ----------
    features:
        Prepared feature matrix with shape (n_samples, n_features).
    targets:
        Prepared target array with shape (n_samples,).
    batch_size:
        Number of examples in each mini-batch inside the block.
    shuffle:
        Whether to shuffle the row order before batching.
    rng:
        Optional random number generator for reproducible shuffling.
    prefetch_batches:
        Number of blocks to keep queued on the device in `cpu_memory` mode.
    data_device_mode:
        Mini-batch loading mode.
    batches_per_block:
        Number of mini-batches grouped into one scanned training step.

    Yields
    ------
    tuple[jax.Array, jax.Array, jax.Array, int]
        Device feature block, target block, mask block, and the number of real
        examples before padding.
    """
    if batches_per_block < 1:
        raise ValueError("batches_per_block must be at least 1.")

    if batches_per_block == 1:
        for batch_features, batch_targets, batch_mask, real_examples in iter_device_batches(
            features,
            targets,
            batch_size,
            shuffle=shuffle,
            rng=rng,
            prefetch_batches=prefetch_batches,
            data_device_mode=data_device_mode,
        ):
            yield (
                batch_features[jnp.newaxis, ...],
                batch_targets[jnp.newaxis, ...],
                batch_mask[jnp.newaxis, ...],
                real_examples,
            )
        return

    mode = resolve_data_device_mode(features, targets, data_device_mode)
    if mode == "gpu_memory":
        yield from _iter_gpu_memory_batch_blocks(
            features,
            targets,
            batch_size,
            shuffle=shuffle,
            rng=rng,
            batches_per_block=batches_per_block,
        )
        return

    yield from _iter_cpu_prefetched_batch_blocks(
        features,
        targets,
        batch_size,
        shuffle=shuffle,
        rng=rng,
        prefetch_batches=prefetch_batches,
        batches_per_block=batches_per_block,
    )


def _iter_cpu_prefetched_batches(
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


def _iter_cpu_prefetched_batch_blocks(
    features: np.ndarray | jax.Array,
    targets: np.ndarray | jax.Array,
    batch_size: int,
    *,
    shuffle: bool,
    rng: np.random.Generator | None,
    prefetch_batches: int,
    batches_per_block: int,
) -> Iterator[tuple[jax.Array, jax.Array, jax.Array, int]]:
    """
    Slice host mini-batch blocks and keep a small device queue ahead of training.
    """
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

    block_size = batch_size * batches_per_block
    block_starts = iter(range(0, len(indices), block_size))
    queue: deque[tuple[jax.Array, jax.Array, jax.Array, int]] = deque()

    def build_block(start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Build one block of host mini-batches with fixed batch-size padding.
        """
        feature_batches: list[np.ndarray] = []
        target_batches: list[np.ndarray] = []
        mask_batches: list[np.ndarray] = []
        real_examples_total = 0

        # Always return the same number of mini-batches per block. The final
        # block is padded and masked, which avoids recompiling the scanned step.
        for block_batch in range(batches_per_block):
            batch_start = start + block_batch * batch_size
            batch_index = indices[batch_start : batch_start + batch_size]
            real_examples = int(len(batch_index))
            real_examples_total += real_examples
            if real_examples == 0:
                batch_index = np.full(batch_size, indices[-1], dtype=indices.dtype)
            else:
                batch_index = _pad_numpy_index(batch_index, batch_size)

            batch_mask = np.zeros(batch_size, dtype=np.float32)
            batch_mask[:real_examples] = 1.0

            feature_batches.append(host_features[batch_index])
            target_batches.append(host_targets[batch_index])
            mask_batches.append(batch_mask)

        return (
            np.stack(feature_batches, axis=0),
            np.stack(target_batches, axis=0),
            np.stack(mask_batches, axis=0),
            real_examples_total,
        )

    def enqueue_next_block() -> bool:
        """
        Slice the next host block, transfer it to the device, and queue it.
        """
        try:
            start = next(block_starts)
        except StopIteration:
            return False

        block_features, block_targets, block_mask, real_examples = build_block(start)
        queue.append(
            (
                jax.device_put(block_features),
                jax.device_put(block_targets),
                jax.device_put(block_mask),
                real_examples,
            )
        )
        return True

    for _ in range(prefetch_batches):
        if not enqueue_next_block():
            break

    while queue:
        device_block = queue.popleft()
        enqueue_next_block()
        yield device_block


def _iter_gpu_memory_batches(
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
            "gpu_memory batching expects JAX arrays that have already been "
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


def _iter_gpu_memory_batch_blocks(
    features: np.ndarray | jax.Array,
    targets: np.ndarray | jax.Array,
    batch_size: int,
    *,
    shuffle: bool,
    rng: np.random.Generator | None,
    batches_per_block: int,
) -> Iterator[tuple[jax.Array, jax.Array, jax.Array, int]]:
    """
    Slice mini-batch blocks from arrays that already live on the JAX device.
    """
    if not isinstance(features, jax.Array) or not isinstance(targets, jax.Array):
        raise ValueError(
            "gpu_memory batching expects JAX arrays that have already been "
            "moved to the device."
        )
    _validate_arrays(features, targets, batch_size)

    indices = jnp.arange(len(features))
    if shuffle:
        seed = 0 if rng is None else int(rng.integers(0, np.iinfo(np.uint32).max))
        indices = jax.random.permutation(jax.random.PRNGKey(seed), indices)

    block_size = batch_size * batches_per_block
    for start in range(0, len(features), block_size):
        feature_batches: list[jax.Array] = []
        target_batches: list[jax.Array] = []
        mask_batches: list[jax.Array] = []
        real_examples_total = 0

        # Keep the leading block dimension fixed so the scanned training step
        # has one compiled shape for the whole epoch.
        for block_batch in range(batches_per_block):
            batch_start = start + block_batch * batch_size
            real_examples = min(batch_size, len(features) - batch_start)
            real_examples = max(real_examples, 0)
            real_examples_total += real_examples
            if real_examples == 0:
                batch_index = jnp.full(batch_size, indices[-1], dtype=indices.dtype)
            else:
                batch_index = indices[batch_start : batch_start + batch_size]
                batch_index = _pad_jax_index(batch_index, batch_size)
            batch_mask = (jnp.arange(batch_size) < real_examples).astype(jnp.float32)

            feature_batches.append(features[batch_index])
            target_batches.append(targets[batch_index])
            mask_batches.append(batch_mask)

        yield (
            jnp.stack(feature_batches, axis=0),
            jnp.stack(target_batches, axis=0),
            jnp.stack(mask_batches, axis=0),
            real_examples_total,
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
