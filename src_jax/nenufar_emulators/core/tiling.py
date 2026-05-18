"""Utilities for tiling spectral data into scalar regression samples."""

from __future__ import annotations

import numpy as np


def tile_spectra(
    parameters: np.ndarray,
    axes: tuple[np.ndarray, ...],
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Tile spectral targets into `[axes, params] -> scalar` samples."""
    params = np.asarray(parameters, dtype=float)
    y = np.asarray(targets, dtype=float)
    if params.ndim != 2:
        raise ValueError("parameters must be a 2D array.")
    if y.ndim != len(axes) + 1:
        raise ValueError("targets must have one sample dimension plus one per axis.")
    if y.shape[0] != params.shape[0]:
        raise ValueError("parameters and targets must share the same sample dimension.")

    axis_arrays = tuple(np.asarray(axis, dtype=float) for axis in axes)
    axis_shape = tuple(len(axis) for axis in axis_arrays)
    if y.shape[1:] != axis_shape:
        raise ValueError("target axis shape does not match provided axes.")

    mesh = np.meshgrid(*axis_arrays, indexing="ij")
    tiled_axes = np.stack([grid.ravel() for grid in mesh], axis=-1)
    repeated_axes = np.tile(tiled_axes, (params.shape[0], 1))
    repeated_params = np.repeat(params, repeats=tiled_axes.shape[0], axis=0)
    features = np.concatenate([repeated_axes, repeated_params], axis=-1)
    flat_targets = y.reshape(-1)
    return features, flat_targets, axis_shape


def reconstruct_spectra(
    flat_predictions: np.ndarray,
    nsamples: int,
    axis_shape: tuple[int, ...],
) -> np.ndarray:
    """Reconstruct spectral outputs from flattened scalar predictions."""
    preds = np.asarray(flat_predictions, dtype=float)
    expected = nsamples * int(np.prod(axis_shape))
    if preds.size != expected:
        raise ValueError("flat_predictions size is inconsistent with requested output shape.")
    return preds.reshape((nsamples, *axis_shape))
