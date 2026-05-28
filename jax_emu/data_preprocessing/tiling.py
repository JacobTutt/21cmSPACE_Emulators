"""
Utilities for tiling spectral data into scalar regression samples.

These workflows predict one scalar value at a time rather than an entire
spectrum at once. The network therefore learns a map of the form:

[axis values, astrophysical parameters] -> one target value

This lets the same dense MLP machinery serve both power-spectrum and
global-signal emulators.
"""

from __future__ import annotations

import numpy as np


# Spectral Tiling
# ---------------
# Logic for flattening grid-based signals into scalar regression rows.

def tile_spectra(
    parameters: np.ndarray,
    axes: tuple[np.ndarray, ...],
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """
    Flatten spectra into scalar-regression training examples.

    This is the practical heart of the emulator formulation used in both
    supported workflows. A full spectrum is converted into many rows of
    the form [axis coordinates, astrophysical parameters] -> scalar target.
    That lets one dense MLP architecture serve both power-spectrum and
    global-signal emulators.

    Parameters
    ----------
    parameters:
        Array of simulation parameters with shape (nsamples, nparams).
    axes:
        Tuple of one-dimensional axis arrays, for example (z, k) for power
        spectra or (z,) for global signals.
    targets:
        Spectral targets with shape (nsamples, *axis_shape).

    Returns
    -------
    features: np.ndarray
        The tiled feature matrix with shape (nsamples * grid_points, n_axes + n_params).
    flat_targets: np.ndarray
        The flattened target array with shape (nsamples * grid_points,).
    axis_shape: tuple[int, ...]
        The shape of the original spectral grid, used for reconstruction.
    """
    # Convert inputs to floating-point NumPy arrays before performing shape checks and tiling.
    params = np.asarray(parameters, dtype=float)
    y = np.asarray(targets, dtype=float)

    # Parameters must be structured as one row per simulation and one column per parameter.
    if params.ndim != 2:
        raise ValueError("parameters must be a 2D array.")

    # Targets must have a leading simulation (sample) axis followed by the physical spectral axes.
    if y.ndim != len(axes) + 1:
        raise ValueError("targets must have one sample dimension plus one per axis.")
    # The number of parameter rows must match the number of simulation target grids.
    if y.shape[0] != params.shape[0]:
        raise ValueError("parameters and targets must share the same sample dimension.")

    # Check that the numerical target grid shape matches the lengths of the supplied axis arrays.
    axis_arrays = tuple(np.asarray(axis, dtype=float) for axis in axes)
    axis_shape = tuple(len(axis) for axis in axis_arrays)
    if y.shape[1:] != axis_shape:
        raise ValueError("target axis shape does not match provided axes.")

    # Build the canonical coordinate grid for the spectra.
    # Using indexing="ij" ensures that the coordinate grid matches the natural axis order
    # found in the stored multi-dimensional target arrays.
    mesh = np.meshgrid(*axis_arrays, indexing="ij")
    # Stack the flattened coordinate grids to get a (grid_points, n_axes) matrix.
    tiled_axes = np.stack([grid.ravel() for grid in mesh], axis=-1)

    # Now broadcast the grid coordinates and parameters to create the full dataset.
    # 1. Repeat the grid coordinate matrix for every simulation in the set.
    repeated_axes = np.tile(tiled_axes, (params.shape[0], 1))
    # 2. Repeat each simulation's parameter vector for every point in the spectral grid.
    repeated_params = np.repeat(params, repeats=tiled_axes.shape[0], axis=0)

    # Final feature order is canonical: axes first, then astrophysical parameters.
    features = np.concatenate([repeated_axes, repeated_params], axis=-1)

    # Targets are flattened in the exact same order as the tiled axis grid.
    flat_targets = y.reshape(-1)
    return features, flat_targets, axis_shape


# Spectral Reconstruction
# -----------------------
# Logic for folding flattened predictions back into grid-based signals.

def reconstruct_spectra(
    flat_predictions: np.ndarray,
    nsamples: int,
    axis_shape: tuple[int, ...],
) -> np.ndarray:
    """
    Restore flattened emulator outputs back to spectrum-shaped arrays.

    This is the shape-level inverse of tile_spectra. It exists because
    the neural network trains on flattened scalar targets, while downstream
    science code usually wants predictions back in per-simulation spectral
    arrays.

    Parameters
    ----------
    flat_predictions:
        The vector of flattened predictions with shape (nsamples * grid_points,).
    nsamples:
        The number of simulations represented in the predictions.
    axis_shape:
        The shape of the spectral grid for each simulation.

    Returns
    -------
    np.ndarray
        The reshaped spectral predictions with shape (nsamples, *axis_shape).
    """
    # Convert to a flat NumPy array before checking the requested output size.
    preds = np.asarray(flat_predictions, dtype=float)

    # The flattened predictions must exactly fill the requested output spectral grid.
    # We calculate the expected total number of elements.
    expected = nsamples * int(np.prod(axis_shape))
    if preds.size != expected:
        raise ValueError("flat_predictions size is inconsistent with requested output shape.")

    # Reshape the flat vector back into simulation-major grid arrays.
    return preds.reshape((nsamples, *axis_shape))
