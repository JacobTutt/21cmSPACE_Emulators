"""
Training-data preparation workflows for the supported emulators.

This module provides the core logic for transforming raw simulation data into
shuffled, scaled, and flattened arrays ready for training a neural network.

1. remove failed simulations (handled before this module is called)
2. apply the configured physical-to-training target transform
3. split simulations into train / validation / test subsets
4. resample each simulation onto one deterministic shared grid
5. scale inputs and optionally divide targets by one global training-set std
6. flatten the grids into scalar regression rows for training
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from jaxemu_21cmSPACE.data_preprocessing.parameters import PreparedFeatures
from jaxemu_21cmSPACE.data_preprocessing.scaling import (
    FeatureScaler,
    FeatureScaling,
    TargetScalingScalar,
)
from jaxemu_21cmSPACE.data_preprocessing.specs import AxisSpec
from jaxemu_21cmSPACE.data_preprocessing.transforms import apply_transform


# Data Containers
# ---------------
# Structures for holding prepared train / validation / test splits.

@dataclass(frozen=True)
class PreparedSplit:
    """
    Prepared train / validation / test arrays ready for model fitting.

    This class bundles the final numerical arrays used in training with the
    scaling metadata required to invert the transforms during inference.
    """

    feature_names: tuple[str, ...]
    train_features: np.ndarray
    train_targets: np.ndarray
    validation_features: np.ndarray
    validation_targets: np.ndarray
    test_features: np.ndarray
    test_targets: np.ndarray
    feature_scaling: tuple[FeatureScaling, ...]
    target_scaling: TargetScalingScalar | None


# Main Workflow
# -------------
# Orchestrates the full data preparation pipeline.

def prepare_fixed_grid_training_split(
    *,
    axes: tuple[np.ndarray, ...],
    axis_specs: tuple[AxisSpec, ...],
    parameters: PreparedFeatures,
    target: np.ndarray,
    feature_scale_methods: dict[str, str],
    data_log: bool,
    offset: float | None,
    train_size: float = 0.6,
    validation_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle_seed: int = 42,
    standardize_target: bool = True,
) -> PreparedSplit:
    """
    Prepare one fixed-grid train / validation / test split.

    The key design choice is that the neural network no longer trains on a
    different random interpolation grid for every simulation. Instead we choose
    one shared grid from the workflow spec, resample every simulation onto that
    grid, then optionally divide all targets by one standard deviation measured
    from the training simulations only.

    Parameters
    ----------
    axes:
        The physical axis coordinates (e.g. z, k) for the raw simulation targets.
    axis_specs:
        Configuration for how each axis should be transformed and sampled.
    parameters:
        The prepared parameter table containing simulation inputs.
    target:
        The raw simulation target arrays (e.g. brightness temperature or power spectra).
    feature_scale_methods:
        Dictionary mapping feature names to scaling labels (e.g. 'zscore').
    data_log:
        Whether to apply a log10 transform to the target data.
    offset:
        Optional constant added to targets before log-transforming.
    train_size, validation_size, test_size:
        Fractional sizes for the data splits. Must sum to 1.
    random_state:
        Seed used for splitting simulations into subsets.
    shuffle_seed:
        Seed used for final row-wise shuffling of the flattened arrays.
    standardize_target:
        Whether to divide targets by one global training-label standard deviation.

    Returns
    -------
    PreparedSplit
        The processed and shuffled datasets ready for the trainer.
    """
    # Step 1: Apply the requested physical-to-training transform (e.g. log10) to targets.
    transformed_target = transform_target(target, data_log=data_log, offset=offset)

    # Step 2: Split the simulations (and their targets) into train, validation, and test sets.
    # This split happens at the simulation level, before resampling or flattening.
    (
        train_parameters,
        validation_parameters,
        test_parameters,
        train_target,
        validation_target,
        test_target,
    ) = split_simulations(
        parameters.values,
        transformed_target,
        train_size=train_size,
        validation_size=validation_size,
        test_size=test_size,
        random_state=random_state,
    )

    # Step 3: Configure the axes for resampling.
    # Determine the transformed physical coordinates and the limits of the shared training grid.
    transformed_axes, transformed_limits = transformed_axis_configuration(axes, axis_specs)
    # Build the deterministic shared grid onto which all simulations will be interpolated.
    sampled_axes = build_fixed_axis_grid(transformed_axes, transformed_limits, axis_specs)

    # Step 4: Construct the feature names.
    # The final feature matrix will contain the axis coordinates followed by the simulation parameters.
    axis_feature_names = tuple(axis.feature_name() for axis in axis_specs)
    feature_names = (*axis_feature_names, *parameters.feature_names)

    # Step 5: Resample every simulation onto the shared training grid.
    # This ensures that every sample in the final dataset is aligned.
    train_target_grid = resample_targets_to_grid(
        train_target,
        transformed_axes=transformed_axes,
        sampled_axes=sampled_axes,
    )
    validation_target_grid = resample_targets_to_grid(
        validation_target,
        transformed_axes=transformed_axes,
        sampled_axes=sampled_axes,
    )
    test_target_grid = resample_targets_to_grid(
        test_target,
        transformed_axes=transformed_axes,
        sampled_axes=sampled_axes,
    )

    # Step 6: Optionally apply global target standardization.
    # This follows globalemu: divide all labels by one std from the training set only.
    target_scaling = None
    if standardize_target:
        target_scaling = TargetScalingScalar.from_targets(train_target_grid)
        train_target_grid = target_scaling.transform_grid(train_target_grid)
        validation_target_grid = target_scaling.transform_grid(validation_target_grid)
        test_target_grid = target_scaling.transform_grid(test_target_grid)

    # Step 7: Flatten the resampled grids into individual regression rows.
    # Each row contains (axis_coords, parameters) as features and a single scalar target value.
    train_features, train_targets = flatten_resampled_rows(
        train_parameters,
        train_target_grid,
        sampled_axes=sampled_axes,
    )
    validation_features, validation_targets = flatten_resampled_rows(
        validation_parameters,
        validation_target_grid,
        sampled_axes=sampled_axes,
    )
    test_features, test_targets = flatten_resampled_rows(
        test_parameters,
        test_target_grid,
        sampled_axes=sampled_axes,
    )

    # Step 8: Build and apply the feature scaler.
    # Like target scaling, statistics are derived from the training set only.
    scaler = build_feature_scaler(
        train_features,
        feature_names=feature_names,
        method_overrides=feature_scale_methods,
    )
    train_features = scaler.transform(train_features).astype(np.float32)
    validation_features = scaler.transform(validation_features).astype(np.float32)
    test_features = scaler.transform(test_features).astype(np.float32)

    # Convert targets to float32 for JAX compatibility.
    train_targets = np.asarray(train_targets, dtype=np.float32)
    validation_targets = np.asarray(validation_targets, dtype=np.float32)
    test_targets = np.asarray(test_targets, dtype=np.float32)

    # Step 9: Final row-wise shuffling.
    # This breaks the block structure created by tiling parameters over the axis grid.
    train_features, train_targets = shuffle_rows(train_features, train_targets, seed=shuffle_seed)
    validation_features, validation_targets = shuffle_rows(
        validation_features,
        validation_targets,
        seed=shuffle_seed,
    )
    test_features, test_targets = shuffle_rows(test_features, test_targets, seed=shuffle_seed)

    # Bundle everything into the final PreparedSplit container.
    return PreparedSplit(
        feature_names=feature_names,
        train_features=train_features,
        train_targets=train_targets,
        validation_features=validation_features,
        validation_targets=validation_targets,
        test_features=test_features,
        test_targets=test_targets,
        feature_scaling=scaler.scaling,
        target_scaling=target_scaling,
    )


# Target Transformation
# ---------------------
# Helpers for applying physical-to-training transforms to simulation outputs.

def transform_target(
    target: np.ndarray,
    *,
    data_log: bool,
    offset: float | None,
) -> np.ndarray:
    """
    Apply the configured target transform before splitting simulations.

    Parameters
    ----------
    target:
        The raw simulation target values.
    data_log:
        Whether to apply log10 to the targets.
    offset:
        Optional constant to add to the targets before logging.

    Returns
    -------
    np.ndarray
        The transformed target array.
    """
    # Work on a copy of the input data to prevent side-effects.
    arr = np.asarray(target, dtype=float).copy()
    # If no log transform is requested, return the array as-is.
    if not data_log:
        return arr
    # If an offset is provided, add it before taking the log.
    if offset is not None:
        return np.log10(arr + offset)

    # If no offset is provided, we must handle zero-valued bins manually.
    non_zero = arr[arr != 0]
    if non_zero.size == 0:
        raise ValueError("Cannot apply zero-truncation to an all-zero target array.")
    # Find the smallest non-zero value to use as a baseline for flooring.
    minimum = float(non_zero.min())
    zero_mask = arr == 0
    if np.any(zero_mask):
        # Keep zero-valued bins finite in log space by replacing them with a
        # small positive floor tied to the smallest non-zero target value.
        arr[zero_mask] = minimum * 1e-3
    return np.log10(arr)


# Dataset Splitting
# -----------------
# Logic for partitioning simulations into training, validation, and test subsets.

def split_simulations(
    parameters: np.ndarray,
    target: np.ndarray,
    *,
    train_size: float,
    validation_size: float,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split simulations once into train / validation / test subsets.

    Parameters
    ----------
    parameters:
        Feature matrix (simulations, parameters).
    target:
        Target arrays (simulations, ...).
    train_size, validation_size, test_size:
        Target fractions for each split.
    random_state:
        Seed for the random permutation of simulation indices.

    Returns
    -------
    tuple of arrays
        (train_p, val_p, test_p, train_t, val_t, test_t)
    """
    # Validation checks for input shapes and split fractions.
    n_samples = len(parameters)
    if len(target) != n_samples:
        raise ValueError("parameters and target must have the same sample count.")
    if not np.isclose(train_size + validation_size + test_size, 1.0):
        raise ValueError("train_size + validation_size + test_size must sum to 1.")

    # Calculate the integer number of samples for each split.
    counts = np.floor(np.array([train_size, validation_size, test_size]) * n_samples).astype(int)
    # Distribute any remainder samples to the splits with the largest fractional parts.
    remainder = n_samples - int(counts.sum())
    fractional = np.array([train_size, validation_size, test_size]) * n_samples - counts
    for idx in np.argsort(-fractional)[:remainder]:
        counts[idx] += 1

    # Generate a random permutation of indices to shuffle the simulations.
    n_train, n_validation, n_test = map(int, counts)
    permutation = np.random.RandomState(random_state).permutation(n_samples)

    # Slice the permutation to get indices for each subset.
    train_indices = permutation[:n_train]
    validation_indices = permutation[n_train : n_train + n_validation]
    test_indices = permutation[n_train + n_validation : n_train + n_validation + n_test]

    # Return the sliced parameter and target arrays.
    return (
        np.asarray(parameters[train_indices], dtype=float),
        np.asarray(parameters[validation_indices], dtype=float),
        np.asarray(parameters[test_indices], dtype=float),
        np.asarray(target[train_indices], dtype=float),
        np.asarray(target[validation_indices], dtype=float),
        np.asarray(target[test_indices], dtype=float),
    )


# Resampling
# ----------
# Functions for interpolating simulations onto a shared coordinate grid.

def resample_targets_to_grid(
    target: np.ndarray,
    *,
    transformed_axes: tuple[np.ndarray, ...],
    sampled_axes: tuple[np.ndarray, ...],
) -> np.ndarray:
    """
    Resample each simulation onto one deterministic shared axis grid.

    Parameters
    ----------
    target:
        The target data for multiple simulations.
    transformed_axes:
        The physical axis coordinates for the input targets.
    sampled_axes:
        The coordinates of the shared grid to interpolate onto.

    Returns
    -------
    np.ndarray
        The resampled target data with shape (simulations, *grid_shape).
    """
    # Pre-calculate the coordinate combinations for the target grid.
    combinations = axis_combinations(sampled_axes)
    axis_shape = tuple(len(axis) for axis in sampled_axes)
    # Initialise the output array.
    resampled = np.empty((len(target), *axis_shape), dtype=float)

    # Loop through each simulation and perform linear interpolation.
    for idx, target_row in enumerate(target):
        interpolator = RegularGridInterpolator(
            transformed_axes,
            target_row,
            method="linear",
            bounds_error=True,
        )
        # Reshape the interpolated values back to the grid structure.
        resampled[idx] = np.asarray(interpolator(combinations), dtype=float).reshape(axis_shape)

    return resampled


def flatten_resampled_rows(
    parameters: np.ndarray,
    target_grid: np.ndarray,
    *,
    sampled_axes: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flatten one resampled target grid into scalar regression rows.

    This function expands the parameter matrix and concatenates it with
    the grid coordinates to create a single large feature matrix.

    Parameters
    ----------
    parameters:
        The simulation parameters with shape (simulations, n_params).
    target_grid:
        The resampled target grid with shape (simulations, *grid_shape).
    sampled_axes:
        The coordinates defining the grid.

    Returns
    -------
    features: np.ndarray
        The flattened feature matrix with shape (sims * grid_points, n_axes + n_params).
    targets: np.ndarray
        The flattened target array with shape (sims * grid_points,).
    """
    # Get all coordinate combinations for the grid.
    combinations = axis_combinations(sampled_axes)
    # Tile the parameters so each grid point for a simulation has the same parameter vector.
    tiled_parameters = np.repeat(parameters, repeats=len(combinations), axis=0)
    # Tile the grid coordinates for every simulation in the set.
    tiled_axes = np.tile(combinations, (len(parameters), 1))
    # Concatenate axes and parameters horizontally, then flatten targets.
    return np.hstack((tiled_axes, tiled_parameters)), target_grid.reshape(-1)


def axis_combinations(sampled_axes: tuple[np.ndarray, ...]) -> np.ndarray:
    """
    Return flattened axis-coordinate combinations for one shared grid.

    Parameters
    ----------
    sampled_axes:
        Tuple of 1D arrays defining each axis.

    Returns
    -------
    np.ndarray
        2D array of coordinate combinations with shape (n_points, n_axes).
    """
    # Use meshgrid to generate a coordinate grid for the provided axes.
    grids = np.meshgrid(*sampled_axes, indexing="ij")
    # Stack the flattened grids into a single coordinate matrix.
    return np.vstack([grid.ravel() for grid in grids]).T


# Axis Configuration
# ------------------
# Helpers for defining how simulation axes are transformed and sampled.

def transformed_axis_configuration(
    axes: tuple[np.ndarray, ...],
    axis_specs: tuple[AxisSpec, ...],
) -> tuple[tuple[np.ndarray, ...], tuple[tuple[float, float], ...]]:
    """
    Return transformed axis arrays and transformed sampling limits.

    Parameters
    ----------
    axes:
        Raw physical axis coordinates.
    axis_specs:
        Specifications for transformations and sampling bounds.

    Returns
    -------
    transformed_axes: tuple[np.ndarray, ...]
        The physical coordinates after applying requested transforms.
    transformed_limits: tuple[tuple[float, float], ...]
        The sampling limits in the transformed coordinate space.
    """
    transformed_axes = []
    transformed_limits = []
    # Loop through each axis and apply its specific configuration.
    for axis_values, axis_spec in zip(axes, axis_specs, strict=True):
        # Apply the transform (e.g. log10 or identity) to the axis values.
        transformed_axes.append(apply_transform(axis_values, axis_spec.transform))
        # If no explicit limits are provided, use the full range of the data.
        if axis_spec.limits is None:
            transformed_limits.append((float(axis_values.min()), float(axis_values.max())))
        else:
            # Otherwise, transform the provided limits into the same space as the data.
            limits = apply_transform(np.asarray(axis_spec.limits, dtype=float), axis_spec.transform)
            transformed_limits.append((float(limits[0]), float(limits[1])))
    return tuple(transformed_axes), tuple(transformed_limits)


def build_fixed_axis_grid(
    transformed_axes: tuple[np.ndarray, ...],
    transformed_limits: tuple[tuple[float, float], ...],
    axis_specs: tuple[AxisSpec, ...],
) -> tuple[np.ndarray, ...]:
    """
    Construct the deterministic axis grid used during training.

    Parameters
    ----------
    transformed_axes:
        The transformed physical coordinates of the raw simulations.
    transformed_limits:
        The bounds within which to sample the axes.
    axis_specs:
        Configuration defining the number of points for each axis.

    Returns
    -------
    tuple[np.ndarray, ...]
        The coordinates for each axis in the shared training grid.
    """
    sampled_axes = []
    for axis_values, (low, high), axis_spec in zip(
        transformed_axes,
        transformed_limits,
        axis_specs,
        strict=True,
    ):
        # If no sampling density is specified, use the subset of raw coordinates within the limits.
        if axis_spec.nsample is None:
            mask = np.logical_and(axis_values >= low, axis_values <= high)
            sampled_axes.append(np.asarray(axis_values[mask], dtype=float))
            continue
        # Otherwise, generate a linear grid with the requested number of points.
        sampled_axes.append(np.linspace(low, high, axis_spec.nsample, dtype=float))
    return tuple(sampled_axes)


# Feature Scaling
# ---------------
# Logic for standardizing feature matrices using training-set statistics.

def build_feature_scaler(
    feature_matrix: np.ndarray,
    *,
    feature_names: tuple[str, ...],
    method_overrides: dict[str, str],
) -> FeatureScaler:
    """
    Build per-feature scaler metadata from the training features.

    Parameters
    ----------
    feature_matrix:
        The matrix of training features used to compute statistics.
    feature_names:
        The names corresponding to each column in the feature matrix.
    method_overrides:
        Dictionary mapping feature names to specific scaling methods.

    Returns
    -------
    FeatureScaler
        A container for the scaling metadata of all features.
    """
    if feature_matrix.shape[1] != len(feature_names):
        raise ValueError("feature_names must match the feature matrix width.")

    scaling: list[FeatureScaling] = []
    # Loop through each feature and determine its scaling parameters.
    for idx, name in enumerate(feature_names):
        # Default to z-score standardization if no override is provided.
        method_name = method_overrides.get(name, "zscore")
        scaling_method = _resolve_scaling_method(method_name)
        # Calculate statistics (e.g. mean, std) from the provided training matrix.
        scaling.append(FeatureScaling.from_values(name, feature_matrix[:, idx], scaling_method))
    return FeatureScaler(tuple(scaling))


def _resolve_scaling_method(method: str) -> str:
    """
    Translate workflow scaling labels into explicit internal scaler names.
    """
    if method == "standardize":
        return "minmax_minus_one_to_one"
    if method == "normalize":
        return "zscore"
    if method == "identity":
        return "identity"
    if method == "zscore":
        return "zscore"
    if method == "minmax_zero_to_one":
        return "minmax_zero_to_one"
    raise ValueError(f"Unsupported scaling method {method!r}.")


# Utilities
# ---------
# General purpose helpers for data manipulation.

def shuffle_rows(features: np.ndarray, targets: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Shuffle rows with a fixed seed for deterministic repeatability.

    Parameters
    ----------
    features:
        The feature matrix to shuffle.
    targets:
        The target array to shuffle aligned with the features.
    seed:
        The random seed for reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The shuffled features and targets.
    """
    # Generate a reproducible permutation of indices.
    indices = np.random.RandomState(seed).permutation(len(targets))
    # Return the shuffled views of the input data.
    return features[indices], targets[indices]
