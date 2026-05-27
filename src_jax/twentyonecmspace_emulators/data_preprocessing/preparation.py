"""Training-data preparation workflows for the supported emulators.

The current repository supports two training products:

- `T21`, represented on one shared redshift grid
- `Delta21`, represented on one shared `(z, k)` grid

Both now follow the same preparation pattern:

1. remove failed simulations
2. apply the configured physical-to-training target transform
3. split simulations into train / validation / test
4. resample each split onto one deterministic shared grid
5. normalize inputs and targets using training-only statistics
6. flatten the grids into scalar regression rows
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from twentyonecmspace_emulators.data_preprocessing.parameters import PreparedFeatures
from twentyonecmspace_emulators.utils.scaling import FeatureScaler, FeatureScaling, TargetScalingSurface
from twentyonecmspace_emulators.utils.specs import AxisSpec
from twentyonecmspace_emulators.utils.transforms import apply_transform


@dataclass(frozen=True)
class PreparedSplit:
    """Prepared train / validation / test arrays ready for model fitting."""

    feature_names: tuple[str, ...]
    train_features: np.ndarray
    train_targets: np.ndarray
    validation_features: np.ndarray
    validation_targets: np.ndarray
    test_features: np.ndarray
    test_targets: np.ndarray
    feature_scaling: tuple[FeatureScaling, ...]
    target_scaling: TargetScalingSurface | None


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
    """Prepare one fixed-grid train / validation / test split.

    The key design choice is that the neural network no longer trains on a
    different random interpolation grid for every simulation. Instead we choose
    one shared grid from the workflow spec, resample every simulation onto that
    grid, then standardize each target bin using the training simulations only.
    """
    transformed_target = transform_target(target, data_log=data_log, offset=offset)
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

    transformed_axes, transformed_limits = transformed_axis_configuration(axes, axis_specs)
    sampled_axes = build_fixed_axis_grid(transformed_axes, transformed_limits, axis_specs)
    axis_feature_names = tuple(axis.feature_name() for axis in axis_specs)
    feature_names = (*axis_feature_names, *parameters.feature_names)

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

    target_scaling = None
    if standardize_target:
        target_scaling = TargetScalingSurface.from_targets(
            axis_names=axis_feature_names,
            axis_values=sampled_axes,
            targets=train_target_grid,
        )
        train_target_grid = target_scaling.transform_grid(train_target_grid)
        validation_target_grid = target_scaling.transform_grid(validation_target_grid)
        test_target_grid = target_scaling.transform_grid(test_target_grid)

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

    scaler = build_feature_scaler(
        train_features,
        feature_names=feature_names,
        method_overrides=feature_scale_methods,
    )
    train_features = scaler.transform(train_features).astype(np.float32)
    validation_features = scaler.transform(validation_features).astype(np.float32)
    test_features = scaler.transform(test_features).astype(np.float32)

    train_targets = np.asarray(train_targets, dtype=np.float32)
    validation_targets = np.asarray(validation_targets, dtype=np.float32)
    test_targets = np.asarray(test_targets, dtype=np.float32)

    train_features, train_targets = shuffle_rows(train_features, train_targets, seed=shuffle_seed)
    validation_features, validation_targets = shuffle_rows(
        validation_features,
        validation_targets,
        seed=shuffle_seed,
    )
    test_features, test_targets = shuffle_rows(test_features, test_targets, seed=shuffle_seed)

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


def transform_target(
    target: np.ndarray,
    *,
    data_log: bool,
    offset: float | None,
) -> np.ndarray:
    """Apply the configured target transform before splitting simulations."""
    arr = np.asarray(target, dtype=float).copy()
    if not data_log:
        return arr
    if offset is not None:
        return np.log10(arr + offset)

    non_zero = arr[arr != 0]
    if non_zero.size == 0:
        raise ValueError("Cannot apply zero-truncation to an all-zero target array.")
    minimum = float(non_zero.min())
    zero_mask = arr == 0
    if np.any(zero_mask):
        # Keep zero-valued bins finite in log space by replacing them with a
        # small positive floor tied to the smallest non-zero target value.
        arr[zero_mask] = minimum * 1e-3
    return np.log10(arr)


def split_simulations(
    parameters: np.ndarray,
    target: np.ndarray,
    *,
    train_size: float,
    validation_size: float,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split simulations once into train / validation / test subsets."""
    n_samples = len(parameters)
    if len(target) != n_samples:
        raise ValueError("parameters and target must have the same sample count.")
    if not np.isclose(train_size + validation_size + test_size, 1.0):
        raise ValueError("train_size + validation_size + test_size must sum to 1.")

    counts = np.floor(np.array([train_size, validation_size, test_size]) * n_samples).astype(int)
    remainder = n_samples - int(counts.sum())
    fractional = np.array([train_size, validation_size, test_size]) * n_samples - counts
    for idx in np.argsort(-fractional)[:remainder]:
        counts[idx] += 1

    n_train, n_validation, n_test = map(int, counts)
    permutation = np.random.RandomState(random_state).permutation(n_samples)

    train_indices = permutation[:n_train]
    validation_indices = permutation[n_train : n_train + n_validation]
    test_indices = permutation[n_train + n_validation : n_train + n_validation + n_test]

    return (
        np.asarray(parameters[train_indices], dtype=float),
        np.asarray(parameters[validation_indices], dtype=float),
        np.asarray(parameters[test_indices], dtype=float),
        np.asarray(target[train_indices], dtype=float),
        np.asarray(target[validation_indices], dtype=float),
        np.asarray(target[test_indices], dtype=float),
    )


def resample_targets_to_grid(
    target: np.ndarray,
    *,
    transformed_axes: tuple[np.ndarray, ...],
    sampled_axes: tuple[np.ndarray, ...],
) -> np.ndarray:
    """Resample each simulation onto one deterministic shared axis grid."""
    combinations = axis_combinations(sampled_axes)
    axis_shape = tuple(len(axis) for axis in sampled_axes)
    resampled = np.empty((len(target), *axis_shape), dtype=float)

    for idx, target_row in enumerate(target):
        interpolator = RegularGridInterpolator(
            transformed_axes,
            target_row,
            method="linear",
            bounds_error=True,
        )
        resampled[idx] = np.asarray(interpolator(combinations), dtype=float).reshape(axis_shape)

    return resampled


def flatten_resampled_rows(
    parameters: np.ndarray,
    target_grid: np.ndarray,
    *,
    sampled_axes: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten one resampled target grid into scalar regression rows."""
    combinations = axis_combinations(sampled_axes)
    tiled_parameters = np.repeat(parameters, repeats=len(combinations), axis=0)
    tiled_axes = np.tile(combinations, (len(parameters), 1))
    return np.hstack((tiled_axes, tiled_parameters)), target_grid.reshape(-1)


def axis_combinations(sampled_axes: tuple[np.ndarray, ...]) -> np.ndarray:
    """Return flattened axis-coordinate combinations for one shared grid."""
    grids = np.meshgrid(*sampled_axes, indexing="ij")
    return np.vstack([grid.ravel() for grid in grids]).T


def transformed_axis_configuration(
    axes: tuple[np.ndarray, ...],
    axis_specs: tuple[AxisSpec, ...],
) -> tuple[tuple[np.ndarray, ...], tuple[tuple[float, float], ...]]:
    """Return transformed axis arrays and transformed sampling limits."""
    transformed_axes = []
    transformed_limits = []
    for axis_values, axis_spec in zip(axes, axis_specs, strict=True):
        transformed_axes.append(apply_transform(axis_values, axis_spec.transform))
        if axis_spec.limits is None:
            transformed_limits.append((float(axis_values.min()), float(axis_values.max())))
        else:
            limits = apply_transform(np.asarray(axis_spec.limits, dtype=float), axis_spec.transform)
            transformed_limits.append((float(limits[0]), float(limits[1])))
    return tuple(transformed_axes), tuple(transformed_limits)


def build_fixed_axis_grid(
    transformed_axes: tuple[np.ndarray, ...],
    transformed_limits: tuple[tuple[float, float], ...],
    axis_specs: tuple[AxisSpec, ...],
) -> tuple[np.ndarray, ...]:
    """Construct the deterministic axis grid used during training."""
    sampled_axes = []
    for axis_values, (low, high), axis_spec in zip(
        transformed_axes,
        transformed_limits,
        axis_specs,
        strict=True,
    ):
        if axis_spec.nsample is None:
            mask = np.logical_and(axis_values >= low, axis_values <= high)
            sampled_axes.append(np.asarray(axis_values[mask], dtype=float))
            continue
        sampled_axes.append(np.linspace(low, high, axis_spec.nsample, dtype=float))
    return tuple(sampled_axes)


def build_feature_scaler(
    feature_matrix: np.ndarray,
    *,
    feature_names: tuple[str, ...],
    method_overrides: dict[str, str],
) -> FeatureScaler:
    """Build per-feature scaler metadata from the training features."""
    if feature_matrix.shape[1] != len(feature_names):
        raise ValueError("feature_names must match the feature matrix width.")

    scaling: list[FeatureScaling] = []
    for idx, name in enumerate(feature_names):
        method_name = method_overrides.get(name, "zscore")
        scaling_method = _resolve_scaling_method(method_name)
        scaling.append(FeatureScaling.from_values(name, feature_matrix[:, idx], scaling_method))
    return FeatureScaler(tuple(scaling))


def shuffle_rows(features: np.ndarray, targets: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle rows with a fixed seed for deterministic repeatability."""
    indices = np.random.RandomState(seed).permutation(len(targets))
    return features[indices], targets[indices]


def _resolve_scaling_method(method: str) -> str:
    """Translate workflow scaling labels into explicit scaler names."""
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
