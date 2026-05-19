"""Training-data preparation workflows for the supported emulators.

This module turns simulation outputs into the scalar regression rows used by
the neural networks. It provides two preparation styles because the supported
emulators genuinely need two different sampling strategies:

- an interpolated row builder for `Delta21`
- a shared-grid row builder for `T21`

Both follow the same overall sequence:

1. remove failed simulations
2. apply the configured target transform
3. split by simulation
4. build scalar regression rows
5. scale the input features
6. shuffle the rows before training
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from nenufar_emulators.conventions import PreparedFeatures
from nenufar_emulators.core.scaling import FeatureScaler, FeatureScaling
from nenufar_emulators.core.specs import AxisSpec
from nenufar_emulators.core.transforms import apply_transform


@dataclass(frozen=True)
class PreparedSplit:
    """Prepared train/validation arrays ready for model fitting."""

    feature_names: tuple[str, ...]
    train_features: np.ndarray
    train_targets: np.ndarray
    validation_features: np.ndarray
    validation_targets: np.ndarray
    feature_scaling: tuple[FeatureScaling, ...]


def prepare_interpolated_training_split(
    *,
    axes: tuple[np.ndarray, ...],
    axis_specs: tuple[AxisSpec, ...],
    parameters: PreparedFeatures,
    target: np.ndarray,
    scale_method: dict[str, str],
    data_log: bool,
    offset: float | None,
    train_size: float = 0.8,
    test_size: float = 0.2,
    random_state: int = 42,
    interpolation_seed: int = 0,
    shuffle_seed: int = 42,
) -> PreparedSplit:
    """Prepare a train/validation split using interpolated training rows.

    Parameters are expected to have already been converted into their final
    model-input representation. Targets are expected to still be in physical
    space so this function can apply the configured target transform before the
    train/validation split.
    """
    transformed_target = transform_target(
        target,
        data_log=data_log,
        offset=offset,
        rng=np.random.default_rng(interpolation_seed),
    )
    train_parameters, validation_parameters, train_target, validation_target = split_simulations(
        parameters.values,
        transformed_target,
        train_size=train_size,
        test_size=test_size,
        random_state=random_state,
    )

    axis_feature_names = tuple(axis.feature_name() for axis in axis_specs)
    feature_names = (*axis_feature_names, *parameters.feature_names)

    train_features, train_targets = generate_training_rows(
        train_parameters,
        train_target,
        axes=axes,
        axis_specs=axis_specs,
        seed=interpolation_seed,
    )
    validation_features, validation_targets = generate_validation_rows(
        validation_parameters,
        validation_target,
        axes=axes,
        axis_specs=axis_specs,
    )

    scaler = build_feature_scaler(
        train_features,
        feature_names=feature_names,
        method_overrides=scale_method,
    )
    scaled_train = scaler.transform(train_features).astype(np.float32)
    scaled_validation = scaler.transform(validation_features).astype(np.float32)
    train_targets = np.asarray(train_targets, dtype=np.float32)
    validation_targets = np.asarray(validation_targets, dtype=np.float32)

    scaled_train, train_targets = shuffle_rows(scaled_train, train_targets, seed=shuffle_seed)
    scaled_validation, validation_targets = shuffle_rows(
        scaled_validation,
        validation_targets,
        seed=shuffle_seed,
    )

    return PreparedSplit(
        feature_names=feature_names,
        train_features=scaled_train,
        train_targets=train_targets,
        validation_features=scaled_validation,
        validation_targets=validation_targets,
        feature_scaling=scaler.scaling,
    )


def prepare_shared_grid_training_split(
    *,
    axes: tuple[np.ndarray, ...],
    axis_specs: tuple[AxisSpec, ...],
    parameters: PreparedFeatures,
    target: np.ndarray,
    scale_method: dict[str, str],
    data_log: bool,
    offset: float | None,
    train_size: float = 0.66,
    test_size: float = 0.34,
    random_state: int = 42,
    shuffle_seed: int = 42,
) -> PreparedSplit:
    """Prepare a train/validation split on one shared axis grid.

    This workflow differs from :func:`prepare_interpolated_training_split` in one
    important way: it does not draw random interpolation points independently
    for each simulation. Instead it constructs one fixed axis grid from the
    declared emulator spec, resamples every training and validation signal onto
    that grid, and then flattens the result into scalar rows.
    """
    transformed_target = transform_target(
        target,
        data_log=data_log,
        offset=offset,
        rng=np.random.default_rng(shuffle_seed),
    )
    train_parameters, validation_parameters, train_target, validation_target = split_simulations(
        parameters.values,
        transformed_target,
        train_size=train_size,
        test_size=test_size,
        random_state=random_state,
    )

    axis_feature_names = tuple(axis.feature_name() for axis in axis_specs)
    feature_names = (*axis_feature_names, *parameters.feature_names)

    train_features, train_targets = generate_resampled_rows(
        train_parameters,
        train_target,
        axes=axes,
        axis_specs=axis_specs,
    )
    validation_features, validation_targets = generate_resampled_rows(
        validation_parameters,
        validation_target,
        axes=axes,
        axis_specs=axis_specs,
    )

    scaler = build_feature_scaler(
        train_features,
        feature_names=feature_names,
        method_overrides=scale_method,
    )
    scaled_train = scaler.transform(train_features).astype(np.float32)
    scaled_validation = scaler.transform(validation_features).astype(np.float32)
    train_targets = np.asarray(train_targets, dtype=np.float32)
    validation_targets = np.asarray(validation_targets, dtype=np.float32)

    scaled_train, train_targets = shuffle_rows(scaled_train, train_targets, seed=shuffle_seed)
    scaled_validation, validation_targets = shuffle_rows(
        scaled_validation,
        validation_targets,
        seed=shuffle_seed,
    )

    return PreparedSplit(
        feature_names=feature_names,
        train_features=scaled_train,
        train_targets=train_targets,
        validation_features=scaled_validation,
        validation_targets=validation_targets,
        feature_scaling=scaler.scaling,
    )


def transform_target(
    target: np.ndarray,
    *,
    data_log: bool,
    offset: float | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply the configured target transform before train/test splitting."""
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
        arr[zero_mask] = minimum * np.power(10.0, rng.uniform(-3.0, 0.0, size=zero_mask.sum()))
    return np.log10(arr)


def split_simulations(
    parameters: np.ndarray,
    target: np.ndarray,
    *,
    train_size: float,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split simulations using the same shuffle-split contract each run."""
    n_samples = len(parameters)
    if len(target) != n_samples:
        raise ValueError("parameters and target must have the same sample count.")

    n_test = int(np.ceil(test_size * n_samples))
    n_train = int(np.floor(train_size * n_samples))
    permutation = np.random.RandomState(random_state).permutation(n_samples)
    test_indices = permutation[:n_test]
    train_indices = permutation[n_test : n_test + n_train]
    return (
        np.asarray(parameters[train_indices], dtype=float),
        np.asarray(parameters[test_indices], dtype=float),
        np.asarray(target[train_indices], dtype=float),
        np.asarray(target[test_indices], dtype=float),
    )


def generate_training_rows(
    parameters: np.ndarray,
    target: np.ndarray,
    *,
    axes: tuple[np.ndarray, ...],
    axis_specs: tuple[AxisSpec, ...],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate interpolated training rows.

    Each simulation contributes `prod(axis.nsample)` scalar training examples.
    Axis sampling happens in feature space rather than in raw physical space,
    which keeps the training rows aligned with the declared axis transforms.
    """
    transformed_axes, transformed_limits = transformed_axis_configuration(axes, axis_specs)
    rng = np.random.default_rng(seed)
    row_features: list[np.ndarray] = []
    row_targets: list[np.ndarray] = []

    for parameter_row, target_row in zip(parameters, target, strict=True):
        priors = [
            np.sort(rng.uniform(low=low, high=high, size=axis_spec.nsample))
            for axis_spec, (low, high) in zip(axis_specs, transformed_limits, strict=True)
        ]
        interpolator = RegularGridInterpolator(transformed_axes, target_row, method="linear")
        grids = np.meshgrid(*priors)
        interpolated = interpolator(tuple(grids))
        stacked = np.stack((*grids, interpolated), axis=-1).reshape(-1, len(axis_specs) + 1)
        tiled_parameters = np.tile(parameter_row, (stacked.shape[0], 1))
        row_features.append(np.hstack((stacked[:, :-1], tiled_parameters)))
        row_targets.append(stacked[:, -1])

    return np.vstack(row_features), np.hstack(row_targets)


def generate_validation_rows(
    parameters: np.ndarray,
    target: np.ndarray,
    *,
    axes: tuple[np.ndarray, ...],
    axis_specs: tuple[AxisSpec, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate validation rows on the cropped physical grid."""
    transformed_axes, transformed_limits = transformed_axis_configuration(axes, axis_specs)
    masks = [
        np.logical_and(axis_values >= low, axis_values <= high)
        for axis_values, (low, high) in zip(transformed_axes, transformed_limits, strict=True)
    ]
    cropped_axes = [axis_values[mask] for axis_values, mask in zip(transformed_axes, masks, strict=True)]

    grids = np.meshgrid(*cropped_axes, indexing="ij")
    combinations = np.vstack([grid.ravel() for grid in grids]).T
    num_parameters = len(parameters)
    num_combinations = len(combinations)

    tiled_parameters = np.repeat(parameters, num_combinations, axis=0)
    tiled_axes = np.tile(combinations, (num_parameters, 1))

    cropped_target = np.asarray(target, dtype=float)
    for axis_index, mask in enumerate(masks, start=1):
        cropped_target = np.take(cropped_target, np.where(mask)[0], axis=axis_index)

    return np.hstack((tiled_axes, tiled_parameters)), cropped_target.ravel()


def generate_resampled_rows(
    parameters: np.ndarray,
    target: np.ndarray,
    *,
    axes: tuple[np.ndarray, ...],
    axis_specs: tuple[AxisSpec, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Resample every simulation onto one shared axis grid, then flatten rows."""
    transformed_axes, transformed_limits = transformed_axis_configuration(axes, axis_specs)
    sampled_axes = build_fixed_axis_grid(
        transformed_axes,
        transformed_limits,
        axis_specs,
    )
    grids = np.meshgrid(*sampled_axes, indexing="ij")
    combinations = np.vstack([grid.ravel() for grid in grids]).T

    row_features: list[np.ndarray] = []
    row_targets: list[np.ndarray] = []
    for parameter_row, target_row in zip(parameters, target, strict=True):
        interpolator = RegularGridInterpolator(transformed_axes, target_row, method="linear")
        interpolated = interpolator(combinations)
        tiled_parameters = np.tile(parameter_row, (combinations.shape[0], 1))
        row_features.append(np.hstack((combinations, tiled_parameters)))
        row_targets.append(np.asarray(interpolated, dtype=float))

    return np.vstack(row_features), np.hstack(row_targets)


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
    """Construct the deterministic axis grid used in fixed-grid workflows.

    If an axis spec declares ``nsample`` we interpolate onto that many evenly
    spaced points in feature space. Otherwise we keep the original axis values
    cropped to the declared limits.
    """
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
    """Build per-feature scaler metadata from the training features.

    The workflow uses the following naming convention:
    - `standardize` meant min-max scaling to `[-1, 1]`
    - `normalize` meant z-score scaling
    """
    if feature_matrix.shape[1] != len(feature_names):
        raise ValueError("feature_names must match the feature matrix width.")

    scaling: list[FeatureScaling] = []
    for idx, name in enumerate(feature_names):
        method_name = method_overrides.get(name, "standardize")
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
    raise ValueError(f"Unsupported scaling method {method!r}.")
