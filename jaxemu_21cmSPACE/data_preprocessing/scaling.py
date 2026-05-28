"""
Feature and target scaling metadata plus application helpers.

Scaling is treated as explicit metadata rather than as an accidental side
effect of training. That makes saved models easier to inspect and ensures the
same input transformations can be reused later during inference. This module
handles:
- per-feature scaling (z-score, min-max, identity)
- global target scaling using one training-set standard deviation
- serialization of scaling metadata for checkpointing
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np


# Type Definitions
# ----------------
# Explicit labels for the supported scaling algorithms.

ScaleMethod = Literal[
    "identity",
    "zscore",
    "minmax_minus_one_to_one",
    "minmax_zero_to_one",
]


# Feature Scaling
# ---------------
# Stores and applies one scaling rule for each model input column.

@dataclass(frozen=True)
class FeatureScaling:
    """
    Storage utility for one named feature scaling rule.

    We store several summary statistics even when only one scaling method is
    used, because later checkpoint readers and diagnostics typically need more
    context than the trainer itself.

    Parameters
    ----------
    name:
        Human-readable name of the feature.
    method:
        The scaling algorithm to apply (e.g. 'zscore').
    minimum, maximum:
        Extrema found in the training data for this feature.
    mean, std:
        Average and standard deviation found in the training data.
    """

    name: str
    method: ScaleMethod
    minimum: float
    maximum: float
    mean: float
    std: float

    def to_dict(self) -> dict[str, float | str]:
        """
        Return a serializer-friendly view of the scaling rule.

        This is mainly used when packaging checkpoints or inspection outputs,
        where plain dictionaries are easier to store and inspect than nested
        dataclass instances.

        Returns
        -------
        dict
            A dictionary containing all scaling metadata.
        """
        # Convert the dataclass instance to a standard Python dictionary.
        return asdict(self)

    @classmethod
    def from_values(cls, name: str, values: np.ndarray, method: ScaleMethod) -> "FeatureScaling":
        """
        Summarize one feature column into reusable scaling metadata.

        The intent is that training code computes these statistics once from
        the training set, stores them in the checkpoint, and then inference
        code reuses the exact same numbers later. Zero-variance inputs are
        assigned a unit standard deviation so z-score scaling stays defined.

        Parameters
        ----------
        name:
            The name of the feature column.
        values:
            The numerical values from the training set.
        method:
            The scaling method that will be applied using these statistics.

        Returns
        -------
        FeatureScaling
            A populated metadata object for the feature.
        """
        # Ensure the input is treated as a floating-point NumPy array.
        arr = np.asarray(values, dtype=float)
        # Compute the standard deviation for z-score scaling.
        std = arr.std()
        # Initialise and return the metadata container.
        return cls(
            name=name,
            method=method,
            minimum=float(arr.min()),
            maximum=float(arr.max()),
            mean=float(arr.mean()),
            # Prevent division-by-zero if the feature is constant in the training set.
            std=float(std if std > 0 else 1.0),
        )


class FeatureScaler:
    """
    Apply per-feature scaling using stored metadata.

    The scaler is intentionally simple and works on matrices in canonical
    feature order. That keeps it easy to compose with synthetic tests and
    future dataset loaders.
    """

    def __init__(self, scaling: tuple[FeatureScaling, ...]) -> None:
        """
        Initialise the scaler with a sequence of rules.

        Parameters
        ----------
        scaling:
            A tuple of FeatureScaling objects, one for each column of the input matrix.
        """
        # Store the feature scaling rules in canonical feature order.
        self.scaling = scaling

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        """
        Scale a feature matrix column by column using stored metadata.

        The expected input is a 2D matrix whose columns are already in the
        canonical emulator feature order. This is the form produced by the
        tiling and feature-preparation utilities before data is sent to the
        neural network.

        Parameters
        ----------
        matrix:
            The physical-space feature matrix with shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            The scaled feature matrix ready for model input.
        """
        # Copy the matrix so scaling does not mutate the caller's array.
        arr = np.asarray(matrix, dtype=float).copy()

        # The feature matrix must be two-dimensional and match the metadata width.
        if arr.ndim != 2:
            raise ValueError("transform expects a 2D matrix.")
        if arr.shape[1] != len(self.scaling):
            raise ValueError("Feature dimension does not match scaling metadata.")

        # Apply each feature's stored rule to the matching column in the matrix.
        for idx, feature in enumerate(self.scaling):
            arr[:, idx] = _apply_scaling(arr[:, idx], feature)
        return arr

    def inverse_transform(self, matrix: np.ndarray) -> np.ndarray:
        """
        Undo scaling on a feature matrix and recover physical-space values.

        This is useful when inspecting saved features, debugging checkpoint
        inputs, or exporting predictions back into a form that is easier to
        compare against the original science tables.

        Parameters
        ----------
        matrix:
            A scaled feature matrix.

        Returns
        -------
        np.ndarray
             The matrix transformed back into its original physical units.
        """
        # Copy the matrix so inverse scaling does not mutate the caller's array.
        arr = np.asarray(matrix, dtype=float).copy()

        # The feature matrix must be two-dimensional and match the metadata width.
        if arr.ndim != 2:
            raise ValueError("inverse_transform expects a 2D matrix.")
        if arr.shape[1] != len(self.scaling):
            raise ValueError("Feature dimension does not match scaling metadata.")

        # Undo each feature's stored rule on the matching column in the matrix.
        for idx, feature in enumerate(self.scaling):
            arr[:, idx] = _invert_scaling(arr[:, idx], feature)
        return arr


# Target Scaling
# --------------
# Stores and applies target scaling for emulator outputs.


@dataclass(frozen=True)
class TargetScalingScalar:
    """
    Storage utility for global target standardization.

    This follows the old globalemu convention: divide all target values by one
    standard deviation calculated from the training labels. It does not subtract
    a mean and it does not store one statistic per redshift or k-bin, so the
    inverse step is just multiplication by the same scalar.

    Parameters
    ----------
    std:
        Standard deviation of all transformed target values in the training set.
    """

    std: float

    @classmethod
    def from_targets(cls, targets: np.ndarray) -> "TargetScalingScalar":
        """
        Summarize training targets into one global scaling value.

        Parameters
        ----------
        targets:
            Training target values in the space seen by the neural network.

        Returns
        -------
        TargetScalingScalar
            A container holding the global training-label standard deviation.
        """
        # Compute one scalar across simulations and all target-grid bins.
        std = float(np.asarray(targets, dtype=float).std())
        # Prevent division-by-zero if all training targets are identical.
        return cls(std=std if std > 0 else 1.0)

    def transform_grid(self, targets: np.ndarray) -> np.ndarray:
        """
        Divide a target grid by the global training-label standard deviation.

        Parameters
        ----------
        targets:
            Target values in transformed physical space.

        Returns
        -------
        np.ndarray
            Target values scaled for model training.
        """
        # Apply one scalar to every target value, regardless of redshift or k-bin.
        return np.asarray(targets, dtype=float) / self.std

    def inverse_grid(self, targets: np.ndarray) -> np.ndarray:
        """
        Undo global target scaling on a full target grid.

        Parameters
        ----------
        targets:
            Globally scaled target values.

        Returns
        -------
        np.ndarray
            Target values restored to transformed physical space.
        """
        # Restore values by multiplying by the same scalar used during training.
        return np.asarray(targets, dtype=float) * self.std

    def transform_rows(
        self,
        targets: np.ndarray,
        axis_coordinates: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Divide flattened target rows by the global training-label std.

        Parameters
        ----------
        targets:
            Flattened target values.
        axis_coordinates:
            Unused. Included so row-based inference can call the same method
            without caring whether targets were grid-shaped during preprocessing.

        Returns
        -------
        np.ndarray
            Scaled target rows.
        """
        return np.asarray(targets, dtype=float) / self.std

    def inverse_rows(
        self,
        targets: np.ndarray,
        axis_coordinates: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Undo global scaling on flattened model predictions.

        Parameters
        ----------
        targets:
            Flattened predictions from the neural network.
        axis_coordinates:
            Unused. Included so row-based inference can call the same method
            without caring whether targets were grid-shaped during preprocessing.

        Returns
        -------
        np.ndarray
            Predictions restored to transformed physical target space.
        """
        return np.asarray(targets, dtype=float) * self.std

    def to_dict(self) -> dict[str, object]:
        """
        Return a serializer-friendly representation of the scalar target scaler.

        Returns
        -------
        dict
            A dictionary containing the scaler type and stored standard deviation.
        """
        return {"kind": "global_std", "std": float(self.std)}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TargetScalingScalar":
        """
        Reconstruct a scalar target scaler from serialized metadata.

        Parameters
        ----------
        payload:
            The serialized metadata dictionary.

        Returns
        -------
        TargetScalingScalar
            A reconstructed scalar target scaler.
        """
        return cls(std=float(payload["std"]))


# Internal Helpers
# ----------------
# Logic for applying and inverting specific scaling algorithms.

def _apply_scaling(values: np.ndarray, feature: FeatureScaling) -> np.ndarray:
    """
    Apply one feature's stored scaling rule to a single column of values.
    """
    # Ensure working with a floating-point array.
    arr = np.asarray(values, dtype=float)

    # Identity leaves the feature in its current units.
    if feature.method == "identity":
        return arr

    # Z-score centres the feature and scales by the training standard deviation.
    if feature.method == "zscore":
        return (arr - feature.mean) / feature.std

    # Min-max scaling to [-1, 1], with a zero output for constant features.
    if feature.method == "minmax_minus_one_to_one":
        denom = feature.maximum - feature.minimum
        if denom == 0:
            return np.zeros_like(arr)
        return (2.0 * (arr - feature.minimum) / denom) - 1.0

    # Min-max scaling to [0, 1], with a zero output for constant features.
    if feature.method == "minmax_zero_to_one":
        denom = feature.maximum - feature.minimum
        if denom == 0:
            return np.zeros_like(arr)
        return (arr - feature.minimum) / denom

    raise ValueError(f"Unsupported scaling method {feature.method}.")


def _invert_scaling(values: np.ndarray, feature: FeatureScaling) -> np.ndarray:
    """
    Undo one feature's stored scaling rule for a single column of values.
    """
    # Ensure working with a floating-point array.
    arr = np.asarray(values, dtype=float)

    # Identity values are already in their original units.
    if feature.method == "identity":
        return arr

    # Undo z-score scaling.
    if feature.method == "zscore":
        return (arr * feature.std) + feature.mean

    # Undo min-max scaling from [-1, 1].
    if feature.method == "minmax_minus_one_to_one":
        return 0.5 * (arr + 1.0) * (feature.maximum - feature.minimum) + feature.minimum

    # Undo min-max scaling from [0, 1].
    if feature.method == "minmax_zero_to_one":
        return arr * (feature.maximum - feature.minimum) + feature.minimum

    raise ValueError(f"Unsupported scaling method {feature.method}.")
