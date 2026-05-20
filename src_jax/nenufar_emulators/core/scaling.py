"""Feature and target scaling metadata plus application helpers.

Scaling is treated as explicit metadata rather than as an accidental side
effect of training. That makes saved models easier to inspect and ensures the
same input transformations can be reused later during inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from scipy.interpolate import RegularGridInterpolator


ScaleMethod = Literal[
    "identity",
    "zscore",
    "minmax_minus_one_to_one",
    "minmax_zero_to_one",
]


@dataclass(frozen=True)
class FeatureScaling:
    """Scaling rule for one named feature.

    We store several summary statistics even when only one scaling method is
    used, because later checkpoint readers and diagnostics typically need more
    context than the trainer itself.
    """

    name: str
    method: ScaleMethod
    minimum: float
    maximum: float
    mean: float
    std: float

    def to_dict(self) -> dict[str, float | str]:
        """Return a serializer-friendly view of the scaling rule.

        This is mainly used when packaging checkpoints or inspection outputs,
        where plain dictionaries are easier to store and inspect than nested
        dataclass instances.
        """
        return asdict(self)

    @classmethod
    def from_values(cls, name: str, values: np.ndarray, method: ScaleMethod) -> "FeatureScaling":
        """Summarize one feature column into reusable scaling metadata.

        The intent is that training code computes these statistics once from
        the training set, stores them in the checkpoint, and then inference
        code reuses the exact same numbers later. Zero-variance inputs are
        assigned a unit standard deviation so z-score scaling stays defined.
        """
        arr = np.asarray(values, dtype=float)
        return cls(
            name=name,
            method=method,
            minimum=float(arr.min()),
            maximum=float(arr.max()),
            mean=float(arr.mean()),
            std=float(arr.std() if arr.std() > 0 else 1.0),
        )


class FeatureScaler:
    """Apply per-feature scaling using stored metadata.

    The scaler is intentionally simple and works on matrices in canonical
    feature order. That keeps it easy to compose with synthetic tests and
    future dataset loaders.
    """

    def __init__(self, scaling: tuple[FeatureScaling, ...]) -> None:
        self.scaling = scaling

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        """Scale a feature matrix column by column using stored metadata.

        The expected input is a 2D matrix whose columns are already in the
        canonical emulator feature order. This is the form produced by the
        tiling and feature-preparation utilities before data is sent to the
        neural network.
        """
        arr = np.asarray(matrix, dtype=float).copy()
        if arr.ndim != 2:
            raise ValueError("transform expects a 2D matrix.")
        if arr.shape[1] != len(self.scaling):
            raise ValueError("Feature dimension does not match scaling metadata.")
        for idx, feature in enumerate(self.scaling):
            arr[:, idx] = _apply_scaling(arr[:, idx], feature)
        return arr

    def inverse_transform(self, matrix: np.ndarray) -> np.ndarray:
        """Undo scaling on a feature matrix and recover physical-space values.

        This is useful when inspecting saved features, debugging checkpoint
        inputs, or exporting predictions back into a form that is easier to
        compare against the original science tables.
        """
        arr = np.asarray(matrix, dtype=float).copy()
        if arr.ndim != 2:
            raise ValueError("inverse_transform expects a 2D matrix.")
        if arr.shape[1] != len(self.scaling):
            raise ValueError("Feature dimension does not match scaling metadata.")
        for idx, feature in enumerate(self.scaling):
            arr[:, idx] = _invert_scaling(arr[:, idx], feature)
        return arr


@dataclass(frozen=True)
class TargetScalingSurface:
    """Axis-aware scaling metadata for emulator targets.

    Input features are scaled column by column, but targets need a different
    treatment. For both supported emulators we now train on signals laid out on
    one shared axis grid. That lets us store one mean and standard deviation
    per grid location, then reuse those statistics both during training and
    later during inference.

    The grid is stored in the same transformed axis space that the network sees
    as input. That keeps the inverse step simple: inference code can use the
    already transformed axis coordinates to recover the correct mean and
    standard deviation for each prediction.
    """

    axis_names: tuple[str, ...]
    axis_values: tuple[np.ndarray, ...]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_targets(
        cls,
        *,
        axis_names: tuple[str, ...],
        axis_values: tuple[np.ndarray, ...],
        targets: np.ndarray,
    ) -> "TargetScalingSurface":
        """Summarize training targets on a shared grid into scaling metadata."""
        arr = np.asarray(targets, dtype=float)
        if arr.ndim != len(axis_values) + 1:
            raise ValueError("targets must have leading sample axis followed by axis-grid dimensions.")
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        return cls(
            axis_names=axis_names,
            axis_values=tuple(np.asarray(axis, dtype=float) for axis in axis_values),
            mean=np.asarray(mean, dtype=float),
            std=np.asarray(std, dtype=float),
        )

    def transform_grid(self, targets: np.ndarray) -> np.ndarray:
        """Standardize a full target grid using the stored mean and std."""
        arr = np.asarray(targets, dtype=float)
        return (arr - self.mean) / self.std

    def inverse_grid(self, targets: np.ndarray) -> np.ndarray:
        """Undo target scaling on a full grid shaped like the stored axes."""
        arr = np.asarray(targets, dtype=float)
        return (arr * self.std) + self.mean

    def transform_rows(self, targets: np.ndarray, axis_coordinates: np.ndarray) -> np.ndarray:
        """Standardize flattened target rows using interpolated grid statistics."""
        mean, std = self._stats_for_rows(axis_coordinates)
        arr = np.asarray(targets, dtype=float)
        return (arr - mean) / std

    def inverse_rows(self, targets: np.ndarray, axis_coordinates: np.ndarray) -> np.ndarray:
        """Undo target scaling on flattened predictions using interpolated stats."""
        mean, std = self._stats_for_rows(axis_coordinates)
        arr = np.asarray(targets, dtype=float)
        return (arr * std) + mean

    def to_dict(self) -> dict[str, object]:
        """Return a serializer-friendly representation of the scaling surface."""
        return {
            "axis_names": list(self.axis_names),
            "axis_values": [np.asarray(axis, dtype=float).tolist() for axis in self.axis_values],
            "mean": np.asarray(self.mean, dtype=float).tolist(),
            "std": np.asarray(self.std, dtype=float).tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TargetScalingSurface":
        """Reconstruct a scaling surface from serialized metadata."""
        return cls(
            axis_names=tuple(str(name) for name in payload["axis_names"]),
            axis_values=tuple(
                np.asarray(axis, dtype=float) for axis in payload["axis_values"]
            ),
            mean=np.asarray(payload["mean"], dtype=float),
            std=np.asarray(payload["std"], dtype=float),
        )

    def _stats_for_rows(self, axis_coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the stored mean and std at a batch of transformed axis points."""
        coords = np.asarray(axis_coordinates, dtype=float)
        if coords.ndim == 1:
            coords = coords[:, None]
        if coords.shape[1] != len(self.axis_values):
            raise ValueError("axis_coordinates width does not match the target-scaling axes.")
        if len(self.axis_values) == 1:
            points = coords[:, 0]
        else:
            points = coords
        mean = RegularGridInterpolator(
            self.axis_values,
            self.mean,
            method="linear",
            bounds_error=True,
        )(points)
        std = RegularGridInterpolator(
            self.axis_values,
            self.std,
            method="linear",
            bounds_error=True,
        )(points)
        return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)


def _apply_scaling(values: np.ndarray, feature: FeatureScaling) -> np.ndarray:
    """Apply one feature's stored scaling rule to a single column of values."""
    arr = np.asarray(values, dtype=float)
    if feature.method == "identity":
        return arr
    if feature.method == "zscore":
        return (arr - feature.mean) / feature.std
    if feature.method == "minmax_minus_one_to_one":
        denom = feature.maximum - feature.minimum
        if denom == 0:
            return np.zeros_like(arr)
        return (2.0 * (arr - feature.minimum) / denom) - 1.0
    if feature.method == "minmax_zero_to_one":
        denom = feature.maximum - feature.minimum
        if denom == 0:
            return np.zeros_like(arr)
        return (arr - feature.minimum) / denom
    raise ValueError(f"Unsupported scaling method {feature.method}.")


def _invert_scaling(values: np.ndarray, feature: FeatureScaling) -> np.ndarray:
    """Undo one feature's stored scaling rule for a single column of values."""
    arr = np.asarray(values, dtype=float)
    if feature.method == "identity":
        return arr
    if feature.method == "zscore":
        return (arr * feature.std) + feature.mean
    if feature.method == "minmax_minus_one_to_one":
        return 0.5 * (arr + 1.0) * (feature.maximum - feature.minimum) + feature.minimum
    if feature.method == "minmax_zero_to_one":
        return arr * (feature.maximum - feature.minimum) + feature.minimum
    raise ValueError(f"Unsupported scaling method {feature.method}.")
