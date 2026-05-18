"""Feature scaling metadata and application helpers.

The old PyTorch code stored enough scaling information to rebuild priors and
perform inference later. This module is the beginning of the same idea in the
new codebase: scaling is treated as explicit metadata, not an accidental side
effect of training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np


ScaleMethod = Literal["identity", "zscore", "minmax_minus_one_to_one"]


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
        """Convert to plain metadata."""
        return asdict(self)

    @classmethod
    def from_values(cls, name: str, values: np.ndarray, method: ScaleMethod) -> "FeatureScaling":
        """Build scaling metadata from observed values.

        Zero-variance inputs are assigned a unit standard deviation so later
        z-score transforms remain numerically well-defined.
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
        """Scale a 2D feature matrix in feature-order."""
        arr = np.asarray(matrix, dtype=float).copy()
        if arr.ndim != 2:
            raise ValueError("transform expects a 2D matrix.")
        if arr.shape[1] != len(self.scaling):
            raise ValueError("Feature dimension does not match scaling metadata.")
        for idx, feature in enumerate(self.scaling):
            arr[:, idx] = _apply_scaling(arr[:, idx], feature)
        return arr

    def inverse_transform(self, matrix: np.ndarray) -> np.ndarray:
        """Invert scaling on a 2D feature matrix in feature-order."""
        arr = np.asarray(matrix, dtype=float).copy()
        if arr.ndim != 2:
            raise ValueError("inverse_transform expects a 2D matrix.")
        if arr.shape[1] != len(self.scaling):
            raise ValueError("Feature dimension does not match scaling metadata.")
        for idx, feature in enumerate(self.scaling):
            arr[:, idx] = _invert_scaling(arr[:, idx], feature)
        return arr


def _apply_scaling(values: np.ndarray, feature: FeatureScaling) -> np.ndarray:
    """Apply the scaling rule stored for one feature."""
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
    raise ValueError(f"Unsupported scaling method {feature.method}.")


def _invert_scaling(values: np.ndarray, feature: FeatureScaling) -> np.ndarray:
    """Invert the scaling rule stored for one feature."""
    arr = np.asarray(values, dtype=float)
    if feature.method == "identity":
        return arr
    if feature.method == "zscore":
        return (arr * feature.std) + feature.mean
    if feature.method == "minmax_minus_one_to_one":
        return 0.5 * (arr + 1.0) * (feature.maximum - feature.minimum) + feature.minimum
    raise ValueError(f"Unsupported scaling method {feature.method}.")
