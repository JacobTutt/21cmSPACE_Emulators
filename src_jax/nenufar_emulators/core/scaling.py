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
    raise ValueError(f"Unsupported scaling method {feature.method}.")
