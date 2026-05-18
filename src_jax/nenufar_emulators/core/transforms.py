"""Named transforms used across configuration and preprocessing."""

from __future__ import annotations

import numpy as np


def apply_transform(values: np.ndarray, transform: str, offset: float = 0.0) -> np.ndarray:
    """Apply a named transform to values."""
    arr = np.asarray(values, dtype=float)
    if transform == "identity":
        return arr
    if transform == "log10":
        return np.log10(arr + offset)
    raise ValueError(f"Unsupported transform {transform}.")


def invert_transform(values: np.ndarray, transform: str, offset: float = 0.0) -> np.ndarray:
    """Invert a named transform."""
    arr = np.asarray(values, dtype=float)
    if transform == "identity":
        return arr
    if transform == "log10":
        return np.power(10.0, arr) - offset
    raise ValueError(f"Unsupported transform {transform}.")
