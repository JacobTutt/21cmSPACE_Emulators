"""Small named transforms used by specs, preprocessing, and inference.

The emulator workflows operate in transformed coordinates for several inputs
and targets, for example ``log10(k)`` or ``log10(Delta21 + 1)``. These helpers
keep those rules explicit so the same transforms can be applied during
training and inverted later for inference or plotting.
"""

from __future__ import annotations

import numpy as np


def apply_transform(values: np.ndarray, transform: str, offset: float = 0.0) -> np.ndarray:
    """Apply a configured transform exactly as the emulator expects it.

    Parameters
    ----------
    values:
        Raw physical values such as axis coordinates, parameters, or targets.
    transform:
        Name of the transform to apply. ``"identity"`` leaves the values
        unchanged, while ``"log10"`` applies base-10 logarithms.
    offset:
        Additive offset applied before ``log10``. This is used for targets
        such as power spectra that may need a positive shift before taking a
        logarithm.
    """
    arr = np.asarray(values, dtype=float)
    if transform == "identity":
        return arr
    if transform == "log10":
        return np.log10(arr + offset)
    raise ValueError(f"Unsupported transform {transform}.")


def invert_transform(values: np.ndarray, transform: str, offset: float = 0.0) -> np.ndarray:
    """Undo a configured transform and recover physical-space values.

    This function exists for the other half of the emulator lifecycle. Training
    often happens in transformed space for numerical reasons, but downstream
    science code usually wants predictions back in the original physical units.
    """
    arr = np.asarray(values, dtype=float)
    if transform == "identity":
        return arr
    if transform == "log10":
        return np.power(10.0, arr) - offset
    raise ValueError(f"Unsupported transform {transform}.")
