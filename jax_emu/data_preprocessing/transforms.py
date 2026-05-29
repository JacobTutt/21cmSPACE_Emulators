"""
Named transforms used by specs, preprocessing, and inference.

The emulator workflows operate in transformed coordinates for several inputs
and targets. Examples are log10(k) for a power-spectrum axis and
log10(Delta21 + 1e-8) for a positive target.

Keeping these transforms here makes the training and inference paths use the
same rules in opposite directions.
"""

from __future__ import annotations

import numpy as np


# Mathematical Transforms
# -----------------------
# Core logic for moving between physical and training coordinate spaces.

def apply_transform(values: np.ndarray, transform: str, offset: float = 0.0) -> np.ndarray:
    """
    Apply a configured transform to physical-space values.

    Parameters
    ----------
    values:
        Raw physical values such as axis coordinates, parameters, or targets.
    transform:
        Name of the transform to apply. "identity" leaves the values
        unchanged, while "log10" applies base-10 logarithms.
    offset:
        Additive offset applied before log10. This is used for targets
        such as power spectra that may need a positive shift before taking a
        logarithm to ensure all values remain finite and positive.

    Returns
    -------
    np.ndarray
        The transformed numerical values.
    """
    # Work with NumPy arrays so preprocessing can run before JAX training starts.
    arr = np.asarray(values, dtype=float)

    # Identity leaves the physical values unchanged.
    if transform == "identity":
        return arr

    # log10 is applied after an optional positive offset.
    if transform == "log10":
        # Add the offset then take the base-10 logarithm.
        return np.log10(arr + offset)

    raise ValueError(f"Unsupported transform {transform}.")


def invert_transform(values: np.ndarray, transform: str, offset: float = 0.0) -> np.ndarray:
    """
    Undo a configured transform and recover physical-space values.

    Training can happen in transformed space for numerical reasons. Inference
    uses this function to move emulator predictions back to the original units.

    Parameters
    ----------
    values:
        Transformed values (e.g. from neural network output).
    transform:
        Name of the transform to invert.
    offset:
        The same additive offset used during the forward transform.

    Returns
    -------
    np.ndarray
        The values restored to their original physical units.
    """
    # Work with NumPy arrays because this is used during preprocessing and inference.
    arr = np.asarray(values, dtype=float)

    # Identity values are already in physical space.
    if transform == "identity":
        return arr

    # Invert log10 and remove the same offset used during the forward transform.
    if transform == "log10":
        # First undo the log-transform using power-of-10.
        # Then subtract the original offset to recover the physical value.
        return np.power(10.0, arr) - offset

    raise ValueError(f"Unsupported transform {transform}.")
