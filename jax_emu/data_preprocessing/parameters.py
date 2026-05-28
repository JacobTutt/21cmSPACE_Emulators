"""
Parameter-table preparation helpers.

This module provides tools to transform raw simulation parameter tables into
feature matrices suitable for neural network training. It handles:
- dropping unused parameters
- applying logarithmic transforms (log10) to specified columns
- recording discrete parameter values for metadata and sampling
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Data Containers
# ---------------
# Structures for holding prepared features and their associated metadata.

@dataclass(frozen=True)
class PreparedFeatures:
    """
    Prepared feature matrix plus associated metadata.

    This class stores the numerical values used as model inputs alongside
    the names of the features and information about discrete parameters.
    """

    feature_names: tuple[str, ...]
    values: np.ndarray
    discrete_values: dict[str, tuple[float, ...]]


# Feature Preparation
# -------------------
# Logic for transforming raw parameter tables into training feature matrices.

def prepare_feature_matrix(
    raw: np.ndarray,
    column_names: tuple[str, ...],
    *,
    transform_params: tuple[str, ...],
    discard_params: tuple[str, ...],
    discrete_params: tuple[str, ...],
) -> PreparedFeatures:
    """
    Prepare model features from a raw parameter table.

    The helper drops unused parameters, applies log10 transforms to selected
    columns, and records discrete parameter values for metadata.

    Parameters
    ----------
    raw:
        Raw 2D numpy array containing simulation parameters.
    column_names:
        The names corresponding to each column in the raw array.
    transform_params:
        A list of parameter names to which log10 should be applied.
    discard_params:
        A list of parameter names to be removed from the feature matrix.
    discrete_params:
        A list of parameter names identified as being sampled from a discrete set.

    Returns
    -------
    PreparedFeatures
        An object containing the prepared numerical matrix and feature metadata.
    """
    # Ensure the input is a 2D numpy array of floats.
    array = np.asarray(raw, dtype=float)
    if array.ndim != 2:
        raise ValueError("raw parameter array must be 2D.")
    # Check that the number of column names matches the width of the input array.
    if array.shape[1] != len(column_names):
        raise ValueError("column_names length does not match raw parameter width.")

    # Map column names to their integer indices in the raw array for fast lookups.
    name_to_index = {name: idx for idx, name in enumerate(column_names)}
    # Identify the subset of parameters we intend to keep in the final matrix.
    keep_names = [name for name in column_names if name not in discard_params]

    # Process and record the unique values for parameters marked as discrete.
    # This is used later to understand the sampling grid.
    discrete_values: dict[str, tuple[float, ...]] = {}
    for name in discrete_params:
        # Extract unique values from the raw data for this parameter.
        values = tuple(float(v) for v in np.unique(array[:, name_to_index[name]]))
        # If the parameter is also being transformed, apply log10 to the discrete values too.
        key = f"log10{name}" if name in transform_params else name
        discrete_values[key] = values if name not in transform_params else tuple(
            float(np.log10(v)) for v in values
        )

    # Initialise lists to hold the columns of the final feature matrix.
    prepared_columns = []
    prepared_names = []
    # Loop through each parameter we are keeping and apply any requested transforms.
    for name in keep_names:
        # Create a copy of the raw column values to avoid in-place modification.
        values = array[:, name_to_index[name]].copy()
        if name in transform_params:
            # Apply log10 transform and update the feature name.
            prepared_columns.append(np.log10(values))
            prepared_names.append(f"log10{name}")
        else:
            # Keep the parameter as-is.
            prepared_columns.append(values)
            prepared_names.append(name)

    # Return the bundled features and metadata.
    return PreparedFeatures(
        feature_names=tuple(prepared_names),
        values=np.stack(prepared_columns, axis=1),
        discrete_values=discrete_values,
    )
