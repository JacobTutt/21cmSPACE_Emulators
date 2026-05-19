"""Shared workflow conventions for model setup and feature preparation.

This module collects the small pieces of configuration and parameter-table
handling that define how the current emulator workflows are built. Keeping
these conventions in one place makes the training setup easy to read and keeps
the rest of the package focused on data flow rather than bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MLPConfig:
    """Configuration for the shared dense emulator network.

    The hidden-layer count is stored in the same split form used by the
    workflow configs: one input-to-hidden layer plus ``n_hidden_blocks``
    repeated hidden layers of the same width.
    """

    input_dim: int
    hidden_dim: int = 100
    n_hidden_blocks: int = 6
    output_dim: int = 1
    activation: str = "relu"

    @property
    def total_hidden_layers(self) -> int:
        """Return the total number of hidden layers for the JAX MLP."""
        return 1 + self.n_hidden_blocks


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer settings for a workflow configuration."""

    optimizer_name: str = "Adam"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class TrainingConfig:
    """Training-loop settings for a workflow configuration."""

    epochs: int
    batch_size: int
    loss_name: str = "MSELoss"
    save_after_epochs: int = 5
    terminate_time_seconds: int = 3600 * 2
    profiling: bool = False
    early_stop: bool = False
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0


@dataclass(frozen=True)
class PreparedFeatures:
    """Prepared feature matrix plus associated metadata.

    We keep the transformed feature names and discrete-value metadata together
    with the numeric array so later loaders and checkpoint code can retain the
    original scientific meaning of each column.
    """

    feature_names: tuple[str, ...]
    values: np.ndarray
    discrete_values: dict[str, tuple[float, ...]]


def prepare_feature_matrix(
    raw: np.ndarray,
    column_names: tuple[str, ...],
    *,
    transform_params: tuple[str, ...],
    discard_params: tuple[str, ...],
    discrete_params: tuple[str, ...],
) -> PreparedFeatures:
    """Prepare a feature matrix from a raw parameter table.

    The preparation recipe used throughout this repository is:

    1. start from a raw parameter table with science-facing column names
    2. drop parameters not used by a given emulator
    3. log-transform selected columns
    4. record which remaining parameters should still be treated as discrete

    This helper makes that recipe reusable and easy to test.

    Parameters
    ----------
    raw:
        Two-dimensional raw parameter table read from disk.
    column_names:
        Names attached to the columns in ``raw``. These define the only valid
        lookup order for the table.
    transform_params:
        Parameters that should be mapped into ``log10`` feature space before
        training.
    discard_params:
        Parameters present in the raw table but intentionally excluded from the
        emulator input for this particular model.
    discrete_params:
        Parameters whose allowed values should be tracked explicitly because
        the workflows treat them as discrete choices.
    """
    array = np.asarray(raw, dtype=float)
    if array.ndim != 2:
        raise ValueError("raw parameter array must be 2D.")
    if array.shape[1] != len(column_names):
        raise ValueError("column_names length does not match raw parameter width.")

    name_to_index = {name: idx for idx, name in enumerate(column_names)}
    # Preserve the declared column order after dropping unused parameters so
    # the resulting matrix has a stable, explicit feature order.
    keep_names = [name for name in column_names if name not in discard_params]

    discrete_values: dict[str, tuple[float, ...]] = {}
    for name in discrete_params:
        values = tuple(float(v) for v in np.unique(array[:, name_to_index[name]]))
        key = f"log10{name}" if name in transform_params else name
        discrete_values[key] = values if name not in transform_params else tuple(
            float(np.log10(v)) for v in values
        )

    prepared_columns = []
    prepared_names = []
    for name in keep_names:
        values = array[:, name_to_index[name]].copy()
        if name in transform_params:
            # Renaming transformed columns to their ``log10...`` form keeps the
            # feature matrix self-describing once it leaves this helper.
            prepared_columns.append(np.log10(values))
            prepared_names.append(f"log10{name}")
        else:
            prepared_columns.append(values)
            prepared_names.append(name)

    prepared = np.stack(prepared_columns, axis=1)
    return PreparedFeatures(
        feature_names=tuple(prepared_names),
        values=prepared,
        discrete_values=discrete_values,
    )
