"""Legacy-compatible configuration and preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LegacyMLPConfig:
    """Configuration matching the old PyTorch MLP semantics."""

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
class LegacyOptimizerConfig:
    """Optimizer defaults mirrored from the old training scripts."""

    optimizer_name: str = "Adam"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class LegacyTrainingConfig:
    """Training defaults mirrored from the old training scripts."""

    epochs: int
    batch_size: int
    loss_name: str = "MSELoss"
    save_after_epochs: int = 5
    terminate_time_seconds: int = 3600 * 2
    profiling: bool = False


@dataclass(frozen=True)
class PreparedFeatures:
    """Prepared feature matrix plus associated metadata."""

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
    """Prepare feature arrays using the same rules as the old code."""
    array = np.asarray(raw, dtype=float)
    if array.ndim != 2:
        raise ValueError("raw parameter array must be 2D.")
    if array.shape[1] != len(column_names):
        raise ValueError("column_names length does not match raw parameter width.")

    name_to_index = {name: idx for idx, name in enumerate(column_names)}
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
