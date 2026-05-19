"""Legacy-compatible configuration and preprocessing helpers.

This module exists to make the migration explicit. Rather than scattering
knowledge of the old repository through ad hoc comments, we encode the legacy
assumptions in small data structures and preparation helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LegacyMLPConfig:
    """Configuration matching the old PyTorch MLP semantics.

    The important subtlety is that the old model defined one input-to-hidden
    layer plus ``n_hidden`` additional hidden blocks. The JAX helper in this
    repository counts hidden layers directly, so we expose both views.
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
class LegacyOptimizerConfig:
    """Optimizer defaults mirrored from the old training scripts.

    These values are stored separately from the model config because the old
    repository often reused the same MLP shape with slightly different trainer
    settings depending on the emulator family.
    """

    optimizer_name: str = "Adam"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class LegacyTrainingConfig:
    """Training defaults mirrored from the old training scripts.

    These are intentionally the *script* defaults, not the synthetic-test
    defaults used for fast verification in this repository.
    """

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
    """Prepare feature arrays using the same rules as the old code.

    The old training scripts all followed the same broad recipe:

    1. start from a raw parameter table with science-facing column names
    2. drop parameters not used by a given emulator
    3. log-transform selected columns
    4. record which remaining parameters should still be treated as discrete

    This helper makes that recipe reusable and easy to test.

    Parameters
    ----------
    raw:
        Two-dimensional raw parameter table read from a legacy file.
    column_names:
        Names attached to the columns in ``raw``. These define the only valid
        lookup order for the table.
    transform_params:
        Parameters that should be mapped into ``log10`` feature space before
        training, mirroring the old scripts.
    discard_params:
        Parameters present in the raw table but intentionally excluded from the
        emulator input for this particular model.
    discrete_params:
        Parameters whose allowed values should be tracked explicitly because
        they were treated as discrete choices in the old pipeline.
    """
    array = np.asarray(raw, dtype=float)
    if array.ndim != 2:
        raise ValueError("raw parameter array must be 2D.")
    if array.shape[1] != len(column_names):
        raise ValueError("column_names length does not match raw parameter width.")

    name_to_index = {name: idx for idx, name in enumerate(column_names)}
    # Preserve the original column order after dropping unused parameters, so
    # the feature matrix remains aligned with the legacy scripts.
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
            # Legacy code generally transformed by renaming the feature to the
            # ``log10...`` form rather than storing a separate transform map.
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
