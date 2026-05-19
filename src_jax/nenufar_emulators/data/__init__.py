"""Data loading and legacy-derived preparation for supported emulators."""

from nenufar_emulators.data.hera_idr4 import (
    HERA_IDR4_COLUMNS,
    HERA_LITTLE_H,
    HeraIdr4Axes,
    HeraIdr4Product,
    load_hera_idr4_axes,
    load_hera_idr4_delta21,
    load_hera_idr4_t21,
    nan_simulation_indices,
)
from nenufar_emulators.data.preparation import (
    LegacyPreparedSplit,
    apply_legacy_target_transform,
    build_fixed_axis_grid,
    build_legacy_feature_scaler,
    generate_resampled_rows,
    generate_training_rows,
    generate_validation_rows,
    prepare_fixed_grid_training_split,
    prepare_legacy_training_split,
    shuffle_rows,
    split_simulations,
    transformed_axis_configuration,
)

__all__ = [
    "HERA_IDR4_COLUMNS",
    "HERA_LITTLE_H",
    "HeraIdr4Axes",
    "HeraIdr4Product",
    "LegacyPreparedSplit",
    "apply_legacy_target_transform",
    "build_fixed_axis_grid",
    "build_legacy_feature_scaler",
    "generate_resampled_rows",
    "generate_training_rows",
    "generate_validation_rows",
    "load_hera_idr4_axes",
    "load_hera_idr4_delta21",
    "load_hera_idr4_t21",
    "nan_simulation_indices",
    "prepare_fixed_grid_training_split",
    "prepare_legacy_training_split",
    "shuffle_rows",
    "split_simulations",
    "transformed_axis_configuration",
]
