"""Data loading and preprocessing for supported emulator workflows."""

from nenufar_emulators.data_preprocessing.hera_idr4 import (
    HERA_IDR4_COLUMNS,
    HERA_LITTLE_H,
    HeraIdr4Axes,
    HeraIdr4Product,
    load_hera_idr4_axes,
    load_hera_idr4_delta21,
    load_hera_idr4_t21,
    nan_simulation_indices,
)
from nenufar_emulators.data_preprocessing.parameters import (
    PreparedFeatures,
    prepare_feature_matrix,
)
from nenufar_emulators.data_preprocessing.preparation import (
    PreparedSplit,
    axis_combinations,
    build_feature_scaler,
    build_fixed_axis_grid,
    flatten_resampled_rows,
    prepare_fixed_grid_training_split,
    resample_targets_to_grid,
    shuffle_rows,
    split_simulations,
    transform_target,
    transformed_axis_configuration,
)

__all__ = [
    "HERA_IDR4_COLUMNS",
    "HERA_LITTLE_H",
    "HeraIdr4Axes",
    "HeraIdr4Product",
    "PreparedFeatures",
    "PreparedSplit",
    "axis_combinations",
    "build_feature_scaler",
    "build_fixed_axis_grid",
    "flatten_resampled_rows",
    "load_hera_idr4_axes",
    "load_hera_idr4_delta21",
    "load_hera_idr4_t21",
    "nan_simulation_indices",
    "prepare_feature_matrix",
    "prepare_fixed_grid_training_split",
    "resample_targets_to_grid",
    "shuffle_rows",
    "split_simulations",
    "transform_target",
    "transformed_axis_configuration",
]
