"""Shared core utilities for emulator specifications and metadata."""

from nenufar_emulators.core.checkpointing import CheckpointMetadata, load, save
from nenufar_emulators.core.datasets import NormalisationPipeline, SpectrumBatch, SpectrumDataset, TiledBatch
from nenufar_emulators.core.hera_idr4 import (
    HERA_IDR4_COLUMNS,
    HERA_LITTLE_H,
    HeraIdr4Axes,
    HeraIdr4Product,
    load_hera_idr4_axes,
    load_hera_idr4_delta21,
    load_hera_idr4_t21,
    nan_simulation_indices,
)
from nenufar_emulators.core.legacy import (
    LegacyMLPConfig,
    LegacyOptimizerConfig,
    LegacyTrainingConfig,
    PreparedFeatures,
    prepare_feature_matrix,
)
from nenufar_emulators.core.legacy_workflow import (
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
from nenufar_emulators.core.network import DenseMLP, forward_mlp, init_mlp
from nenufar_emulators.core.normalisation import (
    DatasetStatistics,
    SpecTransformPipeline,
    StandardizationPipeline,
)
from nenufar_emulators.core.scaling import FeatureScaler, FeatureScaling
from nenufar_emulators.core.specs import AxisSpec, EmulatorSpec, ParameterSpec
from nenufar_emulators.core.tiling import reconstruct_spectra, tile_spectra

__all__ = [
    "AxisSpec",
    "CheckpointMetadata",
    "DatasetStatistics",
    "DenseMLP",
    "EmulatorSpec",
    "FeatureScaler",
    "FeatureScaling",
    "HERA_IDR4_COLUMNS",
    "HERA_LITTLE_H",
    "HeraIdr4Axes",
    "HeraIdr4Product",
    "LegacyPreparedSplit",
    "LegacyMLPConfig",
    "LegacyOptimizerConfig",
    "LegacyTrainingConfig",
    "NormalisationPipeline",
    "ParameterSpec",
    "PreparedFeatures",
    "SpecTransformPipeline",
    "SpectrumBatch",
    "SpectrumDataset",
    "StandardizationPipeline",
    "TiledBatch",
    "build_fixed_axis_grid",
    "forward_mlp",
    "generate_resampled_rows",
    "init_mlp",
    "load",
    "load_hera_idr4_axes",
    "load_hera_idr4_delta21",
    "load_hera_idr4_t21",
    "nan_simulation_indices",
    "prepare_feature_matrix",
    "prepare_fixed_grid_training_split",
    "prepare_legacy_training_split",
    "reconstruct_spectra",
    "save",
    "shuffle_rows",
    "split_simulations",
    "tile_spectra",
    "transformed_axis_configuration",
]
