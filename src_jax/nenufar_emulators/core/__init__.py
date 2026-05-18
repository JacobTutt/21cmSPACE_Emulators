"""Shared core utilities for emulator specifications and metadata."""

from nenufar_emulators.core.checkpointing import CheckpointMetadata
from nenufar_emulators.core.legacy import (
    LegacyMLPConfig,
    LegacyOptimizerConfig,
    LegacyTrainingConfig,
    PreparedFeatures,
    prepare_feature_matrix,
)
from nenufar_emulators.core.network import forward_mlp, init_mlp
from nenufar_emulators.core.scaling import FeatureScaler, FeatureScaling
from nenufar_emulators.core.specs import AxisSpec, EmulatorSpec, ParameterSpec
from nenufar_emulators.core.tiling import reconstruct_spectra, tile_spectra

__all__ = [
    "AxisSpec",
    "CheckpointMetadata",
    "EmulatorSpec",
    "FeatureScaler",
    "FeatureScaling",
    "LegacyMLPConfig",
    "LegacyOptimizerConfig",
    "LegacyTrainingConfig",
    "ParameterSpec",
    "PreparedFeatures",
    "forward_mlp",
    "init_mlp",
    "prepare_feature_matrix",
    "reconstruct_spectra",
    "tile_spectra",
]
