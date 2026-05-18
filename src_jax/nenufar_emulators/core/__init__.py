"""Shared core utilities for emulator specifications and metadata."""

from nenufar_emulators.core.checkpointing import CheckpointMetadata
from nenufar_emulators.core.datasets import NormalisationPipeline, SpectrumBatch, SpectrumDataset, TiledBatch
from nenufar_emulators.core.legacy import (
    LegacyMLPConfig,
    LegacyOptimizerConfig,
    LegacyTrainingConfig,
    PreparedFeatures,
    prepare_feature_matrix,
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
    "forward_mlp",
    "init_mlp",
    "prepare_feature_matrix",
    "reconstruct_spectra",
    "tile_spectra",
]
