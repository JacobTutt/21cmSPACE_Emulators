"""
Data preprocessing and preparation utilities for 21cmSpace emulators.

This sub-package provides high-level workflows to transform raw simulation outputs
into structured, scaled, and shuffled arrays ready for neural network training.
It includes helpers for:
- simulation parameter table processing (parameters.py)
- fixed-grid resampling and dataset splitting (preparation.py)
- emulator input contracts and transforms (specs.py, transforms.py)
- feature and target scaling (scaling.py)
- spectral tiling and reconstruction (tiling.py)
"""

from jaxemu_21cmSPACE.data_preprocessing.scaling import (
    FeatureScaler,
    FeatureScaling,
    TargetScalingScalar,
)
from jaxemu_21cmSPACE.data_preprocessing.specs import AxisSpec, EmulatorSpec, ParameterSpec
from jaxemu_21cmSPACE.data_preprocessing.tiling import reconstruct_spectra, tile_spectra
from jaxemu_21cmSPACE.data_preprocessing.transforms import apply_transform, invert_transform
from jaxemu_21cmSPACE.data_preprocessing.parameters import (
    PreparedFeatures,
    prepare_feature_matrix,
)
from jaxemu_21cmSPACE.data_preprocessing.preparation import (
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
    "AxisSpec",
    "EmulatorSpec",
    "FeatureScaler",
    "FeatureScaling",
    "ParameterSpec",
    "PreparedFeatures",
    "PreparedSplit",
    "TargetScalingScalar",
    "apply_transform",
    "axis_combinations",
    "build_feature_scaler",
    "build_fixed_axis_grid",
    "flatten_resampled_rows",
    "invert_transform",
    "prepare_feature_matrix",
    "prepare_fixed_grid_training_split",
    "reconstruct_spectra",
    "resample_targets_to_grid",
    "shuffle_rows",
    "split_simulations",
    "tile_spectra",
    "transform_target",
    "transformed_axis_configuration",
]
