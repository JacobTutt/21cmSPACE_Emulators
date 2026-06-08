"""Reusable JAX emulator infrastructure."""

from jax_emu.utils.checkpointing import CheckpointMetadata, load, save
from jax_emu.architectures.mlp import DenseMLP
from jax_emu.utils.config import MLPConfig, OptimizerConfig, TrainingConfig
from jax_emu.data_preprocessing.specs import AxisSpec, EmulatorSpec, ParameterSpec
from jax_emu.infer import Emulator, FixedGridEmulator
from jax_emu.training import (
    TrainingHistory,
    build_learning_rate_schedule,
    train_mlp_regressor,
)

__all__ = [
    "AxisSpec",
    "CheckpointMetadata",
    "DenseMLP",
    "Emulator",
    "EmulatorSpec",
    "FixedGridEmulator",
    "MLPConfig",
    "OptimizerConfig",
    "ParameterSpec",
    "TrainingConfig",
    "TrainingHistory",
    "build_learning_rate_schedule",
    "load",
    "save",
    "train_mlp_regressor",
]
