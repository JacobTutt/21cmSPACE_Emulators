"""Reusable JAX emulator infrastructure."""

from jax_emu.utils.checkpointing import CheckpointMetadata, load, save
from jax_emu.architectures.mlp import DenseMLP
from jax_emu.utils.config import MLPConfig, OptimizerConfig, TrainingConfig
from jax_emu.data_preprocessing.specs import AxisSpec, EmulatorSpec, ParameterSpec
from jax_emu.infer import Emulator
from jax_emu.training.trainer import TrainingHistory, train_mlp_regressor

__all__ = [
    "AxisSpec",
    "CheckpointMetadata",
    "DenseMLP",
    "Emulator",
    "EmulatorSpec",
    "MLPConfig",
    "OptimizerConfig",
    "ParameterSpec",
    "TrainingConfig",
    "TrainingHistory",
    "load",
    "save",
    "train_mlp_regressor",
]
