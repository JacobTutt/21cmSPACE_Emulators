"""
T21 model and optimizer defaults.

This module provides the default configuration parameters for the T21 emulator,
bundling architecture (MLP), optimizer, and training loop settings.
"""

from __future__ import annotations

from dataclasses import dataclass

from jax_emu.utils.config import MLPConfig, OptimizerConfig, TrainingConfig


# Workflow Configuration
# ----------------------
# Bundled settings for the T21 emulator workflow.

@dataclass(frozen=True)
class T21Config:
    """
    Model, optimizer, and trainer defaults for the current T21 workflow.

    Parameters
    ----------
    mlp:
        Dense MLP architecture settings.
    optimizer:
        Optax optimizer configuration.
    training:
        High-level training loop settings.
    """

    mlp: MLPConfig
    optimizer: OptimizerConfig
    training: TrainingConfig


# Configuration Defaults
# ----------------------
# Predefined baseline settings for the T21 emulator.

def t21_config() -> T21Config:
    """
    Return the default configuration for the current T21 workflow.

    This uses a moderately wider GELU MLP for the cubic-interpolation T21
    workflow: four hidden layers of width 64.

    Returns
    -------
    T21Config
        The default baseline configuration.
    """
    return T21Config(
        # Baseline MLP architecture for brightness temperature regression.
        mlp=MLPConfig(
            input_dim=10,
            hidden_dim=64,
            n_hidden_blocks=3,
            activation="gelu",
        ),
        # Default Adam optimizer settings.
        optimizer=OptimizerConfig(),
        # Default training loop behavior.
        training=TrainingConfig(
            epochs=10000,
            batch_size=1000,
            save_after_epochs=5,
            data_device_mode="gpu_memory",
            terminate_time_seconds=3600 * 2,
            early_stop=True,
            early_stopping_patience=50,
        ),
    )
