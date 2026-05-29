"""
Delta21 model and optimizer defaults.

This module provides the default configuration parameters for the Delta21
emulator, bundling architecture (MLP), optimizer, and training loop settings.
"""

from __future__ import annotations

from dataclasses import dataclass

from jax_emu.utils.config import MLPConfig, OptimizerConfig, TrainingConfig


# Workflow Configuration
# ----------------------
# Bundled settings for the Delta21 emulator workflow.

@dataclass(frozen=True)
class Delta21Config:
    """
    Model, optimizer, and trainer defaults for the current Delta21 workflow.

    This dataclass gathers the model, optimizer, and loop settings that define
    the current Delta21 workflow.

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
# Predefined baseline settings for the Delta21 emulator.

def delta21_config() -> Delta21Config:
    """
    Return the default 21cmSPACE `Delta21` architecture and trainer.

    This matches the hidden-layer width and activation used by the JWST
    synergies Dsq21 emulator: four hidden layers of width 100 with tanh
    activations.

    Returns
    -------
    Delta21Config
        The default baseline configuration.
    """
    return Delta21Config(
        # Baseline MLP architecture for power-spectrum regression.
        # Note: input_dim=11 accounts for (z, log10k) + 9 parameters.
        mlp=MLPConfig(
            input_dim=11,
            hidden_dim=100,
            n_hidden_blocks=3,
            activation="tanh",
        ),
        # Default Adam optimizer settings.
        optimizer=OptimizerConfig(),
        # Default training loop behavior.
        training=TrainingConfig(
            epochs=10000,
            batch_size=10000,
            save_after_epochs=5,
            terminate_time_seconds=3600 * 2,
            early_stop=True,
            early_stopping_patience=50,
        ),
    )
