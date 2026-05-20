"""T21 model and optimizer defaults."""

from __future__ import annotations

from dataclasses import dataclass

from nenufar_emulators.conventions import MLPConfig, OptimizerConfig, TrainingConfig


@dataclass(frozen=True)
class T21Config:
    """Model, optimizer, and trainer defaults for the current T21 workflow."""

    mlp: MLPConfig
    optimizer: OptimizerConfig
    training: TrainingConfig


def t21_config() -> T21Config:
    """Return the default configuration for the current T21 workflow.

    This configuration follows the later PyTorch scalar-MLP regime more
    closely: a wider ReLU network and a much larger batch size than the older
    global-signal-specific setup.
    """
    return T21Config(
        mlp=MLPConfig(
            input_dim=10,
            hidden_dim=100,
            n_hidden_blocks=6,
            activation="relu",
        ),
        optimizer=OptimizerConfig(),
        training=TrainingConfig(
            epochs=10000,
            batch_size=20000,
            save_after_epochs=5,
            terminate_time_seconds=3600 * 2,
        ),
    )
