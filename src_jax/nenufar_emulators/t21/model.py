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

    The T21 workflow uses a smaller tanh network because the task is
    one-dimensional in redshift and does not need the same capacity as the
    Delta21 power-spectrum model.
    """
    return T21Config(
        mlp=MLPConfig(
            input_dim=10,
            hidden_dim=20,
            n_hidden_blocks=3,
            activation="tanh",
        ),
        optimizer=OptimizerConfig(),
        training=TrainingConfig(
            epochs=1000,
            batch_size=769,
            save_after_epochs=5,
            terminate_time_seconds=3600 * 2,
            early_stop=True,
            early_stopping_patience=50,
        ),
    )
