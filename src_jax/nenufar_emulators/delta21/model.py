"""Delta21 model and optimizer defaults."""

from __future__ import annotations

from dataclasses import dataclass

from nenufar_emulators.conventions import MLPConfig, OptimizerConfig, TrainingConfig


@dataclass(frozen=True)
class Delta21Config:
    """Model, optimizer, and trainer defaults for the current Delta21 workflow.

    This dataclass gathers the model, optimizer, and loop settings that define
    the current Delta21 workflow.
    """

    mlp: MLPConfig
    optimizer: OptimizerConfig
    training: TrainingConfig


def delta21_config() -> Delta21Config:
    """Return the default HERA IDR4 `Delta21` architecture and trainer.

    The Delta21 workflow uses four hidden layers of width 100 with ReLU
    activations. That gives the power-spectrum model enough capacity to learn
    variation across both redshift and wavenumber while keeping the layout
    simple to inspect.
    """
    return Delta21Config(
        mlp=MLPConfig(
            input_dim=11,
            hidden_dim=100,
            n_hidden_blocks=3,
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
