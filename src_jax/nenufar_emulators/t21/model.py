"""T21 model and optimizer defaults."""

from __future__ import annotations

from dataclasses import dataclass

from nenufar_emulators.legacy import (
    LegacyMLPConfig,
    LegacyOptimizerConfig,
    LegacyTrainingConfig,
)

@dataclass(frozen=True)
class T21Config:
    """Model, optimizer, and trainer defaults for the current T21 workflow."""

    mlp: LegacyMLPConfig
    optimizer: LegacyOptimizerConfig
    training: LegacyTrainingConfig


def t21_config() -> T21Config:
    """Return the HERA IDR4 `T21` defaults with a `globalemu`-like network.

    For the HERA IDR4 migration we prefer the narrower `tanh` architecture
    recorded in the old `globalemu` run metadata over the later deep-ReLU
    unified PyTorch path.
    """
    return T21Config(
        mlp=LegacyMLPConfig(
            input_dim=10,
            hidden_dim=20,
            n_hidden_blocks=3,
            activation="tanh",
        ),
        optimizer=LegacyOptimizerConfig(),
        training=LegacyTrainingConfig(
            epochs=1000,
            batch_size=769,
            save_after_epochs=5,
            terminate_time_seconds=3600 * 2,
            early_stop=True,
            early_stopping_patience=50,
        ),
    )
