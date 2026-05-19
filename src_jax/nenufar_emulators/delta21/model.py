"""Delta21 model and optimizer defaults."""

from __future__ import annotations

from dataclasses import dataclass

from nenufar_emulators.legacy import (
    LegacyMLPConfig,
    LegacyOptimizerConfig,
    LegacyTrainingConfig,
)

@dataclass(frozen=True)
class Delta21Config:
    """Model, optimizer, and trainer defaults for the current Delta21 workflow.

    The numbers in this dataclass preserve the legacy Delta21 learning problem
    while presenting it as one coherent workflow configuration.
    """

    mlp: LegacyMLPConfig
    optimizer: LegacyOptimizerConfig
    training: LegacyTrainingConfig


def delta21_config() -> Delta21Config:
    """Return the paper-like HERA IDR4 `Delta21` architecture and trainer.

    This intentionally follows the older `poweremu`-style hidden-layer layout
    rather than the later deeper PyTorch refactor: four hidden layers of width
    100 with ReLU activations.
    """
    return Delta21Config(
        mlp=LegacyMLPConfig(
            input_dim=11,
            hidden_dim=100,
            n_hidden_blocks=3,
            activation="relu",
        ),
        optimizer=LegacyOptimizerConfig(),
        training=LegacyTrainingConfig(
            epochs=10000,
            batch_size=20000,
            save_after_epochs=5,
            terminate_time_seconds=3600 * 2,
        ),
    )
