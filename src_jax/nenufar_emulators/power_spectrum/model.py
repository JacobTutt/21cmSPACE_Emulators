"""Power-spectrum model definitions."""

from __future__ import annotations

from dataclasses import dataclass

from nenufar_emulators.core.legacy import (
    LegacyMLPConfig,
    LegacyOptimizerConfig,
    LegacyTrainingConfig,
)

@dataclass(frozen=True)
class PowerSpectrumModelConfig:
    """Development-time MLP configuration for power-spectrum emulators."""

    hidden_features: int = 100
    hidden_layers: int = 6
    activation: str = "relu"


@dataclass(frozen=True)
class LegacyPowerSpectrumBundle:
    """Legacy-aligned model and training defaults for one emulator."""

    name: str
    mlp: LegacyMLPConfig
    optimizer: LegacyOptimizerConfig
    training: LegacyTrainingConfig


def delta21_frad_legacy_bundle() -> LegacyPowerSpectrumBundle:
    """Return defaults matching the old `Delta21` PyTorch script."""
    return LegacyPowerSpectrumBundle(
        name="Delta21",
        mlp=LegacyMLPConfig(input_dim=11, hidden_dim=100, n_hidden_blocks=6),
        optimizer=LegacyOptimizerConfig(),
        training=LegacyTrainingConfig(
            epochs=10000,
            batch_size=20000,
            save_after_epochs=5,
            terminate_time_seconds=3600 * 2,
        ),
    )


def sdc3b_pk_legacy_bundle() -> LegacyPowerSpectrumBundle:
    """Return defaults matching the old `SDC3b_Pk` PyTorch script."""
    return LegacyPowerSpectrumBundle(
        name="SDC3b_Pk",
        mlp=LegacyMLPConfig(input_dim=7, hidden_dim=100, n_hidden_blocks=6),
        optimizer=LegacyOptimizerConfig(),
        training=LegacyTrainingConfig(
            epochs=10000,
            batch_size=20000,
            save_after_epochs=10,
            terminate_time_seconds=3600 * 3,
        ),
    )
