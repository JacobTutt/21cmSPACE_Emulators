"""Global-signal model definitions."""

from __future__ import annotations

from dataclasses import dataclass

from nenufar_emulators.core.legacy import (
    LegacyMLPConfig,
    LegacyOptimizerConfig,
    LegacyTrainingConfig,
)

@dataclass(frozen=True)
class GlobalSignalModelConfig:
    """Development-time MLP configuration for global-signal emulators."""

    hidden_features: int = 100
    hidden_layers: int = 6
    activation: str = "relu"


@dataclass(frozen=True)
class LegacyGlobalSignalBundle:
    """Legacy-aligned model and training defaults for one emulator."""

    name: str
    mlp: LegacyMLPConfig
    optimizer: LegacyOptimizerConfig
    training: LegacyTrainingConfig


def t21_arad_legacy_bundle() -> LegacyGlobalSignalBundle:
    """Return defaults matching the old `T21` PyTorch script."""
    return LegacyGlobalSignalBundle(
        name="T21",
        mlp=LegacyMLPConfig(input_dim=10, hidden_dim=100, n_hidden_blocks=6),
        optimizer=LegacyOptimizerConfig(),
        training=LegacyTrainingConfig(
            epochs=10000,
            batch_size=20000,
            save_after_epochs=5,
            terminate_time_seconds=3600 * 2,
        ),
    )


def ts_arad_legacy_bundle() -> LegacyGlobalSignalBundle:
    """Return defaults matching the old `Ts` PyTorch script."""
    return LegacyGlobalSignalBundle(
        name="Ts",
        mlp=LegacyMLPConfig(input_dim=10, hidden_dim=100, n_hidden_blocks=6),
        optimizer=LegacyOptimizerConfig(),
        training=LegacyTrainingConfig(
            epochs=10000,
            batch_size=20000,
            save_after_epochs=2,
            terminate_time_seconds=3600 * 2,
        ),
    )


def tk_frad_legacy_bundle() -> LegacyGlobalSignalBundle:
    """Return defaults matching the old `TK` PyTorch script."""
    return LegacyGlobalSignalBundle(
        name="TK",
        mlp=LegacyMLPConfig(input_dim=9, hidden_dim=100, n_hidden_blocks=6),
        optimizer=LegacyOptimizerConfig(),
        training=LegacyTrainingConfig(
            epochs=10000,
            batch_size=20000,
            save_after_epochs=10,
            terminate_time_seconds=3600 * 2,
        ),
    )


def trad_frad_legacy_bundle() -> LegacyGlobalSignalBundle:
    """Return defaults matching the old `Trad` PyTorch script."""
    return LegacyGlobalSignalBundle(
        name="Trad",
        mlp=LegacyMLPConfig(input_dim=10, hidden_dim=100, n_hidden_blocks=6),
        optimizer=LegacyOptimizerConfig(),
        training=LegacyTrainingConfig(
            epochs=10000,
            batch_size=20000,
            save_after_epochs=5,
            terminate_time_seconds=3600 * 2,
        ),
    )


def t_today_frad_legacy_bundle() -> LegacyGlobalSignalBundle:
    """Return defaults matching the old `T_today` PyTorch script."""
    return LegacyGlobalSignalBundle(
        name="T_today",
        mlp=LegacyMLPConfig(input_dim=10, hidden_dim=100, n_hidden_blocks=6),
        optimizer=LegacyOptimizerConfig(),
        training=LegacyTrainingConfig(
            epochs=10000,
            batch_size=20000,
            save_after_epochs=50,
            terminate_time_seconds=3600 * 2,
        ),
    )
