"""Global-signal model definitions.

As with the power-spectrum module, this file keeps development-time defaults
separate from bundles that are meant to mirror the old repository exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

from nenufar_emulators.core.legacy import (
    LegacyMLPConfig,
    LegacyOptimizerConfig,
    LegacyTrainingConfig,
)

@dataclass(frozen=True)
class GlobalSignalModelConfig:
    """Readable default MLP config for new global-signal development work."""

    hidden_features: int = 100
    hidden_layers: int = 6
    activation: str = "relu"


@dataclass(frozen=True)
class LegacyGlobalSignalBundle:
    """Named group of old-script defaults for one global-signal emulator."""

    name: str
    mlp: LegacyMLPConfig
    optimizer: LegacyOptimizerConfig
    training: LegacyTrainingConfig


def t21_arad_legacy_bundle() -> LegacyGlobalSignalBundle:
    """Return the legacy defaults for the old `T21` training script."""
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


def t21_frad_legacy_bundle() -> LegacyGlobalSignalBundle:
    """Return the HERA IDR4 `T21` defaults with the old training settings.

    The architecture and trainer settings match the old `T21` branch. The
    practical difference from the older Arad variant is the scientific meaning
    of the radio-efficiency parameter, not the MLP shape.
    """
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
    """Return the legacy defaults for the old `Ts` training script."""
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
    """Return the legacy defaults for the old `TK` training script."""
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
    """Return the legacy defaults for the old `Trad` training script."""
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
    """Return the legacy defaults for the old `T_today` training script."""
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
