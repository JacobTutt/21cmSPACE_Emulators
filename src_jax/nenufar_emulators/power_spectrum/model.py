"""Power-spectrum model definitions.

This module separates two ideas that are easy to conflate during migration:
the lightweight development-time defaults used by the new code, and the
legacy-aligned bundles that intentionally mirror the old scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

from nenufar_emulators.core.legacy import (
    LegacyMLPConfig,
    LegacyOptimizerConfig,
    LegacyTrainingConfig,
)

@dataclass(frozen=True)
class PowerSpectrumModelConfig:
    """Small readable config for the shared power-spectrum MLP shape.

    This is the sort of object you would pass around inside the new codebase
    when you do not need exact old-script parity.
    """

    hidden_features: int = 100
    hidden_layers: int = 6
    activation: str = "relu"


@dataclass(frozen=True)
class LegacyPowerSpectrumBundle:
    """One named package of old-script defaults for a power-spectrum emulator.

    Each bundle answers a practical question: "if I want to reproduce the old
    script's architecture and trainer settings, which numbers should I use?"
    """

    name: str
    mlp: LegacyMLPConfig
    optimizer: LegacyOptimizerConfig
    training: LegacyTrainingConfig


def delta21_frad_legacy_bundle() -> LegacyPowerSpectrumBundle:
    """Return the exact default bundle for the legacy `Delta21` training path."""
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
    """Return the exact default bundle for the legacy `SDC3b_Pk` training path."""
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
