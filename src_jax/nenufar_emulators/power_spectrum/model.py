"""Power-spectrum model definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PowerSpectrumModelConfig:
    """Default MLP configuration for power-spectrum emulators."""

    hidden_features: int = 100
    hidden_layers: int = 6
    activation: str = "relu"
