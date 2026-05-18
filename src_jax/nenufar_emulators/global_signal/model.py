"""Global-signal model definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GlobalSignalModelConfig:
    """Default MLP configuration for global-signal emulators."""

    hidden_features: int = 100
    hidden_layers: int = 6
    activation: str = "relu"
