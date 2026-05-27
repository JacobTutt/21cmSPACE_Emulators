"""Shared configuration dataclasses for model and training setup."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MLPConfig:
    """Configuration for the shared dense emulator network."""

    input_dim: int
    hidden_dim: int = 100
    n_hidden_blocks: int = 6
    output_dim: int = 1
    activation: str = "relu"

    @property
    def total_hidden_layers(self) -> int:
        """Return the total number of hidden layers for the JAX MLP."""
        return 1 + self.n_hidden_blocks


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer settings for a workflow configuration."""

    optimizer_name: str = "Adam"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class TrainingConfig:
    """Training-loop settings for a workflow configuration."""

    epochs: int
    batch_size: int
    loss_name: str = "MSELoss"
    save_after_epochs: int = 5
    terminate_time_seconds: int = 3600 * 2
    profiling: bool = False
    early_stop: bool = False
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
