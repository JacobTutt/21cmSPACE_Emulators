"""Training loops for prepared emulator arrays."""

from jax_emu.training.trainer import (
    TrainingHistory,
    build_learning_rate_schedule,
    evaluate_mlp_regressor,
    train_mlp_regressor,
)

__all__ = [
    "TrainingHistory",
    "build_learning_rate_schedule",
    "evaluate_mlp_regressor",
    "train_mlp_regressor",
]
