"""Training loops for prepared emulator arrays."""

from jax_emu.training.dataloader import (
    iter_device_batch_blocks,
    iter_device_batches,
    move_arrays_to_device,
    normalise_data_device_mode,
    resolve_data_device_mode,
)
from jax_emu.training.scheduler import (
    build_learning_rate_schedule,
    count_steps_per_epoch,
)
from jax_emu.training.trainer import (
    TrainingHistory,
    evaluate_mlp_regressor,
    train_mlp_regressor,
)

__all__ = [
    "TrainingHistory",
    "build_learning_rate_schedule",
    "count_steps_per_epoch",
    "evaluate_mlp_regressor",
    "iter_device_batch_blocks",
    "iter_device_batches",
    "move_arrays_to_device",
    "normalise_data_device_mode",
    "resolve_data_device_mode",
    "train_mlp_regressor",
]
