"""
Shared configuration dataclasses for emulator training.

This module stores the small pieces of configuration that define a workflow:
- the MLP architecture
- the optimizer settings
- the training-loop settings

The workflow-specific modules choose the default values. These dataclasses keep
those values named and easy to save in checkpoint metadata.
"""

from __future__ import annotations

from dataclasses import dataclass


# Architecture Configuration
# --------------------------
# Parameters governing the structural layout of the neural network.

@dataclass(frozen=True)
class MLPConfig:
    """
    Storage utility for dense MLP architecture settings.

    Parameters
    ----------
    input_dim:
        Number of input features (axes + parameters).
    hidden_dim:
        Width of each hidden layer.
    n_hidden_blocks:
        Number of hidden linear blocks after the first layer.
    output_dim:
        Number of output predictions (usually 1 for scalar regression).
    activation:
        The name of the non-linear activation function to use.
    """

    input_dim: int
    hidden_dim: int = 100
    n_hidden_blocks: int = 6
    output_dim: int = 1
    activation: str = "relu"

    @property
    def total_hidden_layers(self) -> int:
        """
        Return the total number of hidden layers used by DenseMLP.

        The workflow config stores the number of extra hidden blocks after the
        first layer. DenseMLP expects the full hidden-layer count directly.

        Returns
        -------
        int
            The total count of hidden layers.
        """
        # Add 1 (the initial layer) to the number of extra blocks.
        return 1 + self.n_hidden_blocks


# Optimization Configuration
# --------------------------
# Parameters governing the weight update process.

@dataclass(frozen=True)
class OptimizerConfig:
    """
    Storage utility for optimizer settings.

    Parameters
    ----------
    optimizer_name:
        The name of the Optax optimizer to use (e.g. 'Adam').
    learning_rate:
        Initial step size for weight updates.
    weight_decay:
        Regularization parameter to prevent overfitting.
    """

    optimizer_name: str = "Adam"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


# Training Workflow Configuration
# -------------------------------
# Parameters governing the high-level training loop behavior.

@dataclass(frozen=True)
class TrainingConfig:
    """
    Storage utility for training-loop settings.

    Parameters
    ----------
    epochs:
        Number of complete passes over the training dataset.
    batch_size:
        Number of samples processed per gradient update.
    loss_name:
        Label for the loss function being used.
    save_after_epochs:
        Frequency (in epochs) at which checkpoints are written to disk.
    terminate_time_seconds:
        Hard time limit for training before forced termination.
    profiling:
        Whether to enable performance profiling.
    prefetch_batches:
        Number of mini-batches to keep queued on the device.
    early_stop:
        Whether to enable early stopping based on validation loss.
    early_stopping_patience:
        Number of epochs to wait for improvement before stopping.
    early_stopping_min_delta:
        Minimum change in validation loss to qualify as an improvement.
    """

    epochs: int
    batch_size: int
    loss_name: str = "MSELoss"
    save_after_epochs: int = 5
    terminate_time_seconds: int = 3600 * 2
    profiling: bool = False
    prefetch_batches: int = 2
    early_stop: bool = False
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
