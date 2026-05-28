"""
Loss and metric helpers shared by training and evaluation code.

This module keeps the regression losses used by the trainer in one place. The
functions are small, but naming them keeps the training step readable and makes
it clear which metric is being reported.
"""

from __future__ import annotations

import jax.numpy as jnp


# Regression Metrics
# ------------------
# Standard error metrics for evaluating emulator performance.

def mse(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """
    Return mean squared error for emulator regression targets.

    This is the default training loss because the dense emulator predicts one
    scalar target per row. Squared error directly compares that prediction with
    the true scalar value.

    Parameters
    ----------
    predictions:
        The predicted values from the neural network.
    targets:
        The ground-truth simulation values.

    Returns
    -------
    jnp.ndarray
        The scalar mean squared error value.
    """
    # Calculate the squared difference between predictions and targets,
    # then take the mean over the entire batch or dataset.
    return jnp.mean(jnp.square(predictions - targets))


def rmse(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """
    Return root mean squared error in the same units as the target.

    Parameters
    ----------
    predictions:
        The predicted values from the neural network.
    targets:
        The ground-truth simulation values.

    Returns
    -------
    jnp.ndarray
        The scalar root mean squared error value.
    """
    # Take the square root after computing the shared MSE definition.
    # This brings the error back into the physical units of the target signal.
    return jnp.sqrt(mse(predictions, targets))
