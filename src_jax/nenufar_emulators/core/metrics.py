"""Loss and metric helpers shared by training and validation code.

These helpers are intentionally tiny, but naming them explicitly keeps the
training loop readable and gives later evaluation code one place to import the
same definitions from.
"""

from __future__ import annotations

import jax.numpy as jnp


def mse(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Return mean squared error for regression targets.

    This is the default training loss because the supported emulators are
    trained as scalar regressors, so squared error is the simplest direct comparison
    between the predicted scalar value and the true scalar value.
    """
    return jnp.mean(jnp.square(predictions - targets))


def rmse(predictions: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Return root mean squared error in the same units as the target."""
    return jnp.sqrt(mse(predictions, targets))
