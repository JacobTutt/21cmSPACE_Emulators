"""
Learning-rate schedules used by the shared training loop.

This module contains the small utilities that convert epoch-level scheduler
settings into Optax-compatible step-wise learning-rate schedules.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax


# Learning-Rate Schedules
# -----------------------
# Builds the step-wise learning-rate rule used by the Optax optimiser.

def count_steps_per_epoch(n_samples: int, batch_size: int) -> int:
    """
    Return the number of fixed-shape mini-batches in one epoch.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    return max(int(np.ceil(n_samples / batch_size)), 1)


def build_learning_rate_schedule(
    *,
    learning_rate: float,
    schedule_name: str = "constant",
    steps_per_epoch: int,
    epochs: int,
    final_fraction: float = 0.1,
    warmup_epochs: int = 0,
) -> Callable[[jax.Array], jax.Array]:
    """
    Build the learning-rate schedule used by AdamW.

    The returned schedule is evaluated once per optimiser update, so epoch
    counts are converted into mini-batch step counts before constructing the
    schedule.

    Supported schedules
    -------------------
    constant:
        Uses `learning_rate` for every optimizer update. Ignores
        `final_fraction` and `warmup_epochs`.
    cosine:
        Starts at `learning_rate` and decays smoothly to
        `learning_rate * final_fraction` over `epochs`.
    warmup_cosine:
        Starts at zero, increases to `learning_rate` over `warmup_epochs`,
        then decays to `learning_rate * final_fraction` over `epochs`.
    exponential_decay:
        Starts at `learning_rate` and decays multiplicatively to
        `learning_rate * final_fraction` over `epochs`.

    Parameters
    ----------
    learning_rate:
        Initial or peak learning rate used by the schedule.
    schedule_name:
        Schedule type: `constant`, `cosine`, `warmup_cosine`, or
        `exponential_decay`.
    steps_per_epoch:
        Number of optimizer updates in one training epoch.
    epochs:
        Number of epochs over which the schedule is defined.
    final_fraction:
        Final learning-rate fraction for decay schedules. For example, `0.05`
        means the final learning rate is 5 percent of `learning_rate`.
    warmup_epochs:
        Number of epochs used to ramp from zero to `learning_rate` for
        `warmup_cosine`. Ignored by the other schedules.

    Returns
    -------
    Callable[[jax.Array], jax.Array]
        A step-indexed Optax-compatible learning-rate schedule.
    """
    # Basic validation keeps scheduler configuration errors visible at startup.
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be at least 1.")
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if final_fraction <= 0 or final_fraction > 1:
        raise ValueError("final_fraction must be in the interval (0, 1].")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative.")

    # Convert epoch-level settings into step-level values.
    schedule = schedule_name.lower()
    total_steps = max(steps_per_epoch * epochs, 1)
    warmup_steps = min(warmup_epochs * steps_per_epoch, total_steps - 1)
    end_value = learning_rate * final_fraction

    if schedule == "constant":
        return optax.constant_schedule(learning_rate)

    if schedule == "cosine":
        return optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=total_steps,
            alpha=final_fraction,
        )

    if schedule == "warmup_cosine":
        if warmup_steps == 0:
            return optax.cosine_decay_schedule(
                init_value=learning_rate,
                decay_steps=total_steps,
                alpha=final_fraction,
            )
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=end_value,
        )

    if schedule == "exponential_decay":

        def exponential_schedule(step: jax.Array) -> jax.Array:
            """
            Decay smoothly from the initial learning rate to the final fraction.
            """
            progress = jnp.minimum(step / max(total_steps - 1, 1), 1.0)
            return learning_rate * final_fraction**progress

        return exponential_schedule

    raise ValueError(
        "learning_rate_schedule must be one of "
        "'constant', 'cosine', 'warmup_cosine', or 'exponential_decay'."
    )
