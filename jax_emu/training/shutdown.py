"""
Graceful shutdown helpers for long training jobs.

This module contains the signal and wall-time utilities used by the trainer to
return cleanly before a Slurm allocation is killed.
"""

from __future__ import annotations

import signal
from time import perf_counter


# Graceful Shutdown
# -----------------
# Lets long Slurm jobs stop cleanly and return to the normal checkpoint save path.

class GracefulShutdown:
    """
    Track whether the training loop should stop after the current epoch.

    This is used for Slurm jobs and other long runs. Signal handlers should stay
    lightweight, so the handler only records that a stop was requested. The
    training loop checks this flag after each epoch and then exits normally.
    """

    def __init__(self, *, log_prefix: str) -> None:
        """
        Initialise the shutdown tracker.

        Parameters
        ----------
        log_prefix:
            Label used when printing training progress and shutdown messages.
        """
        self.log_prefix = log_prefix
        self.stop_requested = False
        self.reason: str | None = None
        self._previous_handlers: dict[int, object] = {}

    def __enter__(self) -> "GracefulShutdown":
        """
        Register signal handlers for clean training shutdown.
        """
        # Catch the usual Slurm termination signal and Ctrl-C style interrupts.
        # Some Slurm scripts may also request SIGUSR1 before wall time expires.
        signal_numbers = [signal.SIGTERM, signal.SIGINT]
        if hasattr(signal, "SIGUSR1"):
            signal_numbers.append(signal.SIGUSR1)

        for signal_number in signal_numbers:
            try:
                self._previous_handlers[signal_number] = signal.getsignal(signal_number)
                signal.signal(signal_number, self._handle_signal)
            except (AttributeError, ValueError):
                # Signal registration is only valid in the main Python thread.
                continue
        return self

    def __exit__(self, *args: object) -> None:
        """
        Restore the signal handlers that were active before training.
        """
        for signal_number, handler in self._previous_handlers.items():
            signal.signal(signal_number, handler)

    def _handle_signal(self, signal_number: int, _frame: object) -> None:
        """
        Request a clean stop after the current epoch.
        """
        self.stop_requested = True
        self.reason = f"received {signal.Signals(signal_number).name}"
        print(
            f"[{self.log_prefix}] {self.reason}; stopping after current epoch.",
            flush=True,
        )


def time_limit_reached(
    *,
    training_start: float,
    max_runtime_seconds: float | None,
    shutdown_margin_seconds: float,
    epoch_seconds: list[float],
) -> bool:
    """
    Check whether there is enough wall time left for another epoch.

    The check is conservative: it uses the slowest recent epoch and adds a
    margin for held-out evaluation and checkpoint writing after training returns.
    """
    if max_runtime_seconds is None:
        return False
    if max_runtime_seconds <= 0:
        raise ValueError("max_runtime_seconds must be positive when provided.")
    if shutdown_margin_seconds < 0:
        raise ValueError("shutdown_margin_seconds must be non-negative.")

    elapsed = perf_counter() - training_start
    recent_epoch_seconds = epoch_seconds[-3:] if epoch_seconds else [0.0]
    next_epoch_estimate = max(recent_epoch_seconds)
    return elapsed + next_epoch_estimate + shutdown_margin_seconds >= max_runtime_seconds
