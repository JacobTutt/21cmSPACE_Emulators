"""Analysis helpers for trained emulator runs."""

from jax_emu.analysis.loss_curves import (
    LossCurveData,
    loss_curves_from_history,
    loss_curves_from_package,
    plot_loss_curves,
    plot_package_losses,
    plot_training_history,
)

__all__ = [
    "LossCurveData",
    "loss_curves_from_history",
    "loss_curves_from_package",
    "plot_loss_curves",
    "plot_package_losses",
    "plot_training_history",
]
