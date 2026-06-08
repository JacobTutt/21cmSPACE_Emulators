"""
Analysis helpers for trained emulator runs.

This module provides small utilities for inspecting training outputs. The main
use case is plotting training and validation loss curves either immediately
after training or later from a saved `.nenemu` package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jax_emu.training import TrainingHistory


__all__ = [
    "LossCurveData",
    "loss_curves_from_history",
    "loss_curves_from_package",
    "plot_loss_curves",
    "plot_package_losses",
    "plot_training_history",
]


# Loss Curves
# -----------
# Storage utility for the data needed to plot a training run.

@dataclass(frozen=True)
class LossCurveData:
    """
    Storage utility for train/validation loss curves and summary metrics.

    Parameters
    ----------
    train_losses:
        Training loss recorded at the end of each epoch.
    validation_losses:
        Validation loss recorded at the end of each epoch.
    test_loss:
        Optional held-out test loss reported after training.
    best_epoch:
        Optional zero-indexed epoch with the best validation loss.
    best_validation_loss:
        Optional best validation loss value.
    model_name:
        Optional name used in the plot title.
    """

    train_losses: list[float]
    validation_losses: list[float]
    test_loss: float | None = None
    best_epoch: int | None = None
    best_validation_loss: float | None = None
    model_name: str | None = None


def loss_curves_from_history(
    history: "TrainingHistory",
    *,
    test_loss: float | None = None,
    model_name: str | None = None,
) -> LossCurveData:
    """
    Build loss-curve data from a live training history object.

    This is useful immediately after calling `train_mlp_regressor`, before the
    model has necessarily been saved to disk.
    """
    return LossCurveData(
        train_losses=list(history.train_losses),
        validation_losses=list(history.validation_losses),
        test_loss=test_loss,
        best_epoch=history.best_epoch,
        best_validation_loss=history.best_validation_loss,
        model_name=model_name,
    )


def loss_curves_from_package(
    package_or_path: str | Path | dict[str, Any],
    *,
    summary_path: str | Path | None = None,
    test_loss: float | None = None,
) -> LossCurveData:
    """
    Build loss-curve data from a saved `.nenemu` package.

    The checkpoint stores training and validation losses. The held-out test loss
    is stored in the adjacent `.summary.json` written by the high-level T21 and
    Delta21 training workflows, so this helper reads that file automatically
    when it is available.
    """
    # Accept either an already loaded package dictionary or a path to a saved package.
    if isinstance(package_or_path, dict):
        package = package_or_path
        package_path = None
    else:
        from jax_emu.utils.checkpointing import load

        package_path = Path(package_or_path)
        package = load(package_path)

    summary = _load_summary(package_path, summary_path)

    return LossCurveData(
        train_losses=[float(value) for value in package["train_losses"]],
        validation_losses=[float(value) for value in package["val_losses"]],
        test_loss=_resolve_test_loss(test_loss, summary),
        best_epoch=_summary_int(summary, "best_epoch"),
        best_validation_loss=_summary_float(summary, "best_validation_loss"),
        model_name=_model_name_from_metadata(package.get("metadata")),
    )


def plot_loss_curves(
    curves: LossCurveData,
    *,
    output_path: str | Path | None = None,
    title: str | None = None,
    log_y: bool = True,
    ax: Any | None = None,
) -> Any:
    """
    Plot training and validation loss curves.

    Parameters
    ----------
    curves:
        Loss-curve data to plot.
    output_path:
        Optional path where the figure should be written.
    title:
        Optional plot title. Defaults to the model name when available.
    log_y:
        Whether to use a logarithmic y-axis.
    ax:
        Optional existing matplotlib axis.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the loss curves.
    """
    plt = _import_pyplot()

    if not curves.train_losses:
        raise ValueError("train_losses is empty.")
    if not curves.validation_losses:
        raise ValueError("validation_losses is empty.")
    if len(curves.train_losses) != len(curves.validation_losses):
        raise ValueError("train_losses and validation_losses must have the same length.")

    epochs = range(1, len(curves.train_losses) + 1)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
    else:
        fig = ax.figure

    ax.plot(epochs, curves.train_losses, label="Training loss", linewidth=2.0)
    ax.plot(epochs, curves.validation_losses, label="Validation loss", linewidth=2.0)

    if curves.best_epoch is not None:
        best_epoch = curves.best_epoch + 1
        ax.axvline(best_epoch, color="0.35", linestyle="--", linewidth=1.2)
        if curves.best_validation_loss is not None:
            ax.scatter(
                [best_epoch],
                [curves.best_validation_loss],
                color="0.15",
                s=28,
                zorder=3,
                label="Best validation",
            )

    if curves.test_loss is not None:
        ax.text(
            0.98,
            0.96,
            f"Test loss: {curves.test_loss:.3e}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "0.75",
                "alpha": 0.9,
            },
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title or curves.model_name or "Training Loss Curves")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200)

    return fig


def plot_training_history(
    history: "TrainingHistory",
    *,
    test_loss: float | None = None,
    model_name: str | None = None,
    output_path: str | Path | None = None,
    title: str | None = None,
    log_y: bool = True,
    ax: Any | None = None,
) -> Any:
    """
    Plot loss curves from a live training history object.
    """
    curves = loss_curves_from_history(
        history,
        test_loss=test_loss,
        model_name=model_name,
    )
    return plot_loss_curves(curves, output_path=output_path, title=title, log_y=log_y, ax=ax)


def plot_package_losses(
    package_or_path: str | Path | dict[str, Any],
    *,
    summary_path: str | Path | None = None,
    test_loss: float | None = None,
    output_path: str | Path | None = None,
    title: str | None = None,
    log_y: bool = True,
    ax: Any | None = None,
) -> Any:
    """
    Plot loss curves from a saved `.nenemu` package.
    """
    curves = loss_curves_from_package(
        package_or_path,
        summary_path=summary_path,
        test_loss=test_loss,
    )
    return plot_loss_curves(curves, output_path=output_path, title=title, log_y=log_y, ax=ax)


# Internal Helpers
# ----------------

def _default_summary_path(package_path: Path | None) -> Path | None:
    """
    Return the default adjacent summary path for a checkpoint package.
    """
    if package_path is None:
        return None
    return package_path.with_suffix(".summary.json")


def _load_summary(
    package_path: Path | None,
    summary_path: str | Path | None,
) -> dict[str, Any] | None:
    """
    Load a training summary JSON file when one is available.
    """
    candidate = (
        Path(summary_path)
        if summary_path is not None
        else _default_summary_path(package_path)
    )
    if candidate is None or not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def _model_name_from_metadata(metadata: Any) -> str | None:
    """
    Read the model name from checkpoint metadata when it exists.
    """
    if metadata is None:
        return None
    if isinstance(metadata, dict):
        return metadata.get("model_name")
    return getattr(metadata, "model_name", None)


def _resolve_test_loss(
    test_loss: float | None,
    summary: dict[str, Any] | None,
) -> float | None:
    """
    Prefer an explicit test loss, otherwise read it from the summary.
    """
    if test_loss is not None:
        return float(test_loss)
    return _summary_float(summary, "test_loss")


def _summary_float(summary: dict[str, Any] | None, key: str) -> float | None:
    """
    Read a nullable float from a training summary.
    """
    if summary is None or summary.get(key) is None:
        return None
    return float(summary[key])


def _summary_int(summary: dict[str, Any] | None, key: str) -> int | None:
    """
    Read a nullable integer from a training summary.
    """
    if summary is None or summary.get(key) is None:
        return None
    return int(summary[key])


def _import_pyplot() -> Any:
    """
    Import matplotlib lazily so non-plotting code avoids pyplot startup cost.
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Plotting loss curves requires matplotlib, which is a default project "
            "dependency. Reinstall the package with your chosen JAX backend extra, "
            "for example `python -m pip install -e '.[cpu]'`."
        ) from exc
    return plt
