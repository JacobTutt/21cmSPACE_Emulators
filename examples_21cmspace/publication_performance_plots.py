"""
Publication-style emulator performance plots.

This script regenerates the main T21 and Delta21 emulator performance figures
from saved model packages and a 21cmSPACE-style dataset. All examples and
summary statistics are computed from the held-out test split, not from the
training or validation data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from examples_21cmspace.delta21.data import (
    DELTA21_LOG_TARGET_FLOOR,
    delta21_low_z_sampled_axes,
    delta21_spec,
    prepare_twentyonecmspace_delta21_parameters,
)
from examples_21cmspace.delta21.emulator import (
    build_delta21_fixed_grid_emulator,
    load_delta21_package,
)
from examples_21cmspace.t21.data import (
    prepare_twentyonecmspace_t21_parameters,
    t21_spec,
)
from examples_21cmspace.t21.emulator import (
    build_t21_fixed_grid_emulator,
    load_t21_package,
)
from examples_21cmspace.twentyonecmspace import (
    load_twentyonecmspace_delta21,
    load_twentyonecmspace_t21,
)
from jax_emu.data_preprocessing.preparation import (
    apply_target_floor,
    build_fixed_axis_grid,
    resample_targets_to_grid,
    split_simulations,
    transform_target,
    transformed_axis_configuration,
)


# Defaults
# --------
# These match the publication-style plots already generated for the emulator.

RANDOM_SEED = 20260609
T21_FE_FLOOR_MK = 25.0
DELTA21_LOG_FE_FLOOR = 0.5
DELTA21_TARGET_OFFSET = 1e-8


# Containers
# ----------
# Store evaluated test-set grids in a form that plotting functions can reuse.

@dataclass(frozen=True)
class T21Evaluation:
    """
    Test-set T21 predictions and fractional errors.
    """

    z: np.ndarray
    parameters: np.ndarray
    truth: np.ndarray
    prediction: np.ndarray
    fractional_error: np.ndarray
    fe_floor_mk: float


@dataclass(frozen=True)
class Delta21Evaluation:
    """
    Test-set Delta21 predictions and fractional errors in log-space.
    """

    z: np.ndarray
    k: np.ndarray
    parameters: np.ndarray
    truth_log: np.ndarray
    prediction_log: np.ndarray
    fractional_error: np.ndarray
    log_fe_floor: float
    log_target_floor: float


# CLI
# ---
# The command can make T21-only, Delta21-only, or combined plot sets.

def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser.
    """
    parser = argparse.ArgumentParser(
        description="Create publication-style T21 and Delta21 emulator performance plots."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the 21cmSPACE-style dataset used for train/validation/test splitting.",
    )
    parser.add_argument("--t21-package", help="Path to the trained T21 .nenemu package.")
    parser.add_argument("--delta21-package", help="Path to the trained Delta21 .nenemu package.")
    parser.add_argument(
        "--output-dir",
        default="publication_plots",
        help="Directory where PNGs and the summary NPZ are written.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--prediction-batch-size", type=int, default=512)
    parser.add_argument("--t21-fe-floor-mk", type=float, default=T21_FE_FLOOR_MK)
    parser.add_argument("--delta21-log-fe-floor", type=float, default=DELTA21_LOG_FE_FLOOR)
    parser.add_argument("--delta21-log-target-floor", type=float, default=DELTA21_LOG_TARGET_FLOOR)
    parser.add_argument("--t21-random-examples", type=int, default=20)
    parser.add_argument("--delta21-random-examples", type=int, default=10)
    return parser


def main() -> None:
    """
    Run the requested evaluations and write publication-style figures.
    """
    args = build_parser().parse_args()
    if args.t21_package is None and args.delta21_package is None:
        raise ValueError("Provide at least one of --t21-package or --delta21-package.")

    configure_matplotlib()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    summary: dict[str, float | int] = {"random_seed": int(args.seed)}

    t21_eval = None
    if args.t21_package is not None:
        t21_eval = evaluate_t21_test_set(
            dataset_root=args.dataset_root,
            package_path=args.t21_package,
            random_state=args.random_state,
            fe_floor_mk=args.t21_fe_floor_mk,
            prediction_batch_size=args.prediction_batch_size,
        )
        plot_t21_random_examples(
            t21_eval,
            rng=rng,
            n_examples=args.t21_random_examples,
            output_path=output_dir / "t21_plot1_random20_truth_emulator_fractional_error.png",
        )
        plot_t21_ranked_examples(
            t21_eval,
            output_path=output_dir / "t21_plot2_ranked_fractional_error_examples.png",
        )
        plot_fractional_error_histogram(
            t21_eval.fractional_error,
            floor_label=f"T21 floor = {args.t21_fe_floor_mk:g} mK",
            title="Global Signal Fractional Error",
            output_path=output_dir / "t21_plot3_fractional_error_histogram.png",
        )
        summary.update(
            {
                "t21_fe_floor_mk": float(args.t21_fe_floor_mk),
                "t21_median_fe_percent": float(np.nanpercentile(t21_eval.fractional_error, 50.0)),
                "t21_p95_fe_percent": float(np.nanpercentile(t21_eval.fractional_error, 95.0)),
            }
        )

    delta21_eval = None
    if args.delta21_package is not None:
        delta21_eval = evaluate_delta21_test_set(
            dataset_root=args.dataset_root,
            package_path=args.delta21_package,
            random_state=args.random_state,
            log_fe_floor=args.delta21_log_fe_floor,
            log_target_floor=args.delta21_log_target_floor,
            prediction_batch_size=args.prediction_batch_size,
        )
        plot_delta21_fractional_error_maps(
            delta21_eval,
            output_path=output_dir / "delta21_plot1_2d_fractional_error_percentiles.png",
        )
        plot_delta21_ranked_2d_examples(
            delta21_eval,
            output_path=output_dir / "delta21_plot2_ranked_2d_truth_emulator_error.png",
        )
        plot_delta21_ranked_k_slices(
            delta21_eval,
            output_path=output_dir / "delta21_plot3_ranked_1d_k_slices.png",
        )
        plot_delta21_random_k_slices(
            delta21_eval,
            rng=rng,
            n_examples=args.delta21_random_examples,
            output_path=output_dir / "delta21_plot4_random10_1d_k_slices_fractional_error.png",
        )
        plot_fractional_error_histogram(
            delta21_eval.fractional_error,
            floor_label=rf"$\log_{{10}}\Delta^2_{{21}}$ FE floor = {args.delta21_log_fe_floor:g}",
            title="Power Spectrum Fractional Error",
            output_path=output_dir / "delta21_plot5_fractional_error_histogram.png",
        )
        summary.update(
            {
                "delta21_log_fe_floor": float(args.delta21_log_fe_floor),
                "delta21_log_target_floor": float(args.delta21_log_target_floor),
                "delta21_median_fe_percent": float(
                    np.nanpercentile(delta21_eval.fractional_error, 50.0)
                ),
                "delta21_p95_fe_percent": float(
                    np.nanpercentile(delta21_eval.fractional_error, 95.0)
                ),
            }
        )

    np.savez(output_dir / "publication_plot_summary.npz", **summary)
    print_summary(summary, output_dir)


# Evaluation
# ----------
# Build the same test split as training, then evaluate saved packages on it.

def evaluate_t21_test_set(
    *,
    dataset_root: str | Path,
    package_path: str | Path,
    random_state: int,
    fe_floor_mk: float,
    prediction_batch_size: int,
) -> T21Evaluation:
    """
    Evaluate a T21 package on the held-out test simulations.
    """
    z_grid, test_parameters, truth = prepare_t21_test_grid(
        dataset_root,
        random_state=random_state,
    )
    package = load_t21_package(package_path)
    emulator = build_t21_fixed_grid_emulator(
        package,
        jnp.asarray(z_grid, dtype=jnp.float32),
        compile_parameters=jnp.asarray(test_parameters[:1], dtype=jnp.float32),
    )
    prediction = predict_in_batches(emulator, test_parameters, prediction_batch_size)
    fractional_error = clipped_fractional_error_percent(
        truth,
        prediction,
        floor=fe_floor_mk,
    )
    return T21Evaluation(
        z=z_grid,
        parameters=test_parameters,
        truth=truth,
        prediction=prediction,
        fractional_error=fractional_error,
        fe_floor_mk=fe_floor_mk,
    )


def evaluate_delta21_test_set(
    *,
    dataset_root: str | Path,
    package_path: str | Path,
    random_state: int,
    log_fe_floor: float,
    log_target_floor: float,
    prediction_batch_size: int,
) -> Delta21Evaluation:
    """
    Evaluate a Delta21 package on the held-out test simulations.
    """
    z_grid, log_k_grid, test_parameters, truth_log = prepare_delta21_test_grid(
        dataset_root,
        random_state=random_state,
        log_target_floor=log_target_floor,
    )
    k_grid = np.power(10.0, log_k_grid)

    package = load_delta21_package(package_path)
    emulator = build_delta21_fixed_grid_emulator(
        package,
        jnp.asarray(z_grid, dtype=jnp.float32),
        jnp.asarray(k_grid, dtype=jnp.float32),
        compile_parameters=jnp.asarray(test_parameters[:1], dtype=jnp.float32),
    )
    prediction_physical = predict_in_batches(emulator, test_parameters, prediction_batch_size)
    prediction_log = physical_delta21_to_log_target(
        prediction_physical,
        offset=DELTA21_TARGET_OFFSET,
        target_floor=log_target_floor,
    )
    fractional_error = clipped_fractional_error_percent(
        truth_log,
        prediction_log,
        floor=log_fe_floor,
    )
    return Delta21Evaluation(
        z=z_grid,
        k=k_grid,
        parameters=test_parameters,
        truth_log=truth_log,
        prediction_log=prediction_log,
        fractional_error=fractional_error,
        log_fe_floor=log_fe_floor,
        log_target_floor=log_target_floor,
    )


def prepare_t21_test_grid(
    dataset_root: str | Path,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return T21 test parameters and physical test targets on the emulator grid.
    """
    product = load_twentyonecmspace_t21(dataset_root)
    parameters = prepare_twentyonecmspace_t21_parameters(product.parameters)
    spec = t21_spec()
    target = transform_target(product.target, data_log=False, offset=None)

    _, _, test_parameters, _, _, test_target = split_simulations(
        parameters.values,
        target,
        train_size=0.6,
        validation_size=0.2,
        test_size=0.2,
        random_state=random_state,
    )
    transformed_axes, transformed_limits = transformed_axis_configuration(
        (product.axes.z,),
        spec.axes,
    )
    sampled_axes = build_fixed_axis_grid(transformed_axes, transformed_limits, spec.axes)
    test_grid = resample_targets_to_grid(
        test_target,
        transformed_axes=transformed_axes,
        sampled_axes=sampled_axes,
        interpolation_method="cubic",
    )
    return sampled_axes[0], test_parameters.astype(np.float32), test_grid.astype(np.float32)


def prepare_delta21_test_grid(
    dataset_root: str | Path,
    *,
    random_state: int,
    log_target_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return Delta21 test parameters and log-space test targets on the emulator grid.
    """
    product = load_twentyonecmspace_delta21(dataset_root)
    parameters = prepare_twentyonecmspace_delta21_parameters(product.parameters)
    spec = delta21_spec()
    transformed_target = transform_target(
        product.target,
        data_log=True,
        offset=DELTA21_TARGET_OFFSET,
        target_min=log_target_floor,
    )

    _, _, test_parameters, _, _, test_target = split_simulations(
        parameters.values,
        transformed_target,
        train_size=0.6,
        validation_size=0.2,
        test_size=0.2,
        random_state=random_state,
    )
    transformed_axes, _ = transformed_axis_configuration(
        (product.axes.z, product.axes.k),
        spec.axes,
    )
    sampled_axes = delta21_low_z_sampled_axes(spec.axes)
    test_grid = resample_targets_to_grid(
        test_target,
        transformed_axes=transformed_axes,
        sampled_axes=sampled_axes,
        interpolation_method="cubic",
    )
    test_grid = apply_target_floor(test_grid, log_target_floor)
    z_grid, log_k_grid = sampled_axes
    return (
        z_grid,
        log_k_grid,
        test_parameters.astype(np.float32),
        test_grid.astype(np.float32),
    )


def predict_in_batches(
    emulator: Any,
    parameters: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """
    Evaluate a fixed-grid emulator over simulations in batches.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    predictions = []
    for start in range(0, len(parameters), batch_size):
        batch = jnp.asarray(parameters[start : start + batch_size], dtype=jnp.float32)
        prediction = emulator.emulate(batch)
        prediction.block_until_ready()
        predictions.append(np.asarray(jax.device_get(prediction), dtype=np.float32))
    return np.concatenate(predictions, axis=0)


def clipped_fractional_error_percent(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    floor: float,
) -> np.ndarray:
    """
    Return clipped fractional error as a percentage.
    """
    denominator = np.maximum(np.abs(truth), floor)
    return 100.0 * np.abs(truth - prediction) / denominator


def physical_delta21_to_log_target(
    values: np.ndarray,
    *,
    offset: float,
    target_floor: float,
) -> np.ndarray:
    """
    Convert physical Delta21 values into the clipped log target space.
    """
    positive = np.maximum(np.asarray(values, dtype=np.float64) + offset, 10.0**target_floor)
    return np.maximum(np.log10(positive), target_floor).astype(np.float32)


# T21 Plots
# ---------
# Global-signal performance figures.

def plot_t21_random_examples(
    evaluation: T21Evaluation,
    *,
    rng: np.random.Generator,
    n_examples: int,
    output_path: str | Path,
) -> None:
    """
    Plot random held-out T21 examples with FE curves and test-set bands.
    """
    indices = random_indices(len(evaluation.truth), n_examples, rng)
    colours = plt.cm.tab20(np.linspace(0.0, 1.0, len(indices)))

    fig, (ax_signal, ax_error) = plt.subplots(
        2,
        1,
        figsize=(10.5, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    for colour, idx in zip(colours, indices, strict=True):
        ax_signal.plot(evaluation.z, evaluation.truth[idx], color=colour, linewidth=1.4, alpha=0.9)
        ax_signal.plot(
            evaluation.z,
            evaluation.prediction[idx],
            color=colour,
            linestyle="--",
            linewidth=1.4,
            alpha=0.9,
        )
        ax_error.plot(
            evaluation.z,
            positive_for_log_axis(evaluation.fractional_error[idx]),
            color=colour,
            linewidth=1.0,
            alpha=0.65,
        )

    p50 = np.nanpercentile(evaluation.fractional_error, 50.0, axis=0)
    p95 = np.nanpercentile(evaluation.fractional_error, 95.0, axis=0)
    ax_error.fill_between(
        evaluation.z,
        positive_for_log_axis(p50),
        positive_for_log_axis(p95),
        color="0.72",
        alpha=0.35,
        label="50-95% test range",
    )
    ax_error.plot(evaluation.z, positive_for_log_axis(p50), color="black", linewidth=1.8, label="50%")
    ax_error.plot(evaluation.z, positive_for_log_axis(p95), color="black", linestyle=":", linewidth=1.6, label="95%")

    ax_signal.plot([], [], color="black", linewidth=1.6, label="Simulation")
    ax_signal.plot([], [], color="black", linestyle="--", linewidth=1.6, label="Emulator")
    ax_signal.set_ylabel(r"$T_{21}$ [mK]")
    ax_signal.set_title("Random Held-Out Global Signal Examples")
    ax_signal.legend(loc="best", frameon=False, ncol=2)
    ax_signal.grid(alpha=0.18)

    ax_error.set_yscale("log")
    ax_error.set_xlabel("Redshift")
    ax_error.set_ylabel("FE [%]")
    ax_error.legend(loc="best", frameon=False, ncol=3)
    ax_error.grid(alpha=0.18, which="both")
    fig.tight_layout()
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def plot_t21_ranked_examples(
    evaluation: T21Evaluation,
    *,
    output_path: str | Path,
    percentiles: tuple[float, ...] = (25.0, 50.0, 75.0, 95.0),
) -> None:
    """
    Plot held-out T21 examples ranked by mean fractional error.
    """
    mean_error = np.nanmean(evaluation.fractional_error, axis=1)
    indices = ranked_indices(mean_error, percentiles)

    fig, axes = plt.subplots(1, len(indices), figsize=(4.4 * len(indices), 4.6), sharey=False)
    axes = np.asarray(axes).ravel()
    colours = ["#0072B2", "#009E73", "#E69F00", "#D55E00"]

    for ax, percentile, idx, colour in zip(axes, percentiles, indices, colours, strict=True):
        ax.plot(evaluation.z, evaluation.truth[idx], color=colour, linewidth=2.0, label="Simulation")
        ax.plot(
            evaluation.z,
            evaluation.prediction[idx],
            color=colour,
            linestyle="--",
            linewidth=2.0,
            label="Emulator",
        )
        ax.set_title(f"{percentile:.0f}th percentile\nmean FE = {mean_error[idx]:.2f}%")
        ax.set_xlabel("Redshift")
        ax.set_ylabel(r"$T_{21}$ [mK]")
        ax.grid(alpha=0.18)
    axes[0].legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


# Delta21 Plots
# -------------
# Power-spectrum performance figures.

def plot_delta21_fractional_error_maps(
    evaluation: Delta21Evaluation,
    *,
    output_path: str | Path,
) -> None:
    """
    Plot 2D maps of 50th and 95th percentile Delta21 FE.
    """
    p50 = np.nanpercentile(evaluation.fractional_error, 50.0, axis=0)
    p95 = np.nanpercentile(evaluation.fractional_error, 95.0, axis=0)
    global_p50 = float(np.nanpercentile(evaluation.fractional_error, 50.0))
    global_p95 = float(np.nanpercentile(evaluation.fractional_error, 95.0))
    vmax = float(np.nanpercentile(p95, 99.0))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, values, title in zip(
        axes,
        (p50, p95),
        ("50th Percentile FE", "95th Percentile FE"),
        strict=True,
    ):
        mesh = ax.pcolormesh(
            evaluation.k,
            evaluation.z,
            values,
            shading="auto",
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_xscale("log")
        ax.set_xlabel(r"$k$ [$h\,\mathrm{cMpc}^{-1}$]")
        ax.set_title(title)
        ax.text(
            0.03,
            0.95,
            f"global 50% = {global_p50:.2f}%\nglobal 95% = {global_p95:.2f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.88},
        )
        colourbar = fig.colorbar(mesh, ax=ax)
        colourbar.set_label("FE [%]")
    axes[0].set_ylabel("Redshift")
    fig.tight_layout()
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def plot_delta21_ranked_2d_examples(
    evaluation: Delta21Evaluation,
    *,
    output_path: str | Path,
    percentiles: tuple[float, ...] = (0.0, 50.0, 95.0),
) -> None:
    """
    Plot ranked Delta21 truth, emulator, and FE maps.
    """
    mean_error = np.nanmean(evaluation.fractional_error, axis=(1, 2))
    indices = ranked_indices(mean_error, percentiles)
    labels = ["Best", "Median", "95th percentile"]
    target_vmin = evaluation.log_target_floor
    target_vmax = float(
        np.nanpercentile(
            np.concatenate(
                [
                    evaluation.truth_log[indices].ravel(),
                    evaluation.prediction_log[indices].ravel(),
                ]
            ),
            99.0,
        )
    )
    error_vmax = float(np.nanpercentile(evaluation.fractional_error[indices], 99.0))

    fig, axes = plt.subplots(len(indices), 3, figsize=(13.5, 11.0), sharex=True, sharey=True)
    for row, (idx, label) in enumerate(zip(indices, labels, strict=True)):
        panels = (
            (evaluation.truth_log[idx], "Simulation", "viridis", target_vmin, target_vmax),
            (evaluation.prediction_log[idx], "Emulator", "viridis", target_vmin, target_vmax),
            (evaluation.fractional_error[idx], "FE [%]", "magma", 0.0, error_vmax),
        )
        for col, (values, title, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row, col]
            mesh = ax.pcolormesh(
                evaluation.k,
                evaluation.z,
                values,
                shading="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xscale("log")
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"{label}\nRedshift")
            if row == len(indices) - 1:
                ax.set_xlabel(r"$k$ [$h\,\mathrm{cMpc}^{-1}$]")
            colourbar = fig.colorbar(mesh, ax=ax)
            colourbar.ax.tick_params(labelsize=9)
        axes[row, 1].text(
            0.03,
            0.95,
            f"mean FE = {mean_error[idx]:.2f}%",
            transform=axes[row, 1].transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.88},
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def plot_delta21_ranked_k_slices(
    evaluation: Delta21Evaluation,
    *,
    output_path: str | Path,
    percentiles: tuple[float, ...] = (0.0, 50.0, 95.0),
) -> None:
    """
    Plot ranked Delta21 examples across redshift for three k slices.
    """
    mean_error = np.nanmean(evaluation.fractional_error, axis=(1, 2))
    indices = ranked_indices(mean_error, percentiles)
    labels = ["Best", "Median", "95th percentile"]
    k_indices = representative_k_indices(evaluation.k)

    fig, axes = plt.subplots(len(indices), len(k_indices), figsize=(13.2, 9.6), sharex=True)
    for row, (idx, label) in enumerate(zip(indices, labels, strict=True)):
        for col, k_idx in enumerate(k_indices):
            ax = axes[row, col]
            ax.plot(
                evaluation.z,
                evaluation.truth_log[idx, :, k_idx],
                color="#0072B2",
                linewidth=2.0,
                label="Simulation" if row == 0 and col == 0 else None,
            )
            ax.plot(
                evaluation.z,
                evaluation.prediction_log[idx, :, k_idx],
                color="#D55E00",
                linestyle="--",
                linewidth=2.0,
                label="Emulator" if row == 0 and col == 0 else None,
            )
            if row == 0:
                ax.set_title(rf"$k={evaluation.k[k_idx]:.3f}$")
            if col == 0:
                ax.set_ylabel(f"{label}\n" + r"$\log_{10}\Delta^2_{21}$")
            if row == len(indices) - 1:
                ax.set_xlabel("Redshift")
            ax.grid(alpha=0.18)
    axes[0, 0].legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


def plot_delta21_random_k_slices(
    evaluation: Delta21Evaluation,
    *,
    rng: np.random.Generator,
    n_examples: int,
    output_path: str | Path,
) -> None:
    """
    Plot random held-out Delta21 k slices with FE bands.
    """
    indices = random_indices(len(evaluation.truth_log), n_examples, rng)
    k_indices = representative_k_indices(evaluation.k)
    colours = plt.cm.tab10(np.linspace(0.0, 1.0, len(indices)))

    fig, axes = plt.subplots(
        2,
        len(k_indices),
        figsize=(13.5, 7.8),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    for col, k_idx in enumerate(k_indices):
        ax_signal = axes[0, col]
        ax_error = axes[1, col]
        for colour, idx in zip(colours, indices, strict=True):
            ax_signal.plot(
                evaluation.z,
                evaluation.truth_log[idx, :, k_idx],
                color=colour,
                linewidth=1.2,
                alpha=0.85,
            )
            ax_signal.plot(
                evaluation.z,
                evaluation.prediction_log[idx, :, k_idx],
                color=colour,
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
            )
            ax_error.plot(
                evaluation.z,
                positive_for_log_axis(evaluation.fractional_error[idx, :, k_idx]),
                color=colour,
                linewidth=0.9,
                alpha=0.6,
            )

        p50 = np.nanpercentile(evaluation.fractional_error[:, :, k_idx], 50.0, axis=0)
        p95 = np.nanpercentile(evaluation.fractional_error[:, :, k_idx], 95.0, axis=0)
        ax_error.fill_between(
            evaluation.z,
            positive_for_log_axis(p50),
            positive_for_log_axis(p95),
            color="0.72",
            alpha=0.35,
            label="50-95% test range" if col == 0 else None,
        )
        ax_error.plot(evaluation.z, positive_for_log_axis(p50), color="black", linewidth=1.5)
        ax_error.plot(evaluation.z, positive_for_log_axis(p95), color="black", linestyle=":", linewidth=1.3)

        ax_signal.set_title(rf"$k={evaluation.k[k_idx]:.3f}$")
        ax_signal.grid(alpha=0.18)
        ax_error.set_yscale("log")
        ax_error.set_xlabel("Redshift")
        ax_error.set_ylabel("FE [%]")
        ax_error.grid(alpha=0.18, which="both")
    axes[0, 0].plot([], [], color="black", label="Simulation")
    axes[0, 0].plot([], [], color="black", linestyle="--", label="Emulator")
    axes[0, 0].set_ylabel(r"$\log_{10}\Delta^2_{21}$")
    axes[0, 0].legend(loc="best", frameon=False)
    axes[1, 0].legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


# Shared Plots
# ------------
# Plot helpers used by both emulator families.

def plot_fractional_error_histogram(
    fractional_error: np.ndarray,
    *,
    floor_label: str,
    title: str,
    output_path: str | Path,
) -> None:
    """
    Plot a log-x histogram of clipped fractional errors.
    """
    values = positive_for_log_axis(np.asarray(fractional_error, dtype=float).ravel())
    median = float(np.nanpercentile(values, 50.0))
    p95 = float(np.nanpercentile(values, 95.0))
    bins = np.logspace(
        np.log10(max(np.nanmin(values), 1e-6)),
        np.log10(max(np.nanmax(values), 1e-5)),
        80,
    )

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    ax.hist(values, bins=bins, color="#4C78A8", alpha=0.78, edgecolor="white", linewidth=0.4)
    ax.axvline(median, color="black", linewidth=1.8, label="50%")
    ax.axvline(p95, color="black", linestyle="--", linewidth=1.8, label="95%")
    ax.set_xscale("log")
    ax.set_xlabel("Fractional error [%]")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.text(
        0.97,
        0.94,
        f"50% = {median:.2f}%\n95% = {p95:.2f}%\n{floor_label}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.92},
    )
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.18, which="both")
    fig.tight_layout()
    fig.savefig(output_path, dpi=260)
    plt.close(fig)


# Utilities
# ---------
# Small helpers for deterministic selection, styling, and summaries.

def random_indices(n_items: int, n_examples: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw deterministic random indices without replacement.
    """
    n_draw = min(n_examples, n_items)
    return np.sort(rng.choice(n_items, size=n_draw, replace=False))


def ranked_indices(values: np.ndarray, percentiles: tuple[float, ...]) -> list[int]:
    """
    Select item indices at requested percentiles of a ranking metric.
    """
    order = np.argsort(values)
    indices = []
    for percentile in percentiles:
        position = int(round((percentile / 100.0) * (len(order) - 1)))
        indices.append(int(order[np.clip(position, 0, len(order) - 1)]))
    return indices


def representative_k_indices(k_values: np.ndarray) -> list[int]:
    """
    Pick low, middle, and high k bins for one-dimensional power-spectrum slices.
    """
    n_k = len(k_values)
    return sorted({int(round(frac * (n_k - 1))) for frac in (0.15, 0.50, 0.85)})


def positive_for_log_axis(values: np.ndarray) -> np.ndarray:
    """
    Clip non-positive plotting values so log axes remain finite.
    """
    return np.clip(np.asarray(values, dtype=float), 1e-6, None)


def configure_matplotlib() -> None:
    """
    Apply publication-style Matplotlib defaults.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 15,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.titlesize": 17,
            "savefig.facecolor": "white",
            "axes.linewidth": 1.1,
        }
    )


def print_summary(summary: dict[str, float | int], output_dir: Path) -> None:
    """
    Print the metrics saved beside the figures.
    """
    print(f"Wrote publication plots to: {output_dir}")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
