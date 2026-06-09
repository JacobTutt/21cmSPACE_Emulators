"""
Plot posterior diagnostics for power-spectrum inference examples.

The script reads anesthetic-compatible nested-sampling outputs, compares the
posterior to the emulator prior, and makes two diagnostic plot families:

1. Delta21 prior/posterior predictive spectra against the observed upper limits.
2. T21 prior/posterior predictive global signals implied by the same posterior.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from examples_21cmspace.delta21.emulator import (
    build_delta21_fixed_point_emulator,
    default_delta21_inference_prior,
    load_delta21_package,
)
from examples_21cmspace.delta21.hera_data import load_hera_power_spectrum_npz
from examples_21cmspace.delta21.hera_diagnostics import (
    CORNER_PARAMETER_NAMES,
    configure_matplotlib,
    draw_prior_samples,
    plot_prior_posterior_corner,
    positive_for_log,
    read_anesthetic_csv,
    select_corner_parameters,
    weighted_resample,
)
from examples_21cmspace.delta21.nenufar_data import load_nenufar_table4_dataset
from examples_21cmspace.t21.emulator import build_t21_fixed_grid_emulator, load_t21_package


@dataclass(frozen=True)
class FitConfig:
    """
    Storage utility for one saved nested-sampling fit.
    """

    name: str
    nested_results: Path
    data_kind: str
    data_path: Path


@dataclass(frozen=True)
class PowerSpectrumPlotData:
    """
    Plot-ready power-spectrum data for one experiment.
    """

    coordinates: np.ndarray
    limit_2sigma: np.ndarray
    window_matrix: np.ndarray | None
    label: str


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface.
    """
    parser = argparse.ArgumentParser(description="Plot power-spectrum posterior diagnostics.")
    parser.add_argument("--delta21-package", required=True, help="Path to the Delta21 .nenemu package.")
    parser.add_argument("--t21-package", required=True, help="Path to the T21 .nenemu package.")
    parser.add_argument(
        "--fit",
        action="append",
        required=True,
        help=(
            "Fit description formatted as name:data_kind:nested_results:data_path. "
            "data_kind must be hera_npz or nenufar_csv."
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Directory for diagnostic PNG files.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-prior", type=int, default=10_000)
    parser.add_argument("--n-corner-points", type=int, default=4_000)
    parser.add_argument("--n-pencil", type=int, default=1_000)
    parser.add_argument("--t21-z-min", type=float, default=6.0)
    parser.add_argument("--t21-z-max", type=float, default=27.0)
    parser.add_argument("--t21-z-points", type=int, default=240)
    return parser


def main() -> None:
    """
    Create posterior diagnostic plots for the requested fits.
    """
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    prior = default_delta21_inference_prior()
    prior_samples = draw_prior_samples(prior, args.n_prior, rng)
    fit_configs = tuple(parse_fit_config(value) for value in args.fit)

    t21_package = load_t21_package(args.t21_package)
    t21_z = np.linspace(args.t21_z_min, args.t21_z_max, args.t21_z_points, dtype=np.float32)
    t21_emulator = build_t21_fixed_grid_emulator(
        t21_package,
        jnp.asarray(t21_z, dtype=jnp.float32),
        compile_parameters=jnp.asarray(prior_samples[:1], dtype=jnp.float32),
    )

    t21_panels = []
    for fit in fit_configs:
        posterior_samples, posterior_weights = read_anesthetic_csv(fit.nested_results, prior.names)
        corner_plot = output_dir / f"{fit.name}_five_parameter_prior_posterior.png"
        plot_prior_posterior_corner(
            select_corner_parameters(prior_samples, prior.names),
            select_corner_parameters(posterior_samples, prior.names),
            posterior_weights,
            names=CORNER_PARAMETER_NAMES,
            output_path=corner_plot,
            rng=rng,
            n_corner_points=args.n_corner_points,
            title=f"{fit.name} Prior and Posterior Samples",
        )

        power_data = load_power_spectrum_plot_data(fit)
        delta_plot = output_dir / f"{fit.name}_delta21_prior_posterior_predictive.png"
        plot_delta21_prior_posterior_predictive(
            package_path=args.delta21_package,
            power_data=power_data,
            prior_samples=prior_samples,
            posterior_samples=posterior_samples,
            posterior_weights=posterior_weights,
            output_path=delta_plot,
            rng=rng,
            n_pencil=args.n_pencil,
        )

        t21_plot = output_dir / f"{fit.name}_t21_prior_posterior_predictive.png"
        t21_panel = plot_t21_prior_posterior_predictive(
            emulator=t21_emulator,
            z_values=t21_z,
            prior_samples=prior_samples,
            posterior_samples=posterior_samples,
            posterior_weights=posterior_weights,
            output_path=t21_plot,
            rng=rng,
            n_pencil=args.n_pencil,
            title=f"{fit.name} T21 Prior and Posterior Predictive",
        )
        t21_panels.append((fit.name, t21_panel))

        print(
            {
                "fit": fit.name,
                "corner_plot": str(corner_plot),
                "delta21_predictive_plot": str(delta_plot),
                "t21_predictive_plot": str(t21_plot),
                "n_posterior_samples": int(posterior_samples.shape[0]),
            }
        )

    combined_t21_plot = output_dir / "all_fits_t21_prior_posterior_predictive.png"
    plot_combined_t21_panels(t21_panels, combined_t21_plot)
    print({"combined_t21_plot": str(combined_t21_plot)})


def parse_fit_config(value: str) -> FitConfig:
    """
    Parse a command-line fit description.
    """
    parts = value.split(":", 3)
    if len(parts) != 4:
        raise ValueError("Each --fit must be formatted as name:data_kind:nested_results:data_path.")
    name, data_kind, nested_results, data_path = parts
    if data_kind not in {"hera_npz", "nenufar_csv"}:
        raise ValueError("data_kind must be either hera_npz or nenufar_csv.")
    return FitConfig(
        name=name,
        nested_results=Path(nested_results),
        data_kind=data_kind,
        data_path=Path(data_path),
    )


def load_power_spectrum_plot_data(fit: FitConfig) -> PowerSpectrumPlotData:
    """
    Load power-spectrum coordinates and plotted 2-sigma limits.
    """
    if fit.data_kind == "hera_npz":
        dataset = load_hera_power_spectrum_npz(fit.data_path)
        power_data = dataset.power_data
        return PowerSpectrumPlotData(
            coordinates=np.asarray(power_data.coordinates, dtype=np.float32),
            limit_2sigma=np.asarray(power_data.upper_limit + 2.0 * power_data.sigma, dtype=np.float64),
            window_matrix=np.asarray(power_data.window_matrix, dtype=np.float32),
            label="HERA",
        )

    dataset = load_nenufar_table4_dataset(fit.data_path)
    power_data = dataset.power_data
    return PowerSpectrumPlotData(
        coordinates=np.asarray(power_data.coordinates, dtype=np.float32),
        limit_2sigma=np.asarray(power_data.upper_limit + 2.0 * power_data.sigma, dtype=np.float64),
        window_matrix=None,
        label="NenuFAR",
    )


def plot_delta21_prior_posterior_predictive(
    *,
    package_path: str | Path,
    power_data: PowerSpectrumPlotData,
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
    posterior_weights: np.ndarray,
    output_path: str | Path,
    rng: np.random.Generator,
    n_pencil: int,
) -> None:
    """
    Plot prior and posterior Delta21 predictions against 2-sigma limits.
    """
    configure_matplotlib()
    coordinates = power_data.coordinates
    prior_draw = prior_samples[
        rng.choice(prior_samples.shape[0], size=min(n_pencil, prior_samples.shape[0]), replace=False)
    ]
    posterior_draw = weighted_resample(posterior_samples, posterior_weights, n_pencil, rng)

    package = load_delta21_package(package_path)
    emulator = build_delta21_fixed_point_emulator(
        package,
        jnp.asarray(coordinates, dtype=jnp.float32),
        compile_parameters=jnp.asarray(posterior_draw[:1], dtype=jnp.float32),
    )

    prior_prediction = evaluate_delta21_predictions(emulator, prior_draw, power_data.window_matrix)
    posterior_prediction = evaluate_delta21_predictions(emulator, posterior_draw, power_data.window_matrix)

    redshifts = np.unique(coordinates[:, 0])
    fig, axes = plt.subplots(1, len(redshifts), figsize=(5.8 * len(redshifts), 4.6), sharey=True)
    if len(redshifts) == 1:
        axes = np.asarray([axes])

    for ax, z_value in zip(axes, redshifts, strict=True):
        mask = np.isclose(coordinates[:, 0], z_value)
        order = np.argsort(coordinates[mask, 1])
        k_values = coordinates[mask, 1][order]
        prior_panel = prior_prediction[:, mask][:, order]
        posterior_panel = posterior_prediction[:, mask][:, order]
        limit_panel = power_data.limit_2sigma[mask][order]

        for values in prior_panel:
            ax.plot(k_values, positive_for_log(values), color="0.65", alpha=0.12, linewidth=0.7)
        for values in posterior_panel:
            ax.plot(k_values, positive_for_log(values), color="#2b6cb0", alpha=0.22, linewidth=0.8)

        ax.scatter(
            k_values,
            positive_for_log(limit_panel),
            marker="v",
            s=56,
            color="black",
            label=f"{power_data.label} 2$\\sigma$ upper limit",
            zorder=5,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$k$ [$h\,\mathrm{cMpc}^{-1}$]")
        ax.set_title(f"$z = {z_value:.2f}$")
        ax.grid(alpha=0.25, which="both")

    axes[0].set_ylabel(r"$\Delta^2_{21}$ [mK$^2$]")
    axes[0].plot([], [], color="0.65", label="Prior predictive")
    axes[0].plot([], [], color="#2b6cb0", label="Posterior predictive")
    axes[0].legend(loc="best", frameon=False)
    fig.suptitle(f"{power_data.label} Prior and Posterior Predictive Power Spectra", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def evaluate_delta21_predictions(
    emulator: object,
    samples: np.ndarray,
    window_matrix: np.ndarray | None,
) -> np.ndarray:
    """
    Evaluate Delta21 predictions and apply an optional window matrix.
    """
    prediction = np.asarray(
        jax.device_get(emulator.emulate(jnp.asarray(samples, dtype=jnp.float32))),
        dtype=np.float64,
    )
    if window_matrix is None:
        return prediction
    return prediction @ window_matrix.T


def plot_t21_prior_posterior_predictive(
    *,
    emulator: object,
    z_values: np.ndarray,
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
    posterior_weights: np.ndarray,
    output_path: str | Path,
    rng: np.random.Generator,
    n_pencil: int,
    title: str,
) -> dict[str, np.ndarray]:
    """
    Plot prior and posterior T21 predictions over redshift.
    """
    configure_matplotlib()
    prior_draw = prior_samples[
        rng.choice(prior_samples.shape[0], size=min(n_pencil, prior_samples.shape[0]), replace=False)
    ]
    posterior_draw = weighted_resample(posterior_samples, posterior_weights, n_pencil, rng)
    prior_prediction = evaluate_t21_predictions(emulator, prior_draw)
    posterior_prediction = evaluate_t21_predictions(emulator, posterior_draw)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    draw_t21_panel(ax, z_values, prior_prediction, posterior_prediction, title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return {
        "z": z_values,
        "prior_prediction": prior_prediction,
        "posterior_prediction": posterior_prediction,
    }


def plot_combined_t21_panels(
    panels: list[tuple[str, dict[str, np.ndarray]]],
    output_path: str | Path,
) -> None:
    """
    Plot T21 prior/posterior predictive panels for all fits.
    """
    configure_matplotlib()
    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 4.6), sharey=True)
    if len(panels) == 1:
        axes = np.asarray([axes])

    for ax, (name, panel) in zip(axes, panels, strict=True):
        draw_t21_panel(
            ax,
            panel["z"],
            panel["prior_prediction"],
            panel["posterior_prediction"],
            name,
            legend=False,
        )

    axes[0].legend(loc="best", frameon=False)
    fig.suptitle("T21 Prior and Posterior Predictive Signals", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def evaluate_t21_predictions(emulator: object, samples: np.ndarray) -> np.ndarray:
    """
    Evaluate T21 predictions for a table of parameter samples.
    """
    return np.asarray(
        jax.device_get(emulator.emulate(jnp.asarray(samples, dtype=jnp.float32))),
        dtype=np.float64,
    )


def draw_t21_panel(
    ax: plt.Axes,
    z_values: np.ndarray,
    prior_prediction: np.ndarray,
    posterior_prediction: np.ndarray,
    title: str,
    *,
    legend: bool = True,
) -> None:
    """
    Draw one T21 prior/posterior predictive panel.
    """
    for values in prior_prediction:
        ax.plot(z_values, values, color="0.65", alpha=0.10, linewidth=0.7)
    for values in posterior_prediction:
        ax.plot(z_values, values, color="#2b6cb0", alpha=0.18, linewidth=0.8)

    prior_median = np.nanmedian(prior_prediction, axis=0)
    posterior_median = np.nanmedian(posterior_prediction, axis=0)
    ax.plot(z_values, prior_median, color="0.25", linewidth=1.8, label="Prior median")
    ax.plot(z_values, posterior_median, color="#1a4f8b", linewidth=2.0, label="Posterior median")
    ax.set_title(title)
    ax.set_xlabel("Redshift")
    ax.set_ylabel(r"$T_{21}$ [mK]")
    ax.grid(alpha=0.22)
    if legend:
        ax.legend(loc="best", frameon=False)


if __name__ == "__main__":
    main()
