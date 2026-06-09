"""
Plot HERA nested-sampling posterior diagnostics.

This module reads an anesthetic-compatible nested-sampling CSV, compares the
posterior to samples drawn from the emulator prior, and plots prior/posterior
predictive power spectra against the HERA 2-sigma upper limits.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from examples_21cmspace.delta21.emulator import (
    build_delta21_fixed_point_emulator,
    default_delta21_hera_prior,
    load_delta21_package,
)
from examples_21cmspace.delta21.hera_data import load_hera_power_spectrum_npz


DEFAULT_PARAMETER_LABELS = {
    "log10fstarII": r"$\log_{10} f_{\star,\mathrm{II}}$",
    "log10fstarIII": r"$\log_{10} f_{\star,\mathrm{III}}$",
    "log10Vc": r"$\log_{10} V_c$",
    "log10fX": r"$\log_{10} f_X$",
    "log10LX_per_SFR": r"$\log_{10}(L_X/\mathrm{SFR})$",
    "alpha": r"$\alpha$",
    "nu_0": r"$\nu_0$",
    "tau": r"$\tau$",
    "log10fradio": r"$\log_{10} f_\mathrm{radio}$",
    "log10Lr_per_SFR": r"$\log_{10}(L_r/\mathrm{SFR})$",
    "pop": "Pop.",
}


CORNER_PARAMETER_NAMES = (
    "log10fstarIII",
    "log10fstarII",
    "log10Vc",
    "log10LX_per_SFR",
    "log10Lr_per_SFR",
)

LOG10_LX_PER_SFR_OFFSET = 40.47712125471966
LOG10_LR_PER_SFR_OFFSET = 22.0


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for HERA diagnostic plots.
    """
    parser = argparse.ArgumentParser(description="Plot HERA nested-sampling diagnostics.")
    parser.add_argument(
        "--nested-results",
        required=True,
        help="Path to nested_sampling_results.csv.",
    )
    parser.add_argument(
        "--package",
        required=True,
        help="Path to the trained Delta21 .nenemu package.",
    )
    parser.add_argument(
        "--hera-npz",
        required=True,
        help="Path to the extracted HERA likelihood NPZ cache.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for plots. Defaults to a diagnostics folder beside the CSV.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-prior", type=int, default=10_000)
    parser.add_argument("--n-corner-points", type=int, default=4_000)
    parser.add_argument("--n-pencil", type=int, default=1_000)
    return parser


def main() -> None:
    """
    Create HERA posterior corner and predictive diagnostic plots.
    """
    args = build_parser().parse_args()
    nested_results = Path(args.nested_results)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else nested_results.parent / "diagnostics"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    prior = default_delta21_hera_prior()
    posterior_samples, posterior_weights = read_anesthetic_csv(nested_results, prior.names)
    prior_samples = draw_prior_samples(prior, args.n_prior, rng)
    corner_prior_samples = select_corner_parameters(prior_samples, prior.names)
    corner_posterior_samples = select_corner_parameters(posterior_samples, prior.names)

    corner_path = output_dir / "hera_prior_posterior_corner.png"
    plot_prior_posterior_corner(
        corner_prior_samples,
        corner_posterior_samples,
        posterior_weights,
        names=CORNER_PARAMETER_NAMES,
        output_path=corner_path,
        rng=rng,
        n_corner_points=args.n_corner_points,
    )

    predictive_path = output_dir / "hera_prior_posterior_predictive.png"
    plot_prior_posterior_predictive(
        package_path=args.package,
        hera_npz=args.hera_npz,
        prior_samples=prior_samples,
        posterior_samples=posterior_samples,
        posterior_weights=posterior_weights,
        output_path=predictive_path,
        rng=rng,
        n_pencil=args.n_pencil,
    )

    print(
        {
            "corner_plot": str(corner_path),
            "predictive_plot": str(predictive_path),
            "n_posterior_samples": int(posterior_samples.shape[0]),
            "n_prior_samples": int(prior_samples.shape[0]),
        }
    )


def read_anesthetic_csv(
    path: str | Path,
    parameter_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read parameter samples and weights from an anesthetic CSV export.
    """
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        next(reader)  # labels
        next(reader)  # weights marker

        index_by_name = {name: header.index(name) for name in parameter_names}
        weights_index = 1
        samples: list[list[float]] = []
        weights: list[float] = []
        for row in reader:
            if not row:
                continue
            samples.append([float(row[index_by_name[name]]) for name in parameter_names])
            weights.append(float(row[weights_index]))

    sample_array = np.asarray(samples, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(sample_array).all(axis=1) & np.isfinite(weight_array) & (weight_array >= 0)
    sample_array = sample_array[valid]
    weight_array = weight_array[valid]
    if sample_array.size == 0:
        raise ValueError(f"No valid posterior samples found in {path}.")
    if np.sum(weight_array) <= 0.0:
        weight_array = np.full(sample_array.shape[0], 1.0 / sample_array.shape[0])
    else:
        weight_array = weight_array / np.sum(weight_array)
    return sample_array, weight_array


def draw_prior_samples(prior: Any, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw samples from the same prior used by the nested sampler.
    """
    unit = rng.uniform(0.0, 1.0, size=(n_samples, prior.ndim)).astype(np.float32)
    return np.asarray(jax.device_get(prior.transform(jnp.asarray(unit))), dtype=np.float64)


def select_corner_parameters(samples: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    """
    Select the paper-style parameters used in the HERA corner plot.
    """
    index = {name: column for column, name in enumerate(names)}
    return np.column_stack(
        [
            samples[:, index["log10fstarIII"]],
            samples[:, index["log10fstarII"]],
            samples[:, index["log10Vc"]],
            samples[:, index["log10fX"]] + LOG10_LX_PER_SFR_OFFSET,
            samples[:, index["log10fradio"]] + LOG10_LR_PER_SFR_OFFSET,
        ]
    )


def plot_prior_posterior_corner(
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
    posterior_weights: np.ndarray,
    *,
    names: tuple[str, ...],
    output_path: str | Path,
    rng: np.random.Generator,
    n_corner_points: int,
) -> None:
    """
    Plot one-dimensional and two-dimensional prior/posterior marginals.
    """
    configure_matplotlib()
    ndim = len(names)
    prior_grid_samples = prior_samples[
        rng.choice(prior_samples.shape[0], size=min(n_corner_points, prior_samples.shape[0]), replace=False)
    ]

    fig, axes = plt.subplots(ndim, ndim, figsize=(2.2 * ndim, 2.2 * ndim))
    for row in range(ndim):
        for col in range(ndim):
            ax = axes[row, col]
            if row < col:
                ax.axis("off")
                continue

            if row == col:
                bins = _histogram_bins(prior_samples[:, col], posterior_samples[:, col])
                ax.hist(
                    prior_samples[:, col],
                    bins=bins,
                    density=True,
                    histtype="step",
                    color="0.45",
                    linewidth=1.2,
                    label="Prior" if row == 0 else None,
                )
                ax.hist(
                    posterior_samples[:, col],
                    bins=bins,
                    weights=posterior_weights,
                    density=True,
                    histtype="stepfilled",
                    color="#2b6cb0",
                    alpha=0.35,
                    label="Posterior" if row == 0 else None,
                )
            else:
                plot_2d_pdf_grid(
                    ax,
                    prior_grid_samples[:, col],
                    prior_grid_samples[:, row],
                    posterior_samples[:, col],
                    posterior_samples[:, row],
                    posterior_weights,
                )

            if row == ndim - 1:
                ax.set_xlabel(DEFAULT_PARAMETER_LABELS.get(names[col], names[col]))
            else:
                ax.set_xticklabels([])
            if col == 0 and row != 0:
                ax.set_ylabel(DEFAULT_PARAMETER_LABELS.get(names[row], names[row]))
            elif col != 0:
                ax.set_yticklabels([])

            ax.tick_params(axis="both", labelsize=8)
            ax.grid(alpha=0.15, linewidth=0.5)

    axes[0, 0].legend(loc="best", frameon=False, fontsize=9)
    fig.suptitle("HERA Prior and Posterior Samples", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def plot_2d_pdf_grid(
    ax: plt.Axes,
    prior_x: np.ndarray,
    prior_y: np.ndarray,
    posterior_x: np.ndarray,
    posterior_y: np.ndarray,
    posterior_weights: np.ndarray,
) -> None:
    """
    Plot prior and posterior two-dimensional PDFs on a shared grid.
    """
    x_bins = _histogram_bins(prior_x, posterior_x)
    y_bins = _histogram_bins(prior_y, posterior_y)
    prior_pdf, _, _ = np.histogram2d(prior_x, prior_y, bins=(x_bins, y_bins), density=True)
    posterior_pdf, _, _ = np.histogram2d(
        posterior_x,
        posterior_y,
        bins=(x_bins, y_bins),
        weights=posterior_weights,
        density=True,
    )

    prior_mesh = np.ma.masked_where(prior_pdf.T <= 0.0, prior_pdf.T)
    posterior_mesh = np.ma.masked_where(posterior_pdf.T <= 0.0, posterior_pdf.T)
    ax.pcolormesh(x_bins, y_bins, prior_mesh, cmap="Greys", alpha=0.18, shading="auto")
    ax.pcolormesh(x_bins, y_bins, posterior_mesh, cmap="Blues", alpha=0.72, shading="auto")


def plot_prior_posterior_predictive(
    *,
    package_path: str | Path,
    hera_npz: str | Path,
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
    posterior_weights: np.ndarray,
    output_path: str | Path,
    rng: np.random.Generator,
    n_pencil: int,
) -> None:
    """
    Plot prior and posterior predictive lines against HERA 2-sigma limits.
    """
    configure_matplotlib()
    hera_dataset = load_hera_power_spectrum_npz(hera_npz)
    power_data = hera_dataset.power_data
    coordinates = np.asarray(power_data.coordinates, dtype=np.float32)
    window_matrix = np.asarray(power_data.window_matrix, dtype=np.float32)
    limit_2sigma = np.asarray(power_data.upper_limit + 2.0 * power_data.sigma, dtype=np.float64)

    prior_draw = prior_samples[
        rng.choice(prior_samples.shape[0], size=min(n_pencil, prior_samples.shape[0]), replace=False)
    ]
    posterior_draw = weighted_resample(
        posterior_samples,
        posterior_weights,
        n_pencil,
        rng,
    )

    package = load_delta21_package(package_path)
    emulator = build_delta21_fixed_point_emulator(
        package,
        jnp.asarray(coordinates, dtype=jnp.float32),
        compile_parameters=jnp.asarray(posterior_draw[:1], dtype=jnp.float32),
    )

    prior_prediction = window_predictions(emulator, prior_draw, window_matrix)
    posterior_prediction = window_predictions(emulator, posterior_draw, window_matrix)

    redshifts = np.unique(coordinates[:, 0])
    fig, axes = plt.subplots(1, len(redshifts), figsize=(6.0 * len(redshifts), 4.6), sharey=True)
    if len(redshifts) == 1:
        axes = np.asarray([axes])

    for ax, z_value in zip(axes, redshifts, strict=True):
        mask = np.isclose(coordinates[:, 0], z_value)
        order = np.argsort(coordinates[mask, 1])
        k_values = coordinates[mask, 1][order]
        prior_panel = prior_prediction[:, mask][:, order]
        posterior_panel = posterior_prediction[:, mask][:, order]
        limit_panel = limit_2sigma[mask][order]

        for values in prior_panel:
            ax.plot(k_values, positive_for_log(values), color="0.65", alpha=0.12, linewidth=0.7)
        for values in posterior_panel:
            ax.plot(k_values, positive_for_log(values), color="#2b6cb0", alpha=0.22, linewidth=0.8)

        ax.scatter(
            k_values,
            positive_for_log(limit_panel),
            marker="v",
            s=52,
            color="black",
            label="HERA 2$\\sigma$ upper limit",
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
    fig.suptitle("HERA Prior and Posterior Predictive Power Spectra", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def window_predictions(emulator: Any, samples: np.ndarray, window_matrix: np.ndarray) -> np.ndarray:
    """
    Evaluate emulator samples and apply the HERA window matrix.
    """
    prediction = np.asarray(
        jax.device_get(emulator.emulate(jnp.asarray(samples, dtype=jnp.float32))),
        dtype=np.float64,
    )
    return prediction @ window_matrix.T


def weighted_resample(
    samples: np.ndarray,
    weights: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw posterior samples using anesthetic sample weights.
    """
    indices = rng.choice(samples.shape[0], size=n_samples, replace=True, p=weights)
    return samples[indices]


def _histogram_bins(prior_values: np.ndarray, posterior_values: np.ndarray) -> np.ndarray:
    """
    Build common bins for prior and posterior histograms.
    """
    values = np.concatenate([prior_values, posterior_values])
    unique = np.unique(values)
    if unique.size <= 12:
        spacing = np.min(np.diff(unique)) if unique.size > 1 else 1.0
        return np.concatenate([[unique[0] - 0.5 * spacing], unique + 0.5 * spacing])
    return np.linspace(np.nanmin(values), np.nanmax(values), 36)


def positive_for_log(values: np.ndarray) -> np.ndarray:
    """
    Clip values to a small positive floor for log-axis plotting.
    """
    return np.clip(values, 1e-6, None)


def configure_matplotlib() -> None:
    """
    Apply a compact publication-style Matplotlib setup.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 15,
            "savefig.facecolor": "white",
        }
    )


if __name__ == "__main__":
    main()
