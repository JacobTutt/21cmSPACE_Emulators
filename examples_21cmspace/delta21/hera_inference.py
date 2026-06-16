"""
Run a HERA-only nested-sampling fit with a trained Delta21 emulator.

The example uses the same likelihood structure as the older HERA analyses:
extract HERA power-spectrum upper limits, evaluate the emulator at the
model-side `(z, k)` points, apply the HERA window matrix, and use a one-sided
upper-limit likelihood.
"""

from __future__ import annotations

import argparse
from pprint import pprint

import jax
import jax.numpy as jnp

from examples_21cmspace.delta21.emulator import (
    build_delta21_fixed_point_emulator,
    default_delta21_hera_prior,
    load_delta21_package,
)
from jax_emu.inference import (
    NestedSamplingConfig,
    PowerSpectrumUpperLimitLikelihood,
    run_nested_sampling,
)
from examples_21cmspace.delta21.hera_data import (
    DEFAULT_HERA_IDR2_ROOT,
    default_h1c_idr2_selections,
    hera_dataset_summary,
    load_hera_power_spectrum_dataset,
    load_hera_power_spectrum_npz,
    save_hera_power_spectrum_npz,
)


def build_parser() -> argparse.ArgumentParser:
    """
    Build the HERA nested-sampling example command-line interface.
    """
    parser = argparse.ArgumentParser(description="Run HERA Delta21 nested sampling.")
    parser.add_argument("--package", required=True, help="Path to a trained Delta21 .nenemu package.")
    parser.add_argument(
        "--hera-npz",
        help="Optional extracted HERA NPZ cache. If omitted, H1C IDR2 HDF5 files are read.",
    )
    parser.add_argument(
        "--hera-idr2-root",
        default=str(DEFAULT_HERA_IDR2_ROOT),
        help="Directory containing pspec_h1c_idr2_field*.h5 files.",
    )
    parser.add_argument("--field", default="1", help="H1C IDR2 field to use.")
    parser.add_argument(
        "--write-hera-cache",
        help="Optional path where the extracted HERA likelihood arrays should be cached.",
    )
    parser.add_argument("--output-dir", default="outputs/hera_nested_sampling")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--n-live-scale", type=int, default=25)
    parser.add_argument("--num-delete-fraction", type=float, default=0.2)
    parser.add_argument("--num-inner-steps-scale", type=int, default=5)
    parser.add_argument("--logz-live-threshold", type=float, default=-3.0)
    parser.add_argument("--theory-fractional-error", type=float, default=0.2)
    parser.add_argument(
        "--log10-radio-min",
        type=float,
        default=-1.0,
        help=(
            "Lower prior bound for the radio-like parameter column. Use -6 for "
            "the cosmic-string/aradio emulator."
        ),
    )
    parser.add_argument(
        "--log10-radio-max",
        type=float,
        default=5.0,
        help=(
            "Upper prior bound for the radio-like parameter column. Use 3 for "
            "the cosmic-string/aradio emulator."
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Load the HERA data and print shapes without running nested sampling.",
    )
    return parser


def main() -> None:
    """
    Run the HERA-only nested-sampling example.
    """
    args = build_parser().parse_args()

    hera_dataset = (
        load_hera_power_spectrum_npz(args.hera_npz)
        if args.hera_npz is not None
        else load_hera_power_spectrum_dataset(
            default_h1c_idr2_selections(args.hera_idr2_root, field=args.field)
        )
    )
    pprint(hera_dataset_summary(hera_dataset))

    if args.write_hera_cache is not None:
        cache_path = save_hera_power_spectrum_npz(hera_dataset, args.write_hera_cache)
        print(f"Wrote HERA cache: {cache_path}")

    if args.summary_only:
        return

    package = load_delta21_package(args.package)
    prior = default_delta21_hera_prior(
        radio_log10_range=(args.log10_radio_min, args.log10_radio_max),
    )

    # Compile the emulator on the HERA model-side coordinates before sampling.
    compile_parameters = prior.transform(jnp.full((prior.ndim,), 0.5, dtype=jnp.float32))
    if compile_parameters.ndim == 1:
        compile_parameters = compile_parameters[None, :]

    emulator = build_delta21_fixed_point_emulator(
        package,
        hera_dataset.power_data.coordinates,
        compile_parameters=compile_parameters,
    )
    likelihood = PowerSpectrumUpperLimitLikelihood(
        emulator=emulator,
        upper_limit=hera_dataset.power_data.upper_limit,
        sigma=hera_dataset.power_data.sigma,
        window_matrix=hera_dataset.power_data.window_matrix,
        theory_fractional_error=args.theory_fractional_error,
    )

    config = NestedSamplingConfig(
        n_live_scale=args.n_live_scale,
        num_delete_fraction=args.num_delete_fraction,
        num_inner_steps_scale=args.num_inner_steps_scale,
        logz_live_threshold=args.logz_live_threshold,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        run_name="hera_delta21",
    )
    result = run_nested_sampling(
        prior=prior,
        likelihood=likelihood,
        key=jax.random.PRNGKey(args.seed),
        config=config,
    )

    pprint(
        {
            "output_dir": args.output_dir,
            "n_steps": result.n_steps,
            "converged": result.converged,
            "logz": None if result.logz is None else float(result.logz),
            "logz_error": None if result.logz_error is None else float(result.logz_error),
        }
    )


if __name__ == "__main__":
    main()
