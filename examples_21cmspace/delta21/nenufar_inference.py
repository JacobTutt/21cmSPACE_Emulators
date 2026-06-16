"""
Run a NenuFAR-only nested-sampling fit with a trained Delta21 emulator.

The workflow uses the published Table 4 power-spectrum points from Munshi et
al. (2025): residual power estimates, 2-sigma upper limits, and their `(z, k)`
coordinates. No window matrix is published with this table, so the emulator is
evaluated directly at the tabulated spherical k bins.
"""

from __future__ import annotations

import argparse
from pprint import pprint

import jax
import jax.numpy as jnp

from examples_21cmspace.delta21.emulator import (
    build_delta21_fixed_point_emulator,
    default_delta21_inference_prior,
    load_delta21_package,
)
from jax_emu.inference import (
    NestedSamplingConfig,
    PowerSpectrumUpperLimitLikelihood,
    run_nested_sampling,
)
from examples_21cmspace.delta21.nenufar_data import (
    DEFAULT_NENUFAR_TABLE4_PATH,
    load_nenufar_table4_dataset,
    nenufar_dataset_summary,
)


def build_parser() -> argparse.ArgumentParser:
    """
    Build the NenuFAR nested-sampling command-line interface.
    """
    parser = argparse.ArgumentParser(description="Run NenuFAR Delta21 nested sampling.")
    parser.add_argument("--package", required=True, help="Path to a trained Delta21 .nenemu package.")
    parser.add_argument(
        "--nenufar-table",
        default=str(DEFAULT_NENUFAR_TABLE4_PATH),
        help="CSV table containing the published NenuFAR Table 4 points.",
    )
    parser.add_argument(
        "--use-reported-upper-limit",
        action="store_true",
        help="Use the reported 2-sigma upper limit as the likelihood threshold.",
    )
    parser.add_argument("--output-dir", default="outputs/nenufar_nested_sampling")
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
        help="Lower prior bound for the radio-amplitude parameter.",
    )
    parser.add_argument(
        "--log10-radio-max",
        type=float,
        default=5.0,
        help="Upper prior bound for the radio-amplitude parameter.",
    )
    parser.add_argument(
        "--radio-parameter-name",
        default="fradio",
        help="Name for the radio-amplitude prior. Use `aradio` for cosmic-string datasets.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Load the NenuFAR data and print shapes without running nested sampling.",
    )
    return parser


def main() -> None:
    """
    Run the NenuFAR-only nested-sampling workflow.
    """
    args = build_parser().parse_args()

    nenufar_dataset = load_nenufar_table4_dataset(
        args.nenufar_table,
        use_reported_upper_limit_as_threshold=args.use_reported_upper_limit,
    )
    pprint(nenufar_dataset_summary(nenufar_dataset))

    if args.summary_only:
        return

    package = load_delta21_package(args.package)
    prior = default_delta21_inference_prior(
        radio_log10_range=(args.log10_radio_min, args.log10_radio_max),
        radio_parameter_name=args.radio_parameter_name,
    )

    # Compile the emulator on the NenuFAR coordinate list before sampling.
    compile_parameters = prior.transform(jnp.full((prior.ndim,), 0.5, dtype=jnp.float32))
    if compile_parameters.ndim == 1:
        compile_parameters = compile_parameters[None, :]

    emulator = build_delta21_fixed_point_emulator(
        package,
        nenufar_dataset.power_data.coordinates,
        compile_parameters=compile_parameters,
    )
    likelihood = PowerSpectrumUpperLimitLikelihood(
        emulator=emulator,
        upper_limit=nenufar_dataset.power_data.upper_limit,
        sigma=nenufar_dataset.power_data.sigma,
        theory_fractional_error=args.theory_fractional_error,
    )

    config = NestedSamplingConfig(
        n_live_scale=args.n_live_scale,
        num_delete_fraction=args.num_delete_fraction,
        num_inner_steps_scale=args.num_inner_steps_scale,
        logz_live_threshold=args.logz_live_threshold,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        run_name="nenufar_delta21",
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
