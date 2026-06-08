"""
T21 inference helpers and CLI entrypoint.

This module provides the logic for using a trained T21 emulator for prediction.
It handles loading versioned model packages, preparing input parameter tables,
executing the model forward pass, and inverting all preprocessing transforms
to recover physical brightness temperature signals.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from jax_emu.infer import Emulator, FixedGridEmulator
from jax_emu.utils.checkpointing import load
from emulators_21cmspace.t21.data import t21_spec


# Loading Utilities
# -----------------
# Helpers for retrieving saved models and numeric arrays from disk.

def _load_array_file(path: str | Path) -> np.ndarray:
    """
    Load a 1D or 2D numeric array from .npy, .npz, or text.
    """
    file_path = Path(path)
    # Handle standard NumPy binary formats.
    if file_path.suffix == ".npy":
        return np.asarray(np.load(file_path), dtype=float)
    if file_path.suffix == ".npz":
        payload = np.load(file_path)
        # Take the first available array from the archive.
        first_key = payload.files[0]
        return np.asarray(payload[first_key], dtype=float)
    # Fallback to space-separated text files.
    return np.asarray(np.loadtxt(file_path), dtype=float)


def load_t21_package(path: str | Path) -> dict[str, Any]:
    """
    Load a saved T21 checkpoint and validate its metadata.

    Parameters
    ----------
    path:
        The directory path to the .nenemu model package.

    Returns
    -------
    dict
        The loaded model and metadata dictionary.
    """
    # Use the shared Orbax loading utility.
    package = load(path)
    return _validate_t21_package(package)


def _validate_t21_package(package: dict[str, Any]) -> dict[str, Any]:
    """
    Validate that a loaded package can be used for T21 inference.
    """
    metadata = package["metadata"]
    # Ensure the checkpoint is not missing the required inference metadata.
    if metadata is None:
        raise ValueError("Saved checkpoint does not contain checkpoint metadata.")
    # Verify that this package actually belongs to the T21 emulator family.
    if metadata.emulator_spec.name != "t21":
        raise ValueError(f"Expected a T21 package, received {metadata.emulator_spec.name!r}.")
    return package


# Prediction Logic
# ----------------
# Core workflow for moving from astrophysical parameters to physical signals.

def build_t21_emulator(
    package_or_path: str | Path | dict[str, Any],
    *,
    compile_inputs: tuple[jax.Array, ...] | None = None,
) -> Emulator:
    """
    Build a reusable T21 emulator object.
    """
    # Resolve and validate the package outside JIT. File I/O and metadata checks
    # should happen once before repeated accelerator-side inference calls.
    package = (
        load_t21_package(package_or_path)
        if isinstance(package_or_path, (str, Path))
        else _validate_t21_package(package_or_path)
    )
    return Emulator(
        package=package,
        parameter_adapter=_prepare_parameter_values,
        compile_inputs=compile_inputs,
    )


def build_t21_predictor(
    package_or_path: str | Path | dict[str, Any],
    *,
    compile_inputs: tuple[jax.Array, ...] | None = None,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """
    Build a reusable JIT-compiled T21 prediction function.
    """
    # Keep the existing function-style API by returning the generic emulator's
    # forward model method.
    emulator = build_t21_emulator(package_or_path, compile_inputs=compile_inputs)
    return emulator.forward_model


def build_t21_fixed_grid_emulator(
    package_or_path: str | Path | dict[str, Any],
    z_values: jax.Array,
    *,
    compile_parameters: jax.Array | None = None,
) -> FixedGridEmulator:
    """
    Build a reusable T21 emulator for one fixed redshift grid.

    The redshift grid is transformed and scaled once during initialization.
    Later calls only pass parameter tables to `emulate(...)` or
    `forward_model(...)`.
    """
    # Resolve and validate the package outside JIT. The fixed-grid wrapper then
    # stores the compiled parameter-only forward model.
    package = (
        load_t21_package(package_or_path)
        if isinstance(package_or_path, (str, Path))
        else _validate_t21_package(package_or_path)
    )
    return FixedGridEmulator(
        package=package,
        axes=(z_values,),
        parameter_adapter=_prepare_parameter_values,
        compile_parameters=compile_parameters,
    )


def predict_t21(
    package_or_path: str | Path | dict[str, Any],
    parameters: jax.Array,
    z_values: jax.Array,
) -> jax.Array:
    """
    Predict T21 while keeping numerical work on the JAX device.

    Parameters
    ----------
    package_or_path:
        Either a path to a model package or an already loaded package dictionary.
    parameters:
        Astrophysical parameters. This can be a raw 12-column 21cmSPACE table
        or an already-prepared 9-column feature table.
    z_values:
        Redshift coordinates at which to evaluate the signal.

    Returns
    -------
    jax.Array
        Device array with shape (n_sims, n_z).
    """
    # Build the compiled predictor and immediately use it. For repeated calls,
    # build the predictor once with build_t21_predictor(...) and reuse it.
    predictor = build_t21_predictor(package_or_path)
    return predictor(parameters, z_values)


# Diagnostics
# -----------
# Helpers for inspecting model packages.

def describe_t21_package(path: str | Path) -> dict[str, Any]:
    """
    Return a small human-readable summary of a saved T21 checkpoint.
    """
    package = load_t21_package(path)
    metadata = package["metadata"]
    return {
        "model_name": metadata.model_name,
        "package_version": metadata.package_version,
        "feature_names": list(metadata.emulator_spec.input_feature_names()),
        "train_epochs": package["hyperparams"]["epochs"],
        "train_losses": len(package["train_losses"]),
        "validation_losses": len(package["val_losses"]),
    }


# CLI Entrypoint
# -------------
# Logic for running inference from the command line.

def build_parser() -> argparse.ArgumentParser:
    """
    Build the T21 inference command-line interface.
    """
    parser = argparse.ArgumentParser(description="T21 inference entrypoint.")
    parser.add_argument("--package", type=str, help="Path to a saved .nenemu checkpoint.")
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print a short description of a saved checkpoint.",
    )
    parser.add_argument(
        "--parameters-file",
        type=str,
        help="Path to a raw 12-column or prepared 9-column parameter table.",
    )
    parser.add_argument("--z-file", type=str, help="Path to a 1D redshift array.")
    parser.add_argument(
        "--output",
        type=str,
        help="Path to the output .npz prediction file.",
    )
    return parser


def main() -> None:
    """
    Run the T21 inference CLI.
    """
    args = build_parser().parse_args()

    # Handle the describe task.
    if args.describe:
        if args.package is None:
            raise SystemExit("Use --package together with --describe.")
        pprint(describe_t21_package(args.package))
        return

    # Ensure all required inputs are provided for prediction.
    required = [args.package, args.parameters_file, args.z_file]
    if not all(required):
        raise SystemExit(
            "Use --package, --parameters-file, and --z-file to run predictions, "
            "or --package --describe to inspect a saved model."
        )

    # Load input data.
    parameter_table = _load_array_file(args.parameters_file)
    z_values = _load_array_file(args.z_file)

    # Run the prediction pipeline.
    predictions = predict_t21(
        args.package,
        jnp.asarray(parameter_table, dtype=jnp.float32),
        jnp.asarray(z_values, dtype=jnp.float32),
    )

    # Save the output to a compressed NumPy archive.
    output_path = Path("t21_prediction.npz") if args.output is None else Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_array = np.asarray(jax.device_get(predictions))
    np.savez(
        output_path,
        t21=prediction_array,
        parameters=parameter_table,
        z=np.asarray(z_values, dtype=float).ravel(),
    )

    # Print success summary.
    pprint(
        {
            "output_path": str(output_path),
            "prediction_shape": list(predictions.shape),
        }
    )


# Internal Helpers
# ----------------

def _prepare_parameter_values(raw_parameters: jax.Array) -> jax.Array:
    """
    Convert raw or pre-prepared parameter tables into JAX model-input space.
    """
    array = raw_parameters
    if array.ndim == 1:
        array = array[None, :]

    expected_width = len(t21_spec().parameters)
    if array.shape[1] == 12:
        return jnp.stack(
            [
                jnp.log10(array[:, 0]),  # fstarII
                jnp.log10(array[:, 1]),  # fstarIII
                jnp.log10(array[:, 2]),  # Vc
                jnp.log10(array[:, 3]),  # fX
                array[:, 4],  # alpha
                array[:, 5],  # nu_0
                array[:, 7],  # tau
                jnp.log10(array[:, 8]),  # fradio
                array[:, 9],  # pop
            ],
            axis=1,
        )
    if array.shape[1] == expected_width:
        return array

    raise ValueError(
        "T21 inference expects either a raw 12-column 21cmSPACE parameter table "
        f"or a pre-prepared {expected_width}-column feature table."
    )
