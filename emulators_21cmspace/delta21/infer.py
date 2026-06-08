"""
Delta21 inference helpers and CLI entrypoint.

This module provides the logic for using a trained Delta21 emulator for
prediction. It handles loading versioned model packages, preparing input
parameter tables and (z, k) grids, executing the model forward pass, and
inverting all preprocessing transforms to recover physical power spectra.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from emulators_21cmspace.delta21.data import (
    delta21_spec,
)
from jax_emu.infer import Emulator, FixedCoordinateEmulator, FixedGridEmulator
from jax_emu.utils.checkpointing import load


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


def load_delta21_package(path: str | Path) -> dict[str, Any]:
    """
    Load a saved Delta21 checkpoint and validate its metadata.

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
    return _validate_delta21_package(package)


def _validate_delta21_package(package: dict[str, Any]) -> dict[str, Any]:
    """
    Validate that a loaded package can be used for Delta21 inference.
    """
    metadata = package["metadata"]
    # Ensure the checkpoint is not missing required inference metadata.
    if metadata is None:
        raise ValueError("Saved checkpoint does not contain checkpoint metadata.")
    # Verify that this package actually belongs to the Delta21 emulator family.
    if metadata.emulator_spec.name != "delta21":
        raise ValueError(
            f"Expected a Delta21 package, received {metadata.emulator_spec.name!r}."
        )
    return package


# Prediction Logic
# ----------------
# Core workflow for moving from astrophysical parameters to physical power spectra.

def build_delta21_emulator(
    package_or_path: str | Path | dict[str, Any],
    *,
    compile_inputs: tuple[jax.Array, ...] | None = None,
) -> Emulator:
    """
    Build a reusable Delta21 emulator object.

    This resolves the checkpoint package and validates the metadata once, then
    delegates the compiled forward model to the generic `jax_emu.Emulator`
    wrapper.

    Parameters
    ----------
    package_or_path:
        Either a path to a model package or an already loaded package dictionary.
    compile_inputs:
        Optional tuple `(parameters, z_values, k_values)` used to force JIT
        compilation during initialization.

    Returns
    -------
    Emulator
        Reusable emulator object with a compiled `forward_model` method.
    """
    # Resolve and validate the package outside JIT. File I/O and metadata checks
    # should happen once before repeated accelerator-side inference calls.
    package = (
        load_delta21_package(package_or_path)
        if isinstance(package_or_path, (str, Path))
        else _validate_delta21_package(package_or_path)
    )
    return Emulator(
        package=package,
        parameter_adapter=_prepare_parameter_values,
        compile_inputs=compile_inputs,
    )


def build_delta21_predictor(
    package_or_path: str | Path | dict[str, Any],
    *,
    compile_inputs: tuple[jax.Array, ...] | None = None,
) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    """
    Build a reusable JIT-compiled Delta21 prediction function.
    """
    # Keep the existing function-style API by returning the generic emulator's
    # forward model method.
    emulator = build_delta21_emulator(package_or_path, compile_inputs=compile_inputs)
    return emulator.forward_model


def build_delta21_fixed_grid_emulator(
    package_or_path: str | Path | dict[str, Any],
    z_values: jax.Array,
    k_values: jax.Array,
    *,
    compile_parameters: jax.Array | None = None,
) -> FixedGridEmulator:
    """
    Build a reusable Delta21 emulator for one fixed `(z, k)` grid.

    The redshift and wavenumber grid is transformed and scaled once during
    initialization. Later calls only pass parameter tables to `emulate(...)` or
    `forward_model(...)`.
    """
    # Resolve and validate the package outside JIT. The fixed-grid wrapper then
    # stores the compiled parameter-only forward model.
    package = (
        load_delta21_package(package_or_path)
        if isinstance(package_or_path, (str, Path))
        else _validate_delta21_package(package_or_path)
    )
    return FixedGridEmulator(
        package=package,
        axes=(z_values, k_values),
        parameter_adapter=_prepare_parameter_values,
        compile_parameters=compile_parameters,
    )


def build_delta21_fixed_coordinate_emulator(
    package_or_path: str | Path | dict[str, Any],
    z_points: jax.Array,
    k_points: jax.Array,
    *,
    compile_parameters: jax.Array | None = None,
) -> FixedCoordinateEmulator:
    """
    Build a reusable Delta21 emulator for one fixed coordinate list.

    This is useful for likelihoods where the requested `(z, k)` points are
    sparse or window-function based rather than a full rectangular grid.
    """
    package = (
        load_delta21_package(package_or_path)
        if isinstance(package_or_path, (str, Path))
        else _validate_delta21_package(package_or_path)
    )
    return FixedCoordinateEmulator(
        package=package,
        coordinates=(z_points, k_points),
        parameter_adapter=_prepare_parameter_values,
        compile_parameters=compile_parameters,
    )


def build_delta21_fixed_point_emulator(
    package_or_path: str | Path | dict[str, Any],
    coordinates: jax.Array,
    *,
    compile_parameters: jax.Array | None = None,
) -> FixedCoordinateEmulator:
    """
    Build a reusable Delta21 emulator from explicit `(z, k)` coordinate pairs.

    `coordinates` must have shape `(n_points, 2)`, with redshift in column 0
    and k in column 1. These points are not expanded into a grid.
    """
    coordinate_array = jnp.asarray(coordinates, dtype=jnp.float32)
    if coordinate_array.ndim != 2 or coordinate_array.shape[1] != 2:
        raise ValueError("Delta21 coordinates must have shape (n_points, 2).")

    return build_delta21_fixed_coordinate_emulator(
        package_or_path,
        coordinate_array[:, 0],
        coordinate_array[:, 1],
        compile_parameters=compile_parameters,
    )


def predict_delta21(
    package_or_path: str | Path | dict[str, Any],
    parameters: jax.Array,
    z_values: jax.Array,
    k_values: jax.Array,
) -> jax.Array:
    """
    Predict Delta21 while keeping numerical work on the JAX device.

    Parameters
    ----------
    package_or_path:
        Either a path to a model package or an already loaded package dictionary.
    parameters:
        Astrophysical parameters. This can be a raw 12-column 21cmSPACE table
        or an already-prepared 9-column feature table.
    z_values:
        Redshift coordinates at which to evaluate the signal.
    k_values:
        Wavenumber coordinates at which to evaluate the signal.

    Returns
    -------
    jax.Array
        Device array with shape (n_sims, n_z, n_k).
    """
    # Build the compiled predictor and immediately use it. For repeated calls,
    # build the predictor once with build_delta21_predictor(...) and reuse it.
    predictor = build_delta21_predictor(package_or_path)
    return predictor(parameters, z_values, k_values)


# Diagnostics
# -----------
# Helpers for inspecting model packages.

def describe_delta21_package(path: str | Path) -> dict[str, Any]:
    """
    Return a small human-readable summary of a saved Delta21 checkpoint.
    """
    package = load_delta21_package(path)
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
    Build the Delta21 inference command-line interface.
    """
    parser = argparse.ArgumentParser(description="Delta21 inference entrypoint.")
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
    parser.add_argument("--k-file", type=str, help="Path to a 1D wave-number array.")
    parser.add_argument(
        "--output",
        type=str,
        help="Path to the output .npz prediction file.",
    )
    return parser


def main() -> None:
    """
    Run the Delta21 inference CLI.
    """
    args = build_parser().parse_args()

    # Handle the describe task.
    if args.describe:
        if args.package is None:
            raise SystemExit("Use --package together with --describe.")
        pprint(describe_delta21_package(args.package))
        return

    # Ensure all required inputs are provided for prediction.
    required = [args.package, args.parameters_file, args.z_file, args.k_file]
    if not all(required):
        raise SystemExit(
            "Use --package, --parameters-file, --z-file, and --k-file to run predictions, "
            "or --package --describe to inspect a saved model."
        )

    # Load inputs once from disk.
    parameter_table = _load_array_file(args.parameters_file)
    z_values = _load_array_file(args.z_file)
    k_values = _load_array_file(args.k_file)

    # Run the prediction pipeline.
    # The CLI is a file-I/O boundary, so NumPy arrays become JAX arrays here.
    predictions = predict_delta21(
        args.package,
        jnp.asarray(parameter_table, dtype=jnp.float32),
        jnp.asarray(z_values, dtype=jnp.float32),
        jnp.asarray(k_values, dtype=jnp.float32),
    )

    # Save the output to a compressed NumPy archive.
    output_path = Path("delta21_prediction.npz") if args.output is None else Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_array = np.asarray(jax.device_get(predictions))
    np.savez(
        output_path,
        delta21=prediction_array,
        parameters=parameter_table,
        z=np.asarray(z_values, dtype=float).ravel(),
        k=np.asarray(k_values, dtype=float).ravel(),
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

    expected_width = len(delta21_spec().parameters)
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
        "Delta21 inference expects either a raw 12-column 21cmSPACE parameter table "
        f"or a pre-prepared {expected_width}-column feature table."
    )
