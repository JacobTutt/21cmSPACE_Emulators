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
from flax import nnx

from jax_emu.data_preprocessing.scaling import FeatureScaling
from emulators_21cmspace.delta21.data import (
    delta21_spec,
)
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

def build_delta21_predictor(
    package_or_path: str | Path | dict[str, Any],
) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    """
    Build a reusable JIT-compiled Delta21 prediction function.

    This resolves the checkpoint package and validates the metadata once. The
    returned function then contains only the numerical inference path: parameter
    preparation, axis tiling, feature scaling, model evaluation, inverse target
    scaling, inverse target transform, and grid reconstruction.

    Parameters
    ----------
    package_or_path:
        Either a path to a model package or an already loaded package dictionary.

    Returns
    -------
    Callable[[jax.Array, jax.Array, jax.Array], jax.Array]
        A compiled prediction function with signature
        ``predict(parameters, z_values, k_values)``.
    """
    # Resolve and validate the package outside JIT. File I/O and metadata checks
    # should happen once before repeated accelerator-side inference calls.
    package = (
        load_delta21_package(package_or_path)
        if isinstance(package_or_path, (str, Path))
        else _validate_delta21_package(package_or_path)
    )
    model = package["model"]
    metadata = package["metadata"]
    spec = metadata.emulator_spec

    # Check feature order once before compilation. The compiled function can then
    # assume that the saved scaling metadata matches the emulator specification.
    expected_names = spec.input_feature_names()
    actual_names = tuple(feature.name for feature in metadata.input_scaling)
    if actual_names != expected_names:
        raise ValueError(
            "Saved feature scaling order does not match the emulator spec. "
            f"Expected {expected_names}, received {actual_names}."
        )

    axis_transforms = (spec.axes[0].transform, spec.axes[1].transform)
    target_transform = spec.target_transform
    target_offset = spec.target_offset
    input_scaling = metadata.input_scaling
    target_std = None if metadata.target_scaling is None else metadata.target_scaling.std

    @nnx.jit
    def _predict(
        model_instance: Any,
        parameters: jax.Array,
        z_values: jax.Array,
        k_values: jax.Array,
    ) -> jax.Array:
        """
        Run the compiled numerical Delta21 inference path.
        """
        # Prepare raw 12-column or already-prepared 9-column parameters.
        prepared_parameters = _prepare_parameter_values(parameters)
        # Flatten coordinate arrays so the following meshgrid is always 1D x 1D.
        z = z_values.ravel()
        k = k_values.ravel()

        # Generate the evaluation grid for (z, k).
        zz, kk = jnp.meshgrid(z, k, indexing="ij")
        # Apply the same coordinate transforms used during training.
        axis_features = jnp.stack(
            [
                _apply_transform_jax(zz.ravel(), axis_transforms[0]),
                _apply_transform_jax(kk.ravel(), axis_transforms[1]),
            ],
            axis=1,
        )

        # Tile coordinates and parameters into scalar regression rows.
        repeated_axes = jnp.tile(axis_features, (prepared_parameters.shape[0], 1))
        repeated_parameters = jnp.repeat(
            prepared_parameters,
            repeats=axis_features.shape[0],
            axis=0,
        )
        features = jnp.concatenate([repeated_axes, repeated_parameters], axis=1)

        # Scale features using training-set statistics and evaluate the model.
        scaled_features = _scale_features_jax(features, input_scaling)
        flat_predictions = model_instance(scaled_features).squeeze(-1)

        # Undo target scaling and target transforms to return physical values.
        if target_std is not None:
            flat_predictions = flat_predictions * target_std
        physical_predictions = _invert_transform_jax(
            flat_predictions,
            target_transform,
            offset=target_offset,
        )

        # Fold the flattened vector back into (nsamples, n_z, n_k).
        return physical_predictions.reshape(
            (prepared_parameters.shape[0], z.shape[0], k.shape[0])
        )

    def predict(parameters: jax.Array, z_values: jax.Array, k_values: jax.Array) -> jax.Array:
        """
        Predict Delta21 using the compiled model-specific inference function.
        """
        return _predict(model, parameters, z_values, k_values)

    return predict


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


def _apply_transform_jax(values: jax.Array, transform: str) -> jax.Array:
    """
    Apply a named coordinate transform on the JAX device.
    """
    if transform == "identity":
        return values
    if transform == "log10":
        return jnp.log10(values)
    raise ValueError(f"Unsupported transform {transform}.")


def _invert_transform_jax(values: jax.Array, transform: str, offset: float = 0.0) -> jax.Array:
    """
    Undo a named target transform on the JAX device.
    """
    if transform == "identity":
        return values
    if transform == "log10":
        return jnp.power(10.0, values) - offset
    raise ValueError(f"Unsupported transform {transform}.")


def _scale_features_jax(
    features: jax.Array,
    scaling: tuple[FeatureScaling, ...],
) -> jax.Array:
    """
    Apply saved input-feature scaling on the JAX device.
    """
    columns = []
    for idx, feature in enumerate(scaling):
        values = features[:, idx]
        if feature.method == "identity":
            scaled = values
        elif feature.method == "zscore":
            scaled = (values - feature.mean) / feature.std
        elif feature.method == "minmax_minus_one_to_one":
            denom = feature.maximum - feature.minimum
            scaled = (
                jnp.zeros_like(values)
                if denom == 0
                else (2.0 * (values - feature.minimum) / denom) - 1.0
            )
        elif feature.method == "minmax_zero_to_one":
            denom = feature.maximum - feature.minimum
            scaled = (
                jnp.zeros_like(values)
                if denom == 0
                else (values - feature.minimum) / denom
            )
        else:
            raise ValueError(f"Unsupported scaling method {feature.method}.")
        columns.append(scaled)
    return jnp.stack(columns, axis=1).astype(jnp.float32)
