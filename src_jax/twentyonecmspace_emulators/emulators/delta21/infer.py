"""Delta21 inference helpers and CLI entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint
from typing import Any

import jax.numpy as jnp
import numpy as np

from twentyonecmspace_emulators.utils.scaling import FeatureScaler
from twentyonecmspace_emulators.utils.tiling import reconstruct_spectra
from twentyonecmspace_emulators.utils.transforms import apply_transform, invert_transform
from twentyonecmspace_emulators.emulators.delta21.data import (
    delta21_spec,
    prepare_twentyonecmspace_delta21_parameters,
)
from twentyonecmspace_emulators.utils.checkpointing import load


def _load_array_file(path: str | Path) -> np.ndarray:
    """Load a 1D or 2D numeric array from ``.npy``, ``.npz``, or text."""
    file_path = Path(path)
    if file_path.suffix == ".npy":
        return np.asarray(np.load(file_path), dtype=float)
    if file_path.suffix == ".npz":
        payload = np.load(file_path)
        first_key = payload.files[0]
        return np.asarray(payload[first_key], dtype=float)
    return np.asarray(np.loadtxt(file_path), dtype=float)


def _prepare_parameter_values(raw_parameters: np.ndarray) -> np.ndarray:
    """Convert raw or pre-prepared parameter tables into model-input space."""
    array = np.asarray(raw_parameters, dtype=float)
    if array.ndim == 1:
        array = array[None, :]

    expected_width = len(delta21_spec().parameters)
    if array.shape[1] == 12:
        return prepare_twentyonecmspace_delta21_parameters(array).values
    if array.shape[1] == expected_width:
        return array
    raise ValueError(
        "Delta21 inference expects either a raw 12-column 21cmSPACE parameter table "
        f"or a pre-prepared {expected_width}-column feature table."
    )


def load_delta21_package(path: str | Path) -> dict[str, Any]:
    """Load a saved Delta21 package and validate that it contains metadata."""
    package = load(path)
    metadata = package["metadata"]
    if metadata is None:
        raise ValueError("Saved package does not contain checkpoint metadata.")
    if metadata.emulator_spec.name != "delta21":
        raise ValueError(
            f"Expected a Delta21 package, received {metadata.emulator_spec.name!r}."
        )
    return package


def predict_delta21(
    package_or_path: str | Path | dict[str, Any],
    parameters: np.ndarray,
    z_values: np.ndarray,
    k_values: np.ndarray,
) -> np.ndarray:
    """Predict Delta21 on a chosen ``(z, k)`` grid for one or more parameter sets."""
    package = (
        load_delta21_package(package_or_path)
        if isinstance(package_or_path, (str, Path))
        else package_or_path
    )
    metadata = package["metadata"]
    spec = metadata.emulator_spec
    scaler = FeatureScaler(metadata.input_scaling)

    prepared_parameters = _prepare_parameter_values(parameters)
    z = np.asarray(z_values, dtype=float).ravel()
    k = np.asarray(k_values, dtype=float).ravel()

    zz, kk = np.meshgrid(z, k, indexing="ij")
    axis_features = np.column_stack(
        [
            apply_transform(zz.ravel(), spec.axes[0].transform),
            apply_transform(kk.ravel(), spec.axes[1].transform),
        ]
    )
    repeated_axes = np.tile(axis_features, (prepared_parameters.shape[0], 1))
    repeated_parameters = np.repeat(
        prepared_parameters,
        repeats=axis_features.shape[0],
        axis=0,
    )
    features = np.concatenate([repeated_axes, repeated_parameters], axis=1)

    expected_names = spec.input_feature_names()
    actual_names = tuple(feature.name for feature in metadata.input_scaling)
    if actual_names != expected_names:
        raise ValueError(
            "Saved feature scaling order does not match the emulator spec. "
            f"Expected {expected_names}, received {actual_names}."
        )

    scaled_features = scaler.transform(features).astype(np.float32)
    flat_predictions = np.asarray(
        package["model"](jnp.asarray(scaled_features))
    ).squeeze(-1)
    if metadata.target_scaling is not None:
        flat_predictions = metadata.target_scaling.inverse_rows(
            flat_predictions,
            repeated_axes,
        )
    physical_predictions = invert_transform(
        flat_predictions,
        spec.target_transform,
        offset=spec.target_offset,
    )
    return reconstruct_spectra(
        physical_predictions,
        nsamples=prepared_parameters.shape[0],
        axis_shape=(len(z), len(k)),
    )


def describe_delta21_package(path: str | Path) -> dict[str, Any]:
    """Return a small human-readable summary of a saved Delta21 package."""
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


def build_parser() -> argparse.ArgumentParser:
    """Build the Delta21 inference command-line interface."""
    parser = argparse.ArgumentParser(description="Delta21 inference entrypoint.")
    parser.add_argument("--package", type=str, help="Path to a saved .nenemu package.")
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print a short description of a saved package.",
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
    """Run the Delta21 inference CLI."""
    args = build_parser().parse_args()
    if args.describe:
        if args.package is None:
            raise SystemExit("Use --package together with --describe.")
        pprint(describe_delta21_package(args.package))
        return

    required = [args.package, args.parameters_file, args.z_file, args.k_file]
    if not all(required):
        raise SystemExit(
            "Use --package, --parameters-file, --z-file, and --k-file to run predictions, "
            "or --package --describe to inspect a saved model."
        )

    predictions = predict_delta21(
        args.package,
        _load_array_file(args.parameters_file),
        _load_array_file(args.z_file),
        _load_array_file(args.k_file),
    )
    output_path = Path("delta21_prediction.npz") if args.output is None else Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        delta21=predictions,
        parameters=_load_array_file(args.parameters_file),
        z=np.asarray(_load_array_file(args.z_file), dtype=float).ravel(),
        k=np.asarray(_load_array_file(args.k_file), dtype=float).ravel(),
    )
    pprint(
        {
            "output_path": str(output_path),
            "prediction_shape": list(predictions.shape),
        }
    )
