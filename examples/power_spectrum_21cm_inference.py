"""
Run file-based inference with a trained 21-cm power-spectrum emulator.

This is an example script rather than source package code. It loads arrays from
disk, calls the reusable Delta21 emulator helper, and writes the prediction to
an `.npz` file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

import jax
import jax.numpy as jnp
import numpy as np

from emulators_21cmspace.delta21.emulator import describe_delta21_package, predict_delta21


def load_array_file(path: str | Path) -> np.ndarray:
    """
    Load a 1D or 2D numeric array from `.npy`, `.npz`, or text.
    """
    file_path = Path(path)
    if file_path.suffix == ".npy":
        return np.asarray(np.load(file_path), dtype=float)
    if file_path.suffix == ".npz":
        payload = np.load(file_path)
        return np.asarray(payload[payload.files[0]], dtype=float)
    return np.asarray(np.loadtxt(file_path), dtype=float)


def build_parser() -> argparse.ArgumentParser:
    """
    Build the example command-line interface.
    """
    parser = argparse.ArgumentParser(description="Run power-spectrum emulator inference.")
    parser.add_argument("--package", type=str, required=True, help="Path to a .nenemu package.")
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print a short description of the saved checkpoint.",
    )
    parser.add_argument(
        "--parameters-file",
        type=str,
        help="Path to a raw 12-column or prepared 9-column parameter table.",
    )
    parser.add_argument("--z-file", type=str, help="Path to a 1D redshift array.")
    parser.add_argument("--k-file", type=str, help="Path to a 1D wave-number array.")
    parser.add_argument("--output", type=str, default="delta21_prediction.npz")
    return parser


def main() -> None:
    """
    Run the example.
    """
    args = build_parser().parse_args()

    if args.describe:
        pprint(describe_delta21_package(args.package))
        return

    if args.parameters_file is None or args.z_file is None or args.k_file is None:
        raise SystemExit("Use --parameters-file, --z-file, and --k-file to run predictions.")

    parameter_table = load_array_file(args.parameters_file)
    z_values = load_array_file(args.z_file)
    k_values = load_array_file(args.k_file)

    predictions = predict_delta21(
        args.package,
        jnp.asarray(parameter_table, dtype=jnp.float32),
        jnp.asarray(z_values, dtype=jnp.float32),
        jnp.asarray(k_values, dtype=jnp.float32),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        delta21=np.asarray(jax.device_get(predictions)),
        parameters=parameter_table,
        z=np.asarray(z_values, dtype=float).ravel(),
        k=np.asarray(k_values, dtype=float).ravel(),
    )
    pprint({"output_path": str(output_path), "prediction_shape": list(predictions.shape)})


if __name__ == "__main__":
    main()
