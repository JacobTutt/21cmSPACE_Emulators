"""
T21 emulator loading and prediction helpers.

This module contains reusable package code for using a trained T21 emulator.
File-based examples and command-line wrappers live outside the source package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp

from jax_emu.inference import Emulator, FixedEmulator
from jax_emu.utils.checkpointing import load
from examples_21cmspace.t21.data import t21_spec


# Loading Utilities
# -----------------
# Helpers for retrieving saved models from disk.


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
    # Keep the function-style API by returning the generic emulator call.
    emulator = build_t21_emulator(package_or_path, compile_inputs=compile_inputs)
    return emulator.emulate


def build_t21_fixed_grid_emulator(
    package_or_path: str | Path | dict[str, Any],
    z_values: jax.Array,
    *,
    compile_parameters: jax.Array | None = None,
) -> FixedEmulator:
    """
    Build a reusable T21 emulator for one fixed redshift grid.

    The redshift grid is transformed and scaled once during initialization.
    Later calls only pass parameter tables to `emulate(...)`.
    """
    # Resolve and validate the package outside JIT. The fixed wrapper then
    # stores the compiled parameter-only emulator call.
    package = (
        load_t21_package(package_or_path)
        if isinstance(package_or_path, (str, Path))
        else _validate_t21_package(package_or_path)
    )
    return FixedEmulator(
        package=package,
        axes=(z_values,),
        parameter_adapter=_prepare_parameter_values,
        compile_parameters=compile_parameters,
    )


def build_t21_fixed_coordinate_emulator(
    package_or_path: str | Path | dict[str, Any],
    z_points: jax.Array,
    *,
    compile_parameters: jax.Array | None = None,
) -> FixedEmulator:
    """
    Build a reusable T21 emulator for one fixed redshift coordinate list.

    This has the same output as the fixed-grid route for one axis, but it gives
    likelihood code a common interface for all emulator families.
    """
    package = (
        load_t21_package(package_or_path)
        if isinstance(package_or_path, (str, Path))
        else _validate_t21_package(package_or_path)
    )
    return FixedEmulator(
        package=package,
        coordinates=(z_points,),
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
                jnp.log10(array[:, 8]),  # radio amplitude: fradio or aradio
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
