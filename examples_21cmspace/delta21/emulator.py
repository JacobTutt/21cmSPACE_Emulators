"""
Delta21 emulator loading and prediction helpers.

This module contains reusable package code for using a trained Delta21
emulator. File-based examples and command-line wrappers live outside the source
package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp

from examples_21cmspace.delta21.data import (
    delta21_spec,
)
from jax_emu.inference import Emulator, FixedEmulator
from jax_emu.inference.prior import DiscretePrior, PriorSpec, UniformPrior
from jax_emu.utils.checkpointing import load


# Loading Utilities
# -----------------
# Helpers for retrieving saved models from disk.


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
    delegates the compiled emulator call to the generic `jax_emu.Emulator`
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
        Reusable emulator object with a compiled `emulate` method.
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
    # Keep the function-style API by returning the generic emulator call.
    emulator = build_delta21_emulator(package_or_path, compile_inputs=compile_inputs)
    return emulator.emulate


def build_delta21_fixed_grid_emulator(
    package_or_path: str | Path | dict[str, Any],
    z_values: jax.Array,
    k_values: jax.Array,
    *,
    compile_parameters: jax.Array | None = None,
) -> FixedEmulator:
    """
    Build a reusable Delta21 emulator for one fixed `(z, k)` grid.

    The redshift and wavenumber grid is transformed and scaled once during
    initialization. Later calls only pass parameter tables to `emulate(...)`.
    """
    # Resolve and validate the package outside JIT. The fixed wrapper then
    # stores the compiled parameter-only emulator call.
    package = (
        load_delta21_package(package_or_path)
        if isinstance(package_or_path, (str, Path))
        else _validate_delta21_package(package_or_path)
    )
    return FixedEmulator(
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
) -> FixedEmulator:
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
    return FixedEmulator(
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
) -> FixedEmulator:
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


# Priors
# ------
# Reference prior used by the power-spectrum nested-sampling examples.

def default_delta21_inference_prior(
    *,
    radio_log10_range: tuple[float, float] = (-1.0, 5.0),
    radio_parameter_name: str = "fradio",
) -> PriorSpec:
    """
    Return a default prior for the prepared Delta21 emulator inputs.

    The Delta21 emulator accepts either a raw 12-column 21cmSPACE table or the
    prepared nine-column table used by the network. This prior samples the
    prepared table directly, so log-scaled parameters are sampled in log10
    space and discrete simulation parameters are sampled from their allowed
    values.

    Parameters
    ----------
    radio_log10_range:
        Prior range for the final radio-like parameter column. The original
        21cmSPACE emulator used `fradio` with `[-1, 5]`; the cosmic-string
        dataset uses the same column position for `aradio`, which should use
        its own training range, for example `[-6, 3]`.
    radio_parameter_name:
        Name used for the radio-amplitude prior. Use `fradio` for the original
        radio-background dataset and `aradio` for the cosmic-string dataset.
    """
    radio_min, radio_max = radio_log10_range
    return PriorSpec(
        [
            UniformPrior("log10fstarII", -3.0, float(jnp.log10(0.5))),
            UniformPrior("log10fstarIII", -3.0, float(jnp.log10(0.5))),
            UniformPrior("log10Vc", float(jnp.log10(4.2)), 2.0),
            UniformPrior("log10fX", -3.0, 3.0),
            DiscretePrior("alpha", [1.0, 1.3, 1.5]),
            DiscretePrior(
                "nu_0",
                [*range(100, 1600, 100), 2000, 3000],
            ),
            UniformPrior("tau", 0.054 - 3.0 * 0.007, 0.054 + 3.0 * 0.007),
            UniformPrior(f"log10{radio_parameter_name}", radio_min, radio_max),
            DiscretePrior("pop", [231.0, 232.0, 233.0]),
        ]
    )


def default_delta21_hera_prior(
    *,
    radio_log10_range: tuple[float, float] = (-1.0, 5.0),
    radio_parameter_name: str = "fradio",
) -> PriorSpec:
    """
    Return the default prior used by the HERA example workflow.
    """
    return default_delta21_inference_prior(
        radio_log10_range=radio_log10_range,
        radio_parameter_name=radio_parameter_name,
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
                jnp.log10(array[:, 8]),  # radio amplitude: fradio or aradio
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
