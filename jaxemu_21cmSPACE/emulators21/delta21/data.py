"""
Delta21 emulator data contracts and preparation helpers.

This module turns raw 21cmSPACE arrays into the scalar regression problem used
to train the current Delta21 emulator. It defines the input contract, the
parameter preparation rules, and the row-building workflow in one place so the
full training setup is easy to inspect. It handles:
- the baseline Delta21 emulator specification (z and k axes)
- parameter processing logic (filtering and transforms)
- the full training data preparation pipeline for power spectra
"""

from __future__ import annotations

import numpy as np

from jaxemu_21cmSPACE.data_preprocessing.specs import AxisSpec, EmulatorSpec, ParameterSpec
from jaxemu_21cmSPACE.emulators21.twentyonecmspace import (
    DIMENSIONLESS_HUBBLE_PARAMETER,
    TWENTYONECMSPACE_COLUMNS,
    load_twentyonecmspace_delta21,
)
from jaxemu_21cmSPACE.data_preprocessing.parameters import PreparedFeatures, prepare_feature_matrix
from jaxemu_21cmSPACE.data_preprocessing.preparation import PreparedSplit, prepare_fixed_grid_training_split


# Emulator Specification
# ----------------------
# Defines the input/output contract for the Delta21 (power spectrum) model.

def delta21_spec() -> EmulatorSpec:
    """
    Return the baseline 21cmSPACE Delta21 emulator contract.

    This uses the established `Delta21` setup: two tiled axes (`z`, `k`) plus nine
    astrophysical parameters after dropping unused columns and applying the
    workflow transforms.

    Returns
    -------
    EmulatorSpec
        The contract defining the model input features and targets.
    """
    return EmulatorSpec(
        name="delta21",
        family="power_spectrum",
        # The Delta21 emulator operates on a 2D redshift-wavenumber grid.
        axes=(
            AxisSpec(name="z", transform="identity", limits=(6.0, 27.0), nsample=20),
            # Wavenumber k is transformed to log10 space for training.
            AxisSpec(
                name="k",
                transform="log10",
                # Limits are specified in physical units and converted to h-scaled units.
                limits=(
                    3e-2 / DIMENSIONLESS_HUBBLE_PARAMETER,
                    0.99 / DIMENSIONLESS_HUBBLE_PARAMETER,
                ),
                nsample=20,
            ),
        ),
        # Astrophysical parameters identified as having a significant impact on Delta21.
        parameters=(
            ParameterSpec(name="fstarII", transform="log10"),
            ParameterSpec(name="fstarIII", transform="log10"),
            ParameterSpec(name="Vc", transform="log10"),
            ParameterSpec(name="fX", transform="log10"),
            # Discrete parameters from the 21cmSPACE sampling grid.
            ParameterSpec(name="alpha", discrete_values=(1.0, 1.3, 1.5)),
            ParameterSpec(
                name="nu_0",
                discrete_values=tuple(float(v) for v in [*range(100, 1600, 100), 2000, 3000]),
            ),
            ParameterSpec(name="tau"),
            ParameterSpec(name="fradio", transform="log10"),
            ParameterSpec(name="pop", discrete_values=(231.0, 232.0, 233.0)),
        ),
        # Target power spectrum is trained in log10 space with an offset for stability.
        target_transform="log10",
        target_offset=1.0,
    )


# Parameter Preparation
# ---------------------
# Logic for processing raw simulation parameters into model features.

def prepare_twentyonecmspace_delta21_parameters(raw_parameters: np.ndarray) -> PreparedFeatures:
    """
    Prepare 21cmSPACE 12-parameter arrays for the `Delta21` emulator.

    The raw table contains 12 columns, but the workflow uses only nine of
    them. It drops `zeta`, `feed`, and `delay`, and logs the star-formation and
    radio-efficiency style parameters, while keeping `alpha`, `nu_0`, and
    `pop` available as explicitly discrete metadata.

    Parameters
    ----------
    raw_parameters:
        The raw 12-column parameter table loaded from the 21cmSPACE dataset.

    Returns
    -------
    PreparedFeatures
        The processed features ready for the neural network.
    """
    return prepare_feature_matrix(
        raw_parameters,
        TWENTYONECMSPACE_COLUMNS,
        # Apply log10 to specified astrophysical parameters.
        transform_params=("fstarII", "fstarIII", "Vc", "fX", "fradio"),
        # Discard parameters that are not used in the current Delta21 architecture.
        discard_params=("zeta", "feed", "delay"),
        # Mark parameters that were sampled from a fixed set of discrete values.
        discrete_params=("alpha", "nu_0", "pop"),
    )


# Data Preparation Pipeline
# -------------------------
# Orchestrates the full loading and transformation workflow for training.

def prepare_twentyonecmspace_delta21_training_split(
    dataset_root: str,
    *,
    random_state: int = 42,
    shuffle_seed: int = 42,
) -> PreparedSplit:
    """
    Prepare 21cmSPACE `Delta21` arrays on one shared `(z, k)` grid.

    Parameters
    ----------
    dataset_root:
        Path to the 21cmSPACE dataset files.
    random_state:
        Seed for splitting simulations into train/val/test sets.
    shuffle_seed:
        Seed for shuffling the final flattened rows.

    Returns
    -------
    PreparedSplit
        Bundled training, validation, and test datasets.
    """
    # Load the raw 21cmSPACE power-spectrum data.
    product = load_twentyonecmspace_delta21(dataset_root)
    # Process the raw parameter table into the transformed feature set.
    prepared_parameters = prepare_twentyonecmspace_delta21_parameters(product.parameters)
    # Get the Delta21 emulator contract.
    spec = delta21_spec()

    # Run the generic preparation workflow with Delta21-specific settings.
    return prepare_fixed_grid_training_split(
        axes=(product.axes.z, product.axes.k),
        axis_specs=spec.axes,
        parameters=prepared_parameters,
        target=product.target,
        # Configure scaling methods for each input feature (coordinates and parameters).
        feature_scale_methods={
            "z": "zscore",
            "log10k": "zscore",
            "log10fstarII": "zscore",
            "log10fstarIII": "zscore",
            "log10Vc": "zscore",
            "log10fX": "zscore",
            "alpha": "minmax_zero_to_one",
            "nu_0": "minmax_zero_to_one",
            "tau": "zscore",
            "log10fradio": "zscore",
            "pop": "minmax_zero_to_one",
        },
        # Delta21 signals are trained in log10 space with a stability offset.
        data_log=True,
        offset=1.0,
        train_size=0.6,
        validation_size=0.2,
        test_size=0.2,
        random_state=random_state,
        shuffle_seed=shuffle_seed,
        # Divide log-space targets by one global training-label std.
        standardize_target=True,
    )
