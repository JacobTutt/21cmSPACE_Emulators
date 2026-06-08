"""
T21 emulator data contracts and preparation helpers.

This module owns the end of the workflow that turns raw 21cmSPACE global
signal arrays into the fixed-grid scalar regression problem used to train the
current T21 emulator. It defines:
- the baseline T21 emulator specification (axes and parameters)
- parameter processing logic (filtering and transforms)
- the full training data preparation pipeline
"""

from __future__ import annotations

import numpy as np

from jax_emu.data_preprocessing.specs import AxisSpec, EmulatorSpec, ParameterSpec
from examples_21cmspace.twentyonecmspace import (
    TWENTYONECMSPACE_COLUMNS,
    load_twentyonecmspace_t21,
)
from jax_emu.data_preprocessing.parameters import PreparedFeatures, prepare_feature_matrix
from jax_emu.data_preprocessing.preparation import (
    PreparedSplit,
    prepare_fixed_grid_training_split,
)


# Emulator Specification
# ----------------------
# Defines the input/output contract for the T21 (brightness temperature) model.

def t21_spec() -> EmulatorSpec:
    """
    Return the baseline 21cmSPACE T21 contract using ``fradio``.

    The network sees one redshift axis plus the transformed astrophysical
    parameters that control the global signal.

    Returns
    -------
    EmulatorSpec
        The contract defining the model input features and targets.
    """
    return EmulatorSpec(
        name="t21",
        family="global_signal",
        # The T21 emulator operates on a single redshift axis.
        axes=(
            AxisSpec(name="z", transform="identity", limits=(6.0, 27.0), nsample=200),
        ),
        # Astrophysical parameters identified as having a significant impact on T21.
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
        # Target brightness temperature is trained in linear physical space.
        target_transform="identity",
        target_offset=0.0,
    )


# Parameter Preparation
# ---------------------
# Logic for processing raw simulation parameters into model features.

def prepare_twentyonecmspace_t21_parameters(raw_parameters: np.ndarray) -> PreparedFeatures:
    """
    Prepare 21cmSPACE parameter tables for the current T21 emulator.

    The helper applies the parameter filtering and log transforms used by the
    current T21 workflow so the resulting feature matrix is ready for training.

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
        # Discard parameters that are not used in the current T21 architecture.
        discard_params=("zeta", "feed", "delay"),
        # Mark parameters that were sampled from a fixed set of discrete values.
        discrete_params=("alpha", "nu_0", "pop"),
    )


# Data Preparation Pipeline
# -------------------------
# Orchestrates the full loading and transformation workflow for training.

def prepare_twentyonecmspace_t21_training_split(
    dataset_root: str,
    *,
    random_state: int = 42,
    shuffle_seed: int = 42,
) -> PreparedSplit:
    """
    Prepare 21cmSPACE `T21` arrays on one shared redshift grid.

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
    # Load the raw 21cmSPACE brightness temperature data.
    product = load_twentyonecmspace_t21(dataset_root)
    # Process the raw parameter table into the transformed feature set.
    prepared_parameters = prepare_twentyonecmspace_t21_parameters(product.parameters)
    # Get the T21 emulator contract.
    spec = t21_spec()

    # Run the generic preparation workflow with T21-specific settings.
    return prepare_fixed_grid_training_split(
        axes=(product.axes.z,),
        axis_specs=spec.axes,
        parameters=prepared_parameters,
        target=product.target,
        # Configure scaling methods for each input feature.
        feature_scale_methods={
            "z": "zscore",
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
        # T21 signals are trained in linear space (no log transform on targets).
        data_log=False,
        offset=None,
        train_size=0.6,
        validation_size=0.2,
        test_size=0.2,
        random_state=random_state,
        shuffle_seed=shuffle_seed,
        interpolation_method="cubic",
        # Divide targets by one global training-label std, following globalemu.
        standardize_target=True,
    )
