"""T21 emulator data contracts and preparation helpers.

This module owns the end of the workflow that turns raw 21cmSPACE global
signal arrays into the fixed-grid scalar regression problem used to train the
current T21 emulator.
"""

from __future__ import annotations

import numpy as np

from twentyonecmspace_emulators.utils.specs import AxisSpec, EmulatorSpec, ParameterSpec
from twentyonecmspace_emulators.data_preprocessing.twentyonecmspace import load_twentyonecmspace_t21
from twentyonecmspace_emulators.data_preprocessing.parameters import PreparedFeatures, prepare_feature_matrix
from twentyonecmspace_emulators.data_preprocessing.preparation import (
    PreparedSplit,
    prepare_fixed_grid_training_split,
)

TWENTYONECMSPACE_COLUMNS = (
    "fstarII",
    "fstarIII",
    "Vc",
    "fX",
    "alpha",
    "nu_0",
    "zeta",
    "tau",
    "fradio",
    "pop",
    "feed",
    "delay",
)

def t21_spec() -> EmulatorSpec:
    """Return the baseline 21cmSPACE T21 contract using ``fradio``.

    The network sees one redshift axis plus the transformed astrophysical
    parameters that control the global signal.
    """
    return EmulatorSpec(
        name="t21",
        family="global_signal",
        axes=(
            AxisSpec(name="z", transform="identity", limits=(6.0, 27.0), nsample=200),
        ),
        parameters=(
            ParameterSpec(name="fstarII", transform="log10"),
            ParameterSpec(name="fstarIII", transform="log10"),
            ParameterSpec(name="Vc", transform="log10"),
            ParameterSpec(name="fX", transform="log10"),
            ParameterSpec(name="alpha", discrete_values=(1.0, 1.3, 1.5)),
            ParameterSpec(
                name="nu_0",
                discrete_values=tuple(float(v) for v in [*range(100, 1600, 100), 2000, 3000]),
            ),
            ParameterSpec(name="tau"),
            ParameterSpec(name="fradio", transform="log10"),
            ParameterSpec(name="pop", discrete_values=(231.0, 232.0, 233.0)),
        ),
        target_transform="identity",
        target_offset=0.0,
    )
def prepare_twentyonecmspace_t21_parameters(raw_parameters: np.ndarray) -> PreparedFeatures:
    """Prepare 21cmSPACE parameter tables for the current T21 emulator.

    The helper applies the parameter filtering and log transforms used by the
    current T21 workflow so the resulting feature matrix is ready for training.
    """
    return prepare_feature_matrix(
        raw_parameters,
        TWENTYONECMSPACE_COLUMNS,
        transform_params=("fstarII", "fstarIII", "Vc", "fX", "fradio"),
        discard_params=("zeta", "feed", "delay"),
        discrete_params=("alpha", "nu_0", "pop"),
    )


def prepare_twentyonecmspace_t21_training_split(
    dataset_root: str,
    *,
    random_state: int = 42,
    shuffle_seed: int = 42,
) -> PreparedSplit:
    """Prepare 21cmSPACE `T21` arrays on one shared redshift grid."""
    product = load_twentyonecmspace_t21(dataset_root)
    prepared_parameters = prepare_twentyonecmspace_t21_parameters(product.parameters)
    spec = t21_spec()
    return prepare_fixed_grid_training_split(
        axes=(product.axes.z,),
        axis_specs=spec.axes,
        parameters=prepared_parameters,
        target=product.target,
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
        data_log=False,
        offset=None,
        train_size=0.6,
        validation_size=0.2,
        test_size=0.2,
        random_state=random_state,
        shuffle_seed=shuffle_seed,
        standardize_target=True,
    )
