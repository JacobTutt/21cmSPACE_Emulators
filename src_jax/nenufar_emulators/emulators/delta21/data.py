"""Delta21 emulator data contracts and preparation helpers.

This module turns raw HERA IDR4 arrays into the scalar regression problem used
to train the current Delta21 emulator. It defines the input contract, the
parameter preparation rules, and the row-building workflow in one place so the
full training setup is easy to inspect.
"""

from __future__ import annotations

import numpy as np

from nenufar_emulators.utils.specs import AxisSpec, EmulatorSpec, ParameterSpec
from nenufar_emulators.data_preprocessing.hera_idr4 import HERA_LITTLE_H, load_hera_idr4_delta21
from nenufar_emulators.data_preprocessing.parameters import PreparedFeatures, prepare_feature_matrix
from nenufar_emulators.data_preprocessing.preparation import PreparedSplit, prepare_fixed_grid_training_split

HERA_IDR4_COLUMNS = (
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

def delta21_spec() -> EmulatorSpec:
    """Return the baseline HERA IDR4 Delta21 emulator contract.

    This uses the established `Delta21` setup: two tiled axes (`z`, `k`) plus nine
    astrophysical parameters after dropping unused columns and applying the
    workflow transforms.
    """
    return EmulatorSpec(
        name="delta21",
        family="power_spectrum",
        axes=(
            AxisSpec(name="z", transform="identity", limits=(6.0, 27.0), nsample=20),
            AxisSpec(
                name="k",
                transform="log10",
                limits=(3e-2 / HERA_LITTLE_H, 0.99 / HERA_LITTLE_H),
                nsample=20,
            ),
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
        target_transform="log10",
        target_offset=1.0,
    )
def prepare_hera_idr4_delta21_parameters(raw_parameters: np.ndarray) -> PreparedFeatures:
    """Prepare HERA IDR4 12-parameter arrays for the `Delta21` emulator.

    The raw table contains 12 columns, but the workflow uses only nine of
    them. It drops `zeta`, `feed`, and `delay`, and logs the star-formation and
    radio-efficiency style parameters, while keeping `alpha`, `nu_0`, and
    `pop` available as explicitly discrete metadata.
    """
    return prepare_feature_matrix(
        raw_parameters,
        HERA_IDR4_COLUMNS,
        transform_params=("fstarII", "fstarIII", "Vc", "fX", "fradio"),
        discard_params=("zeta", "feed", "delay"),
        discrete_params=("alpha", "nu_0", "pop"),
    )


def prepare_hera_idr4_delta21_training_split(
    dataset_root: str,
    *,
    random_state: int = 42,
    shuffle_seed: int = 42,
) -> PreparedSplit:
    """Prepare HERA IDR4 `Delta21` arrays on one shared `(z, k)` grid."""
    product = load_hera_idr4_delta21(dataset_root)
    prepared_parameters = prepare_hera_idr4_delta21_parameters(product.parameters)
    spec = delta21_spec()
    return prepare_fixed_grid_training_split(
        axes=(product.axes.z, product.axes.k),
        axis_specs=spec.axes,
        parameters=prepared_parameters,
        target=product.target,
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
        data_log=True,
        offset=1.0,
        train_size=0.6,
        validation_size=0.2,
        test_size=0.2,
        random_state=random_state,
        shuffle_seed=shuffle_seed,
        standardize_target=True,
    )
