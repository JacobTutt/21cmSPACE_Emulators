"""Delta21 emulator data contracts and preparation helpers.

This module turns raw HERA IDR4 arrays into the scalar regression problem used
to train the current Delta21 emulator. It defines the input contract, the
parameter preparation rules, and the row-building workflow in one place so the
full training setup is easy to inspect.
"""

from __future__ import annotations

import numpy as np

from nenufar_emulators.core.datasets import NormalisationPipeline, SpectrumDataset
from nenufar_emulators.core.normalisation import SpecTransformPipeline
from nenufar_emulators.core.specs import AxisSpec, EmulatorSpec, ParameterSpec
from nenufar_emulators.data.hera_idr4 import HERA_LITTLE_H, load_hera_idr4_delta21
from nenufar_emulators.conventions import PreparedFeatures, prepare_feature_matrix
from nenufar_emulators.data.preparation import PreparedSplit, prepare_interpolated_training_split

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
    interpolation_seed: int = 0,
) -> PreparedSplit:
    """Prepare HERA IDR4 `Delta21` arrays for model fitting.

    The resulting rows follow the interpolated sampling workflow used for the
    current Delta21 emulator: split simulations first, draw training samples in
    transformed axis space, and keep validation on a deterministic cropped
    grid.
    """
    product = load_hera_idr4_delta21(dataset_root)
    prepared_parameters = prepare_hera_idr4_delta21_parameters(product.parameters)
    spec = delta21_spec()
    return prepare_interpolated_training_split(
        axes=(product.axes.z, product.axes.k),
        axis_specs=spec.axes,
        parameters=prepared_parameters,
        target=product.target,
        scale_method={"tau": "normalize"},
        data_log=True,
        offset=1.0,
        random_state=random_state,
        interpolation_seed=interpolation_seed,
    )
def build_delta21_dataset(
    spectra: np.ndarray,
    axes: tuple[np.ndarray, ...],
    parameters: PreparedFeatures | np.ndarray,
    *,
    spec: EmulatorSpec | None = None,
    parameter_names: tuple[str, ...] | None = None,
    forward_pipeline: NormalisationPipeline | list[NormalisationPipeline] | None = None,
    tiling: bool = True,
) -> SpectrumDataset:
    """Build a power-spectrum dataset using the declared workflow contract.

    The dataset always includes a :class:`SpecTransformPipeline` so axis and
    target transforms follow the declared emulator spec. Parameter transforms
    are also applied if raw parameter names are supplied rather than a
    pre-transformed :class:`PreparedFeatures` object.
    """
    emulator_spec = delta21_spec() if spec is None else spec
    if isinstance(parameters, PreparedFeatures):
        parameter_values = parameters.values
        parameter_names = parameters.feature_names
    else:
        if parameter_names is None:
            parameter_names = emulator_spec.parameter_names()
        parameter_values = np.asarray(parameters, dtype=float)

    pipelines: list[NormalisationPipeline] = [SpecTransformPipeline(emulator_spec)]
    if forward_pipeline is not None:
        pipelines.extend(
            forward_pipeline if isinstance(forward_pipeline, list) else [forward_pipeline]
        )
    return SpectrumDataset(
        spectra=np.asarray(spectra, dtype=float),
        axes=tuple(np.asarray(axis, dtype=float) for axis in axes),
        parameters=np.asarray(parameter_values, dtype=float),
        axis_names=emulator_spec.axis_names(),
        parameter_names=parameter_names,
        forward_pipeline=pipelines,
        tiling=tiling,
    )
