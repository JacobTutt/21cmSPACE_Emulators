"""T21 emulator data contracts and preparation helpers.

This module owns the end of the workflow that turns raw HERA IDR4 global
signal arrays into the fixed-grid scalar regression problem used to train the
current T21 emulator.
"""

from __future__ import annotations

import numpy as np

from nenufar_emulators.core.datasets import NormalisationPipeline, SpectrumDataset
from nenufar_emulators.core.normalisation import SpecTransformPipeline
from nenufar_emulators.core.specs import AxisSpec, EmulatorSpec, ParameterSpec
from nenufar_emulators.data.hera_idr4 import load_hera_idr4_t21
from nenufar_emulators.conventions import PreparedFeatures, prepare_feature_matrix
from nenufar_emulators.data.preparation import (
    PreparedSplit,
    prepare_shared_grid_training_split,
)

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

def t21_spec() -> EmulatorSpec:
    """Return the baseline HERA IDR4 T21 contract using ``fradio``.

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
def prepare_hera_idr4_t21_parameters(raw_parameters: np.ndarray) -> PreparedFeatures:
    """Prepare HERA IDR4 parameter tables for the current T21 emulator.

    The helper applies the parameter filtering and log transforms used by the
    current T21 workflow so the resulting feature matrix is ready for training.
    """
    return prepare_feature_matrix(
        raw_parameters,
        HERA_IDR4_COLUMNS,
        transform_params=("fstarII", "fstarIII", "Vc", "fX", "fradio"),
        discard_params=("zeta", "feed", "delay"),
        discrete_params=("alpha", "nu_0", "pop"),
    )


def prepare_hera_idr4_t21_training_split(
    dataset_root: str,
    *,
    random_state: int = 42,
    shuffle_seed: int = 42,
) -> PreparedSplit:
    """Prepare HERA IDR4 `T21` arrays using the fixed-grid T21 workflow.

    Unlike the power-spectrum emulator, `T21` is not prepared through
    per-simulation random interpolation. We split by simulation, resample all
    signals onto one shared redshift grid declared by the emulator spec, and
    then flatten those deterministic rows for training.
    """
    product = load_hera_idr4_t21(dataset_root)
    prepared_parameters = prepare_hera_idr4_t21_parameters(product.parameters)
    spec = t21_spec()
    return prepare_shared_grid_training_split(
        axes=(product.axes.z,),
        axis_specs=spec.axes,
        parameters=prepared_parameters,
        target=product.target,
        scale_method={"tau": "normalize"},
        data_log=False,
        offset=None,
        train_size=0.66,
        test_size=0.34,
        random_state=random_state,
        shuffle_seed=shuffle_seed,
    )
def build_t21_dataset(
    spectra: np.ndarray,
    axes: tuple[np.ndarray, ...],
    parameters: PreparedFeatures | np.ndarray,
    *,
    spec: EmulatorSpec | None = None,
    parameter_names: tuple[str, ...] | None = None,
    forward_pipeline: NormalisationPipeline | list[NormalisationPipeline] | None = None,
    tiling: bool = True,
) -> SpectrumDataset:
    """Build a global-signal dataset using the declared emulator contract.

    As with the power-spectrum helper, the spec-driven transform pipeline is
    attached by default so axis, parameter, and target conventions stay aligned
    with the workflow definition used throughout this repository.
    """
    emulator_spec = t21_spec() if spec is None else spec
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
