"""Power-spectrum emulator specifications.

This module groups together two kinds of information:

- repository-native emulator contracts used by the new code
- legacy parameter-table preparation rules inherited from the old scripts

Keeping both in one place makes it obvious which legacy conventions each new
power-spectrum emulator is trying to reproduce.
"""

from __future__ import annotations

import numpy as np

from nenufar_emulators.core.datasets import NormalisationPipeline, SpectrumDataset
from nenufar_emulators.core.hera_idr4 import HERA_LITTLE_H, load_hera_idr4_delta21
from nenufar_emulators.core.legacy import PreparedFeatures, prepare_feature_matrix
from nenufar_emulators.core.legacy_workflow import LegacyPreparedSplit, prepare_legacy_training_split
from nenufar_emulators.core.normalisation import SpecTransformPipeline
from nenufar_emulators.core.specs import AxisSpec, EmulatorSpec, ParameterSpec

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

SDC3B_COLUMNS = ("zeta_eff", "zeta_exp", "rmfp", "Vc")


def default_power_spectrum_spec() -> EmulatorSpec:
    """Return the baseline HERA-style power-spectrum emulator contract.

    This mirrors the old `Delta21` setup: two tiled axes (`z`, `k`) plus nine
    astrophysical parameters after dropping unused columns and applying the
    legacy log transforms.
    """
    return EmulatorSpec(
        name="delta21_power_spectrum",
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


def sdc3b_power_spectrum_spec() -> EmulatorSpec:
    """Return the baseline SDC3b power-spectrum emulator contract.

    The SDC3b emulator uses a different scientific parameterization and a
    three-axis power-spectrum target, so its spec is kept separate from the
    HERA-style Delta21 path.
    """
    return EmulatorSpec(
        name="sdc3b_power_spectrum",
        family="power_spectrum",
        axes=(
            AxisSpec(name="z", limits=(6.1036, 8.4044), nsample=20),
            AxisSpec(name="kperp", transform="log10", limits=(5e-2, 5e-1), nsample=20),
            AxisSpec(name="kpar", transform="log10", limits=(5e-2, 5e-1), nsample=20),
        ),
        parameters=(
            ParameterSpec(name="zeta_eff"),
            ParameterSpec(name="zeta_exp"),
            ParameterSpec(name="rmfp"),
            ParameterSpec(name="Vc"),
        ),
    )


def prepare_hera_idr4_delta21_parameters(raw_parameters: np.ndarray) -> PreparedFeatures:
    """Prepare HERA IDR4 12-parameter arrays for the old `Delta21` emulator.

    The raw table contains 12 columns, but the legacy emulator dropped `zeta`,
    `feed`, and `delay` before training. It also logged the star-formation and
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
) -> LegacyPreparedSplit:
    """Prepare HERA IDR4 `Delta21` arrays using the old training recipe.

    This reproduces the scientifically important steps from the PyTorch
    pipeline while intentionally swapping back to the HERA IDR4 `k` axis rather
    than the mixed cosmic-string axis line that was uncommented in one legacy
    script revision.
    """
    product = load_hera_idr4_delta21(dataset_root)
    prepared_parameters = prepare_hera_idr4_delta21_parameters(product.parameters)
    spec = default_power_spectrum_spec()
    return prepare_legacy_training_split(
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


def prepare_sdc3b_parameters(raw_parameters: np.ndarray) -> PreparedFeatures:
    """Prepare raw SDC3b parameters for the SDC3b power-spectrum emulator.

    This path is simpler than the HERA one because the legacy SDC3b setup did
    not discard columns or apply log transforms at this stage.
    """
    return prepare_feature_matrix(
        raw_parameters,
        SDC3B_COLUMNS,
        transform_params=(),
        discard_params=(),
        discrete_params=(),
    )


def build_power_spectrum_dataset(
    spectra: np.ndarray,
    axes: tuple[np.ndarray, ...],
    parameters: PreparedFeatures | np.ndarray,
    *,
    spec: EmulatorSpec | None = None,
    parameter_names: tuple[str, ...] | None = None,
    forward_pipeline: NormalisationPipeline | list[NormalisationPipeline] | None = None,
    tiling: bool = True,
) -> SpectrumDataset:
    """Build a power-spectrum dataset using old-code transform conventions.

    The dataset always includes a :class:`SpecTransformPipeline` so axis and
    target transforms follow the declared emulator spec. Parameter transforms
    are also applied if raw parameter names are supplied rather than a
    pre-transformed :class:`PreparedFeatures` object.
    """
    emulator_spec = default_power_spectrum_spec() if spec is None else spec
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
