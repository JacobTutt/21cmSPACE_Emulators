"""Global-signal emulator specifications.

The old repository used several distinct one-dimensional emulators under the
broader "global signal" label. This module makes those variants explicit so the
new code can migrate them one by one without losing track of their differing
parameterizations and target transforms.
"""

from __future__ import annotations

import numpy as np

from nenufar_emulators.core.legacy import PreparedFeatures, prepare_feature_matrix
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

HERA_COSMIC_STRING_ARAD_COLUMNS = (
    "fstarII",
    "fstarIII",
    "Vc",
    "fX",
    "alpha",
    "nu_0",
    "zeta",
    "tau",
    "Arad",
    "pop",
    "feed",
    "delay",
)

LEGACY_FRAD_COLUMNS = (
    "Rmfp",
    "fstar",
    "Vc",
    "fX",
    "alpha",
    "nu_0",
    "zeta",
    "tau",
    "fradio",
)


def default_global_signal_spec() -> EmulatorSpec:
    """Return the baseline HERA-style global-signal contract using ``fradio``.

    This is the closest equivalent to the standard modern global-signal setup
    in the old repository: one redshift axis plus the transformed astrophysical
    parameters used for ``T21``-like emulators.
    """
    return EmulatorSpec(
        name="t21_global_signal",
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
            ParameterSpec(name="nu_0", discrete_values=tuple(float(v) for v in np.arange(0.1, 1.6, 0.1))),
            ParameterSpec(name="tau"),
            ParameterSpec(name="fradio", transform="log10"),
            ParameterSpec(name="pop", discrete_values=(2.0, 3.0)),
        ),
        target_transform="identity",
        target_offset=0.0,
    )


def t21_arad_spec() -> EmulatorSpec:
    """Return the old `T21`/`Ts` style Arad global-signal emulator contract.

    `T21` and `Ts` share the same input parameterization in the legacy code but
    differ in their target transforms and sampling density.
    """
    return EmulatorSpec(
        name="t21_arad_global_signal",
        family="global_signal",
        axes=(AxisSpec(name="z", limits=(6.0, 27.0), nsample=200),),
        parameters=(
            ParameterSpec(name="fstarII", transform="log10"),
            ParameterSpec(name="fstarIII", transform="log10"),
            ParameterSpec(name="Vc", transform="log10"),
            ParameterSpec(name="fX", transform="log10"),
            ParameterSpec(name="alpha", discrete_values=(1.0, 1.3, 1.5)),
            ParameterSpec(name="nu_0", discrete_values=tuple(float(v) for v in np.arange(0.1, 1.6, 0.1))),
            ParameterSpec(name="tau"),
            ParameterSpec(name="Arad", transform="log10"),
            ParameterSpec(name="pop", discrete_values=(2.0, 3.0)),
        ),
        target_transform="identity",
    )


def ts_arad_spec() -> EmulatorSpec:
    """Return the old `Ts` Arad emulator contract.

    The main practical difference from ``T21`` is that the target was trained
    in ``log10`` space and sampled on a denser redshift grid.
    """
    return EmulatorSpec(
        name="ts_arad_global_signal",
        family="global_signal",
        axes=(AxisSpec(name="z", limits=(6.0, 27.0), nsample=400),),
        parameters=t21_arad_spec().parameters,
        target_transform="log10",
        target_offset=1.0,
    )


def trad_frad_spec() -> EmulatorSpec:
    """Return the old ``Trad`` emulator contract using ``fradio`` inputs."""
    return EmulatorSpec(
        name="trad_frad_global_signal",
        family="global_signal",
        axes=(AxisSpec(name="z", limits=(6.0, 27.0), nsample=400),),
        parameters=default_global_signal_spec().parameters,
        target_transform="log10",
        target_offset=1.0,
    )


def tk_frad_spec() -> EmulatorSpec:
    """Return the old `TK` frad global-signal emulator contract.

    `TK` is the main legacy oddball in this family because it uses the older
    9-parameter frad table rather than the newer 12-parameter HERA table.
    """
    return EmulatorSpec(
        name="tk_frad_global_signal",
        family="global_signal",
        axes=(AxisSpec(name="z", limits=(6.0, 27.0), nsample=200),),
        parameters=(
            ParameterSpec(name="Rmfp"),
            ParameterSpec(name="fstar", transform="log10"),
            ParameterSpec(name="Vc", transform="log10"),
            ParameterSpec(name="fX", transform="log10"),
            ParameterSpec(name="alpha", discrete_values=(1.0, 1.3, 1.5)),
            ParameterSpec(name="nu_0", discrete_values=tuple(float(v) for v in np.arange(0.1, 1.6, 0.1))),
            ParameterSpec(name="tau"),
            ParameterSpec(name="fradio", transform="log10"),
        ),
        target_transform="log10",
    )


def prepare_hera_idr4_frad_parameters(raw_parameters: np.ndarray) -> PreparedFeatures:
    """Prepare HERA IDR4 tables for ``Trad`` and ``T_today`` style emulators.

    This applies the same feature dropping and log transforms used by the old
    scripts so that later JAX training code sees the same effective inputs.
    """
    return prepare_feature_matrix(
        raw_parameters,
        HERA_IDR4_COLUMNS,
        transform_params=("fstarII", "fstarIII", "Vc", "fX", "fradio"),
        discard_params=("zeta", "feed", "delay"),
        discrete_params=("alpha", "nu_0", "pop"),
    )


def prepare_hera_cosmic_string_arad_parameters(raw_parameters: np.ndarray) -> PreparedFeatures:
    """Prepare HERA cosmic-string Arad arrays for T21/Ts-style emulators.

    The legacy scripts used `Arad` here instead of `fradio`, so the transformed
    feature name becomes `log10Arad` rather than `log10fradio`.
    """
    return prepare_feature_matrix(
        raw_parameters,
        HERA_COSMIC_STRING_ARAD_COLUMNS,
        transform_params=("fstarII", "fstarIII", "Vc", "fX", "Arad"),
        discard_params=("zeta", "feed", "delay"),
        discrete_params=("alpha", "nu_0", "pop"),
    )


def prepare_legacy_frad_parameters(raw_parameters: np.ndarray) -> PreparedFeatures:
    """Prepare the older 9-parameter ``fradio`` tables for ``TK`` emulators.

    This path exists because ``TK`` did not use the newer 12-column HERA table
    that the other migrated global-signal emulators expect.
    """
    return prepare_feature_matrix(
        raw_parameters,
        LEGACY_FRAD_COLUMNS,
        transform_params=("fstar", "Vc", "fX", "fradio"),
        discard_params=("zeta",),
        discrete_params=("alpha", "nu_0"),
    )
