"""Power-spectrum emulator specifications.

This module groups together two kinds of information:

- repository-native emulator contracts used by the new code
- legacy parameter-table preparation rules inherited from the old scripts

Keeping both in one place makes it obvious which legacy conventions each new
power-spectrum emulator is trying to reproduce.
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
            AxisSpec(name="k", transform="log10", limits=(3e-2, 0.99), nsample=20),
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
    `feed`, and `delay` before training.
    """
    return prepare_feature_matrix(
        raw_parameters,
        HERA_IDR4_COLUMNS,
        transform_params=("fstarII", "fstarIII", "Vc", "fX", "fradio"),
        discard_params=("zeta", "feed", "delay"),
        discrete_params=("alpha", "nu_0", "pop"),
    )


def prepare_sdc3b_parameters(raw_parameters: np.ndarray) -> PreparedFeatures:
    """Prepare SDC3b parameter arrays for the old SDC3b power emulator."""
    return prepare_feature_matrix(
        raw_parameters,
        SDC3B_COLUMNS,
        transform_params=(),
        discard_params=(),
        discrete_params=(),
    )
