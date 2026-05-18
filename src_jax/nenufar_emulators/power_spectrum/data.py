"""Power-spectrum emulator specifications."""

from __future__ import annotations

from nenufar_emulators.core.specs import AxisSpec, EmulatorSpec, ParameterSpec


def default_power_spectrum_spec() -> EmulatorSpec:
    """Return the baseline power-spectrum emulator contract."""
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
            ParameterSpec(name="nu_0"),
            ParameterSpec(name="tau"),
            ParameterSpec(name="fradio", transform="log10"),
            ParameterSpec(name="pop"),
        ),
        target_transform="log10",
        target_offset=1.0,
    )
