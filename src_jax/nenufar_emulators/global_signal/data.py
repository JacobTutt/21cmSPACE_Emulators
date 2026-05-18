"""Global-signal emulator specifications."""

from __future__ import annotations

from nenufar_emulators.core.specs import AxisSpec, EmulatorSpec, ParameterSpec


def default_global_signal_spec() -> EmulatorSpec:
    """Return the baseline global-signal emulator contract."""
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
            ParameterSpec(name="nu_0"),
            ParameterSpec(name="tau"),
            ParameterSpec(name="fradio", transform="log10"),
            ParameterSpec(name="pop"),
        ),
        target_transform="identity",
        target_offset=0.0,
    )
