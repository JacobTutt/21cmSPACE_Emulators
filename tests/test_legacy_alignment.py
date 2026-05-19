"""Tests for old-code parameter and config alignment."""

from __future__ import annotations

import numpy as np

from nenufar_emulators.global_signal.data import (
    prepare_hera_cosmic_string_arad_parameters,
    prepare_legacy_frad_parameters,
    t21_arad_spec,
    tk_frad_spec,
    trad_frad_spec,
    ts_arad_spec,
)
from nenufar_emulators.global_signal.model import (
    t21_arad_legacy_bundle,
    t21_frad_legacy_bundle,
    t_today_frad_legacy_bundle,
    tk_frad_legacy_bundle,
    trad_frad_legacy_bundle,
    ts_arad_legacy_bundle,
)
from nenufar_emulators.power_spectrum.data import (
    default_power_spectrum_spec,
    prepare_hera_idr4_delta21_parameters,
    sdc3b_power_spectrum_spec,
)
from nenufar_emulators.power_spectrum.model import (
    delta21_frad_legacy_bundle,
    sdc3b_pk_legacy_bundle,
)


def test_delta21_spec_matches_old_feature_count_and_discrete_params() -> None:
    spec = default_power_spectrum_spec()
    assert len(spec.input_feature_names()) == 11
    assert spec.parameters[4].name == "alpha"
    assert spec.parameters[5].name == "nu_0"
    assert spec.parameters[8].name == "pop"
    assert spec.parameters[5].discrete_values[-2:] == (2000.0, 3000.0)
    assert spec.parameters[8].discrete_values == (231.0, 232.0, 233.0)
    assert spec.target_transform == "log10"
    assert spec.target_offset == 1.0


def test_prepare_hera_idr4_delta21_parameters_matches_old_transform_names() -> None:
    raw = np.array(
        [
            [1e-2, 1e-3, 10.0, 100.0, 1.0, 0.1, 30.0, 0.05, 1e2, 2.0, 0.0, 0.0],
            [1e-1, 1e-2, 20.0, 1000.0, 1.3, 0.2, 30.0, 0.06, 1e3, 3.0, 0.0, 0.0],
        ]
    )
    prepared = prepare_hera_idr4_delta21_parameters(raw)
    assert prepared.feature_names == (
        "log10fstarII",
        "log10fstarIII",
        "log10Vc",
        "log10fX",
        "alpha",
        "nu_0",
        "tau",
        "log10fradio",
        "pop",
    )
    assert "alpha" in prepared.discrete_values
    assert "nu_0" in prepared.discrete_values
    assert "pop" in prepared.discrete_values


def test_prepare_legacy_frad_parameters_matches_old_tk_contract() -> None:
    raw = np.array(
        [
            [30.0, 1e-2, 10.0, 100.0, 1.0, 0.1, 30.0, 0.05, 1e2],
            [40.0, 1e-1, 20.0, 1000.0, 1.3, 0.2, 30.0, 0.06, 1e3],
        ]
    )
    prepared = prepare_legacy_frad_parameters(raw)
    assert prepared.feature_names == (
        "Rmfp",
        "log10fstar",
        "log10Vc",
        "log10fX",
        "alpha",
        "nu_0",
        "tau",
        "log10fradio",
    )


def test_legacy_bundles_match_old_training_defaults() -> None:
    assert delta21_frad_legacy_bundle().mlp.input_dim == 11
    assert delta21_frad_legacy_bundle().training.batch_size == 20000
    assert sdc3b_pk_legacy_bundle().mlp.input_dim == 7
    assert t21_arad_legacy_bundle().training.save_after_epochs == 5
    assert t21_frad_legacy_bundle().training.save_after_epochs == 5
    assert ts_arad_legacy_bundle().training.save_after_epochs == 2
    assert tk_frad_legacy_bundle().mlp.input_dim == 9
    assert trad_frad_legacy_bundle().training.save_after_epochs == 5
    assert t_today_frad_legacy_bundle().training.save_after_epochs == 50


def test_global_specs_capture_old_target_transform_choices() -> None:
    assert t21_arad_spec().target_transform == "identity"
    assert ts_arad_spec().target_transform == "log10"
    assert ts_arad_spec().target_offset == 1.0
    assert trad_frad_spec().target_transform == "log10"
    assert tk_frad_spec().target_transform == "log10"


def test_sdc3b_spec_matches_old_input_dimension() -> None:
    assert len(sdc3b_power_spectrum_spec().input_feature_names()) == 7


def test_prepare_hera_cosmic_string_arad_parameters_uses_arad_feature_name() -> None:
    raw = np.array(
        [
            [1e-2, 1e-3, 10.0, 100.0, 1.0, 0.1, 30.0, 0.05, 1e2, 2.0, 0.0, 0.0],
        ]
    )
    prepared = prepare_hera_cosmic_string_arad_parameters(raw)
    assert "log10Arad" in prepared.feature_names
