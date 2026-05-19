"""Tests for Delta21 and T21 legacy-derived workflow contracts."""

from __future__ import annotations

import numpy as np

from nenufar_emulators.delta21.data import delta21_spec, prepare_hera_idr4_delta21_parameters
from nenufar_emulators.delta21.model import delta21_config
from nenufar_emulators.t21.data import prepare_hera_idr4_t21_parameters, t21_spec
from nenufar_emulators.t21.model import t21_config


def test_delta21_spec_matches_expected_feature_contract() -> None:
    spec = delta21_spec()
    assert len(spec.input_feature_names()) == 11
    assert spec.parameters[4].name == "alpha"
    assert spec.parameters[5].name == "nu_0"
    assert spec.parameters[8].name == "pop"
    assert spec.parameters[5].discrete_values[-2:] == (2000.0, 3000.0)
    assert spec.parameters[8].discrete_values == (231.0, 232.0, 233.0)
    assert spec.target_transform == "log10"
    assert spec.target_offset == 1.0


def test_t21_spec_matches_expected_feature_contract() -> None:
    spec = t21_spec()
    assert len(spec.input_feature_names()) == 10
    assert spec.parameters[4].name == "alpha"
    assert spec.parameters[5].name == "nu_0"
    assert spec.parameters[8].name == "pop"
    assert spec.target_transform == "identity"
    assert spec.target_offset == 0.0


def test_prepare_hera_idr4_delta21_parameters_matches_old_transform_names() -> None:
    raw = np.array(
        [
            [1e-2, 1e-3, 10.0, 100.0, 1.0, 100.0, 30.0, 0.05, 1e2, 231.0, 0.0, 0.0],
            [1e-1, 1e-2, 20.0, 1000.0, 1.3, 200.0, 30.0, 0.06, 1e3, 233.0, 0.0, 0.0],
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


def test_prepare_hera_idr4_t21_parameters_matches_old_transform_names() -> None:
    raw = np.array(
        [
            [1e-2, 1e-3, 10.0, 100.0, 1.0, 100.0, 30.0, 0.05, 1e2, 231.0, 0.0, 0.0],
            [1e-1, 1e-2, 20.0, 1000.0, 1.3, 200.0, 30.0, 0.06, 1e3, 233.0, 0.0, 0.0],
        ]
    )
    prepared = prepare_hera_idr4_t21_parameters(raw)
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


def test_delta21_and_t21_configs_match_current_architecture_choices() -> None:
    delta21 = delta21_config()
    t21 = t21_config()

    assert delta21.mlp.input_dim == 11
    assert delta21.mlp.total_hidden_layers == 4
    assert delta21.mlp.hidden_dim == 100
    assert delta21.mlp.activation == "relu"
    assert delta21.training.batch_size == 20000

    assert t21.mlp.input_dim == 10
    assert t21.mlp.total_hidden_layers == 4
    assert t21.mlp.hidden_dim == 20
    assert t21.mlp.activation == "tanh"
    assert t21.training.batch_size == 769
    assert t21.training.epochs == 1000
    assert t21.training.early_stop is True
    assert t21.training.early_stopping_patience == 50
