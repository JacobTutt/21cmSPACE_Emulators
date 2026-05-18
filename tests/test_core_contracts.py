"""Tests for core metadata and transform contracts."""

from __future__ import annotations

import numpy as np

from nenufar_emulators.core.checkpointing import CheckpointMetadata
from nenufar_emulators.core.scaling import FeatureScaler, FeatureScaling
from nenufar_emulators.core.specs import AxisSpec, EmulatorSpec, ParameterSpec
from nenufar_emulators.core.transforms import apply_transform, invert_transform


def test_transform_round_trip_log10() -> None:
    values = np.array([1.0, 10.0, 100.0])
    transformed = apply_transform(values, "log10")
    recovered = invert_transform(transformed, "log10")
    assert np.allclose(recovered, values)


def test_emulator_spec_feature_order() -> None:
    spec = EmulatorSpec(
        name="delta21",
        family="power_spectrum",
        axes=(AxisSpec(name="z"), AxisSpec(name="k", transform="log10")),
        parameters=(
            ParameterSpec(name="fstarII", transform="log10"),
            ParameterSpec(name="alpha"),
        ),
        target_transform="log10",
        target_offset=1.0,
    )
    assert spec.input_feature_names() == ("z", "log10k", "log10fstarII", "alpha")


def test_feature_scaler_round_trip() -> None:
    scaling = (
        FeatureScaling.from_values("z", np.array([6.0, 10.0, 20.0]), "minmax_minus_one_to_one"),
        FeatureScaling.from_values("tau", np.array([0.04, 0.05, 0.06]), "zscore"),
    )
    scaler = FeatureScaler(scaling)
    matrix = np.array([[6.0, 0.04], [20.0, 0.06]])
    recovered = scaler.inverse_transform(scaler.transform(matrix))
    assert np.allclose(recovered, matrix)


def test_checkpoint_metadata_to_dict() -> None:
    spec = EmulatorSpec(
        name="t21",
        family="global_signal",
        axes=(AxisSpec(name="z"),),
        parameters=(ParameterSpec(name="fradio", transform="log10"),),
    )
    metadata = CheckpointMetadata(
        model_name="t21-test",
        package_version="0.1.0",
        emulator_spec=spec,
        input_scaling=(
            FeatureScaling.from_values("z", np.array([6.0, 20.0]), "identity"),
            FeatureScaling.from_values("log10fradio", np.array([1.0, 2.0]), "zscore"),
        ),
        training_config={"epochs": 5},
    )
    serialized = metadata.to_dict()
    assert serialized["model_name"] == "t21-test"
    assert serialized["emulator_spec"]["name"] == "t21"
    assert len(serialized["input_scaling"]) == 2
