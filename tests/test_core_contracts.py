"""Tests for core metadata and transform contracts."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from nenufar_emulators.utils.checkpointing import CheckpointMetadata, load, save
from nenufar_emulators.utils.scaling import FeatureScaler, FeatureScaling, TargetScalingSurface
from nenufar_emulators.utils.specs import AxisSpec, EmulatorSpec, ParameterSpec
from nenufar_emulators.utils.transforms import apply_transform, invert_transform
from nenufar_emulators.architectures.mlp import init_mlp
from nenufar_emulators.emulators.t21.data import t21_spec


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
        target_scaling=None,
        training_config={"epochs": 5},
    )
    serialized = metadata.to_dict()
    assert serialized["model_name"] == "t21-test"
    assert serialized["emulator_spec"]["name"] == "t21"
    assert len(serialized["input_scaling"]) == 2


def test_checkpoint_package_round_trip(tmp_path) -> None:
    spec = t21_spec()
    spectra = np.array(
        [
            np.linspace(-1.0, 1.0, 5),
            np.linspace(-0.5, 1.5, 5),
        ]
    )
    z = np.linspace(6.0, 10.0, 5)
    metadata = CheckpointMetadata(
        model_name="t21-test",
        package_version="0.1.0",
        emulator_spec=spec,
        input_scaling=(
            FeatureScaling.from_values("z", z, "identity"),
            FeatureScaling.from_values("log10fstarII", np.array([-2.0, -1.0]), "zscore"),
        ),
        target_scaling=TargetScalingSurface.from_targets(
            axis_names=("z",),
            axis_values=(z,),
            targets=spectra,
        ),
        training_config={"epochs": 2},
    )
    model = init_mlp(
        jax.random.PRNGKey(0),
        in_features=len(spec.input_feature_names()),
        hidden_features=8,
        hidden_layers=2,
    )
    features = jnp.asarray(
        [
            [0.0, -2.0, -3.0, 1.0, 2.0, 1.0, 100.0, 0.05, 2.0, 231.0],
            [1.0, -1.0, -2.0, 1.3, 3.0, 1.3, 200.0, 0.06, 3.0, 233.0],
        ],
        dtype=jnp.float32,
    )
    expected = model(features)
    package_path = save(
        tmp_path / "demo_model",
        model,
        train_losses=[1.0, 0.5],
        val_losses=[1.2, 0.6],
        loss="mse",
        metadata=metadata,
        epochs=2,
        patience=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
    )
    loaded = load(package_path)
    recovered = loaded["model"](features)

    assert package_path.name.endswith(".nenemu")
    assert np.allclose(np.asarray(recovered), np.asarray(expected))
    assert loaded["metadata"].model_name == "t21-test"
    assert loaded["hyperparams"]["hidden_features"] == 8
    assert loaded["metadata"].target_scaling is not None
