"""Tests for core metadata and transform contracts."""

from __future__ import annotations

import json

import numpy as np
import jax
import jax.numpy as jnp

from jax_emu.analysis import loss_curves_from_history, loss_curves_from_package
from jax_emu.utils.checkpointing import CheckpointMetadata, load, save
from jax_emu.data_preprocessing.scaling import (
    FeatureScaler,
    FeatureScaling,
    TargetScalingScalar,
)
from jax_emu.data_preprocessing.specs import AxisSpec, EmulatorSpec, ParameterSpec
from jax_emu.data_preprocessing.transforms import apply_transform, invert_transform
from jax_emu.training import TrainingHistory, build_learning_rate_schedule
from flax import nnx

from jax_emu.architectures.mlp import DenseMLP
from examples_21cmspace.t21.data import t21_spec


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


def test_target_scaling_scalar_round_trip() -> None:
    targets = np.array([[-2.0, 0.0, 2.0], [1.0, 3.0, 5.0]])
    scaler = TargetScalingScalar.from_targets(targets)
    recovered = scaler.inverse_grid(scaler.transform_grid(targets))

    assert scaler.std > 0
    assert np.allclose(recovered, targets)


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
        target_scaling=TargetScalingScalar.from_targets(spectra),
        training_config={"epochs": 2},
    )
    model = DenseMLP(
        in_features=len(spec.input_feature_names()),
        hidden_features=8,
        hidden_layers=2,
        rngs=nnx.Rngs(jax.random.PRNGKey(0)),
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
    assert package_path.is_dir()
    assert (package_path / "0" / "state").exists()
    assert (package_path / "0" / "config").exists()
    assert np.allclose(np.asarray(recovered), np.asarray(expected))
    assert loaded["metadata"].model_name == "t21-test"
    assert loaded["hyperparams"]["hidden_features"] == 8
    assert loaded["metadata"].target_scaling is not None


def test_loss_curves_from_history() -> None:
    history = TrainingHistory(
        train_losses=[2.0, 1.0],
        validation_losses=[2.5, 1.5],
        best_epoch=1,
        best_validation_loss=1.5,
    )
    curves = loss_curves_from_history(history, test_loss=1.25, model_name="demo")

    assert curves.train_losses == [2.0, 1.0]
    assert curves.validation_losses == [2.5, 1.5]
    assert curves.test_loss == 1.25
    assert curves.best_epoch == 1
    assert curves.best_validation_loss == 1.5
    assert curves.model_name == "demo"


def test_loss_curves_from_package_reads_adjacent_summary(tmp_path) -> None:
    spec = t21_spec()
    metadata = CheckpointMetadata(
        model_name="t21-loss-demo",
        package_version="0.1.0",
        emulator_spec=spec,
        input_scaling=(
            FeatureScaling.from_values("z", np.array([6.0, 20.0]), "identity"),
            FeatureScaling.from_values("log10fstarII", np.array([-2.0, -1.0]), "zscore"),
        ),
        target_scaling=None,
        training_config={"epochs": 2},
    )
    model = DenseMLP(
        in_features=len(spec.input_feature_names()),
        hidden_features=8,
        hidden_layers=2,
        rngs=nnx.Rngs(jax.random.PRNGKey(0)),
    )
    package_path = save(
        tmp_path / "loss_demo",
        model,
        train_losses=[3.0, 2.0],
        val_losses=[4.0, 1.5],
        loss="mse",
        metadata=metadata,
        epochs=2,
        patience=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
    )
    package_path.with_suffix(".summary.json").write_text(
        json.dumps(
            {
                "test_loss": 1.25,
                "best_epoch": 1,
                "best_validation_loss": 1.5,
            }
        )
    )

    curves = loss_curves_from_package(package_path)

    assert curves.train_losses == [3.0, 2.0]
    assert curves.validation_losses == [4.0, 1.5]
    assert curves.test_loss == 1.25
    assert curves.best_epoch == 1
    assert curves.best_validation_loss == 1.5
    assert curves.model_name == "t21-loss-demo"


def test_constant_learning_rate_schedule() -> None:
    schedule = build_learning_rate_schedule(
        learning_rate=1e-3,
        schedule_name="constant",
        steps_per_epoch=10,
        epochs=5,
    )

    assert np.isclose(float(schedule(0)), 1e-3)
    assert np.isclose(float(schedule(49)), 1e-3)


def test_cosine_learning_rate_schedule_reaches_final_fraction() -> None:
    schedule = build_learning_rate_schedule(
        learning_rate=1e-3,
        schedule_name="cosine",
        steps_per_epoch=10,
        epochs=5,
        final_fraction=0.1,
    )

    assert np.isclose(float(schedule(0)), 1e-3)
    assert float(schedule(50)) <= 1.1e-4


def test_warmup_cosine_learning_rate_schedule_warms_up() -> None:
    schedule = build_learning_rate_schedule(
        learning_rate=1e-3,
        schedule_name="warmup_cosine",
        steps_per_epoch=10,
        epochs=5,
        final_fraction=0.1,
        warmup_epochs=1,
    )

    assert np.isclose(float(schedule(0)), 0.0)
    assert np.isclose(float(schedule(10)), 1e-3)
    assert float(schedule(50)) <= 1.1e-4


def test_exponential_learning_rate_schedule_reaches_final_fraction() -> None:
    schedule = build_learning_rate_schedule(
        learning_rate=1e-3,
        schedule_name="exponential_decay",
        steps_per_epoch=10,
        epochs=5,
        final_fraction=0.1,
    )

    assert np.isclose(float(schedule(0)), 1e-3)
    assert np.isclose(float(schedule(49)), 1e-4)
