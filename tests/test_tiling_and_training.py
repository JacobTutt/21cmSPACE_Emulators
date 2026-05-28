"""Tests for tiling helpers and synthetic JAX training."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from jaxemu_21cmSPACE.data_preprocessing.tiling import reconstruct_spectra, tile_spectra
from jaxemu_21cmSPACE.emulators21.delta21.data import delta21_spec
from jaxemu_21cmSPACE.emulators21.t21.data import t21_spec
from jaxemu_21cmSPACE.architectures.mlp import DenseMLP
from jaxemu_21cmSPACE.training.trainer import train_mlp_regressor


def test_tile_and_reconstruct_shapes() -> None:
    parameters = np.array([[1.0, 2.0], [3.0, 4.0]])
    axes = (np.array([10.0, 20.0]), np.array([0.1, 0.2, 0.3]))
    targets = np.arange(12.0).reshape(2, 2, 3)
    features, flat_targets, axis_shape = tile_spectra(parameters, axes, targets)
    assert features.shape == (12, 4)
    assert flat_targets.shape == (12,)
    reconstructed = reconstruct_spectra(flat_targets, nsamples=2, axis_shape=axis_shape)
    assert reconstructed.shape == targets.shape
    assert np.allclose(reconstructed, targets)


def test_synthetic_training_smoke() -> None:
    rng = np.random.default_rng(0)
    train_features = rng.normal(size=(256, 3))
    validation_features = rng.normal(size=(64, 3))
    weights = np.array([2.0, -1.0, 0.5])
    train_targets = train_features @ weights + 0.1
    validation_targets = validation_features @ weights + 0.1

    model = DenseMLP(
        in_features=train_features.shape[1],
        hidden_features=16,
        hidden_layers=2,
        rngs=nnx.Rngs(jax.random.PRNGKey(0)),
    )
    model, history = train_mlp_regressor(
        model,
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        epochs=40,
        batch_size=32,
        learning_rate=5e-3,
        weight_decay=0.0,
        seed=0,
    )
    preds = model(jnp.asarray(validation_features)).squeeze(-1)
    mse = jnp.mean(jnp.square(preds - validation_targets))
    assert history.train_losses[-1] < history.train_losses[0]
    assert float(mse) < 0.1


def test_training_accepts_device_arrays_without_epoch_logging() -> None:
    rng = np.random.default_rng(2)
    features = rng.normal(size=(64, 2)).astype(np.float32)
    targets = (features[:, 0] - 0.25 * features[:, 1]).astype(np.float32)

    model = DenseMLP(
        in_features=features.shape[1],
        hidden_features=8,
        hidden_layers=1,
        rngs=nnx.Rngs(jax.random.PRNGKey(2)),
    )
    _, history = train_mlp_regressor(
        model,
        jnp.asarray(features),
        jnp.asarray(targets),
        jnp.asarray(features[:16]),
        jnp.asarray(targets[:16]),
        epochs=3,
        batch_size=16,
        learning_rate=1e-3,
        weight_decay=0.0,
        seed=2,
        log_every=None,
    )

    assert len(history.train_losses) == 3
    assert len(history.validation_losses) == 3


def test_early_stopping_records_and_restores_best_epoch() -> None:
    rng = np.random.default_rng(4)
    train_features = rng.normal(size=(32, 2))
    validation_features = np.zeros((8, 2))
    train_targets = np.zeros(32)
    validation_targets = np.ones(8)

    model = DenseMLP(
        in_features=train_features.shape[1],
        hidden_features=4,
        hidden_layers=1,
        rngs=nnx.Rngs(jax.random.PRNGKey(3)),
    )
    _, history = train_mlp_regressor(
        model,
        train_features,
        train_targets,
        validation_features,
        validation_targets,
        epochs=12,
        batch_size=8,
        learning_rate=0.0,
        weight_decay=0.0,
        seed=3,
        early_stopping_patience=2,
    )
    assert history.best_epoch == 0
    assert history.best_validation_loss == history.validation_losses[0]
    assert len(history.validation_losses) < 12


def test_default_specs_have_expected_axis_counts() -> None:
    assert len(delta21_spec().axes) == 2
    assert len(t21_spec().axes) == 1
