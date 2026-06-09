"""End-to-end tests for training-package and inference workflows."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat, savemat

from examples_21cmspace.delta21.emulator import (
    build_delta21_fixed_coordinate_emulator,
    build_delta21_fixed_grid_emulator,
    build_delta21_emulator,
    build_delta21_predictor,
    predict_delta21,
)
from examples_21cmspace.delta21.train import train_delta21_from_dataset_root
from examples_21cmspace.t21.emulator import (
    build_t21_fixed_coordinate_emulator,
    build_t21_fixed_grid_emulator,
    build_t21_emulator,
    predict_t21,
)
from examples_21cmspace.t21.train import train_t21_from_dataset_root


def write_mock_twentyonecmspace_dataset(tmp_path: Path) -> Path:
    """Write a compact 21cmSPACE-like dataset for training and inference tests."""
    root = tmp_path / "21cmSPACE_Emulator_Data"
    root.mkdir()

    z = np.array([[6.0, 10.0, 20.0, 30.0]])
    k = np.array([[0.02, 0.05, 0.99]])
    nu_keV = np.array([[0.1, 1.0, 10.0]])
    parameters = np.array(
        [
            [1e-3, 2e-3, 5.0, 1e-2, 1.0, 100.0, 20.0, 0.04, 1e-1, 231.0, 1.0, 0.75],
            [2e-3, 3e-3, 10.0, 1e-1, 1.3, 200.0, 30.0, 0.05, 1e0, 232.0, 1.0, 0.75],
            [3e-3, 4e-3, 20.0, 1e0, 1.5, 300.0, 40.0, 0.06, 1e1, 233.0, 0.0, 0.0],
            [4e-3, 5e-3, 30.0, 1e1, 1.0, 400.0, 50.0, 0.07, 1e2, 231.0, 0.0, 0.0],
            [5e-3, 6e-3, 40.0, 1e2, 1.3, 500.0, 60.0, 0.08, 1e3, 232.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    delta21 = np.empty((5, 4, 3), dtype=float)
    t21 = np.empty((5, 4), dtype=float)
    for idx in range(5):
        base = (idx + 1) * 0.2
        zz, kk = np.meshgrid(z.ravel(), k.ravel(), indexing="ij")
        delta21[idx] = base + (zz - 5.0) * (kk + 0.5)
        t21[idx] = -100.0 + base * 10.0 + 0.5 * z.ravel()

    savemat(root / "21cmspace_z_mat.mat", {"z21cm": z})
    savemat(root / "21cmspace_k_mat.mat", {"ks": k})
    savemat(root / "21cmspace_nu_mat.mat", {"nu_keV": nu_keV})
    savemat(root / "21cmspace_parameters_mat.mat", {"parameters": parameters})
    savemat(root / "21cmspace_Deltak_mat.mat", {"combined_Deltaks": delta21})
    savemat(root / "21cmspace_T21_mat.mat", {"combined_T21s": t21})
    return root


def test_delta21_training_and_inference_round_trip(tmp_path: Path) -> None:
    dataset_root = write_mock_twentyonecmspace_dataset(tmp_path)
    package_path = tmp_path / "delta21_model.nenemu"

    summary = train_delta21_from_dataset_root(
        str(dataset_root),
        output_path=package_path,
        epochs=2,
        batch_size=128,
        shuffle_seed=3,
    )

    assert package_path.exists()
    assert package_path.with_suffix(".summary.json").exists()
    assert summary["package_path"] == str(package_path)

    raw_parameters = jnp.asarray(
        loadmat(dataset_root / "21cmspace_parameters_mat.mat")["parameters"][:2],
        dtype=jnp.float32,
    )
    z = jnp.array([6.0, 10.0], dtype=jnp.float32)
    k = jnp.array([0.05 / 0.6704, 0.99 / 0.6704], dtype=jnp.float32)
    predictions = predict_delta21(package_path, raw_parameters, z, k)
    predictor = build_delta21_predictor(package_path)
    compiled_predictions = predictor(raw_parameters, z, k)
    emulator = build_delta21_emulator(
        package_path,
        compile_inputs=(raw_parameters, z, k),
    )
    emulator_predictions = emulator.emulate(raw_parameters, z, k)
    fixed_grid_emulator = build_delta21_fixed_grid_emulator(
        package_path,
        z,
        k,
        compile_parameters=raw_parameters,
    )
    fixed_grid_predictions = fixed_grid_emulator.emulate(raw_parameters)
    zz, kk = jnp.meshgrid(z, k, indexing="ij")
    fixed_coordinate_emulator = build_delta21_fixed_coordinate_emulator(
        package_path,
        zz.ravel(),
        kk.ravel(),
        compile_parameters=raw_parameters,
    )
    fixed_coordinate_predictions = fixed_coordinate_emulator.emulate(raw_parameters)

    assert isinstance(predictions, jax.Array)
    assert predictions.shape == (2, 2, 2)
    assert np.isfinite(np.asarray(jax.device_get(predictions))).all()
    assert isinstance(compiled_predictions, jax.Array)
    assert compiled_predictions.shape == (2, 2, 2)
    assert np.isfinite(np.asarray(jax.device_get(compiled_predictions))).all()
    assert isinstance(emulator_predictions, jax.Array)
    assert emulator_predictions.shape == (2, 2, 2)
    assert np.isfinite(np.asarray(jax.device_get(emulator_predictions))).all()
    assert isinstance(fixed_grid_predictions, jax.Array)
    assert fixed_grid_predictions.shape == (2, 2, 2)
    assert np.isfinite(np.asarray(jax.device_get(fixed_grid_predictions))).all()
    assert isinstance(fixed_coordinate_predictions, jax.Array)
    assert fixed_coordinate_predictions.shape == (2, 4)
    assert np.isfinite(np.asarray(jax.device_get(fixed_coordinate_predictions))).all()
    np.testing.assert_allclose(
        np.asarray(jax.device_get(fixed_grid_predictions)),
        np.asarray(jax.device_get(compiled_predictions)),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(fixed_coordinate_predictions)),
        np.asarray(jax.device_get(fixed_grid_predictions)).reshape(2, -1),
        rtol=1e-5,
        atol=1e-5,
    )


def test_t21_training_and_inference_round_trip(tmp_path: Path) -> None:
    dataset_root = write_mock_twentyonecmspace_dataset(tmp_path)
    package_path = tmp_path / "t21_model.nenemu"

    summary = train_t21_from_dataset_root(
        str(dataset_root),
        output_path=package_path,
        epochs=3,
        batch_size=128,
        shuffle_seed=7,
    )

    assert package_path.exists()
    assert package_path.with_suffix(".summary.json").exists()
    assert summary["package_path"] == str(package_path)

    raw_parameters = jnp.asarray(
        loadmat(dataset_root / "21cmspace_parameters_mat.mat")["parameters"][:2],
        dtype=jnp.float32,
    )
    z = jnp.array([6.0, 10.0, 20.0, 27.0], dtype=jnp.float32)
    predictions = predict_t21(package_path, raw_parameters, z)
    emulator = build_t21_emulator(
        package_path,
        compile_inputs=(raw_parameters, z),
    )
    emulator_predictions = emulator.emulate(raw_parameters, z)
    fixed_grid_emulator = build_t21_fixed_grid_emulator(
        package_path,
        z,
        compile_parameters=raw_parameters,
    )
    fixed_grid_predictions = fixed_grid_emulator.emulate(raw_parameters)
    fixed_coordinate_emulator = build_t21_fixed_coordinate_emulator(
        package_path,
        z,
        compile_parameters=raw_parameters,
    )
    fixed_coordinate_predictions = fixed_coordinate_emulator.emulate(raw_parameters)

    assert isinstance(predictions, jax.Array)
    assert predictions.shape == (2, 4)
    assert np.isfinite(np.asarray(jax.device_get(predictions))).all()
    assert isinstance(emulator_predictions, jax.Array)
    assert emulator_predictions.shape == (2, 4)
    assert np.isfinite(np.asarray(jax.device_get(emulator_predictions))).all()
    assert isinstance(fixed_grid_predictions, jax.Array)
    assert fixed_grid_predictions.shape == (2, 4)
    assert np.isfinite(np.asarray(jax.device_get(fixed_grid_predictions))).all()
    assert isinstance(fixed_coordinate_predictions, jax.Array)
    assert fixed_coordinate_predictions.shape == (2, 4)
    assert np.isfinite(np.asarray(jax.device_get(fixed_coordinate_predictions))).all()
    np.testing.assert_allclose(
        np.asarray(jax.device_get(fixed_grid_predictions)),
        np.asarray(jax.device_get(emulator_predictions)),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(jax.device_get(fixed_coordinate_predictions)),
        np.asarray(jax.device_get(fixed_grid_predictions)),
        rtol=1e-5,
        atol=1e-5,
    )
