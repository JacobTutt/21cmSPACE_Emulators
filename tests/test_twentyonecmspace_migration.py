"""Tests for 21cmSPACE preparation paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

from jax_emu.data_preprocessing.scaling import TargetScalingScalar
from emulators_21cmspace.delta21.data import prepare_twentyonecmspace_delta21_training_split
from emulators_21cmspace.t21.data import prepare_twentyonecmspace_t21_training_split


def test_prepare_twentyonecmspace_delta21_split_matches_expected_shape_rules(tmp_path: Path) -> None:
    dataset_root = write_mock_twentyonecmspace_dataset(tmp_path)
    prepared = prepare_twentyonecmspace_delta21_training_split(dataset_root, shuffle_seed=7)

    assert prepared.feature_names == (
        "z",
        "log10k",
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
    assert prepared.train_features.shape == (7500, 11)
    assert prepared.train_targets.shape == (7500,)
    assert prepared.validation_features.shape == (2500, 11)
    assert prepared.validation_targets.shape == (2500,)
    assert prepared.test_features.shape == (2500, 11)
    assert prepared.test_targets.shape == (2500,)
    assert prepared.train_features.dtype == np.float32
    assert prepared.train_targets.dtype == np.float32
    scaling_by_name = {feature.name: feature.method for feature in prepared.feature_scaling}
    assert scaling_by_name["tau"] == "zscore"
    assert scaling_by_name["z"] == "zscore"
    assert scaling_by_name["log10k"] == "zscore"
    assert scaling_by_name["alpha"] == "minmax_zero_to_one"
    assert isinstance(prepared.target_scaling, TargetScalingScalar)


def test_prepare_twentyonecmspace_t21_split_matches_expected_shape_rules(tmp_path: Path) -> None:
    dataset_root = write_mock_twentyonecmspace_dataset(tmp_path)
    prepared = prepare_twentyonecmspace_t21_training_split(dataset_root, shuffle_seed=11)

    assert prepared.feature_names == (
        "z",
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
    assert prepared.train_features.shape == (600, 10)
    assert prepared.train_targets.shape == (600,)
    assert prepared.validation_features.shape == (200, 10)
    assert prepared.validation_targets.shape == (200,)
    assert prepared.test_features.shape == (200, 10)
    assert prepared.test_targets.shape == (200,)
    scaling_by_name = {feature.name: feature.method for feature in prepared.feature_scaling}
    assert scaling_by_name["tau"] == "zscore"
    assert scaling_by_name["z"] == "zscore"
    assert scaling_by_name["pop"] == "minmax_zero_to_one"
    assert isinstance(prepared.target_scaling, TargetScalingScalar)


def write_mock_twentyonecmspace_dataset(tmp_path: Path) -> Path:
    """Write a small 21cmSPACE-like dataset for workflow tests."""
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
