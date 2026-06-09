"""Tests for prior transforms and likelihood contracts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_emu.inference import (
    DiscretePrior,
    FixedPrior,
    GaussianLikelihood,
    GlobalSignalForegroundLikelihood,
    GlobalSignalLikelihood,
    JointLikelihood,
    LogUniformPrior,
    NestedSamplingConfig,
    NestedSamplingResult,
    PowerSpectrumData,
    PowerSpectrumGaussianLikelihood,
    PowerSpectrumUpperLimitLikelihood,
    PriorSpec,
    UniformPrior,
    resolve_nested_sampling_settings,
    save_anesthetic_samples,
)
from examples_21cmspace.delta21.hera_data import (
    HERAObservation,
    combine_hera_observations,
    default_h1c_idr2_selections,
    hera_dataset_summary,
    load_hera_power_spectrum_dataset,
    load_hera_power_spectrum_npz,
    save_hera_power_spectrum_npz,
)
from examples_21cmspace.delta21.nenufar_data import load_nenufar_table4_dataset


class DummyEmulator:
    """Small emulator stub used to test likelihood math."""

    def __init__(self, prediction: jax.Array) -> None:
        self.prediction = prediction

    def emulate(self, parameters: jax.Array) -> jax.Array:
        array = parameters if parameters.ndim == 2 else parameters[None, :]
        return jnp.tile(self.prediction[None, :], (array.shape[0], 1))


def test_prior_spec_transforms_unit_cube_to_physical_values() -> None:
    prior = PriorSpec(
        [
            UniformPrior("x", -1.0, 1.0),
            LogUniformPrior("scale", 1.0, 100.0),
            DiscretePrior("alpha", [1.0, 1.3, 1.5]),
        ]
    )

    theta = prior.transform(jnp.array([0.25, 0.5, 0.7]))

    assert prior.ndim == 3
    assert prior.names == ("x", "scale", "alpha")
    np.testing.assert_allclose(np.asarray(theta), np.array([-0.5, 10.0, 1.5]), rtol=1e-6)


def test_prior_spec_can_return_grouped_parameters() -> None:
    prior = PriorSpec(
        {
            "astro": [
                UniformPrior("x", -1.0, 1.0),
            ],
            "foreground": [
                UniformPrior("a0", 3.0, 4.0),
                UniformPrior("a1", -1.0, 1.0),
            ],
            "noise": [
                LogUniformPrior("sigma", 0.1, 10.0),
            ],
        }
    )

    theta = prior.transform(jnp.array([0.5, 0.0, 0.75, 0.5]))

    assert prior.names == ("astro.x", "foreground.a0", "foreground.a1", "noise.sigma")
    np.testing.assert_allclose(np.asarray(theta["astro"]), np.array([0.0]), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(theta["foreground"]), np.array([3.0, 0.5]), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(theta["noise"]), np.array([1.0]), rtol=1e-6)


def test_nested_sampling_settings_scale_with_prior_dimension() -> None:
    prior = PriorSpec(
        [
            UniformPrior("x", -1.0, 1.0),
            LogUniformPrior("scale", 1.0, 100.0),
            DiscretePrior("alpha", [1.0, 1.3, 1.5]),
            FixedPrior("fixed", 2.0),
        ]
    )

    settings = resolve_nested_sampling_settings(
        prior,
        NestedSamplingConfig(
            n_live_scale=20,
            num_delete_fraction=0.1,
            num_inner_steps_scale=5,
        ),
    )

    assert prior.ndim == 3
    assert settings.n_live == 60
    assert settings.num_delete == 6
    assert settings.num_inner_steps == 15


def test_nested_sampling_default_settings_match_reference_recipe() -> None:
    prior = PriorSpec(
        [
            UniformPrior("x", -1.0, 1.0),
            UniformPrior("y", -1.0, 1.0),
            UniformPrior("z", -1.0, 1.0),
            FixedPrior("fixed", 2.0),
        ]
    )

    settings = resolve_nested_sampling_settings(prior)

    assert prior.ndim == 3
    assert settings.n_live == 75
    assert settings.num_delete == 15
    assert settings.num_inner_steps == 15


def test_anesthetic_export_writes_expected_files(tmp_path) -> None:
    pytest.importorskip("anesthetic")
    prior = PriorSpec(
        [
            UniformPrior("x", -1.0, 1.0),
            FixedPrior("fixed", 2.0),
        ]
    )
    result = NestedSamplingResult(
        state=None,
        final_state=None,
        info=None,
        physical_samples=jnp.array([[0.0, 2.0], [0.5, 2.0], [1.0, 2.0]]),
        loglikelihood=jnp.array([-3.0, -1.0, -2.0]),
        loglikelihood_birth=jnp.array([-jnp.inf, -3.0, -2.0]),
        logz=jnp.array(-1.2),
        logz_error=jnp.array(0.1),
        n_steps=2,
        converged=True,
        settings=resolve_nested_sampling_settings(
            prior,
            NestedSamplingConfig(n_live=10, num_delete=1, num_inner_steps=3),
        ),
    )

    csv_path = save_anesthetic_samples(result, prior, tmp_path)

    assert csv_path.name == "nested_sampling_results.csv"
    assert csv_path.exists()
    assert (tmp_path / "parameter_names.json").exists()
    assert (tmp_path / "sampler_config.json").exists()
    assert (tmp_path / "test_stats.txt").exists()


def test_gaussian_likelihood_matches_manual_diagonal_result() -> None:
    likelihood = GaussianLikelihood(
        data=jnp.array([1.0, 2.0]),
        sigma=jnp.array([0.5, 1.0]),
        include_normalization=False,
    )

    loglike = likelihood(jnp.array([[1.5, 1.0]]))

    assert np.isclose(float(loglike), -0.625)


def test_global_signal_foreground_likelihood_uses_nuisance_groups() -> None:
    likelihood = GlobalSignalForegroundLikelihood(
        emulator=DummyEmulator(jnp.array([1.0, 2.0, 3.0])),
        data=jnp.array([11.0, 12.0, 13.0]),
        reduced_frequency=jnp.array([-1.0, 0.0, 1.0]),
    )
    parameters = {
        "astro": jnp.array([0.0]),
        "foreground": jnp.array([1.0, 0.0]),
        "noise": jnp.array([1.0]),
    }

    loglike = likelihood(parameters)

    np.testing.assert_allclose(float(loglike), -0.5 * 3 * np.log(2 * np.pi), rtol=1e-6)


def test_global_signal_likelihood_can_use_noise_group() -> None:
    likelihood = GlobalSignalLikelihood(
        emulator=DummyEmulator(jnp.array([1.0, 2.0])),
        data=jnp.array([1.0, 2.0]),
    )
    parameters = {
        "astro": jnp.array([0.0]),
        "noise": jnp.array([1.0]),
    }

    loglike = likelihood(parameters)

    np.testing.assert_allclose(float(loglike), -0.5 * 2 * np.log(2 * np.pi), rtol=1e-6)


def test_global_signal_foreground_likelihood_can_use_fixed_sigma() -> None:
    likelihood = GlobalSignalForegroundLikelihood(
        emulator=DummyEmulator(jnp.array([1.0])),
        data=jnp.array([11.0]),
        reduced_frequency=jnp.array([0.0]),
        sigma=jnp.array([1.0]),
    )
    parameters = {
        "astro": jnp.array([0.0]),
        "foreground": jnp.array([1.0]),
    }

    loglike = likelihood(parameters)

    np.testing.assert_allclose(float(loglike), -0.5 * np.log(2 * np.pi), rtol=1e-6)


def test_power_spectrum_upper_limit_penalizes_models_above_the_limit() -> None:
    parameters = jnp.array([0.0])
    sigma = jnp.array([1.0, 1.0])
    upper_limit = jnp.array([10.0, 10.0])

    low_model = PowerSpectrumUpperLimitLikelihood(
        emulator=DummyEmulator(jnp.array([1.0, 2.0])),
        upper_limit=upper_limit,
        sigma=sigma,
    )
    high_model = PowerSpectrumUpperLimitLikelihood(
        emulator=DummyEmulator(jnp.array([20.0, 30.0])),
        upper_limit=upper_limit,
        sigma=sigma,
    )

    assert float(low_model(parameters)) > float(high_model(parameters))
    assert float(low_model(parameters)) > -1e-6


def test_power_spectrum_data_uses_explicit_coordinate_pairs() -> None:
    data = PowerSpectrumData(
        coordinates=jnp.array(
            [
                [7.9, 0.12],
                [7.9, 0.18],
                [10.4, 0.09],
            ]
        ),
        upper_limit=jnp.array([10.0, 20.0, 30.0]),
        sigma=jnp.array([1.0, 2.0, 3.0]),
    )

    np.testing.assert_allclose(np.asarray(data.z_model_points), np.array([7.9, 7.9, 10.4]))
    np.testing.assert_allclose(np.asarray(data.k_model_points), np.array([0.12, 0.18, 0.09]))


def test_power_spectrum_data_rejects_grid_shaped_coordinates() -> None:
    with pytest.raises(ValueError, match="shape"):
        PowerSpectrumData(
            coordinates=jnp.ones((2, 3)),
            upper_limit=jnp.ones(2),
            sigma=jnp.ones(2),
        )


def test_power_spectrum_likelihood_applies_window_matrix() -> None:
    likelihood = PowerSpectrumGaussianLikelihood(
        emulator=DummyEmulator(jnp.array([1.0, 3.0])),
        data=jnp.array([2.0]),
        sigma=jnp.array([1.0]),
        window_matrix=jnp.array([[0.5, 0.5]]),
    )

    loglike = likelihood(jnp.array([0.0]))

    assert np.isfinite(float(loglike))
    np.testing.assert_allclose(float(loglike), -0.5 * np.log(2 * np.pi), rtol=1e-6)


def test_hera_observations_combine_into_block_window_dataset(tmp_path) -> None:
    first = HERAObservation(
        z=7.9,
        k_model=np.array([0.1, 0.2], dtype=np.float32),
        upper_limit=np.array([10.0], dtype=np.float32),
        sigma=np.array([1.0], dtype=np.float32),
        window_matrix=np.array([[0.25, 0.75]], dtype=np.float32),
        k_data=np.array([0.1], dtype=np.float32),
        source_file="field1.h5",
        band=1,
        field="1",
    )
    second = HERAObservation(
        z=10.4,
        k_model=np.array([0.3, 0.4, 0.5], dtype=np.float32),
        upper_limit=np.array([20.0, 30.0], dtype=np.float32),
        sigma=np.array([2.0, 3.0], dtype=np.float32),
        window_matrix=np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]],
            dtype=np.float32,
        ),
        k_data=np.array([0.3, 0.4], dtype=np.float32),
        source_file="field1.h5",
        band=2,
        field="1",
    )

    dataset = combine_hera_observations([first, second])

    np.testing.assert_allclose(
        np.asarray(dataset.power_data.coordinates),
        np.array(
            [
                [7.9, 0.1],
                [7.9, 0.2],
                [10.4, 0.3],
                [10.4, 0.4],
                [10.4, 0.5],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        np.asarray(dataset.power_data.window_matrix),
        np.array(
            [
                [0.25, 0.75, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5, 0.5],
            ],
            dtype=np.float32,
        ),
    )
    assert hera_dataset_summary(dataset)["n_data_bins"] == 3

    cache_path = save_hera_power_spectrum_npz(dataset, tmp_path / "hera_cache.npz")
    cached = load_hera_power_spectrum_npz(cache_path)

    np.testing.assert_allclose(
        np.asarray(cached.power_data.coordinates),
        np.asarray(dataset.power_data.coordinates),
    )
    np.testing.assert_allclose(
        np.asarray(cached.power_data.window_matrix),
        np.asarray(dataset.power_data.window_matrix),
    )


def test_hera_hdf5_loader_returns_expected_idr2_arrays() -> None:
    dataset = load_hera_power_spectrum_dataset(default_h1c_idr2_selections(field="1"))
    summary = hera_dataset_summary(dataset)

    assert summary["n_model_points"] == 21
    assert summary["n_data_bins"] == 21
    assert summary["window_shape"] == [21, 21]
    np.testing.assert_allclose(summary["redshifts"], [7.9287629, 10.3721304], rtol=1e-6)
    np.testing.assert_allclose(summary["k_min"], 0.128, rtol=1e-6)
    np.testing.assert_allclose(summary["k_max"], 1.408, rtol=1e-6)

    first, second = dataset.observations
    np.testing.assert_allclose(first.k_model[:3], [0.128, 0.256, 0.384], rtol=1e-6)
    np.testing.assert_allclose(second.k_model[:3], [0.192, 0.320, 0.448], rtol=1e-6)
    assert first.window_matrix.shape == (11, 11)
    assert second.window_matrix.shape == (10, 10)


def test_nenufar_table4_loader_returns_direct_upper_limit_data() -> None:
    dataset = load_nenufar_table4_dataset()

    assert dataset.power_data.coordinates.shape == (12, 2)
    assert dataset.power_data.upper_limit.shape == (12,)
    assert dataset.power_data.sigma.shape == (12,)
    assert dataset.power_data.window_matrix is None
    assert np.all(np.asarray(dataset.power_data.sigma) > 0.0)


def test_joint_likelihood_sums_individual_modules() -> None:
    parameters = jnp.array([0.0])
    first = GlobalSignalLikelihood(
        emulator=DummyEmulator(jnp.array([1.0])),
        data=jnp.array([1.0]),
        sigma=jnp.array([1.0]),
    )
    second = GlobalSignalLikelihood(
        emulator=DummyEmulator(jnp.array([2.0])),
        data=jnp.array([1.0]),
        sigma=jnp.array([1.0]),
    )
    joint = JointLikelihood([first, second])

    expected = first(parameters) + second(parameters)

    np.testing.assert_allclose(float(joint(parameters)), float(expected))
    assert set(joint.contributions(parameters)) == {"global_signal", "global_signal_1"}


def test_joint_likelihood_allows_shared_astro_and_nuisance_groups() -> None:
    foreground_like = GlobalSignalForegroundLikelihood(
        emulator=DummyEmulator(jnp.array([1.0])),
        data=jnp.array([11.0]),
        reduced_frequency=jnp.array([0.0]),
    )
    power_like = PowerSpectrumUpperLimitLikelihood(
        emulator=DummyEmulator(jnp.array([1.0])),
        upper_limit=jnp.array([10.0]),
        sigma=jnp.array([1.0]),
    )
    joint = JointLikelihood([foreground_like, power_like])
    parameters = {
        "astro": jnp.array([0.0]),
        "foreground": jnp.array([1.0]),
        "noise": jnp.array([1.0]),
    }

    expected = foreground_like(parameters) + power_like(parameters)

    np.testing.assert_allclose(float(joint(parameters)), float(expected))
