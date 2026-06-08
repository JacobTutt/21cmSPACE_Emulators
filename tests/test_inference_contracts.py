"""Tests for prior transforms and likelihood contracts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_emu.inference import (
    DiscretePrior,
    GaussianLikelihood,
    GlobalSignalForegroundLikelihood,
    GlobalSignalLikelihood,
    JointLikelihood,
    LogUniformPrior,
    PowerSpectrumData,
    PowerSpectrumGaussianLikelihood,
    PowerSpectrumUpperLimitLikelihood,
    PriorSpec,
    UniformPrior,
)


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
