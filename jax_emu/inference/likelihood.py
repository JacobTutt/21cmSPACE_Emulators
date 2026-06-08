"""
Likelihood utilities for emulator-based inference.

The likelihood layer compares emulator predictions against data. It does not
know how the emulator was trained; it only assumes the supplied emulator object
has an `emulate(parameters)` method that returns physical predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
from jax.scipy.special import erf


class EmulatorLike(Protocol):
    """
    Minimal interface required by the likelihood classes.
    """

    def emulate(self, parameters: jax.Array) -> jax.Array:
        """
        Return physical predictions for one or more parameter rows.
        """


@dataclass(frozen=True)
class PowerSpectrumData:
    """
    Storage utility for pointwise power-spectrum data.

    The coordinates are explicit `(z, k)` pairs, not a rectangular grid. This
    supports ragged observations where different redshifts have different k
    values.

    Parameters
    ----------
    coordinates:
        Array with shape `(n_model_points, 2)`. Column 0 is redshift and column
        1 is k.
    upper_limit:
        Upper limits or measured values in data-bin space.
    sigma:
        Observational uncertainties in data-bin space.
    window_matrix:
        Optional matrix mapping model coordinate points onto data bins. If
        omitted, the data bins are assumed to match the model points directly.
    """

    coordinates: jax.Array
    upper_limit: jax.Array
    sigma: jax.Array
    window_matrix: jax.Array | None = None

    @property
    def z_model_points(self) -> jax.Array:
        """
        Return redshift coordinates as a 1D array.
        """
        return _coordinate_array(self.coordinates)[:, 0]

    @property
    def k_model_points(self) -> jax.Array:
        """
        Return k coordinates as a 1D array.
        """
        return _coordinate_array(self.coordinates)[:, 1]


@dataclass(frozen=True)
class GaussianLikelihood:
    """
    Diagonal Gaussian likelihood for measured data points.

    Parameters
    ----------
    data:
        Observed values.
    sigma:
        One-sigma observational uncertainty for each value.
    include_normalization:
        Whether to include the Gaussian normalization term.
    """

    data: jax.Array
    sigma: jax.Array
    include_normalization: bool = True

    def __call__(self, theory: jax.Array, theory_sigma: jax.Array | float = 0.0) -> jax.Array:
        """
        Evaluate the log likelihood for one or more theory predictions.
        """
        data = jnp.asarray(self.data, dtype=jnp.float32)
        sigma = _total_sigma(self.sigma, theory_sigma)
        residual = data - theory

        loglike = -0.5 * jnp.sum((residual / sigma) ** 2, axis=-1)
        if self.include_normalization:
            loglike -= 0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * sigma**2), axis=-1)
        return _squeeze_single(loglike)


@dataclass(frozen=True)
class UpperLimitLikelihood:
    """
    One-sided upper-limit likelihood.

    This is the form used by HERA-style 21-cm power-spectrum constraints. A
    model below the reported upper limit is weakly penalized, while a model
    above the limit receives a rapidly decreasing likelihood.

    Parameters
    ----------
    upper_limit:
        Data values defining the upper limits.
    sigma:
        One-sigma observational uncertainty for each upper limit.
    min_probability:
        Numerical floor applied before taking the logarithm.
    """

    upper_limit: jax.Array
    sigma: jax.Array
    min_probability: float = 1e-50

    def __call__(self, theory: jax.Array, theory_sigma: jax.Array | float = 0.0) -> jax.Array:
        """
        Evaluate the one-sided log likelihood.
        """
        upper_limit = jnp.asarray(self.upper_limit, dtype=jnp.float32)
        sigma = _total_sigma(self.sigma, theory_sigma)
        standardized = (upper_limit - theory) / (jnp.sqrt(2.0) * sigma)
        probability = 0.5 * (1.0 + erf(standardized))
        probability = jnp.clip(probability, self.min_probability, 1.0)
        return _squeeze_single(jnp.sum(jnp.log(probability), axis=-1))


@dataclass(frozen=True)
class GlobalSignalLikelihood:
    """
    Gaussian likelihood for a global 21-cm signal emulator.

    The emulator should already be initialized on the redshift points used by
    the data. Calling the likelihood then only requires astrophysical
    parameters.
    """

    emulator: EmulatorLike
    data: jax.Array
    sigma: jax.Array
    theory_fractional_error: float | jax.Array = 0.0
    name: str = "global_signal"

    def __call__(self, parameters: jax.Array) -> jax.Array:
        """
        Evaluate the global-signal log likelihood.
        """
        prediction = self.emulator.emulate(parameters)
        theory_sigma = _fractional_theory_sigma(prediction, self.theory_fractional_error)
        return GaussianLikelihood(self.data, self.sigma)(prediction, theory_sigma)


@dataclass(frozen=True)
class PowerSpectrumGaussianLikelihood:
    """
    Gaussian likelihood for power-spectrum detections.

    This is useful for mock detected power spectra or future datasets with
    symmetric measurement errors. It is not the default HERA upper-limit
    likelihood.
    """

    emulator: EmulatorLike
    data: jax.Array
    sigma: jax.Array
    window_matrix: jax.Array | None = None
    theory_fractional_error: float | jax.Array = 0.0
    name: str = "power_spectrum_gaussian"

    def __call__(self, parameters: jax.Array) -> jax.Array:
        """
        Evaluate the power-spectrum Gaussian log likelihood.
        """
        prediction = _apply_window(self.emulator.emulate(parameters), self.window_matrix)
        theory_sigma = _fractional_theory_sigma(prediction, self.theory_fractional_error)
        return GaussianLikelihood(self.data, self.sigma)(prediction, theory_sigma)


@dataclass(frozen=True)
class PowerSpectrumUpperLimitLikelihood:
    """
    HERA-style upper-limit likelihood for 21-cm power spectra.

    The emulator should be initialized on the model k-bins required by the
    observation. If the data provide a window matrix, it is applied as
    `theory_data_bins = W @ theory_model_bins` before evaluating the
    upper-limit probability.
    """

    emulator: EmulatorLike
    upper_limit: jax.Array
    sigma: jax.Array
    window_matrix: jax.Array | None = None
    theory_fractional_error: float | jax.Array = 0.0
    min_probability: float = 1e-50
    name: str = "power_spectrum_upper_limit"

    def __call__(self, parameters: jax.Array) -> jax.Array:
        """
        Evaluate the one-sided power-spectrum log likelihood.
        """
        prediction = _apply_window(self.emulator.emulate(parameters), self.window_matrix)
        theory_sigma = _fractional_theory_sigma(prediction, self.theory_fractional_error)
        return UpperLimitLikelihood(
            self.upper_limit,
            self.sigma,
            min_probability=self.min_probability,
        )(prediction, theory_sigma)


class JointLikelihood:
    """
    Sum a set of independent likelihood modules.

    This mirrors the structure used in the older inference code: each dataset
    owns its own likelihood calculation, and the joint constraint is the sum of
    those log likelihoods.
    """

    def __init__(self, likelihoods: list[object] | tuple[object, ...]) -> None:
        """
        Store the likelihood modules in evaluation order.
        """
        if len(likelihoods) == 0:
            raise ValueError("JointLikelihood requires at least one likelihood.")
        self.likelihoods = tuple(likelihoods)

    def __call__(self, parameters: jax.Array) -> jax.Array:
        """
        Evaluate the summed log likelihood.
        """
        total = 0.0
        for likelihood in self.likelihoods:
            total = total + likelihood(parameters)
        return total

    def contributions(self, parameters: jax.Array) -> dict[str, jax.Array]:
        """
        Return individual likelihood terms for diagnostics.
        """
        outputs = {}
        for index, likelihood in enumerate(self.likelihoods):
            name = getattr(likelihood, "name", f"likelihood_{index}")
            if name in outputs:
                name = f"{name}_{index}"
            outputs[name] = likelihood(parameters)
        return outputs


def _apply_window(prediction: jax.Array, window_matrix: jax.Array | None) -> jax.Array:
    """
    Apply an optional data window matrix to model predictions.
    """
    if window_matrix is None:
        return prediction
    window = jnp.asarray(window_matrix, dtype=jnp.float32)
    return prediction @ window.T


def _coordinate_array(coordinates: jax.Array) -> jax.Array:
    """
    Validate and return an explicit `(z, k)` coordinate table.
    """
    array = jnp.asarray(coordinates, dtype=jnp.float32)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("Power-spectrum coordinates must have shape (n_points, 2).")
    return array


def _fractional_theory_sigma(prediction: jax.Array, fraction: float | jax.Array) -> jax.Array:
    """
    Convert a fractional emulator/theory error into an absolute uncertainty.
    """
    return jnp.asarray(fraction, dtype=jnp.float32) * jnp.abs(prediction)


def _total_sigma(observation_sigma: jax.Array, theory_sigma: jax.Array | float) -> jax.Array:
    """
    Combine observational and theory uncertainties in quadrature.
    """
    sigma = jnp.asarray(observation_sigma, dtype=jnp.float32)
    theory = jnp.asarray(theory_sigma, dtype=jnp.float32)
    return jnp.sqrt(sigma**2 + theory**2)


def _squeeze_single(values: jax.Array) -> jax.Array:
    """
    Return a scalar for one parameter row, otherwise keep the batch axis.
    """
    return jnp.squeeze(values) if values.shape == (1,) else values
