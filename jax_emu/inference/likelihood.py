"""
Likelihood utilities for emulator-based inference.

The likelihood layer compares emulator predictions against data. It does not
know how the emulator was trained; it only assumes the supplied emulator object
has an `emulate(parameters)` method that returns physical predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Callable, Protocol

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

    def __post_init__(self) -> None:
        coordinates = _coordinate_array(self.coordinates)
        upper_limit = _array_1d(self.upper_limit, "upper_limit")
        sigma = _array_1d(self.sigma, "sigma")
        if upper_limit.shape != sigma.shape:
            raise ValueError("upper_limit and sigma must have matching shapes.")

        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "upper_limit", upper_limit)
        object.__setattr__(self, "sigma", sigma)
        if self.window_matrix is not None:
            window_matrix = jnp.asarray(self.window_matrix, dtype=jnp.float32)
            if window_matrix.shape[1] != coordinates.shape[0]:
                raise ValueError(
                    "window_matrix must have one column per model coordinate point."
                )
            if window_matrix.shape[0] != upper_limit.shape[0]:
                raise ValueError("window_matrix must have one row per data value.")
            object.__setattr__(
                self,
                "window_matrix",
                window_matrix,
            )
        elif upper_limit.shape[0] != coordinates.shape[0]:
            raise ValueError(
                "Without a window matrix, data values must match model coordinate points."
            )

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
    _loglikelihood: Callable[[jax.Array, jax.Array | float], jax.Array] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        data = jnp.asarray(self.data, dtype=jnp.float32)
        sigma = jnp.asarray(self.sigma, dtype=jnp.float32)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(
            self,
            "_loglikelihood",
            _build_gaussian_loglikelihood(data, sigma, self.include_normalization),
        )

    def __call__(self, theory: jax.Array, theory_sigma: jax.Array | float = 0.0) -> jax.Array:
        """
        Evaluate the log likelihood for one or more theory predictions.
        """
        return self._loglikelihood(theory, theory_sigma)


@dataclass(frozen=True)
class UpperLimitLikelihood:
    """
    One-sided upper-limit likelihood.

    A model below the reported upper limit is weakly penalized, while a model
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
    _loglikelihood: Callable[[jax.Array, jax.Array | float], jax.Array] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        upper_limit = jnp.asarray(self.upper_limit, dtype=jnp.float32)
        sigma = jnp.asarray(self.sigma, dtype=jnp.float32)
        object.__setattr__(self, "upper_limit", upper_limit)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(
            self,
            "_loglikelihood",
            _build_upper_limit_loglikelihood(upper_limit, sigma, self.min_probability),
        )

    def __call__(self, theory: jax.Array, theory_sigma: jax.Array | float = 0.0) -> jax.Array:
        """
        Evaluate the one-sided log likelihood.
        """
        return self._loglikelihood(theory, theory_sigma)


@dataclass(frozen=True)
class GlobalSignalLikelihood:
    """
    Gaussian likelihood for a global 21-cm signal emulator.

    The emulator should already be initialized on the redshift points used by
    the data. The noise model is either fixed by `sigma` at initialization or
    supplied later as `parameters["noise"]`.
    """

    emulator: EmulatorLike
    data: jax.Array
    sigma: jax.Array | None = None
    theory_fractional_error: float | jax.Array = 0.0
    name: str = "global_signal"
    _loglikelihood: Callable[[jax.Array], jax.Array] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        data = jnp.asarray(self.data, dtype=jnp.float32)
        sigma = _optional_array(self.sigma)
        theory_fractional_error = jnp.asarray(self.theory_fractional_error, dtype=jnp.float32)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "theory_fractional_error", theory_fractional_error)
        object.__setattr__(
            self,
            "_loglikelihood",
            _build_global_signal_loglikelihood(
                self.emulator,
                data,
                sigma,
                theory_fractional_error,
            ),
        )

    def __call__(self, parameters: jax.Array | Mapping[str, jax.Array]) -> jax.Array:
        """
        Evaluate the global-signal log likelihood.
        """
        return self._loglikelihood(parameters)


@dataclass(frozen=True)
class GlobalSignalForegroundLikelihood:
    """
    Global-signal likelihood with a smooth foreground and noise nuisance term.

    The parameter input should contain:
    - `astro`: parameters used by the emulator
    - `foreground`: polynomial foreground coefficients
    - `noise`: one positive noise standard deviation

    The foreground model is:

    `foreground = 10 ** sum(a_i * x**i)`

    where `x` is reduced log-frequency on `[-1, 1]`.
    """

    emulator: EmulatorLike
    data: jax.Array
    frequency: jax.Array | None = None
    reduced_frequency: jax.Array | None = None
    sigma: jax.Array | None = None
    theory_fractional_error: float | jax.Array = 0.0
    signal_scale: float = 1.0
    name: str = "global_signal_foreground"
    _loglikelihood: Callable[[Mapping[str, jax.Array]], jax.Array] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        data = jnp.asarray(self.data, dtype=jnp.float32)
        reduced_frequency = _resolve_reduced_frequency(self.frequency, self.reduced_frequency)
        if data.shape[-1] != reduced_frequency.shape[0]:
            raise ValueError("data and reduced_frequency must have matching lengths.")

        sigma = _optional_array(self.sigma)
        theory_fractional_error = jnp.asarray(self.theory_fractional_error, dtype=jnp.float32)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "reduced_frequency", reduced_frequency)
        object.__setattr__(self, "theory_fractional_error", theory_fractional_error)
        object.__setattr__(
            self,
            "_loglikelihood",
            _build_global_signal_foreground_loglikelihood(
                self.emulator,
                data,
                reduced_frequency,
                sigma,
                theory_fractional_error,
                self.signal_scale,
            ),
        )

    def __call__(self, parameters: Mapping[str, jax.Array]) -> jax.Array:
        """
        Evaluate the foreground-marginalized global-signal log likelihood.
        """
        return self._loglikelihood(parameters)


@dataclass(frozen=True)
class PowerSpectrumGaussianLikelihood:
    """
    Gaussian likelihood for power-spectrum detections.

    This is useful for mock detected power spectra or future datasets with
    symmetric measurement errors.
    """

    emulator: EmulatorLike
    data: jax.Array
    sigma: jax.Array
    window_matrix: jax.Array | None = None
    theory_fractional_error: float | jax.Array = 0.0
    name: str = "power_spectrum_gaussian"
    _loglikelihood: Callable[[jax.Array], jax.Array] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        data = jnp.asarray(self.data, dtype=jnp.float32)
        sigma = jnp.asarray(self.sigma, dtype=jnp.float32)
        window_matrix = _optional_array(self.window_matrix)
        theory_fractional_error = jnp.asarray(self.theory_fractional_error, dtype=jnp.float32)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "window_matrix", window_matrix)
        object.__setattr__(self, "theory_fractional_error", theory_fractional_error)
        object.__setattr__(
            self,
            "_loglikelihood",
            _build_power_spectrum_gaussian_loglikelihood(
                self.emulator,
                data,
                sigma,
                window_matrix,
                theory_fractional_error,
            ),
        )

    def __call__(self, parameters: jax.Array | Mapping[str, jax.Array]) -> jax.Array:
        """
        Evaluate the power-spectrum Gaussian log likelihood.
        """
        return self._loglikelihood(parameters)


@dataclass(frozen=True)
class PowerSpectrumUpperLimitLikelihood:
    """
    Upper-limit likelihood for 21-cm power spectra.

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
    _loglikelihood: Callable[[jax.Array], jax.Array] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        upper_limit = jnp.asarray(self.upper_limit, dtype=jnp.float32)
        sigma = jnp.asarray(self.sigma, dtype=jnp.float32)
        window_matrix = _optional_array(self.window_matrix)
        theory_fractional_error = jnp.asarray(self.theory_fractional_error, dtype=jnp.float32)
        object.__setattr__(self, "upper_limit", upper_limit)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "window_matrix", window_matrix)
        object.__setattr__(self, "theory_fractional_error", theory_fractional_error)
        object.__setattr__(
            self,
            "_loglikelihood",
            _build_power_spectrum_upper_limit_loglikelihood(
                self.emulator,
                upper_limit,
                sigma,
                window_matrix,
                theory_fractional_error,
                self.min_probability,
            ),
        )

    def __call__(self, parameters: jax.Array | Mapping[str, jax.Array]) -> jax.Array:
        """
        Evaluate the one-sided power-spectrum log likelihood.
        """
        return self._loglikelihood(parameters)


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
        self._loglikelihood = _build_joint_loglikelihood(self.likelihoods)

    def __call__(self, parameters: jax.Array | Mapping[str, jax.Array]) -> jax.Array:
        """
        Evaluate the summed log likelihood.
        """
        return self._loglikelihood(parameters)

    def contributions(self, parameters: jax.Array | Mapping[str, jax.Array]) -> dict[str, jax.Array]:
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
    return prediction @ window_matrix.T


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
    return fraction * jnp.abs(prediction)


def _total_sigma(observation_sigma: jax.Array, theory_sigma: jax.Array | float) -> jax.Array:
    """
    Combine observational and theory uncertainties in quadrature.
    """
    return jnp.sqrt(observation_sigma**2 + theory_sigma**2)


def _squeeze_single(values: jax.Array) -> jax.Array:
    """
    Return a scalar for one parameter row, otherwise keep the batch axis.
    """
    return jnp.squeeze(values) if values.shape == (1,) else values


def _array_1d(values: jax.Array, name: str) -> jax.Array:
    """
    Convert a value array to one float32 dimension.
    """
    array = jnp.asarray(values, dtype=jnp.float32).ravel()
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array


def _optional_array(values: jax.Array | None) -> jax.Array | None:
    """
    Convert an optional array once during likelihood setup.
    """
    return None if values is None else jnp.asarray(values, dtype=jnp.float32)


def _astro_parameters(parameters: jax.Array | Mapping[str, jax.Array]) -> jax.Array:
    """
    Return emulator parameters from either flat or grouped inputs.
    """
    if isinstance(parameters, Mapping):
        return parameters["astro"]
    return parameters


def _parameter_group(parameters: Mapping[str, jax.Array], group: str) -> jax.Array:
    """
    Return one named nuisance parameter group.
    """
    if group not in parameters:
        raise KeyError(f"Grouped likelihood parameters require a {group!r} entry.")
    return parameters[group]


def _resolve_reduced_frequency(
    frequency: jax.Array | None,
    reduced_frequency: jax.Array | None,
) -> jax.Array:
    """
    Return the reduced log-frequency coordinate used by the foreground model.
    """
    if reduced_frequency is not None:
        return jnp.asarray(reduced_frequency, dtype=jnp.float32).ravel()
    if frequency is None:
        raise ValueError("Provide either frequency or reduced_frequency.")

    log_frequency = jnp.log10(jnp.asarray(frequency, dtype=jnp.float32).ravel())
    return 2.0 * (
        (log_frequency - jnp.min(log_frequency))
        / (jnp.max(log_frequency) - jnp.min(log_frequency))
    ) - 1.0


def _polynomial_foreground(
    reduced_frequency: jax.Array,
    coefficients: jax.Array,
) -> jax.Array:
    """
    Evaluate `10 ** sum(a_i * x**i)` for foreground coefficients.
    """
    coeffs = jnp.asarray(coefficients, dtype=jnp.float32)
    powers = reduced_frequency[None, :] ** jnp.arange(coeffs.shape[-1])[:, None]
    log_foreground = coeffs @ powers
    return jnp.power(10.0, log_foreground)


def _noise_sigma(noise: jax.Array) -> jax.Array:
    """
    Return a broadcastable positive noise standard deviation.
    """
    noise_array = jnp.asarray(noise, dtype=jnp.float32)
    if noise_array.ndim == 0:
        return noise_array
    return noise_array[..., 0][..., None]


def _likelihood_noise_sigma(
    parameters: jax.Array | Mapping[str, jax.Array],
    fixed_sigma: jax.Array | None,
) -> jax.Array:
    """
    Return fixed noise or the grouped `noise` nuisance parameter.
    """
    if fixed_sigma is not None:
        return fixed_sigma
    if not isinstance(parameters, Mapping):
        raise ValueError("Global-signal likelihood requires either fixed sigma or theta['noise'].")
    return _noise_sigma(_parameter_group(parameters, "noise"))


def _build_gaussian_loglikelihood(
    data: jax.Array,
    sigma: jax.Array,
    include_normalization: bool,
) -> Callable[[jax.Array, jax.Array | float], jax.Array]:
    """
    Build the compiled diagonal Gaussian likelihood.
    """

    @jax.jit
    def loglikelihood(theory: jax.Array, theory_sigma: jax.Array | float = 0.0) -> jax.Array:
        total_sigma = _total_sigma(sigma, theory_sigma)
        residual = data - theory
        loglike = -0.5 * jnp.sum((residual / total_sigma) ** 2, axis=-1)
        if include_normalization:
            loglike -= 0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * total_sigma**2), axis=-1)
        return _squeeze_single(loglike)

    return loglikelihood


def _build_upper_limit_loglikelihood(
    upper_limit: jax.Array,
    sigma: jax.Array,
    min_probability: float,
) -> Callable[[jax.Array, jax.Array | float], jax.Array]:
    """
    Build the compiled one-sided upper-limit likelihood.
    """

    @jax.jit
    def loglikelihood(theory: jax.Array, theory_sigma: jax.Array | float = 0.0) -> jax.Array:
        total_sigma = _total_sigma(sigma, theory_sigma)
        standardized = (upper_limit - theory) / (jnp.sqrt(2.0) * total_sigma)
        probability = 0.5 * (1.0 + erf(standardized))
        probability = jnp.clip(probability, min_probability, 1.0)
        return _squeeze_single(jnp.sum(jnp.log(probability), axis=-1))

    return loglikelihood


def _build_global_signal_loglikelihood(
    emulator: EmulatorLike,
    data: jax.Array,
    sigma: jax.Array | None,
    theory_fractional_error: jax.Array,
) -> Callable[[jax.Array], jax.Array]:
    """
    Build the compiled global-signal likelihood.
    """

    @jax.jit
    def loglikelihood(parameters: jax.Array | Mapping[str, jax.Array]) -> jax.Array:
        prediction = emulator.emulate(_astro_parameters(parameters))
        theory_sigma = _fractional_theory_sigma(prediction, theory_fractional_error)
        total_sigma = _total_sigma(_likelihood_noise_sigma(parameters, sigma), theory_sigma)
        residual = data - prediction
        loglike = (
            -0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * total_sigma**2), axis=-1)
            -0.5 * jnp.sum((residual / total_sigma) ** 2, axis=-1)
        )
        return _squeeze_single(loglike)

    return loglikelihood


def _build_global_signal_foreground_loglikelihood(
    emulator: EmulatorLike,
    data: jax.Array,
    reduced_frequency: jax.Array,
    sigma: jax.Array | None,
    theory_fractional_error: jax.Array,
    signal_scale: float,
) -> Callable[[Mapping[str, jax.Array]], jax.Array]:
    """
    Build the compiled global-signal foreground likelihood.
    """

    @jax.jit
    def loglikelihood(parameters: Mapping[str, jax.Array]) -> jax.Array:
        astro = _parameter_group(parameters, "astro")
        foreground_coefficients = _parameter_group(parameters, "foreground")

        signal = emulator.emulate(astro) * signal_scale
        foreground = _polynomial_foreground(reduced_frequency, foreground_coefficients)
        theory_sigma = _fractional_theory_sigma(signal, theory_fractional_error)
        total_sigma = _total_sigma(_likelihood_noise_sigma(parameters, sigma), theory_sigma)
        residual = data - foreground - signal
        loglike = (
            -0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * total_sigma**2), axis=-1)
            -0.5 * jnp.sum((residual / total_sigma) ** 2, axis=-1)
        )
        return _squeeze_single(loglike)

    return loglikelihood


def _build_power_spectrum_gaussian_loglikelihood(
    emulator: EmulatorLike,
    data: jax.Array,
    sigma: jax.Array,
    window_matrix: jax.Array | None,
    theory_fractional_error: jax.Array,
) -> Callable[[jax.Array], jax.Array]:
    """
    Build the compiled power-spectrum Gaussian likelihood.
    """
    gaussian = _build_gaussian_loglikelihood(data, sigma, include_normalization=True)

    @jax.jit
    def loglikelihood(parameters: jax.Array | Mapping[str, jax.Array]) -> jax.Array:
        prediction = _apply_window(emulator.emulate(_astro_parameters(parameters)), window_matrix)
        theory_sigma = _fractional_theory_sigma(prediction, theory_fractional_error)
        return gaussian(prediction, theory_sigma)

    return loglikelihood


def _build_power_spectrum_upper_limit_loglikelihood(
    emulator: EmulatorLike,
    upper_limit: jax.Array,
    sigma: jax.Array,
    window_matrix: jax.Array | None,
    theory_fractional_error: jax.Array,
    min_probability: float,
) -> Callable[[jax.Array], jax.Array]:
    """
    Build the compiled power-spectrum upper-limit likelihood.
    """
    upper_limit_likelihood = _build_upper_limit_loglikelihood(
        upper_limit,
        sigma,
        min_probability,
    )

    @jax.jit
    def loglikelihood(parameters: jax.Array | Mapping[str, jax.Array]) -> jax.Array:
        prediction = _apply_window(emulator.emulate(_astro_parameters(parameters)), window_matrix)
        theory_sigma = _fractional_theory_sigma(prediction, theory_fractional_error)
        return upper_limit_likelihood(prediction, theory_sigma)

    return loglikelihood


def _build_joint_loglikelihood(
    likelihoods: tuple[object, ...],
) -> Callable[[jax.Array], jax.Array]:
    """
    Build the compiled joint likelihood.
    """

    @jax.jit
    def loglikelihood(parameters: jax.Array | Mapping[str, jax.Array]) -> jax.Array:
        total = 0.0
        for likelihood in likelihoods:
            total = total + likelihood(parameters)
        return total

    return loglikelihood
