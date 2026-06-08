"""
BlackJAX nested-sampling adapter.

This module keeps sampler-specific code separate from emulator likelihoods.
The likelihood receives physical parameters; the sampler works on the unit
cube and uses `PriorSpec.transform(...)` to move between the two spaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

from jax_emu.inference.prior import PriorSpec


@dataclass(frozen=True)
class NestedSamplingResult:
    """
    Storage utility for nested-sampling outputs.

    Parameters
    ----------
    state:
        Final raw BlackJAX sampler state.
    info:
        Final raw BlackJAX step information.
    unit_samples:
        Samples in unit-cube coordinates when available.
    physical_samples:
        Samples transformed into physical parameter space when available.
    loglikelihood:
        Log-likelihood values when available.
    logz:
        Evidence estimate when exposed by the BlackJAX state.
    logz_error:
        Evidence uncertainty when exposed by the BlackJAX state.
    """

    state: Any
    info: Any
    unit_samples: jax.Array | None = None
    physical_samples: jax.Array | None = None
    loglikelihood: jax.Array | None = None
    logz: jax.Array | None = None
    logz_error: jax.Array | None = None


def make_unit_cube_loglikelihood(
    prior: PriorSpec,
    likelihood: Callable[[jax.Array], jax.Array],
) -> Callable[[jax.Array], jax.Array]:
    """
    Build the sampler-facing log-likelihood function.

    BlackJAX explores unit-cube coordinates. The emulator likelihood expects
    physical astrophysical parameters, so this closure performs the prior
    transform before calling the likelihood.
    """

    def loglikelihood(unit_cube: jax.Array) -> jax.Array:
        physical_parameters = prior.transform(unit_cube)
        return likelihood(physical_parameters)

    return loglikelihood


def run_nested_sampling(
    *,
    prior: PriorSpec,
    likelihood: Callable[[jax.Array], jax.Array],
    key: jax.Array,
    n_live: int = 1000,
    max_steps: int = 1000,
    num_delete: int = 50,
    num_inner_steps: int | None = None,
    initial_live_points: jax.Array | None = None,
) -> NestedSamplingResult:
    """
    Run a BlackJAX nested sampler over emulator parameters.

    Parameters
    ----------
    prior:
        Prior transform from the unit cube to physical parameters.
    likelihood:
        Callable returning the log likelihood for physical parameters.
    key:
        JAX random key.
    n_live:
        Number of live points.
    max_steps:
        Maximum number of nested-sampling iterations.
    num_delete:
        Number of live points replaced per iteration for APIs that expose this
        setting.
    num_inner_steps:
        Number of inner MCMC steps used to generate replacement live points.
        Defaults to `3 * prior.ndim`.
    initial_live_points:
        Optional user-supplied unit-cube live points with shape
        `(n_live, prior.ndim)`.

    Returns
    -------
    NestedSamplingResult
        Raw final sampler state plus extracted samples/evidence when available.
    """
    try:
        import blackjax
    except ImportError as exc:
        raise ImportError(
            "BlackJAX is required for nested sampling. Install the inference "
            "extra, for example `python -m pip install -e '.[cpu,inference]'`."
        ) from exc

    if prior.ndim <= 0:
        raise ValueError("Nested sampling requires at least one sampled prior dimension.")

    n_inner = 3 * prior.ndim if num_inner_steps is None else num_inner_steps
    unit_loglikelihood = make_unit_cube_loglikelihood(prior, likelihood)
    unit_logprior = _unit_cube_logprior

    algorithm_builder = getattr(blackjax, "nss", None)
    if algorithm_builder is None:
        raise AttributeError(
            "This BlackJAX installation does not expose `blackjax.nss`. "
            "The inference layer is ready for BlackJAX nested sampling, but "
            "the exact sampler API must match the installed BlackJAX version."
        )

    algorithm = algorithm_builder(
        logprior_fn=unit_logprior,
        loglikelihood_fn=unit_loglikelihood,
        num_delete=num_delete,
        num_inner_steps=n_inner,
    )

    if initial_live_points is None:
        key, init_key = jax.random.split(key)
        live_points = jax.random.uniform(init_key, shape=(n_live, prior.ndim))
    else:
        live_points = jnp.asarray(initial_live_points, dtype=jnp.float32)
        if live_points.shape != (n_live, prior.ndim):
            raise ValueError(
                "initial_live_points must have shape "
                f"({n_live}, {prior.ndim}), received {live_points.shape}."
            )

    state = algorithm.init(live_points)
    step = jax.jit(algorithm.step)
    info = None

    for _ in range(max_steps):
        key, step_key = jax.random.split(key)
        state, info = step(step_key, state)

    return _result_from_state(prior, state, info)


def _unit_cube_logprior(unit_cube: jax.Array) -> jax.Array:
    """
    Uniform log prior on the unit cube.
    """
    cube = jnp.asarray(unit_cube)
    inside = jnp.all((cube >= 0.0) & (cube <= 1.0), axis=-1)
    return jnp.where(inside, 0.0, -jnp.inf)


def _result_from_state(prior: PriorSpec, state: Any, info: Any) -> NestedSamplingResult:
    """
    Extract common nested-sampling fields from a BlackJAX state.
    """
    unit_samples = _first_existing_attribute(state, ("particles", "live_points", "samples"))
    loglikelihood = _first_existing_attribute(
        state,
        ("loglikelihood", "log_likelihood", "log_likelihoods", "logl"),
    )
    logz = _first_existing_attribute(state, ("logZ", "logz", "log_evidence"))
    logz_error = _first_existing_attribute(state, ("logZerr", "logz_error", "log_evidence_error"))

    physical_samples = None
    if unit_samples is not None:
        physical_samples = prior.transform(unit_samples)

    return NestedSamplingResult(
        state=state,
        info=info,
        unit_samples=unit_samples,
        physical_samples=physical_samples,
        loglikelihood=loglikelihood,
        logz=logz,
        logz_error=logz_error,
    )


def _first_existing_attribute(obj: Any, names: tuple[str, ...]) -> Any | None:
    """
    Return the first available attribute from an object or mapping.
    """
    if obj is None:
        return None
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None
