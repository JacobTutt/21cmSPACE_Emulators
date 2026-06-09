"""
BlackJAX nested-sampling adapter.

This module keeps sampler-specific code separate from emulator likelihoods.
The likelihood receives physical parameters; the sampler works on the unit
cube and uses `PriorSpec.transform(...)` to move between the two spaces.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from anesthetic import NestedSamples
from blackjax.ns.utils import finalise, log_weights
from tqdm import tqdm

from jax_emu.inference.prior import PriorSpec


@dataclass(frozen=True)
class NestedSamplingConfig:
    """
    Settings for a BlackJAX nested-sampling run.

    Parameters
    ----------
    n_live:
        Explicit number of live points. If omitted, this is calculated as
        `prior.ndim * n_live_scale`.
    n_live_scale:
        Number of live points per sampled dimension.
    num_delete:
        Explicit number of live points replaced at each nested-sampling step.
        If omitted, this is calculated from `num_delete_fraction`.
    num_delete_fraction:
        Fraction of live points replaced at each nested-sampling step.
    num_inner_steps:
        Explicit number of inner MCMC steps used to find replacement live
        points. If omitted, this is calculated as
        `prior.ndim * num_inner_steps_scale`.
    num_inner_steps_scale:
        Number of inner MCMC steps per sampled dimension.
    logz_live_threshold:
        Stop when `logZ_live - logZ` falls below this value.
    max_steps:
        Optional safety limit on the number of nested-sampling iterations.
    finalise_on_cpu:
        Move finalisation of dead and live points to CPU to avoid device memory
        spikes for large runs.
    log_weight_samples:
        Number of stochastic log-weight realisations used to estimate evidence
        uncertainty.
    output_dir:
        Optional directory where anesthetic-compatible outputs are written.
    run_name:
        Name stored in output metadata.
    progress_bar:
        Whether to print a tqdm progress bar during nested sampling.
    """

    n_live: int | None = None
    n_live_scale: int = 25
    num_delete: int | None = None
    num_delete_fraction: float = 0.2
    num_inner_steps: int | None = None
    num_inner_steps_scale: int = 5
    logz_live_threshold: float = -3.0
    max_steps: int | None = None
    finalise_on_cpu: bool = True
    log_weight_samples: int = 100
    output_dir: str | Path | None = None
    run_name: str = "nested_sampling"
    progress_bar: bool = True


@dataclass(frozen=True)
class NestedSamplingSettings:
    """
    Concrete sampler settings resolved from a config and prior dimension.
    """

    n_live: int
    num_delete: int
    num_inner_steps: int
    logz_live_threshold: float
    max_steps: int | None
    finalise_on_cpu: bool
    log_weight_samples: int
    output_dir: str | Path | None
    run_name: str
    progress_bar: bool


@dataclass(frozen=True)
class NestedSamplingResult:
    """
    Storage utility for nested-sampling outputs.

    Parameters
    ----------
    state:
        Final live-point BlackJAX sampler state.
    final_state:
        Finalised dead plus live BlackJAX nested-sampling information.
    info:
        Final raw BlackJAX step information.
    dead_points:
        Raw dead-point information collected during the run.
    unit_samples:
        Samples in unit-cube coordinates when available.
    physical_samples:
        Samples transformed into physical parameter space when available.
    loglikelihood:
        Log-likelihood values when available.
    loglikelihood_birth:
        Birth contour log-likelihood values used by anesthetic.
    logprior:
        Log-prior values for the unit-cube samples.
    logz:
        Evidence estimate.
    logz_error:
        Evidence uncertainty.
    n_steps:
        Number of nested-sampling iterations run.
    converged:
        Whether the evidence stopping rule was met before `max_steps`.
    settings:
        Concrete sampler settings used for this run.
    """

    state: Any
    final_state: Any | None
    info: Any
    dead_points: tuple[Any, ...] = ()
    unit_samples: jax.Array | None = None
    physical_samples: jax.Array | None = None
    loglikelihood: jax.Array | None = None
    loglikelihood_birth: jax.Array | None = None
    logprior: jax.Array | None = None
    logz: jax.Array | None = None
    logz_error: jax.Array | None = None
    n_steps: int = 0
    converged: bool = False
    settings: NestedSamplingSettings | None = None


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
    config: NestedSamplingConfig | None = None,
    n_live: int | None = None,
    max_steps: int | None = None,
    num_delete: int | None = None,
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
    config:
        Optional nested-sampling settings. Explicit keyword arguments below
        override matching values in the config.
    n_live:
        Number of live points. Defaults to `prior.ndim * config.n_live_scale`.
    max_steps:
        Optional safety limit on the number of nested-sampling iterations.
    num_delete:
        Number of live points replaced per iteration. Defaults to a fraction of
        `n_live`.
    num_inner_steps:
        Number of inner MCMC steps used to generate replacement live points.
        Defaults to `prior.ndim * config.num_inner_steps_scale`.
    initial_live_points:
        Optional user-supplied unit-cube live points with shape
        `(n_live, prior.ndim)`.

    Returns
    -------
    NestedSamplingResult
        Raw final sampler state plus extracted samples/evidence when available.
    """
    if prior.ndim <= 0:
        raise ValueError("Nested sampling requires at least one sampled prior dimension.")

    base_config = NestedSamplingConfig() if config is None else config
    if n_live is not None:
        base_config = replace(base_config, n_live=n_live)
    if max_steps is not None:
        base_config = replace(base_config, max_steps=max_steps)
    if num_delete is not None:
        base_config = replace(base_config, num_delete=num_delete)
    if num_inner_steps is not None:
        base_config = replace(base_config, num_inner_steps=num_inner_steps)

    settings = resolve_nested_sampling_settings(prior, base_config)
    unit_loglikelihood = make_unit_cube_loglikelihood(prior, likelihood)

    algorithm = blackjax.nss(
        logprior_fn=_unit_cube_logprior,
        loglikelihood_fn=unit_loglikelihood,
        num_delete=settings.num_delete,
        num_inner_steps=settings.num_inner_steps,
    )

    if initial_live_points is None:
        key, init_key = jax.random.split(key)
        live_points = jax.random.uniform(init_key, shape=(settings.n_live, prior.ndim))
    else:
        live_points = jnp.asarray(initial_live_points, dtype=jnp.float32)
        if live_points.shape != (settings.n_live, prior.ndim):
            raise ValueError(
                "initial_live_points must have shape "
                f"({settings.n_live}, {prior.ndim}), received {live_points.shape}."
            )

    state = algorithm.init(live_points)
    step = jax.jit(algorithm.step)
    info = None
    dead_points: list[Any] = []
    n_steps = 0

    progress = tqdm(
        total=settings.max_steps,
        desc=settings.run_name,
        unit="step",
        disable=not settings.progress_bar,
    )
    while _should_continue_nested_sampling(state, settings, n_steps):
        key, step_key = jax.random.split(key)
        state, info = step(step_key, state)
        dead_points.append(info)
        n_steps += 1
        progress.update(1)
        progress.set_postfix(
            logz=float(state.logZ),
            delta_logz=_evidence_delta(state),
            refresh=False,
        )
    progress.close()

    converged = _evidence_delta(state) < settings.logz_live_threshold
    result = _result_from_state(
        prior,
        state,
        tuple(dead_points),
        info,
        settings,
        key,
        n_steps,
        converged,
    )

    if settings.output_dir is not None:
        save_anesthetic_samples(result, prior, settings.output_dir)

    return result


def resolve_nested_sampling_settings(
    prior: PriorSpec,
    config: NestedSamplingConfig | None = None,
) -> NestedSamplingSettings:
    """
    Resolve dimension-scaled nested-sampling settings into concrete integers.
    """
    cfg = NestedSamplingConfig() if config is None else config
    if prior.ndim <= 0:
        raise ValueError("Nested sampling requires at least one sampled prior dimension.")

    n_live = cfg.n_live if cfg.n_live is not None else prior.ndim * cfg.n_live_scale
    num_delete = (
        cfg.num_delete
        if cfg.num_delete is not None
        else max(1, int(n_live * cfg.num_delete_fraction))
    )
    num_inner_steps = (
        cfg.num_inner_steps
        if cfg.num_inner_steps is not None
        else prior.ndim * cfg.num_inner_steps_scale
    )

    if n_live < 1:
        raise ValueError("n_live must be at least 1.")
    if num_delete < 1:
        raise ValueError("num_delete must be at least 1.")
    if num_delete > n_live:
        raise ValueError("num_delete cannot exceed n_live.")
    if num_inner_steps < 1:
        raise ValueError("num_inner_steps must be at least 1.")
    if cfg.log_weight_samples < 1:
        raise ValueError("log_weight_samples must be at least 1.")
    if cfg.max_steps is not None and cfg.max_steps < 0:
        raise ValueError("max_steps must be non-negative when provided.")

    return NestedSamplingSettings(
        n_live=int(n_live),
        num_delete=int(num_delete),
        num_inner_steps=int(num_inner_steps),
        logz_live_threshold=float(cfg.logz_live_threshold),
        max_steps=cfg.max_steps,
        finalise_on_cpu=cfg.finalise_on_cpu,
        log_weight_samples=cfg.log_weight_samples,
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,
        progress_bar=cfg.progress_bar,
    )


def save_anesthetic_samples(
    result: NestedSamplingResult,
    prior: PriorSpec,
    output_dir: str | Path,
) -> Path:
    """
    Save nested-sampling results in an anesthetic-friendly directory.

    The main output is `nested_sampling_results.csv`, which stores physical
    parameters, `logL`, and `logL_birth` in the format read by
    `anesthetic.NestedSamples`.
    """
    if result.physical_samples is None:
        raise ValueError("Cannot save anesthetic samples without physical samples.")
    if result.loglikelihood is None:
        raise ValueError("Cannot save anesthetic samples without log-likelihood values.")
    if result.loglikelihood_birth is None:
        raise ValueError("Cannot save anesthetic samples without log-likelihood birth values.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    names = prior.names
    labels = dict(zip(names, prior.labels, strict=True))
    nested_samples = NestedSamples(
        data=np.asarray(result.physical_samples),
        logL=np.asarray(result.loglikelihood),
        logL_birth=np.asarray(result.loglikelihood_birth),
        columns=list(names),
        labels=labels,
    )

    csv_path = output_path / "nested_sampling_results.csv"
    nested_samples.to_csv(csv_path)

    _write_parameter_names(output_path / "parameter_names.json", prior)
    _write_sampler_config(output_path / "sampler_config.json", result)
    _write_test_stats(output_path / "test_stats.txt", nested_samples, result, prior)

    return csv_path


def _unit_cube_logprior(unit_cube: jax.Array) -> jax.Array:
    """
    Uniform log prior on the unit cube.
    """
    cube = jnp.asarray(unit_cube)
    inside = jnp.all((cube >= 0.0) & (cube <= 1.0), axis=-1)
    return jnp.where(inside, 0.0, -jnp.inf)


def _result_from_state(
    prior: PriorSpec,
    state: Any,
    dead_points: tuple[Any, ...],
    info: Any,
    settings: NestedSamplingSettings,
    key: jax.Array,
    n_steps: int,
    converged: bool,
) -> NestedSamplingResult:
    """
    Finalise and extract common nested-sampling fields from a BlackJAX state.
    """
    if settings.finalise_on_cpu:
        with jax.default_device(jax.devices("cpu")[0]):
            final_state = finalise(state, list(dead_points))
    else:
        final_state = finalise(state, list(dead_points))

    unit_samples = _first_existing_attribute(final_state, ("particles", "live_points", "samples"))
    loglikelihood = _first_existing_attribute(
        final_state,
        ("loglikelihood", "log_likelihood", "log_likelihoods", "logl"),
    )
    loglikelihood_birth = _first_existing_attribute(
        final_state,
        ("loglikelihood_birth", "log_likelihood_birth", "logl_birth"),
    )
    logprior = _first_existing_attribute(final_state, ("logprior", "log_prior"))
    logz = _first_existing_attribute(state, ("logZ", "logz", "log_evidence"))
    logz_error = _first_existing_attribute(state, ("logZerr", "logz_error", "log_evidence_error"))

    physical_samples = None
    if unit_samples is not None:
        physical_samples = _physical_samples_array(prior, unit_samples)

    if settings.log_weight_samples > 0:
        log_weight = log_weights(key, final_state, shape=settings.log_weight_samples)
        logz_samples = jax.scipy.special.logsumexp(log_weight, axis=0)
        logz = jnp.mean(logz_samples)
        logz_error = jnp.std(logz_samples)

    return NestedSamplingResult(
        state=state,
        final_state=final_state,
        info=info,
        dead_points=dead_points,
        unit_samples=unit_samples,
        physical_samples=physical_samples,
        loglikelihood=loglikelihood,
        loglikelihood_birth=loglikelihood_birth,
        logprior=logprior,
        logz=logz,
        logz_error=logz_error,
        n_steps=n_steps,
        converged=converged,
        settings=settings,
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


def _physical_samples_array(prior: PriorSpec, unit_samples: jax.Array) -> jax.Array:
    """
    Transform unit-cube samples into a flat physical parameter matrix.
    """
    transformed = prior.transform(unit_samples)
    if isinstance(transformed, dict):
        return jnp.concatenate([transformed[group] for group in prior.groups], axis=-1)
    return transformed


def _should_continue_nested_sampling(
    state: Any,
    settings: NestedSamplingSettings,
    n_steps: int,
) -> bool:
    """
    Return whether the nested sampler should take another step.
    """
    if settings.max_steps is not None and n_steps >= settings.max_steps:
        return False
    return _evidence_delta(state) >= settings.logz_live_threshold


def _evidence_delta(state: Any) -> float:
    """
    Return the live evidence contribution relative to accumulated evidence.
    """
    return float(state.logZ_live - state.logZ)


def _write_parameter_names(path: Path, prior: PriorSpec) -> None:
    """
    Save parameter names grouped in the same way as the prior definition.
    """
    if prior.is_grouped:
        grouped = {
            group: [f"{group}.{param.name}" for param in params]
            for group, params in zip(prior.groups, prior.grouped_priors, strict=True)
        }
    else:
        grouped = {"parameters": list(prior.names)}
    grouped["all"] = list(prior.names)
    grouped["sampled"] = list(prior.sampled_names)

    with path.open("w", encoding="utf-8") as file:
        json.dump(grouped, file, indent=2)


def _write_sampler_config(path: Path, result: NestedSamplingResult) -> None:
    """
    Save sampler settings and run status as JSON.
    """
    payload: dict[str, Any] = {
        "n_steps": result.n_steps,
        "converged": result.converged,
        "logz": None if result.logz is None else float(result.logz),
        "logz_error": None if result.logz_error is None else float(result.logz_error),
    }
    if result.settings is not None:
        payload["settings"] = asdict(result.settings)
        if payload["settings"]["output_dir"] is not None:
            payload["settings"]["output_dir"] = str(payload["settings"]["output_dir"])

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _write_test_stats(
    path: Path,
    nested_samples: Any,
    result: NestedSamplingResult,
    prior: PriorSpec,
) -> None:
    """
    Save a compact text summary for quick run inspection.
    """
    with path.open("w", encoding="utf-8") as file:
        file.write("# Nested Sampling Test Statistics\n")
        file.write(f"Number of Dimensions: {prior.ndim}\n")
        if result.settings is not None:
            file.write(f"Number of Live Points: {result.settings.n_live}\n")
            file.write(f"Number Deleted Per Step: {result.settings.num_delete}\n")
            file.write(f"Number of Inner Steps: {result.settings.num_inner_steps}\n")
        file.write(f"Converged: {result.converged}\n")
        file.write(f"Steps: {result.n_steps}\n")
        if result.logz is not None and result.logz_error is not None:
            file.write(f"LogZ: {float(result.logz):.6f} +/- {float(result.logz_error):.6f}\n")
        for name in prior.names:
            mean = nested_samples[name].mean()
            std = nested_samples[name].std()
            file.write(f"{name}: Mean = {mean}, Std = {std}\n")
