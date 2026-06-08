"""
Prior definitions for emulator inference.

Nested samplers usually explore a unit cube. These utilities define how each
unit-cube coordinate maps onto the physical astrophysical parameters passed to
the emulator likelihood.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp


# Type Definitions
# ----------------
# Supported prior families.

PriorKind = Literal["uniform", "log_uniform", "discrete", "fixed"]


@dataclass(frozen=True)
class ParameterPrior:
    """
    Storage utility for one prior transform.

    Parameters
    ----------
    name:
        Name of the physical parameter.
    kind:
        Prior family used to transform the unit-cube coordinate.
    lower, upper:
        Bounds for continuous priors.
    values:
        Allowed values for discrete priors.
    value:
        Constant value for fixed parameters.
    label:
        Optional plotting label.
    """

    name: str
    kind: PriorKind
    lower: float | None = None
    upper: float | None = None
    values: tuple[float, ...] | None = None
    value: float | None = None
    label: str | None = None

    @property
    def ndim(self) -> int:
        """
        Return the number of unit-cube dimensions consumed by this prior.
        """
        # Fixed parameters do not need to be sampled.
        return 0 if self.kind == "fixed" else 1

    def transform(self, unit_value: jax.Array | None = None) -> jax.Array:
        """
        Transform one unit-cube coordinate into a physical parameter value.
        """
        if self.kind == "fixed":
            if self.value is None:
                raise ValueError(f"Fixed prior {self.name!r} requires value.")
            return jnp.asarray(self.value, dtype=jnp.float32)

        if unit_value is None:
            raise ValueError(f"Prior {self.name!r} requires a unit-cube value.")

        u = jnp.asarray(unit_value, dtype=jnp.float32)

        if self.kind == "uniform":
            lower, upper = _require_bounds(self)
            return lower + u * (upper - lower)

        if self.kind == "log_uniform":
            lower, upper = _require_bounds(self)
            return jnp.exp(jnp.log(lower) + u * (jnp.log(upper) - jnp.log(lower)))

        if self.kind == "discrete":
            if self.values is None or len(self.values) == 0:
                raise ValueError(f"Discrete prior {self.name!r} requires values.")
            values = jnp.asarray(self.values, dtype=jnp.float32)
            index = jnp.floor(u * values.size).astype(jnp.int32)
            index = jnp.clip(index, 0, values.size - 1)
            return values[index]

        raise ValueError(f"Unsupported prior kind {self.kind!r}.")


def UniformPrior(name: str, lower: float, upper: float, *, label: str | None = None) -> ParameterPrior:
    """
    Build a uniform prior over a linear parameter interval.
    """
    return ParameterPrior(name=name, kind="uniform", lower=lower, upper=upper, label=label)


def LogUniformPrior(
    name: str,
    lower: float,
    upper: float,
    *,
    label: str | None = None,
) -> ParameterPrior:
    """
    Build a log-uniform prior over a positive physical parameter.
    """
    return ParameterPrior(name=name, kind="log_uniform", lower=lower, upper=upper, label=label)


def DiscretePrior(
    name: str,
    values: tuple[float, ...] | list[float],
    *,
    label: str | None = None,
) -> ParameterPrior:
    """
    Build a categorical prior from a finite set of allowed values.
    """
    return ParameterPrior(name=name, kind="discrete", values=tuple(values), label=label)


def FixedPrior(name: str, value: float, *, label: str | None = None) -> ParameterPrior:
    """
    Build a fixed parameter value that is not sampled.
    """
    return ParameterPrior(name=name, kind="fixed", value=value, label=label)


@dataclass(frozen=True)
class PriorSpec:
    """
    Ordered collection of parameter priors.

    The order defines the unit-cube sampling coordinates. Priors can be supplied
    either as a flat list or as named groups such as `astro`, `foreground`, and
    `noise`.
    """

    priors: tuple[ParameterPrior, ...]
    groups: tuple[str, ...]
    grouped_priors: tuple[tuple[ParameterPrior, ...], ...]

    def __init__(
        self,
        priors: (
            tuple[ParameterPrior, ...]
            | list[ParameterPrior]
            | Mapping[str, tuple[ParameterPrior, ...] | list[ParameterPrior]]
        ),
    ) -> None:
        if isinstance(priors, Mapping):
            groups = tuple(priors.keys())
            grouped_priors = tuple(tuple(group_priors) for group_priors in priors.values())
            flat_priors = tuple(prior for group in grouped_priors for prior in group)
        else:
            groups = ("parameters",)
            grouped_priors = (tuple(priors),)
            flat_priors = tuple(priors)

        object.__setattr__(self, "priors", flat_priors)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "grouped_priors", grouped_priors)
        self._validate()

    @property
    def names(self) -> tuple[str, ...]:
        """
        Return physical parameter names in likelihood order.
        """
        if self.is_grouped:
            return tuple(
                f"{group}.{prior.name}"
                for group, priors in zip(self.groups, self.grouped_priors, strict=True)
                for prior in priors
            )
        return tuple(prior.name for prior in self.priors)

    @property
    def sampled_names(self) -> tuple[str, ...]:
        """
        Return names for parameters represented in the unit cube.
        """
        if self.is_grouped:
            return tuple(
                f"{group}.{prior.name}"
                for group, priors in zip(self.groups, self.grouped_priors, strict=True)
                for prior in priors
                if prior.ndim == 1
            )
        return tuple(prior.name for prior in self.priors if prior.ndim == 1)

    @property
    def ndim(self) -> int:
        """
        Return the number of unit-cube dimensions required by this prior.
        """
        return sum(prior.ndim for prior in self.priors)

    @property
    def labels(self) -> tuple[str, ...]:
        """
        Return plotting labels in physical parameter order.
        """
        return tuple(prior.label or prior.name for prior in self.priors)

    @property
    def is_grouped(self) -> bool:
        """
        Return whether the transform produces a structured parameter object.
        """
        return self.groups != ("parameters",)

    def transform(self, unit_cube: jax.Array) -> jax.Array | dict[str, jax.Array]:
        """
        Map a unit-cube vector onto physical parameter values.
        """
        cube = jnp.asarray(unit_cube, dtype=jnp.float32)
        if cube.shape[-1] != self.ndim:
            raise ValueError(f"Expected unit cube with width {self.ndim}, received {cube.shape[-1]}.")

        cube_index = 0
        grouped_outputs = {}
        for group_name, group_priors in zip(self.groups, self.grouped_priors, strict=True):
            outputs = []
            for prior in group_priors:
                if prior.ndim == 0:
                    fixed_value = prior.transform()
                    if cube.shape[:-1]:
                        fixed_value = jnp.broadcast_to(fixed_value, cube.shape[:-1])
                    outputs.append(fixed_value)
                else:
                    outputs.append(prior.transform(cube[..., cube_index]))
                    cube_index += 1
            grouped_outputs[group_name] = jnp.stack(outputs, axis=-1)

        if self.is_grouped:
            return grouped_outputs
        return grouped_outputs["parameters"]

    def log_prior(self, physical_parameters: jax.Array | Mapping[str, jax.Array]) -> jax.Array:
        """
        Return zero inside the prior support and `-inf` outside it.

        Nested sampling mainly needs the unit-cube transform. This helper is
        still useful for diagnostics or samplers that expect a log prior.
        """
        if self.is_grouped:
            if not isinstance(physical_parameters, Mapping):
                raise ValueError("Grouped priors require grouped physical parameters.")
            params_by_group = {
                group: jnp.asarray(physical_parameters[group], dtype=jnp.float32)
                for group in self.groups
            }
        else:
            params_by_group = {
                "parameters": jnp.asarray(physical_parameters, dtype=jnp.float32)
            }

        first_params = next(iter(params_by_group.values()))
        in_support = jnp.ones(first_params.shape[:-1], dtype=bool)
        for group, group_priors in zip(self.groups, self.grouped_priors, strict=True):
            params = params_by_group[group]
            if params.shape[-1] != len(group_priors):
                raise ValueError(
                    f"Expected group {group!r} with width {len(group_priors)}, "
                    f"received {params.shape[-1]}."
                )
            for index, prior in enumerate(group_priors):
                values = params[..., index]
                if prior.kind in {"uniform", "log_uniform"}:
                    lower, upper = _require_bounds(prior)
                    in_support = in_support & (values >= lower) & (values <= upper)
                elif prior.kind == "discrete":
                    allowed = jnp.asarray(prior.values, dtype=jnp.float32)
                    in_support = in_support & jnp.any(values[..., None] == allowed, axis=-1)
                elif prior.kind == "fixed":
                    in_support = in_support & (values == prior.value)

        return jnp.where(in_support, 0.0, -jnp.inf)

    def _validate(self) -> None:
        """
        Check that parameter names are unique and prior definitions are valid.
        """
        names = list(self.names)
        if len(set(names)) != len(names):
            raise ValueError("Prior names must be unique.")
        if len(set(self.groups)) != len(self.groups):
            raise ValueError("Prior group names must be unique.")
        for prior in self.priors:
            if prior.kind in {"uniform", "log_uniform"}:
                lower, upper = _require_bounds(prior)
                if lower >= upper:
                    raise ValueError(f"Prior {prior.name!r} has invalid bounds.")
                if prior.kind == "log_uniform" and lower <= 0:
                    raise ValueError(f"Log-uniform prior {prior.name!r} needs lower > 0.")
            if prior.kind == "discrete" and (prior.values is None or len(prior.values) == 0):
                raise ValueError(f"Discrete prior {prior.name!r} requires values.")
            if prior.kind == "fixed" and prior.value is None:
                raise ValueError(f"Fixed prior {prior.name!r} requires value.")


def _require_bounds(prior: ParameterPrior) -> tuple[jax.Array, jax.Array]:
    """
    Return continuous prior bounds as JAX arrays.
    """
    if prior.lower is None or prior.upper is None:
        raise ValueError(f"Prior {prior.name!r} requires lower and upper bounds.")
    return (
        jnp.asarray(prior.lower, dtype=jnp.float32),
        jnp.asarray(prior.upper, dtype=jnp.float32),
    )
