"""Repository-level emulator specifications.

These dataclasses are intentionally small, but they carry a large part of the
design intent for the repository:

- they describe the *physical* inputs seen by an emulator
- they record which variables are transformed before training
- they define the canonical feature order expected by model code

Keeping this logic in one place makes later data-loader and checkpoint code
easier to reason about and avoids repeating legacy naming conventions in
multiple modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


TransformName = Literal["identity", "log10"]
FamilyName = Literal["power_spectrum", "global_signal"]


@dataclass(frozen=True)
class AxisSpec:
    """Specification for a physical axis used by an emulator.

    Examples are redshift, wavenumber, or observed frequency. The `transform`
    field describes how the axis is represented in feature space, not how it is
    stored in raw science files.
    """

    name: str
    transform: TransformName = "identity"
    limits: tuple[float, float] | None = None
    nsample: int | None = None

    def feature_name(self) -> str:
        """Return the feature name after transformation.

        The old code frequently referred to transformed quantities as
        ``log10k`` or ``log10fradio``. We preserve that convention here so the
        new code can mirror legacy feature ordering and remain readable.
        """
        return self.name if self.transform == "identity" else f"{self.transform}{self.name}"

    def validate(self) -> None:
        """Validate internal consistency."""
        if not self.name:
            raise ValueError("Axis name must be non-empty.")
        if self.limits is not None and self.limits[0] >= self.limits[1]:
            raise ValueError(f"Axis {self.name} has invalid limits {self.limits}.")
        if self.nsample is not None and self.nsample <= 0:
            raise ValueError(f"Axis {self.name} must have positive nsample.")


@dataclass(frozen=True)
class ParameterSpec:
    """Specification for a model parameter.

    `discrete_values` is used for parameters that were treated as discrete in
    the legacy training scripts, such as `alpha`, `nu_0`, and `pop`.
    """

    name: str
    transform: TransformName = "identity"
    discrete_values: tuple[float, ...] | None = None

    def feature_name(self) -> str:
        """Return the parameter name after transformation."""
        return self.name if self.transform == "identity" else f"{self.transform}{self.name}"

    def validate(self) -> None:
        """Validate internal consistency."""
        if not self.name:
            raise ValueError("Parameter name must be non-empty.")
        if self.discrete_values is not None and len(self.discrete_values) == 0:
            raise ValueError(f"Parameter {self.name} cannot have empty discrete_values.")


@dataclass(frozen=True)
class EmulatorSpec:
    """High-level emulator definition used by training and inference code.

    This is the contract we want real loaders, trainers, checkpoints, and
    inference code to agree on. The main benefit is that the *scientific* input
    semantics are described once, while the implementation details can evolve
    underneath.
    """

    name: str
    family: FamilyName
    axes: tuple[AxisSpec, ...] = field(default_factory=tuple)
    parameters: tuple[ParameterSpec, ...] = field(default_factory=tuple)
    target_transform: TransformName = "identity"
    target_offset: float = 0.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate emulator configuration consistency.

        Most checks here are guarding against ambiguous feature ordering. That
        matters because even a correct model architecture will behave
        nonsensically if two features collide after log-transformation or if an
        axis name is accidentally reused as a parameter name.
        """
        if not self.name:
            raise ValueError("Emulator name must be non-empty.")
        if not self.axes:
            raise ValueError(f"Emulator {self.name} must define at least one axis.")
        if not self.parameters:
            raise ValueError(f"Emulator {self.name} must define at least one parameter.")

        axis_names = [axis.name for axis in self.axes]
        parameter_names = [parameter.name for parameter in self.parameters]
        transformed_names = self.input_feature_names()

        if len(set(axis_names)) != len(axis_names):
            raise ValueError(f"Emulator {self.name} has duplicate axis names.")
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError(f"Emulator {self.name} has duplicate parameter names.")
        if set(axis_names) & set(parameter_names):
            raise ValueError(f"Emulator {self.name} reuses names between axes and parameters.")
        if len(set(transformed_names)) != len(transformed_names):
            raise ValueError(
                f"Emulator {self.name} has colliding transformed input feature names."
            )

        for axis in self.axes:
            axis.validate()
        for parameter in self.parameters:
            parameter.validate()

    def input_feature_names(self) -> tuple[str, ...]:
        """Return transformed input feature names in model-input order.

        The order is always ``axes`` first and then ``parameters``. This is the
        same ordering assumed by the tiling code and is meant to mirror the old
        scalar-regression emulator formulation.
        """
        names = [axis.feature_name() for axis in self.axes]
        names.extend(parameter.feature_name() for parameter in self.parameters)
        return tuple(names)

    def parameter_names(self) -> tuple[str, ...]:
        """Return untransformed parameter names."""
        return tuple(parameter.name for parameter in self.parameters)

    def axis_names(self) -> tuple[str, ...]:
        """Return untransformed axis names."""
        return tuple(axis.name for axis in self.axes)
