"""Repository-level emulator specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


TransformName = Literal["identity", "log10"]
FamilyName = Literal["power_spectrum", "global_signal"]


@dataclass(frozen=True)
class AxisSpec:
    """Specification for a physical axis used by an emulator."""

    name: str
    transform: TransformName = "identity"
    limits: tuple[float, float] | None = None
    nsample: int | None = None

    def feature_name(self) -> str:
        """Return the feature name after transformation."""
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
    """Specification for a model parameter."""

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
    """High-level emulator definition used by training and inference code."""

    name: str
    family: FamilyName
    axes: tuple[AxisSpec, ...] = field(default_factory=tuple)
    parameters: tuple[ParameterSpec, ...] = field(default_factory=tuple)
    target_transform: TransformName = "identity"
    target_offset: float = 0.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate emulator configuration consistency."""
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
        """Return transformed input feature names in model-input order."""
        names = [axis.feature_name() for axis in self.axes]
        names.extend(parameter.feature_name() for parameter in self.parameters)
        return tuple(names)

    def parameter_names(self) -> tuple[str, ...]:
        """Return untransformed parameter names."""
        return tuple(parameter.name for parameter in self.parameters)

    def axis_names(self) -> tuple[str, ...]:
        """Return untransformed axis names."""
        return tuple(axis.name for axis in self.axes)
