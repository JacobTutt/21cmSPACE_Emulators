"""Dataclasses that describe what an emulator takes as input.

These specs keep track of the axes, parameters, transforms, and target
transform used by an emulator. They also define the order of the model input
columns: axes first, then parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


TransformName = Literal["identity", "log10"]
FamilyName = Literal["power_spectrum", "global_signal"]


@dataclass(frozen=True)
class AxisSpec:
    """One axis of the emulator output.

    Examples are redshift ``z`` and wave number ``k``. The transform says how
    this axis should appear in the model input, for example ``k`` as
    ``log10k``.
    """

    name: str
    transform: TransformName = "identity"
    limits: tuple[float, float] | None = None
    nsample: int | None = None

    def feature_name(self) -> str:
        """Return the column name used for this axis in model inputs.

        For transformed axes this includes the transform name, such as
        ``log10k``.
        """
        return self.name if self.transform == "identity" else f"{self.transform}{self.name}"

    def validate(self) -> None:
        """Check that the axis settings are valid."""
        if not self.name:
            raise ValueError("Axis name must be non-empty.")
        if self.limits is not None and self.limits[0] >= self.limits[1]:
            raise ValueError(f"Axis {self.name} has invalid limits {self.limits}.")
        if self.nsample is not None and self.nsample <= 0:
            raise ValueError(f"Axis {self.name} must have positive nsample.")


@dataclass(frozen=True)
class ParameterSpec:
    """One astrophysical parameter used by the emulator.

    The transform says how the parameter should appear in the model input.
    `discrete_values` records allowed values for parameters such as ``alpha``,
    ``nu_0``, and ``pop``.
    """

    name: str
    transform: TransformName = "identity"
    discrete_values: tuple[float, ...] | None = None

    def feature_name(self) -> str:
        """Return the column name used for this parameter in model inputs.

        For transformed parameters this includes the transform name, such as
        ``log10fradio``.
        """
        return self.name if self.transform == "identity" else f"{self.transform}{self.name}"

    def validate(self) -> None:
        """Check that the parameter settings are valid."""
        if not self.name:
            raise ValueError("Parameter name must be non-empty.")
        if self.discrete_values is not None and len(self.discrete_values) == 0:
            raise ValueError(f"Parameter {self.name} cannot have empty discrete_values.")


@dataclass(frozen=True)
class EmulatorSpec:
    """Description of the inputs and target transform for one emulator.

    This does not load data or train a model. It only records which axes and
    parameters are used, what transforms are applied, and how the target is
    transformed before training.
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
        """Check that the emulator spec is internally consistent.

        This mainly catches duplicate names and transformed names that would
        make the model input columns ambiguous.
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
        """Return model input column names in the order used for training.

        The order is transformed axes first, then transformed parameters.
        """
        names = [axis.feature_name() for axis in self.axes]
        names.extend(parameter.feature_name() for parameter in self.parameters)
        return tuple(names)

    def parameter_names(self) -> tuple[str, ...]:
        """Return the raw physical parameter names in their declared order.

        These are the names used before transforms are applied.
        """
        return tuple(parameter.name for parameter in self.parameters)

    def axis_names(self) -> tuple[str, ...]:
        """Return the raw physical axis names in their declared order."""
        return tuple(axis.name for axis in self.axes)
