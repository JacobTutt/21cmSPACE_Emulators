"""
Dataclasses that describe the inputs and target transform for an emulator.

These specs keep track of the axes, parameters, transforms, and target
transform used by an emulator. They also define the order of the model input
columns:
- axes first
- parameters second

The spec does not load data or train a model. It defines the contract that
preprocessing, training, checkpointing, and inference must all follow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# Type Definitions
# ----------------
# Standard labels for coordinate and target transforms.

TransformName = Literal["identity", "log10"]
FamilyName = Literal["power_spectrum", "global_signal"]


# Axis Specification
# ------------------
# Describes one output axis and how it appears in the model input table.

@dataclass(frozen=True)
class AxisSpec:
    """
    Storage utility for one emulator output axis.

    Examples are redshift z and wavenumber k. The transform says how this axis
    should appear in the model input, for example k as log10k.

    Parameters
    ----------
    name:
        Physical name of the axis (e.g. 'z', 'k').
    transform:
        The transform to apply to this axis before it enters the model.
    limits:
        Optional (min, max) bounds to restrict the sampling of this axis.
    nsample:
        Optional number of points to sample along this axis for the training grid.
    """

    name: str
    transform: TransformName = "identity"
    limits: tuple[float, float] | None = None
    nsample: int | None = None

    def feature_name(self) -> str:
        """
        Return the column name used for this axis in model inputs.

        For transformed axes this includes the transform name, such as
        log10k.

        Returns
        -------
        str
            The name of the feature column representing this axis.
        """
        # Identity axes keep their physical name as-is.
        return self.name if self.transform == "identity" else f"{self.transform}{self.name}"

    def validate(self) -> None:
        """
        Check that the axis settings are valid.
        """
        # Empty names would create unusable feature columns later.
        if not self.name:
            raise ValueError("Axis name must be non-empty.")

        # Limits must be ordered low to high when they are provided.
        if self.limits is not None and self.limits[0] >= self.limits[1]:
            raise ValueError(f"Axis {self.name} has invalid limits {self.limits}.")

        # The stored grid size must be positive when it is provided.
        if self.nsample is not None and self.nsample <= 0:
            raise ValueError(f"Axis {self.name} must have positive nsample.")


# Parameter Specification
# -----------------------
# Describes one astrophysical parameter and how it appears in the model input table.

@dataclass(frozen=True)
class ParameterSpec:
    """
    Storage utility for one astrophysical emulator parameter.

    The transform says how the parameter should appear in the model input.
    discrete_values records allowed values for parameters such as alpha, nu_0,
    and pop.

    Parameters
    ----------
    name:
        Physical name of the simulation parameter.
    transform:
        The transform to apply to this parameter before it enters the model.
    discrete_values:
        Optional tuple of allowed discrete values for this parameter.
    """

    name: str
    transform: TransformName = "identity"
    discrete_values: tuple[float, ...] | None = None

    def feature_name(self) -> str:
        """
        Return the column name used for this parameter in model inputs.

        For transformed parameters this includes the transform name, such as
        log10fradio.

        Returns
        -------
        str
            The name of the feature column representing this parameter.
        """
        # Identity parameters keep their physical name as-is.
        return self.name if self.transform == "identity" else f"{self.transform}{self.name}"

    def validate(self) -> None:
        """
        Check that the parameter settings are valid.
        """
        # Empty names would create unusable feature columns.
        if not self.name:
            raise ValueError("Parameter name must be non-empty.")

        # If discrete values are supplied, the allowed set cannot be empty.
        if self.discrete_values is not None and len(self.discrete_values) == 0:
            raise ValueError(f"Parameter {self.name} cannot have empty discrete_values.")


# Emulator Specification
# ----------------------
# Defines the full model input contract for one emulator workflow.

@dataclass(frozen=True)
class EmulatorSpec:
    """
    Storage utility for one emulator input and target contract.

    This does not load data or train a model. It only records which axes and
    parameters are used, what transforms are applied, and how the target is
    transformed before training.

    Parameters
    ----------
    name:
        A unique name for the emulator configuration.
    family:
        The type of signal being emulated (power spectrum or global signal).
    axes:
        The sequence of output axes.
    parameters:
        The sequence of input astrophysical parameters.
    target_transform:
        The transform applied to the target signal before training.
    target_offset:
        An optional offset added to targets before applying a log transform.
    """

    name: str
    family: FamilyName
    axes: tuple[AxisSpec, ...] = field(default_factory=tuple)
    parameters: tuple[ParameterSpec, ...] = field(default_factory=tuple)
    target_transform: TransformName = "identity"
    target_offset: float = 0.0

    def __post_init__(self) -> None:
        # Validate immediately so invalid specs fail close to where they are built.
        self.validate()

    def validate(self) -> None:
        """
        Check that the emulator spec is internally consistent.

        This mainly catches duplicate names and transformed names that would
        make the model input columns ambiguous.
        """
        # A spec needs a name, at least one axis, and at least one parameter.
        if not self.name:
            raise ValueError("Emulator name must be non-empty.")
        if not self.axes:
            raise ValueError(f"Emulator {self.name} must define at least one axis.")
        if not self.parameters:
            raise ValueError(f"Emulator {self.name} must define at least one parameter.")

        # Extract raw names to check for collisions.
        axis_names = [axis.name for axis in self.axes]
        parameter_names = [parameter.name for parameter in self.parameters]
        # Get the final feature names that will appear in the training matrix.
        transformed_names = self.input_feature_names()

        # Raw axis and parameter names must not collide.
        if len(set(axis_names)) != len(axis_names):
            raise ValueError(f"Emulator {self.name} has duplicate axis names.")
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError(f"Emulator {self.name} has duplicate parameter names.")
        if set(axis_names) & set(parameter_names):
            raise ValueError(f"Emulator {self.name} reuses names between axes and parameters.")

        # Transformed feature names must also be unique after prefixes are applied.
        # This prevents collisions between e.g. an axis 'log10k' and a parameter 'log10k'.
        if len(set(transformed_names)) != len(transformed_names):
            raise ValueError(
                f"Emulator {self.name} has colliding transformed input feature names."
            )

        # Validate each child spec (axes and parameters) after checking the emulator-level contract.
        for axis in self.axes:
            axis.validate()
        for parameter in self.parameters:
            parameter.validate()

    def input_feature_names(self) -> tuple[str, ...]:
        """
        Return model input column names in the order used for training.

        The order is transformed axes first, then transformed parameters.

        Returns
        -------
        tuple[str, ...]
            The sequence of feature column names.
        """
        # Axis columns always come before parameter columns in tiled/flattened features.
        names = [axis.feature_name() for axis in self.axes]
        names.extend(parameter.feature_name() for parameter in self.parameters)
        return tuple(names)

    def parameter_names(self) -> tuple[str, ...]:
        """
        Return the raw physical parameter names in their declared order.

        These are the names used before transforms are applied.

        Returns
        -------
        tuple[str, ...]
             The raw parameter names.
        """
        return tuple(parameter.name for parameter in self.parameters)

    def axis_names(self) -> tuple[str, ...]:
        """
        Return the raw physical axis names in their declared order.

        Returns
        -------
        tuple[str, ...]
            The raw axis names.
        """
        return tuple(axis.name for axis in self.axes)
