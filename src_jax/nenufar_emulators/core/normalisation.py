"""Preprocessing pipelines for spectral datasets.

This module provides the dataset-level transform objects that sit between raw
science arrays and the tiled neural-network inputs. The conventions here are
tuned to this package's Delta21 and T21 workflows:

- axis transforms follow the declared :class:`~nenufar_emulators.core.specs.AxisSpec`
- parameter transforms follow the declared :class:`~nenufar_emulators.core.specs.ParameterSpec`
- target transforms follow the emulator spec, including any configured offset
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nenufar_emulators.core.datasets import NormalisationPipeline, SpectrumBatch
from nenufar_emulators.core.specs import EmulatorSpec
from nenufar_emulators.core.transforms import apply_transform, invert_transform


def _safe_std(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return a standard deviation with zeros replaced by ones."""
    std = np.asarray(values).std(axis=axis)
    return np.where(std > 0, std, 1.0)


class SpecTransformPipeline(NormalisationPipeline):
    """Apply the transforms declared by an :class:`EmulatorSpec`.

    This pipeline converts raw physical axes, parameters, and
    targets into the representation actually seen by the neural network.
    """

    def __init__(self, spec: EmulatorSpec) -> None:
        self.spec = spec

    def forward(self, batch: SpectrumBatch) -> SpectrumBatch:
        """Transform axes, parameters, and targets into model space."""
        axis_lookup = {name: idx for idx, name in enumerate(batch.axis_names)}
        parameter_lookup = {name: idx for idx, name in enumerate(batch.parameter_names)}

        transformed_axes = []
        transformed_axis_names = []
        for axis_spec in self.spec.axes:
            transformed_name = axis_spec.feature_name()
            if transformed_name in axis_lookup:
                idx = axis_lookup[transformed_name]
                axis_values = batch.axes[idx]
            elif axis_spec.name in axis_lookup:
                idx = axis_lookup[axis_spec.name]
                axis_values = apply_transform(batch.axes[idx], axis_spec.transform)
            else:
                raise ValueError(f"Axis {axis_spec.name!r} is missing from the batch.")
            transformed_axes.append(np.asarray(axis_values, dtype=float))
            transformed_axis_names.append(transformed_name)

        transformed_parameters = []
        transformed_parameter_names = []
        for parameter_spec in self.spec.parameters:
            transformed_name = parameter_spec.feature_name()
            if transformed_name in parameter_lookup:
                idx = parameter_lookup[transformed_name]
                parameter_values = batch.parameters[:, idx]
            elif parameter_spec.name in parameter_lookup:
                idx = parameter_lookup[parameter_spec.name]
                parameter_values = apply_transform(
                    batch.parameters[:, idx],
                    parameter_spec.transform,
                )
            else:
                raise ValueError(f"Parameter {parameter_spec.name!r} is missing from the batch.")
            transformed_parameters.append(np.asarray(parameter_values, dtype=float))
            transformed_parameter_names.append(transformed_name)

        spectra = batch.spectra
        target_transformed = batch.target_transformed
        if not batch.target_transformed:
            spectra = apply_transform(
                batch.spectra,
                self.spec.target_transform,
                offset=self.spec.target_offset,
            )
            target_transformed = True

        return SpectrumBatch(
            spectra=np.asarray(spectra, dtype=float),
            axes=tuple(transformed_axes),
            parameters=np.stack(transformed_parameters, axis=1),
            axis_names=tuple(transformed_axis_names),
            parameter_names=tuple(transformed_parameter_names),
            target_transformed=target_transformed,
        )

    def backward(self, batch: SpectrumBatch) -> SpectrumBatch:
        """Undo the spec-defined transforms and recover physical-space arrays."""
        axis_lookup = {name: idx for idx, name in enumerate(batch.axis_names)}
        parameter_lookup = {name: idx for idx, name in enumerate(batch.parameter_names)}

        recovered_axes = []
        recovered_axis_names = []
        for axis_spec in self.spec.axes:
            transformed_name = axis_spec.feature_name()
            if transformed_name not in axis_lookup:
                raise ValueError(f"Transformed axis {transformed_name!r} is missing from the batch.")
            idx = axis_lookup[transformed_name]
            recovered_axes.append(
                invert_transform(batch.axes[idx], axis_spec.transform)
            )
            recovered_axis_names.append(axis_spec.name)

        recovered_parameters = []
        recovered_parameter_names = []
        for parameter_spec in self.spec.parameters:
            transformed_name = parameter_spec.feature_name()
            if transformed_name not in parameter_lookup:
                raise ValueError(
                    f"Transformed parameter {transformed_name!r} is missing from the batch."
                )
            idx = parameter_lookup[transformed_name]
            recovered_parameters.append(
                invert_transform(batch.parameters[:, idx], parameter_spec.transform)
            )
            recovered_parameter_names.append(parameter_spec.name)

        spectra = batch.spectra
        target_transformed = batch.target_transformed
        if batch.target_transformed:
            spectra = invert_transform(
                batch.spectra,
                self.spec.target_transform,
                offset=self.spec.target_offset,
            )
            target_transformed = False

        return SpectrumBatch(
            spectra=np.asarray(spectra, dtype=float),
            axes=tuple(np.asarray(axis, dtype=float) for axis in recovered_axes),
            parameters=np.stack(recovered_parameters, axis=1),
            axis_names=tuple(recovered_axis_names),
            parameter_names=tuple(recovered_parameter_names),
            target_transformed=target_transformed,
        )


@dataclass(frozen=True)
class DatasetStatistics:
    """Summary statistics used by :class:`StandardizationPipeline`."""

    spectra_mean: np.ndarray
    spectra_std: np.ndarray
    axis_mean: tuple[np.ndarray, ...]
    axis_std: tuple[np.ndarray, ...]
    parameter_mean: np.ndarray
    parameter_std: np.ndarray


class StandardizationPipeline(NormalisationPipeline):
    """Standardise spectra, axes, and parameters using stored dataset statistics.

    This is the dataset-level version of the usual preprocessing step before
    training: shift and scale selected quantities so the network sees a
    numerically friendlier problem, while still allowing the transform to be
    inverted later.
    """

    def __init__(
        self,
        statistics: DatasetStatistics,
        *,
        standardize_spectra: bool = False,
        standardize_axes: bool = False,
        standardize_parameters: bool = False,
    ) -> None:
        self.statistics = statistics
        self.standardize_spectra = standardize_spectra
        self.standardize_axes = standardize_axes
        self.standardize_parameters = standardize_parameters

    @classmethod
    def from_batch(
        cls,
        batch: SpectrumBatch,
        *,
        standardize_spectra: bool = False,
        standardize_axes: bool = False,
        standardize_parameters: bool = False,
    ) -> "StandardizationPipeline":
        """Build a standardisation pipeline from one representative batch.

        In practice this batch is usually the full non-tiled training set after
        spec-driven log transforms have already been applied.
        """
        statistics = DatasetStatistics(
            spectra_mean=np.asarray(batch.spectra.mean(axis=0), dtype=float),
            spectra_std=np.asarray(_safe_std(batch.spectra, axis=0), dtype=float),
            axis_mean=tuple(np.asarray(axis.mean(axis=0), dtype=float) for axis in batch.axes),
            axis_std=tuple(np.asarray(_safe_std(axis, axis=0), dtype=float) for axis in batch.axes),
            parameter_mean=np.asarray(batch.parameters.mean(axis=0), dtype=float),
            parameter_std=np.asarray(_safe_std(batch.parameters, axis=0), dtype=float),
        )
        return cls(
            statistics,
            standardize_spectra=standardize_spectra,
            standardize_axes=standardize_axes,
            standardize_parameters=standardize_parameters,
        )

    def forward(self, batch: SpectrumBatch) -> SpectrumBatch:
        """Apply standardisation to the requested parts of the batch."""
        spectra = batch.spectra
        axes = batch.axes
        parameters = batch.parameters

        if self.standardize_spectra:
            spectra = (spectra - self.statistics.spectra_mean) / self.statistics.spectra_std
        if self.standardize_axes:
            axes = tuple(
                (axis - mean) / std
                for axis, mean, std in zip(
                    batch.axes,
                    self.statistics.axis_mean,
                    self.statistics.axis_std,
                    strict=True,
                )
            )
        if self.standardize_parameters:
            parameters = (
                batch.parameters - self.statistics.parameter_mean
            ) / self.statistics.parameter_std

        return SpectrumBatch(
            spectra=np.asarray(spectra, dtype=float),
            axes=tuple(np.asarray(axis, dtype=float) for axis in axes),
            parameters=np.asarray(parameters, dtype=float),
            axis_names=batch.axis_names,
            parameter_names=batch.parameter_names,
            target_transformed=batch.target_transformed,
        )

    def backward(self, batch: SpectrumBatch) -> SpectrumBatch:
        """Undo standardisation and recover the previous numeric scale."""
        spectra = batch.spectra
        axes = batch.axes
        parameters = batch.parameters

        if self.standardize_spectra:
            spectra = spectra * self.statistics.spectra_std + self.statistics.spectra_mean
        if self.standardize_axes:
            axes = tuple(
                axis * std + mean
                for axis, mean, std in zip(
                    batch.axes,
                    self.statistics.axis_mean,
                    self.statistics.axis_std,
                    strict=True,
                )
            )
        if self.standardize_parameters:
            parameters = (
                parameters * self.statistics.parameter_std
            ) + self.statistics.parameter_mean

        return SpectrumBatch(
            spectra=np.asarray(spectra, dtype=float),
            axes=tuple(np.asarray(axis, dtype=float) for axis in axes),
            parameters=np.asarray(parameters, dtype=float),
            axis_names=batch.axis_names,
            parameter_names=batch.parameter_names,
            target_transformed=batch.target_transformed,
        )
