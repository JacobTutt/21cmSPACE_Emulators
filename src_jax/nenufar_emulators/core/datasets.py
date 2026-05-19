"""Dataset abstractions for tiled spectral emulators.

Datasets own batching, preprocessing happens through explicit pipelines, and
tiling is treated as part of the dataset contract rather than as ad hoc logic
inside training scripts.

Axis names, parameter names, transforms, and target semantics are all carried
alongside the arrays so the workflow remains scientifically explicit from data
loading through model fitting.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from nenufar_emulators.core.tiling import tile_spectra


@dataclass(frozen=True)
class SpectrumBatch:
    """Untiled batch of spectra, axes, and parameters.

    This is the natural representation before the emulator data is flattened
    into scalar regression rows. Pipelines such as target transforms or
    standardisation operate on this object because they need access to the
    original spectral shape.
    """

    spectra: np.ndarray
    axes: tuple[np.ndarray, ...]
    parameters: np.ndarray
    axis_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    target_transformed: bool = False

    def __post_init__(self) -> None:
        spectra = np.asarray(self.spectra, dtype=float)
        parameters = np.asarray(self.parameters, dtype=float)
        if spectra.ndim < 2:
            raise ValueError("spectra must have shape (nsamples, *axis_shape).")
        if parameters.ndim != 2:
            raise ValueError("parameters must have shape (nsamples, nparams).")
        if spectra.shape[0] != parameters.shape[0]:
            raise ValueError("spectra and parameters must have the same sample dimension.")
        if len(self.axes) != spectra.ndim - 1:
            raise ValueError("number of axes must match spectral rank.")
        if len(self.axis_names) != len(self.axes):
            raise ValueError("axis_names must align with the axes tuple.")
        if len(self.parameter_names) != parameters.shape[1]:
            raise ValueError("parameter_names must align with the parameter columns.")
        for axis, expected_length in zip(self.axes, spectra.shape[1:], strict=True):
            axis_array = np.asarray(axis, dtype=float)
            if axis_array.ndim != 2:
                raise ValueError("each axis array must have shape (nsamples, axis_length).")
            if axis_array.shape[0] != spectra.shape[0]:
                raise ValueError("axis sample dimension must match spectra.")
            if axis_array.shape[1] != expected_length:
                raise ValueError("axis lengths must match the spectral axis lengths.")

    @property
    def nsamples(self) -> int:
        """Return the number of spectra in the batch."""
        return int(self.spectra.shape[0])

    @property
    def axis_shape(self) -> tuple[int, ...]:
        """Return the per-spectrum axis shape."""
        return tuple(int(length) for length in self.spectra.shape[1:])

    def input_feature_names(self) -> tuple[str, ...]:
        """Return the model-input feature names for this batch.

        The convention matches the rest of the repository: axes come first and
        parameters follow afterward.
        """
        return (*self.axis_names, *self.parameter_names)


@dataclass(frozen=True)
class TiledBatch:
    """Mini-batch already flattened into scalar-regression training rows."""

    targets: jnp.ndarray
    features: jnp.ndarray
    axis_shape: tuple[int, ...]
    feature_names: tuple[str, ...]


class SpectrumDataset:
    """Dataset for spectral emulator training and preprocessing.

    In practical terms this class owns four concerns:

    - holding untiled spectra, axes, and parameters together
    - applying preprocessing pipelines in a transparent order
    - yielding either untiled or tiled batches
    - preserving the names attached to axes and parameter columns

    The class is intentionally general enough to serve both one-dimensional and
    two-dimensional spectral workflows while still carrying the names and
    transforms that define the model contract.
    """

    def __init__(
        self,
        spectra: np.ndarray,
        axes: tuple[np.ndarray, ...],
        parameters: np.ndarray,
        *,
        axis_names: tuple[str, ...],
        parameter_names: tuple[str, ...],
        forward_pipeline: "NormalisationPipeline | Sequence[NormalisationPipeline] | None" = None,
        tiling: bool = True,
    ) -> None:
        self.spectra = np.asarray(spectra, dtype=float)
        self.axes = tuple(np.asarray(axis, dtype=float) for axis in axes)
        self.parameters = np.asarray(parameters, dtype=float)
        self.axis_names = axis_names
        self.parameter_names = parameter_names
        self.forward_pipeline = (
            list(forward_pipeline)
            if isinstance(forward_pipeline, Sequence)
            else [forward_pipeline]
            if forward_pipeline is not None
            else []
        )
        self.tiling = tiling

        # Validate the dataset once up front so later batch creation can stay
        # straightforward and the error messages remain local to data loading.
        for axis in self.axes:
            if axis.ndim != 1:
                raise ValueError("dataset axes must be stored as one-dimensional base arrays.")
        self._make_batch(np.arange(self.spectra.shape[0]))

    def __len__(self) -> int:
        """Return the number of spectra stored in the dataset."""
        return int(self.spectra.shape[0])

    def __getitem__(self, idx: int) -> SpectrumBatch:
        """Return one spectrum packaged as a single-sample batch."""
        return self._make_batch(np.array([idx], dtype=int))

    def as_batch(self, *, apply_pipelines: bool = True) -> SpectrumBatch:
        """Return the full dataset as one untiled batch.

        This is mainly useful for deriving normalisation statistics. The common
        pattern is to build a non-tiled dataset, apply the spec-driven
        transforms, and then compute a standardisation pipeline from the
        resulting full batch.
        """
        batch = self._make_batch(np.arange(len(self)))
        return self._apply_forward_pipelines(batch) if apply_pipelines else batch

    def get_batch_iterator(
        self,
        batch_size: int,
        *,
        shuffle: bool = True,
        key: jax.Array | None = None,
    ) -> Generator[SpectrumBatch | TiledBatch, None, None]:
        """Yield untiled or tiled batches from the dataset.

        When ``tiling=True`` the yielded object is a :class:`TiledBatch` with
        flattened targets and feature rows ready for the shared MLP trainer.
        When ``tiling=False`` the yielded object remains a :class:`SpectrumBatch`
        so callers can inspect axis shapes or compute dataset statistics before
        flattening.
        """
        n = len(self)
        indices = jnp.arange(n)
        if shuffle:
            if key is None:
                key = jax.random.PRNGKey(0)
            indices = jax.random.permutation(key, indices)

        for start in range(0, n, batch_size):
            batch_indices = np.asarray(indices[start : start + batch_size], dtype=int)
            batch = self._make_batch(batch_indices)
            batch = self._apply_forward_pipelines(batch)
            if not self.tiling:
                yield batch
                continue

            features, flat_targets, axis_shape = tile_spectra(
                batch.parameters,
                tuple(axis[0] for axis in batch.axes),
                batch.spectra,
            )
            if shuffle:
                key, subkey = jax.random.split(key)
                perm = np.asarray(jax.random.permutation(subkey, len(flat_targets)))
                flat_targets = flat_targets[perm]
                features = features[perm]
            yield TiledBatch(
                targets=jnp.asarray(flat_targets),
                features=jnp.asarray(features),
                axis_shape=axis_shape,
                feature_names=batch.input_feature_names(),
            )

    def _make_batch(self, indices: np.ndarray) -> SpectrumBatch:
        """Construct an untiled batch for the selected spectrum indices."""
        spectra = self.spectra[indices]
        axes = tuple(
            np.repeat(axis[None, :], repeats=len(indices), axis=0) for axis in self.axes
        )
        parameters = self.parameters[indices]
        return SpectrumBatch(
            spectra=spectra,
            axes=axes,
            parameters=parameters,
            axis_names=self.axis_names,
            parameter_names=self.parameter_names,
        )

    def _apply_forward_pipelines(self, batch: SpectrumBatch) -> SpectrumBatch:
        """Apply the configured preprocessing pipelines in declaration order."""
        transformed = batch
        for pipeline in self.forward_pipeline:
            transformed = pipeline.forward(transformed)
        return transformed


class NormalisationPipeline:
    """Base class for preprocessing pipelines applied to :class:`SpectrumBatch`.

    These objects are used for more than just statistical normalisation: they
    are where transform and scaling logic becomes explicit, reversible, and
    testable.
    """

    def forward(self, batch: SpectrumBatch) -> SpectrumBatch:
        """Apply the pipeline to a batch before training or validation."""
        raise NotImplementedError

    def backward(self, batch: SpectrumBatch) -> SpectrumBatch:
        """Undo the pipeline and recover the previous data representation."""
        raise NotImplementedError
