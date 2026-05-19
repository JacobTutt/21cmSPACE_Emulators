"""Tests for the astroemu-style dataset and preprocessing architecture."""

from __future__ import annotations

import numpy as np

from nenufar_emulators.core.datasets import SpectrumDataset, TiledBatch
from nenufar_emulators.core.normalisation import SpecTransformPipeline, StandardizationPipeline
from nenufar_emulators.core.specs import AxisSpec, EmulatorSpec, ParameterSpec
from nenufar_emulators.delta21.data import build_delta21_dataset, delta21_spec
from nenufar_emulators.t21.data import build_t21_dataset


def test_spec_transform_pipeline_applies_axis_parameter_and_target_rules() -> None:
    spec = EmulatorSpec(
        name="demo",
        family="global_signal",
        axes=(AxisSpec(name="z"), AxisSpec(name="k", transform="log10")),
        parameters=(
            ParameterSpec(name="fstarII", transform="log10"),
            ParameterSpec(name="alpha"),
        ),
        target_transform="log10",
        target_offset=1.0,
    )
    raw_batch = SpectrumDataset(
        spectra=np.array([[[3.0, 9.0]]]),
        axes=(np.array([6.0]), np.array([0.1, 1.0])),
        parameters=np.array([[1e-2, 1.3]]),
        axis_names=("z", "k"),
        parameter_names=("fstarII", "alpha"),
        forward_pipeline=[SpecTransformPipeline(spec)],
        tiling=False,
    ).as_batch()

    assert raw_batch.axis_names == ("z", "log10k")
    assert raw_batch.parameter_names == ("log10fstarII", "alpha")
    assert np.allclose(raw_batch.axes[1], np.array([[-1.0, 0.0]]))
    assert np.allclose(raw_batch.parameters, np.array([[-2.0, 1.3]]))
    assert np.allclose(raw_batch.spectra, np.log10(np.array([[[4.0, 10.0]]])))


def test_standardization_pipeline_round_trips_batch_values() -> None:
    dataset = SpectrumDataset(
        spectra=np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
        axes=(np.array([6.0, 8.0, 10.0]),),
        parameters=np.array([[1.0, 10.0], [3.0, 30.0]]),
        axis_names=("z",),
        parameter_names=("alpha", "fX"),
        tiling=False,
    )
    batch = dataset.as_batch()
    pipeline = StandardizationPipeline.from_batch(
        batch,
        standardize_spectra=True,
        standardize_axes=True,
        standardize_parameters=True,
    )
    standardized = pipeline.forward(batch)
    recovered = pipeline.backward(standardized)
    assert np.allclose(recovered.spectra, batch.spectra)
    assert np.allclose(recovered.axes[0], batch.axes[0])
    assert np.allclose(recovered.parameters, batch.parameters)


def test_power_dataset_builder_tiles_with_transformed_feature_names() -> None:
    spec = delta21_spec()
    dataset = build_delta21_dataset(
        spectra=np.ones((2, 3, 4)),
        axes=(np.array([6.0, 8.0, 10.0]), np.array([0.1, 0.2, 0.5, 1.0])),
        parameters=np.column_stack(
            [
                np.array([1e-2, 1e-1]),
                np.array([1e-3, 1e-2]),
                np.array([10.0, 20.0]),
                np.array([100.0, 1000.0]),
                np.array([1.0, 1.3]),
                np.array([100.0, 200.0]),
                np.array([0.05, 0.06]),
                np.array([1e2, 1e3]),
                np.array([231.0, 233.0]),
            ]
        ),
        spec=spec,
        tiling=True,
    )
    tiled = next(dataset.get_batch_iterator(batch_size=2, shuffle=False))
    assert isinstance(tiled, TiledBatch)
    assert tiled.features.shape == (24, 11)
    assert tiled.feature_names == spec.input_feature_names()


def test_global_dataset_builder_supports_non_tiled_statistics_flow() -> None:
    dataset = build_t21_dataset(
        spectra=np.array(
            [
                np.linspace(-1.0, 1.0, 5),
                np.linspace(-0.5, 1.5, 5),
            ]
        ),
        axes=(np.linspace(6.0, 10.0, 5),),
        parameters=np.column_stack(
            [
                np.array([1e-2, 1e-1]),
                np.array([1e-3, 1e-2]),
                np.array([10.0, 20.0]),
                np.array([100.0, 1000.0]),
                np.array([1.0, 1.3]),
                np.array([100.0, 200.0]),
                np.array([0.05, 0.06]),
                np.array([1e2, 1e3]),
                np.array([231.0, 233.0]),
            ]
        ),
        tiling=False,
    )
    batch = dataset.as_batch()
    assert batch.axis_shape == (5,)
    assert batch.axis_names == ("z",)
    assert batch.parameter_names[0] == "log10fstarII"
