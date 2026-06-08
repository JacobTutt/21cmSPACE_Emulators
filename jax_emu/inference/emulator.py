"""
Reusable inference wrapper for trained JAX emulators.

The `Emulator` class combines a trained model with its checkpoint metadata and
builds a compiled forward model. It owns the generic inference route:
physical parameters and independent axes -> transformed/scaled network inputs
-> model prediction -> inverse target scaling/transform -> physical output.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from jax_emu.data_preprocessing.scaling import FeatureScaling
from jax_emu.utils.checkpointing import CheckpointMetadata


# Type Definitions
# ----------------
# Parameter adapters convert user-facing physical parameter arrays into the
# transformed parameter columns expected by the emulator specification.

ParameterAdapter = Callable[[jax.Array], jax.Array]


# Emulator
# --------
# Reusable compiled forward models for trained emulators.

class Emulator:
    """
    Generic compiled inference wrapper for a trained emulator.

    The wrapper can be constructed from either a loaded checkpoint package or a
    live model plus `CheckpointMetadata`. The package-specific code is only
    responsible for loading/checking the package and, if needed, providing a
    parameter adapter for a project-specific raw parameter table.

    Parameters
    ----------
    package:
        Loaded checkpoint package containing at least `model` and `metadata`.
    model:
        Live trained model. Used when `package` is not supplied.
    metadata:
        Checkpoint metadata. Used when `package` is not supplied.
    parameter_adapter:
        Optional function that maps incoming physical parameter arrays into the
        transformed parameter columns expected by the model. If omitted, the
        input is assumed to already follow `metadata.emulator_spec.parameters`
        in physical space and the generic spec transforms are applied.
    compile_inputs:
        Optional tuple `(parameters, axis_0, axis_1, ...)` used to force JIT
        compilation during initialization. If omitted, compilation happens on
        the first call.
    """

    def __init__(
        self,
        *,
        package: dict[str, Any] | None = None,
        model: Any | None = None,
        metadata: CheckpointMetadata | None = None,
        parameter_adapter: ParameterAdapter | None = None,
        compile_inputs: tuple[jax.Array, ...] | None = None,
    ) -> None:
        # Accept either a loaded checkpoint package or explicit model/metadata.
        if package is not None:
            model = package["model"]
            metadata = package["metadata"]

        if model is None:
            raise ValueError("Emulator requires a model or a package containing a model.")
        if metadata is None:
            raise ValueError("Emulator requires checkpoint metadata.")

        self.model = model
        self.metadata = metadata
        self.spec = metadata.emulator_spec
        self.parameter_adapter = parameter_adapter

        self._validate_feature_order()
        self._predict = self._build_compiled_predictor()

        # Optional warm-up call. This compiles the model for the supplied shapes
        # so later calls with the same shapes do not pay the compilation cost.
        if compile_inputs is not None:
            self.compile(*compile_inputs)

    def forward_model(self, parameters: jax.Array, *axes: jax.Array) -> jax.Array:
        """
        Evaluate the physical emulator prediction.

        Parameters
        ----------
        parameters:
            Parameter table for one or more simulations.
        *axes:
            One independent coordinate array per emulator axis, for example
            `z` for T21 or `(z, k)` for Delta21.

        Returns
        -------
        jax.Array
            Physical prediction array with shape `(n_sims, *axis_lengths)`.
        """
        # Keep the public method small: validation and compilation details are
        # handled at initialization, while `_predict` owns the numerical route.
        return self._predict(self.model, parameters, *axes)

    def forwardmodel(self, parameters: jax.Array, *axes: jax.Array) -> jax.Array:
        """
        Alias for `forward_model`.

        This keeps the call site compact for users who prefer
        `emulator.forwardmodel(...)`.
        """
        return self.forward_model(parameters, *axes)

    def compile(self, parameters: jax.Array, *axes: jax.Array) -> None:
        """
        Compile the forward model for a representative set of input shapes.

        JAX compiles on first use for each input shape. Calling this method is
        useful before an MCMC or repeated-inference loop where the same shapes
        will be used many times.
        """
        # Block so the compilation cost is paid here rather than hidden inside
        # the first real inference call.
        self.forward_model(parameters, *axes).block_until_ready()

    def _validate_feature_order(self) -> None:
        """
        Ensure saved feature scaling follows the emulator input contract.
        """
        # Scaling metadata is applied column-by-column. If its order differs
        # from the spec, predictions would silently use the wrong statistics.
        expected_names = self.spec.input_feature_names()
        actual_names = tuple(feature.name for feature in self.metadata.input_scaling)
        if actual_names != expected_names:
            raise ValueError(
                "Saved feature scaling order does not match the emulator spec. "
                f"Expected {expected_names}, received {actual_names}."
            )

    def _build_compiled_predictor(self) -> Callable[..., jax.Array]:
        """
        Build the NNX-jitted numerical inference function.
        """
        spec = self.spec
        input_scaling = self.metadata.input_scaling
        target_scaling = self.metadata.target_scaling
        target_std = None if target_scaling is None else target_scaling.std
        parameter_adapter = self.parameter_adapter

        @nnx.jit
        def _predict(model_instance: Any, parameters: jax.Array, *axes: jax.Array) -> jax.Array:
            """
            Run the compiled numerical inference path.
            """
            if len(axes) != len(spec.axes):
                raise ValueError(
                    f"Expected {len(spec.axes)} independent axes, received {len(axes)}."
                )

            # Convert parameters into transformed parameter columns.
            prepared_parameters = (
                _prepare_parameters_from_spec(parameters, spec.parameters)
                if parameter_adapter is None
                else parameter_adapter(parameters)
            )

            # Flatten each independent axis so meshgrid builds a regular grid.
            flat_axes = tuple(axis.ravel() for axis in axes)
            mesh_axes = jnp.meshgrid(*flat_axes, indexing="ij")

            # Apply the same axis transforms used during training.
            axis_columns = [
                _apply_transform_jax(mesh_axis.ravel(), axis_spec.transform)
                for mesh_axis, axis_spec in zip(mesh_axes, spec.axes, strict=True)
            ]
            axis_features = jnp.stack(axis_columns, axis=1)

            # Tile axes and parameters into scalar regression rows.
            repeated_axes = jnp.tile(axis_features, (prepared_parameters.shape[0], 1))
            repeated_parameters = jnp.repeat(
                prepared_parameters,
                repeats=axis_features.shape[0],
                axis=0,
            )
            features = jnp.concatenate([repeated_axes, repeated_parameters], axis=1)

            # Apply saved feature scaling and evaluate the network.
            scaled_features = _scale_features_jax(features, input_scaling)
            flat_predictions = model_instance(scaled_features).squeeze(-1)

            # Return from network space to physical target space.
            if target_std is not None:
                flat_predictions = flat_predictions * target_std
            physical_predictions = _invert_transform_jax(
                flat_predictions,
                spec.target_transform,
                offset=spec.target_offset,
            )

            # Fold the flat predictions back onto the independent-axis grid.
            output_shape = (prepared_parameters.shape[0], *(axis.shape[0] for axis in flat_axes))
            return physical_predictions.reshape(output_shape)

        return _predict


class FixedGridEmulator:
    """
    Generic compiled inference wrapper for one fixed output grid.

    This is the inference path to use when the redshift / k grid is known in
    advance, for example inside an MCMC likelihood. The independent-axis grid is
    transformed and scaled once at initialization. Each later call only has to
    transform the physical parameters, tile them against the stored grid, run
    the network, and invert the target transform.

    Parameters
    ----------
    axes:
        Fixed physical coordinate arrays, for example `(z,)` for T21 or
        `(z, k)` for Delta21.
    package:
        Loaded checkpoint package containing at least `model` and `metadata`.
    model:
        Live trained model. Used when `package` is not supplied.
    metadata:
        Checkpoint metadata. Used when `package` is not supplied.
    parameter_adapter:
        Optional function that maps incoming physical parameter arrays into the
        transformed parameter columns expected by the model.
    compile_parameters:
        Optional parameter array used to force JIT compilation during
        initialization. If omitted, compilation happens on the first call.
    """

    def __init__(
        self,
        *,
        axes: tuple[jax.Array, ...],
        package: dict[str, Any] | None = None,
        model: Any | None = None,
        metadata: CheckpointMetadata | None = None,
        parameter_adapter: ParameterAdapter | None = None,
        compile_parameters: jax.Array | None = None,
    ) -> None:
        # Accept either a loaded checkpoint package or explicit model/metadata.
        if package is not None:
            model = package["model"]
            metadata = package["metadata"]

        if model is None:
            raise ValueError("FixedGridEmulator requires a model or a package containing a model.")
        if metadata is None:
            raise ValueError("FixedGridEmulator requires checkpoint metadata.")

        self.model = model
        self.metadata = metadata
        self.spec = metadata.emulator_spec
        self.parameter_adapter = parameter_adapter
        self.axis_shape = tuple(int(jnp.asarray(axis).size) for axis in axes)

        self._validate_feature_order()
        self.scaled_axis_features = _build_scaled_axis_features_jax(
            axes,
            self.spec.axes,
            self.metadata.input_scaling,
        )
        self._predict = self._build_compiled_predictor()

        # Optional warm-up call. This compiles the parameter-only call path for
        # the stored grid and the supplied parameter shape.
        if compile_parameters is not None:
            self.compile(compile_parameters)

    def forward_model(self, parameters: jax.Array) -> jax.Array:
        """
        Evaluate the physical emulator prediction on the stored grid.

        Parameters
        ----------
        parameters:
            Parameter table for one or more simulations.

        Returns
        -------
        jax.Array
            Physical prediction array with shape `(n_sims, *axis_shape)`.
        """
        return self._predict(self.model, parameters, self.scaled_axis_features)

    def forwardmodel(self, parameters: jax.Array) -> jax.Array:
        """
        Alias for `forward_model`.
        """
        return self.forward_model(parameters)

    def emulate(self, parameters: jax.Array) -> jax.Array:
        """
        Alias for `forward_model`.

        This gives fixed-grid inference a compact call site:
        `emulator.emulate(parameters)`.
        """
        return self.forward_model(parameters)

    def compile(self, parameters: jax.Array) -> None:
        """
        Compile the fixed-grid forward model for a representative parameter shape.
        """
        self.forward_model(parameters).block_until_ready()

    def _validate_feature_order(self) -> None:
        """
        Ensure saved feature scaling follows the emulator input contract.
        """
        expected_names = self.spec.input_feature_names()
        actual_names = tuple(feature.name for feature in self.metadata.input_scaling)
        if actual_names != expected_names:
            raise ValueError(
                "Saved feature scaling order does not match the emulator spec. "
                f"Expected {expected_names}, received {actual_names}."
            )

    def _build_compiled_predictor(self) -> Callable[..., jax.Array]:
        """
        Build the NNX-jitted fixed-grid inference function.
        """
        spec = self.spec
        input_scaling = self.metadata.input_scaling
        target_scaling = self.metadata.target_scaling
        target_std = None if target_scaling is None else target_scaling.std
        parameter_adapter = self.parameter_adapter
        n_axes = len(spec.axes)
        parameter_scaling = input_scaling[n_axes:]
        axis_shape = self.axis_shape

        @nnx.jit
        def _predict(
            model_instance: Any,
            parameters: jax.Array,
            scaled_axis_features: jax.Array,
        ) -> jax.Array:
            """
            Run the compiled fixed-grid numerical inference path.
            """
            # Convert parameters into transformed parameter columns.
            prepared_parameters = (
                _prepare_parameters_from_spec(parameters, spec.parameters)
                if parameter_adapter is None
                else parameter_adapter(parameters)
            )

            # Scale the parameter columns. The axis columns were already
            # transformed and scaled when this fixed-grid emulator was created.
            scaled_parameters = _scale_features_jax(prepared_parameters, parameter_scaling)

            # Tile the stored grid against the incoming parameter table.
            repeated_axes = jnp.tile(scaled_axis_features, (scaled_parameters.shape[0], 1))
            repeated_parameters = jnp.repeat(
                scaled_parameters,
                repeats=scaled_axis_features.shape[0],
                axis=0,
            )
            scaled_features = jnp.concatenate([repeated_axes, repeated_parameters], axis=1)

            # Evaluate the network and return to physical target space.
            flat_predictions = model_instance(scaled_features).squeeze(-1)
            if target_std is not None:
                flat_predictions = flat_predictions * target_std
            physical_predictions = _invert_transform_jax(
                flat_predictions,
                spec.target_transform,
                offset=spec.target_offset,
            )

            # Fold the flat predictions back onto the stored independent-axis grid.
            output_shape = (prepared_parameters.shape[0], *axis_shape)
            return physical_predictions.reshape(output_shape)

        return _predict


class FixedCoordinateEmulator:
    """
    Generic compiled inference wrapper for one fixed coordinate list.

    This is the inference path to use when the requested output points are not
    a full rectangular grid. For example, an observational likelihood may need
    predictions only at selected model-side coordinate points.

    Parameters
    ----------
    coordinates:
        One coordinate array per emulator axis. All arrays must have the same
        length, for example `(z_points, k_points)` for Delta21.
    package:
        Loaded checkpoint package containing at least `model` and `metadata`.
    model:
        Live trained model. Used when `package` is not supplied.
    metadata:
        Checkpoint metadata. Used when `package` is not supplied.
    parameter_adapter:
        Optional function that maps incoming physical parameter arrays into the
        transformed parameter columns expected by the model.
    compile_parameters:
        Optional parameter array used to force JIT compilation during
        initialization. If omitted, compilation happens on the first call.
    """

    def __init__(
        self,
        *,
        coordinates: tuple[jax.Array, ...],
        package: dict[str, Any] | None = None,
        model: Any | None = None,
        metadata: CheckpointMetadata | None = None,
        parameter_adapter: ParameterAdapter | None = None,
        compile_parameters: jax.Array | None = None,
    ) -> None:
        # Accept either a loaded checkpoint package or explicit model/metadata.
        if package is not None:
            model = package["model"]
            metadata = package["metadata"]

        if model is None:
            raise ValueError(
                "FixedCoordinateEmulator requires a model or a package containing a model."
            )
        if metadata is None:
            raise ValueError("FixedCoordinateEmulator requires checkpoint metadata.")

        self.model = model
        self.metadata = metadata
        self.spec = metadata.emulator_spec
        self.parameter_adapter = parameter_adapter
        self.n_coordinates = _coordinate_count(coordinates)

        self._validate_feature_order()
        self.scaled_axis_features = _build_scaled_coordinate_features_jax(
            coordinates,
            self.spec.axes,
            self.metadata.input_scaling,
        )
        self._predict = self._build_compiled_predictor()

        # Optional warm-up call. This compiles the parameter-only call path for
        # the stored coordinate list and the supplied parameter shape.
        if compile_parameters is not None:
            self.compile(compile_parameters)

    def forward_model(self, parameters: jax.Array) -> jax.Array:
        """
        Evaluate the physical emulator prediction on the stored coordinates.

        Parameters
        ----------
        parameters:
            Parameter table for one or more simulations.

        Returns
        -------
        jax.Array
            Physical prediction array with shape `(n_sims, n_coordinates)`.
        """
        return self._predict(self.model, parameters, self.scaled_axis_features)

    def forwardmodel(self, parameters: jax.Array) -> jax.Array:
        """
        Alias for `forward_model`.
        """
        return self.forward_model(parameters)

    def emulate(self, parameters: jax.Array) -> jax.Array:
        """
        Alias for `forward_model`.
        """
        return self.forward_model(parameters)

    def compile(self, parameters: jax.Array) -> None:
        """
        Compile the fixed-coordinate forward model for one parameter shape.
        """
        self.forward_model(parameters).block_until_ready()

    def _validate_feature_order(self) -> None:
        """
        Ensure saved feature scaling follows the emulator input contract.
        """
        expected_names = self.spec.input_feature_names()
        actual_names = tuple(feature.name for feature in self.metadata.input_scaling)
        if actual_names != expected_names:
            raise ValueError(
                "Saved feature scaling order does not match the emulator spec. "
                f"Expected {expected_names}, received {actual_names}."
            )

    def _build_compiled_predictor(self) -> Callable[..., jax.Array]:
        """
        Build the NNX-jitted fixed-coordinate inference function.
        """
        spec = self.spec
        input_scaling = self.metadata.input_scaling
        target_scaling = self.metadata.target_scaling
        target_std = None if target_scaling is None else target_scaling.std
        parameter_adapter = self.parameter_adapter
        n_axes = len(spec.axes)
        parameter_scaling = input_scaling[n_axes:]
        n_coordinates = self.n_coordinates

        @nnx.jit
        def _predict(
            model_instance: Any,
            parameters: jax.Array,
            scaled_axis_features: jax.Array,
        ) -> jax.Array:
            """
            Run the compiled fixed-coordinate numerical inference path.
            """
            # Convert parameters into transformed parameter columns.
            prepared_parameters = (
                _prepare_parameters_from_spec(parameters, spec.parameters)
                if parameter_adapter is None
                else parameter_adapter(parameters)
            )

            # Scale parameter columns per call. The coordinate columns were
            # transformed and scaled once when the wrapper was initialized.
            scaled_parameters = _scale_features_jax(prepared_parameters, parameter_scaling)

            # Pair every parameter row with every stored coordinate row.
            repeated_axes = jnp.tile(scaled_axis_features, (scaled_parameters.shape[0], 1))
            repeated_parameters = jnp.repeat(
                scaled_parameters,
                repeats=scaled_axis_features.shape[0],
                axis=0,
            )
            scaled_features = jnp.concatenate([repeated_axes, repeated_parameters], axis=1)

            # Evaluate the network and return to physical target space.
            flat_predictions = model_instance(scaled_features).squeeze(-1)
            if target_std is not None:
                flat_predictions = flat_predictions * target_std
            physical_predictions = _invert_transform_jax(
                flat_predictions,
                spec.target_transform,
                offset=spec.target_offset,
            )

            return physical_predictions.reshape((prepared_parameters.shape[0], n_coordinates))

        return _predict


# JAX Helpers
# -----------
# Small numerical helpers used inside the compiled inference path.

def _prepare_parameters_from_spec(parameters: jax.Array, parameter_specs: tuple[Any, ...]) -> jax.Array:
    """
    Transform a physical parameter table using the emulator parameter specs.
    """
    array = parameters
    if array.ndim == 1:
        array = array[None, :]

    if array.shape[1] != len(parameter_specs):
        raise ValueError(
            f"Expected {len(parameter_specs)} parameter columns, received {array.shape[1]}."
        )

    return jnp.stack(
        [
            _apply_transform_jax(array[:, idx], parameter.transform)
            for idx, parameter in enumerate(parameter_specs)
        ],
        axis=1,
    )


def _apply_transform_jax(values: jax.Array, transform: str) -> jax.Array:
    """
    Apply a named transform on the JAX device.
    """
    if transform == "identity":
        return values
    if transform == "log10":
        return jnp.log10(values)
    raise ValueError(f"Unsupported transform {transform}.")


def _invert_transform_jax(values: jax.Array, transform: str, offset: float = 0.0) -> jax.Array:
    """
    Undo a named target transform on the JAX device.
    """
    if transform == "identity":
        return values
    if transform == "log10":
        return jnp.power(10.0, values) - offset
    raise ValueError(f"Unsupported transform {transform}.")


def _scale_features_jax(
    features: jax.Array,
    scaling: tuple[FeatureScaling, ...],
) -> jax.Array:
    """
    Apply saved input-feature scaling on the JAX device.
    """
    columns = []
    for idx, feature in enumerate(scaling):
        values = features[:, idx]
        if feature.method == "identity":
            scaled = values
        elif feature.method == "zscore":
            scaled = (values - feature.mean) / feature.std
        elif feature.method == "minmax_minus_one_to_one":
            denom = feature.maximum - feature.minimum
            scaled = (
                jnp.zeros_like(values)
                if denom == 0
                else (2.0 * (values - feature.minimum) / denom) - 1.0
            )
        elif feature.method == "minmax_zero_to_one":
            denom = feature.maximum - feature.minimum
            scaled = (
                jnp.zeros_like(values)
                if denom == 0
                else (values - feature.minimum) / denom
            )
        else:
            raise ValueError(f"Unsupported scaling method {feature.method}.")
        columns.append(scaled)
    return jnp.stack(columns, axis=1).astype(jnp.float32)


def _build_scaled_axis_features_jax(
    axes: tuple[jax.Array, ...],
    axis_specs: tuple[Any, ...],
    input_scaling: tuple[FeatureScaling, ...],
) -> jax.Array:
    """
    Build the transformed and scaled feature rows for one fixed axis grid.
    """
    if len(axes) != len(axis_specs):
        raise ValueError(f"Expected {len(axis_specs)} independent axes, received {len(axes)}.")

    # Build the regular coordinate grid once. The result has one row per grid
    # point and one column per independent axis.
    flat_axes = tuple(jnp.asarray(axis).ravel() for axis in axes)
    mesh_axes = jnp.meshgrid(*flat_axes, indexing="ij")
    axis_columns = [
        _apply_transform_jax(mesh_axis.ravel(), axis_spec.transform)
        for mesh_axis, axis_spec in zip(mesh_axes, axis_specs, strict=True)
    ]
    axis_features = jnp.stack(axis_columns, axis=1)

    # Only scale the axis columns here. Parameter columns are handled per call.
    axis_scaling = input_scaling[: len(axis_specs)]
    return _scale_features_jax(axis_features, axis_scaling)


def _coordinate_count(coordinates: tuple[jax.Array, ...]) -> int:
    """
    Return the number of points in a fixed coordinate list.
    """
    if not coordinates:
        raise ValueError("At least one coordinate array is required.")

    lengths = tuple(int(jnp.asarray(coordinate).size) for coordinate in coordinates)
    if len(set(lengths)) != 1:
        raise ValueError("All coordinate arrays must have the same length.")
    return lengths[0]


def _build_scaled_coordinate_features_jax(
    coordinates: tuple[jax.Array, ...],
    axis_specs: tuple[Any, ...],
    input_scaling: tuple[FeatureScaling, ...],
) -> jax.Array:
    """
    Build transformed and scaled feature rows for one coordinate list.
    """
    if len(coordinates) != len(axis_specs):
        raise ValueError(
            f"Expected {len(axis_specs)} coordinate arrays, received {len(coordinates)}."
        )
    _coordinate_count(coordinates)

    # Store one row per requested coordinate. Unlike the fixed-grid route, this
    # does not expand the axes into a meshgrid.
    axis_columns = [
        _apply_transform_jax(jnp.asarray(coordinate).ravel(), axis_spec.transform)
        for coordinate, axis_spec in zip(coordinates, axis_specs, strict=True)
    ]
    axis_features = jnp.stack(axis_columns, axis=1)

    # Only scale the axis columns here. Parameter columns are handled per call.
    axis_scaling = input_scaling[: len(axis_specs)]
    return _scale_features_jax(axis_features, axis_scaling)
