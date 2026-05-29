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
# Reusable compiled forward model for trained emulators.

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
