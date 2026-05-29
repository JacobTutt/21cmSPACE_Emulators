# Preprocessing and Normalisation

Preprocessing turns raw simulation outputs into the arrays used by the neural
network. It defines the contract between physical quantities and emulator
inputs/outputs.

For scalar-output emulators, the core map is:

```text
physical parameters + coordinates -> network features -> scalar target
```

The same preprocessing contract must be reused at inference time. Physical
inputs are transformed before the network, and network outputs are transformed
back afterwards.

## Motivation

The network should learn in a numerically well-behaved space, not necessarily
in the raw physical units. Preprocessing can make the learning problem simpler:

```text
wide physical ranges -> compact training ranges
positive skewed values -> log-space values
heterogeneous columns -> comparable network inputs
```

The goal is not to change the science target. The goal is to present the same
physical problem in a coordinate system that is easier for the optimizer.

## Workflow

A typical workflow is:

1. Load raw simulation parameters, output axes, and target arrays.
2. Clean failed simulations or NaN targets.
3. Apply deterministic transforms to selected parameters, axes, or targets.
4. Split simulations into train, validation, and test sets.
5. Resample targets onto a shared grid when fixed output axes are needed.
6. Flatten spectra or grids into scalar regression rows.
7. Compute feature and target scaling from the training split only.
8. Reuse the same metadata for validation, test, and inference.

In short:

```text
raw arrays -> clean -> transform -> split -> resample -> flatten -> scale
```

## Why Transform and Normalise?

Neural networks train more reliably when inputs and targets occupy comparable
numerical ranges. Physical simulation parameters often span many orders of
magnitude, so the raw values may be a poor coordinate system for optimisation.

| Operation | Why use it? | Example |
| --- | --- | --- |
| `log10` transform | Compresses positive quantities spanning orders of magnitude. | `f_star -> log10f_star` |
| `log10(y + offset)` | Keeps positive targets finite before taking a logarithm. | power spectrum values |
| `zscore` normalisation | Centres continuous features and scales by training-set scatter. | redshift or log-parameters |
| `minmax_zero_to_one` | Maps bounded or discrete values onto a compact range. | discrete model choices |
| `identity` | Leaves a value unchanged when its physical scale is already suitable. | already-normalised inputs |

## Parameter Transforms

Raw parameter tables often need a small amount of structure:

```text
drop unused columns -> transform selected columns -> record discrete values
```

```python
from jax_emu.data_preprocessing import prepare_feature_matrix

prepared_parameters = prepare_feature_matrix(
    raw_parameters,
    column_names=("f_star", "f_x", "alpha", "unused"),
    transform_params=("f_star", "f_x"),
    discard_params=("unused",),
    discrete_params=("alpha",),
)
```

This function turns the raw parameter table into a `PreparedFeatures` object.
That object is the first product passed into the rest of the preprocessing
pipeline.

| Product | Meaning |
| --- | --- |
| `values` | Numerical parameter matrix after discarded columns and log transforms. |
| `feature_names` | Column names in the exact order expected by the network. |
| `discrete_values` | Allowed values for discrete parameters, useful for metadata and checks. |

## Axis and Target Transforms

Transforms define the coordinate system used for training. If a transform is
applied before training, the inverse transform is needed after inference.

```python
from jax_emu.data_preprocessing import apply_transform, invert_transform

transformed_targets = apply_transform(target, transform="log10", offset=1e-8)
physical_target = invert_transform(transformed_targets, transform="log10", offset=1e-8)
```

`apply_transform` is the forward map into training space. `invert_transform`
is the matching map back to physical space. The optional `offset` keeps
log-transformed positive targets finite.

Transform and normalisation are separate steps:

```text
physical value -> transform -> normalise -> network
```

## Dataset Splitting

Split at the simulation level before flattening. This keeps each parameter row
paired with its full target array.

```python
from jax_emu.data_preprocessing import split_simulations

(
    train_parameters,
    validation_parameters,
    test_parameters,
    train_targets,
    validation_targets,
    test_targets,
) = split_simulations(
    prepared_parameters.values,
    transformed_targets,
    train_size=0.6,
    validation_size=0.2,
    test_size=0.2,
    random_state=42,
)
```

The products are still simulation-level arrays. Each parameter row remains
paired with its full target grid, so no single simulation can leak across train,
validation, and test.

Fit preprocessing statistics from the training split only. Reuse them for
validation, test, and inference.

## Fixed Grid Resampling

Many simulations store targets on an axis grid such as redshift, frequency, or
wavenumber. For fixed-grid training, every split is interpolated onto the same
axis coordinates before flattening.

```python
from jax_emu.data_preprocessing import (
    AxisSpec,
    build_fixed_axis_grid,
    resample_targets_to_grid,
    transformed_axis_configuration,
)

axis_specs = (
    AxisSpec(name="z", transform="identity", limits=(6.0, 27.0), nsample=200),
    AxisSpec(name="k", transform="log10", limits=(0.1, 1.0), nsample=20),
)

transformed_axes, transformed_limits = transformed_axis_configuration(
    axes=(z_axis, k_axis),
    axis_specs=axis_specs,
)
sampled_axes = build_fixed_axis_grid(
    transformed_axes,
    transformed_limits,
    axis_specs,
)

train_target_grid = resample_targets_to_grid(
    train_targets,
    transformed_axes=transformed_axes,
    sampled_axes=sampled_axes,
)
validation_target_grid = resample_targets_to_grid(
    validation_targets,
    transformed_axes=transformed_axes,
    sampled_axes=sampled_axes,
)
test_target_grid = resample_targets_to_grid(
    test_targets,
    transformed_axes=transformed_axes,
    sampled_axes=sampled_axes,
)

feature_names = (
    *(axis.feature_name() for axis in axis_specs),
    *prepared_parameters.feature_names,
)
```

The products are aligned target grids and the axis coordinates used by the
network. `feature_names` records the final feature order: axes first, then
simulation parameters.

## Target Scaling

Targets can also be scaled. The reusable target scaler stores one global
standard deviation measured from transformed training targets.

```python
from jax_emu.data_preprocessing import TargetScalingScalar

target_scaling = TargetScalingScalar.from_targets(train_target_grid)

scaled_train_target_grid = target_scaling.transform_grid(train_target_grid)
scaled_validation_target_grid = target_scaling.transform_grid(validation_target_grid)
scaled_test_target_grid = target_scaling.transform_grid(test_target_grid)
```

`TargetScalingScalar.from_targets` fits one global standard deviation from the
training targets. The transformed target arrays are the labels used by the
trainer. The scaler itself is saved so predictions can be multiplied back to the
unscaled target space.

After inference, invert target scaling before inverting the physical target
transform:

```python
predicted_target_grid = target_scaling.inverse_grid(model_output_grid)
physical_target_grid = invert_transform(
    predicted_target_grid,
    transform="log10",
    offset=1e-8,
)
```

## Tiling Scalar Rows

The dense MLP trains on scalar rows. Grid-valued targets are flattened:

```text
[axis coordinates, physical parameters] -> one target value
```

```python
from jax_emu.data_preprocessing import flatten_resampled_rows, reconstruct_spectra

train_features, train_target_rows = flatten_resampled_rows(
    train_parameters,
    scaled_train_target_grid,
    sampled_axes=sampled_axes,
)
validation_features, validation_target_rows = flatten_resampled_rows(
    validation_parameters,
    scaled_validation_target_grid,
    sampled_axes=sampled_axes,
)
test_features, test_target_rows = flatten_resampled_rows(
    test_parameters,
    scaled_test_target_grid,
    sampled_axes=sampled_axes,
)

axis_shape = tuple(len(axis) for axis in sampled_axes)
predicted_grid = reconstruct_spectra(
    flat_predictions,
    nsamples=len(test_parameters),
    axis_shape=axis_shape,
)
```

`flatten_resampled_rows` produces the final scalar-regression training product:

```text
train_features:     (n_simulations * n_grid_points, n_axes + n_parameters)
train_target_rows:  (n_simulations * n_grid_points,)
axis_shape:         original grid shape needed for reconstruction
```

`reconstruct_spectra` performs the shape inverse after prediction. It folds the
flat network outputs back into per-simulation grids.

This is a shape transform. It does not change the physical values.

## Feature Scaling

Feature scaling maps each input column into the numerical space seen by the
network.

| Method | Operation | Typical use |
| --- | --- | --- |
| `identity` | `x` | Already suitable values |
| `zscore` | `(x - mean) / std` | Continuous features |
| `minmax_zero_to_one` | `(x - min) / (max - min)` | Bounded or discrete features |
| `minmax_minus_one_to_one` | Scale to `[-1, 1]` | Symmetric bounded features |

```python
from jax_emu.data_preprocessing import FeatureScaler, FeatureScaling

feature_scale_methods = {
    "z": "zscore",
    "log10k": "zscore",
    "log10f_star": "zscore",
    "log10f_x": "zscore",
    "alpha": "minmax_zero_to_one",
}

scaling = tuple(
    FeatureScaling.from_values(
        name,
        train_features[:, idx],
        feature_scale_methods[name],
    )
    for idx, name in enumerate(feature_names)
)

feature_scaler = FeatureScaler(scaling)

scaled_train_features = feature_scaler.transform(train_features)
scaled_validation_features = feature_scaler.transform(validation_features)
scaled_test_features = feature_scaler.transform(test_features)
```

`FeatureScaling.from_values` fits one column rule from the training rows.
`FeatureScaler` applies the ordered tuple of rules to a full feature matrix.
The important products are:

| Product | Used for |
| --- | --- |
| `scaling` | Metadata saved with the emulator checkpoint. |
| `feature_scaler` | Object that applies the same rules to any split or inference input. |
| `scaled_*_features` | Arrays passed directly to the network. |

Scaling metadata is fitted once from training rows:

```text
training rows -> scaling metadata -> all splits and inference inputs
```

## Full Preparation Helper

For fixed-grid workflows, `prepare_fixed_grid_training_split` combines the
standard steps:

```text
target transform -> split -> resample -> target scale -> flatten -> feature scale
```

```python
from jax_emu.data_preprocessing import AxisSpec, prepare_fixed_grid_training_split

prepared = prepare_fixed_grid_training_split(
    axes=(z_axis, k_axis),
    axis_specs=(
        AxisSpec(name="z", transform="identity", limits=(6.0, 27.0), nsample=200),
        AxisSpec(name="k", transform="log10", limits=(0.1, 1.0), nsample=20),
    ),
    parameters=prepared_parameters,
    target=target,
    feature_scale_methods={
        "z": "zscore",
        "log10k": "zscore",
        "log10f_star": "zscore",
        "log10f_x": "zscore",
        "alpha": "minmax_zero_to_one",
    },
    data_log=False,
    offset=None,
    train_size=0.6,
    validation_size=0.2,
    test_size=0.2,
    random_state=42,
    shuffle_seed=42,
)
```

The returned `PreparedSplit` is the hand-off to training code. It contains:

| Product | Meaning |
| --- | --- |
| `train_*`, `validation_*`, `test_*` | Final feature and target arrays for each split. |
| `feature_names` | Feature order used by the network. |
| `feature_scaling` | Input scaling metadata to save with the emulator. |
| `target_scaling` | Optional output scaling metadata to invert predictions. |

## Remembering the Contract

The model weights are not enough for inference. The checkpoint also needs the
preprocessing contract:

| Metadata | What it remembers |
| --- | --- |
| `emulator_spec` | Axis, parameter, and target transforms. |
| `input_scaling` | Per-feature means, standard deviations, and min/max values. |
| `target_scaling` | Optional target scaling used after the network output. |
| `feature_names` | The column order expected by the trained model. |

Training code stores this metadata beside the model weights:

```python
from jax_emu.utils import CheckpointMetadata, save

metadata = CheckpointMetadata(
    model_name="my_emulator",
    package_version="0.1.0",
    emulator_spec=emulator_spec,
    input_scaling=prepared.feature_scaling,
    target_scaling=prepared.target_scaling,
    training_config={
        "feature_names": list(prepared.feature_names),
    },
)

save(
    "my_emulator.nenemu",
    model,
    train_losses=history.train_losses,
    val_losses=history.validation_losses,
    loss="mse",
    metadata=metadata,
)
```

The saved checkpoint is therefore:

```text
model weights + architecture + preprocessing metadata
```

That is what lets the inference code apply the same transforms without guessing
how the model was trained.

## Training

During training, preprocessing maps physical simulation data into the numerical
space seen by the network.

```text
physical inputs -> transforms -> normalisation -> network inputs
physical targets -> transforms -> normalisation -> network targets
```

For a scalar-output emulator, each training row is:

```text
[transformed and normalised inputs] -> transformed and normalised scalar target
```

The trainer only sees these arrays. The original physical units are represented
by the preprocessing metadata.

![Training preprocessing flow](assets/preprocessing-training.svg)

## Inference

At inference time, the same metadata maps new physical inputs into the network
and maps network outputs back to physical units.

```text
new physical parameters
-> input transforms
-> input normalisation
-> DenseMLP
-> inverse target normalisation
-> inverse target transform
-> physical prediction
```

The key point is that the network predicts in training space, not directly in
physical space. The emulator is the network plus the saved transform and
normalisation metadata.

The inference path loads the model and the remembered preprocessing metadata.
The metadata is the link between the physical input and the trained network:

```python
from jax_emu.utils import load

package = load("my_emulator.nenemu")

model = package["model"]
metadata = package["metadata"]
spec = metadata.emulator_spec

axis_specs = spec.axes
parameter_specs = spec.parameters
expected_feature_order = spec.input_feature_names()
feature_scaling = metadata.input_scaling
target_scaling = metadata.target_scaling
target_transform = spec.target_transform
target_offset = spec.target_offset
```

The important point is where each rule comes from. Feature scaling comes from
`metadata.input_scaling`; the target inverse comes from `metadata.target_scaling`;
and the physical target transform comes from `metadata.emulator_spec`.

Concrete inference functions then use those objects to build the same feature
matrix used in training, run the model, and invert the output:

```python
from emulators_21cmspace.delta21.infer import load_delta21_package, predict_delta21

package = load_delta21_package("delta21_model.nenemu")

delta21 = predict_delta21(
    package,
    parameters=physical_parameters,
    z_values=z_grid,
    k_values=k_grid,
)
```

Inside `predict_delta21`, the saved spec supplies the axis and target transforms,
`metadata.input_scaling` supplies the feature normalisation, and
`metadata.target_scaling` supplies the inverse output scaling. The same pattern
is used by the T21 inference helper.

![Inference preprocessing flow](assets/preprocessing-inference.svg)
