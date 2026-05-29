# Preprocessing and Normalisation

Preprocessing is the step that turns raw simulation outputs into arrays that a
neural network can train on. It defines the numerical contract between physical
simulation quantities and the values seen by the emulator.

For scalar-output emulators, preprocessing usually has two jobs:

1. Build the feature matrix that goes into the network.
2. Transform the target values that the network is trained to predict.

The same operations must be available at inference time. A trained emulator is
only meaningful if physical inputs are transformed into the same feature space
used during training, and network outputs are transformed back into physical
units in the correct order.

![Preprocessing workflow](assets/preprocessing-flow.svg)

## Workflow

A typical emulator preprocessing workflow is:

1. Load raw simulation parameters, output axes, and target arrays.
2. Clean the data, for example by removing failed simulations or NaN targets.
3. Apply deterministic transforms to selected parameters, axes, or targets.
4. Split simulations into train, validation, and test sets.
5. Resample targets onto a shared grid if the emulator needs fixed output axes.
6. Flatten grid-based targets into scalar regression rows.
7. Compute feature and target scaling from the training split only.
8. Apply the same scaling rules to validation, test, and inference inputs.

The exact transforms depend on the observable. For example, a positive target
may be easier to emulate in `log10(y + offset)`, while a coordinate or physical
parameter may be left in its original units.

## Parameter Transforms

Raw simulation parameters are often not used directly. Some columns may be
dropped, some may be log-transformed, and some may be recorded as discrete
parameters.

```python
from jax_emu.data_preprocessing import prepare_feature_matrix

prepared_parameters = prepare_feature_matrix(
    raw_parameters,
    column_names=("f_star", "f_x", "alpha", "unused"),
    transform_params=("f_star", "f_x"),
    discard_params=("unused",),
    discrete_params=("alpha",),
)

print(prepared_parameters.feature_names)
print(prepared_parameters.values.shape)
print(prepared_parameters.discrete_values)
```

This produces a numerical parameter matrix plus metadata describing the feature
names and any discrete values. The feature names matter because they define the
column order expected by the network and by any saved checkpoint.

## Axis and Target Transforms

Transforms move physical values into the coordinate system used for training.
The same transform must be inverted after inference if the prediction needs to
be reported in physical units.

```python
from jax_emu.data_preprocessing import apply_transform, invert_transform

log_target = apply_transform(target, transform="log10", offset=1e-8)
physical_target = invert_transform(log_target, transform="log10", offset=1e-8)
```

The transform itself is separate from scaling. A log transform changes the
coordinate system. Scaling then rescales the transformed values to a numerical
range that is easier for the network to optimize.

## Dataset Splitting

Splitting is done at the simulation level before flattening into scalar rows.
This keeps each simulation's parameters paired with its full target array.

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

After this split, preprocessing statistics should be fitted from the training
split only. Validation, test, and inference data should reuse the training
metadata.

## Feature Scaling

Feature scaling maps each input column into the numerical space seen by the
network. Common choices are z-score scaling for continuous quantities and
min-max scaling for discrete or bounded quantities.

```python
from jax_emu.data_preprocessing import FeatureScaler, FeatureScaling

scaling = (
    FeatureScaling.from_values("z", train_features[:, 0], "zscore"),
    FeatureScaling.from_values("log10f_star", train_features[:, 1], "zscore"),
    FeatureScaling.from_values("alpha", train_features[:, 2], "minmax_zero_to_one"),
)

feature_scaler = FeatureScaler(scaling)

scaled_train_features = feature_scaler.transform(train_features)
scaled_validation_features = feature_scaler.transform(validation_features)
```

The important rule is that scaling statistics are fitted once from the training
split and then reused. Recomputing scaling separately for validation, test, or
inference would change the meaning of the network inputs.

## Target Scaling

Targets can also be scaled before training. The current reusable target scaler
stores one global standard deviation measured from the transformed training
targets.

```python
from jax_emu.data_preprocessing import TargetScalingScalar

target_scaling = TargetScalingScalar.from_targets(train_target_grid)

scaled_train_targets = target_scaling.transform_grid(train_target_grid)
scaled_validation_targets = target_scaling.transform_grid(validation_target_grid)
```

After inference, the operation is inverted before any physical target transform
is undone:

```python
predicted_target_grid = target_scaling.inverse_grid(model_output_grid)
physical_target_grid = invert_transform(
    predicted_target_grid,
    transform="log10",
    offset=1e-8,
)
```

## Tiling Scalar Rows

The dense MLP is trained on scalar regression rows. If the raw target is a
spectrum or grid, preprocessing flattens it into rows of the form:

```text
[axis coordinates, physical parameters] -> one target value
```

```python
from jax_emu.data_preprocessing import tile_spectra, reconstruct_spectra

features, target_rows, axis_shape = tile_spectra(
    train_parameters,
    axes=(z_grid, k_grid),
    targets=scaled_train_targets,
)

predicted_grid = reconstruct_spectra(
    flat_predictions,
    nsamples=len(test_parameters),
    axis_shape=axis_shape,
)
```

This is the shape transform that connects a grid-valued observable to a
scalar-output neural network.

## Full Preparation Helper

For workflows with fixed output axes, `prepare_fixed_grid_training_split`
combines the common steps: target transform, simulation split, fixed-grid
resampling, target scaling, row flattening, feature scaling, and row shuffling.

```python
from jax_emu.data_preprocessing import AxisSpec, prepare_fixed_grid_training_split

prepared = prepare_fixed_grid_training_split(
    axes=(z_axis,),
    axis_specs=(
        AxisSpec(name="z", transform="identity", limits=(6.0, 27.0), nsample=200),
    ),
    parameters=prepared_parameters,
    target=target,
    feature_scale_methods={
        "z": "zscore",
        "log10f_star": "zscore",
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

The returned object contains both arrays and metadata:

```python
prepared.train_features
prepared.train_targets
prepared.validation_features
prepared.validation_targets
prepared.test_features
prepared.test_targets
prepared.feature_names
prepared.feature_scaling
prepared.target_scaling
```

## Training and Inference Contract

The network does not know about physical units by itself. It only sees the
feature matrix and target values produced by preprocessing. A complete emulator
therefore needs to save enough metadata to repeat the same transforms later.

At training time:

```text
physical parameters
-> parameter and axis transforms
-> feature scaling
-> DenseMLP
-> scaled target values
```

At inference time:

```text
new physical parameters
-> same parameter and axis transforms
-> same feature scaling
-> DenseMLP prediction
-> inverse target scaling
-> inverse target transform
-> physical prediction
```

This is why preprocessing metadata is part of the emulator, not just a setup
detail. If the transform or scaling contract changes, the trained network is no
longer being evaluated on the same problem it learned.
