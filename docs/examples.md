# Examples

This page shows how the preprocessing, architecture, training, checkpointing,
and inference pieces fit together in Python.

The current 21cmSPACE examples are:

- global 21-cm brightness temperature, `T21(z)`
- 21-cm power spectrum, `Delta21(z, k)`

## Dataset Root

Both examples expect a 21cmSPACE dataset root containing the MATLAB `.mat`
files:

```python
# Path containing the 21cmSPACE .mat files.
dataset_root = "/path/to/21cmspace-data"
```

The loader reads:

| File | MATLAB key | Used by |
| --- | --- | --- |
| `21cmspace_z_mat.mat` | `z21cm` | T21 and Delta21 |
| `21cmspace_k_mat.mat` | `ks` | Delta21 |
| `21cmspace_nu_mat.mat` | `nu_keV` | dataset loader metadata |
| `21cmspace_parameters_mat.mat` | `parameters` | T21 and Delta21 |
| `21cmspace_T21_mat.mat` | `combined_T21s` | T21 |
| `21cmspace_Deltak_mat.mat` | `combined_Deltaks` | Delta21 |

The raw parameter table has 12 columns:

```text
fstarII, fstarIII, Vc, fX, alpha, nu_0, zeta, tau, fradio, pop, feed, delay
```

The emulator uses 9 prepared parameters after dropping `zeta`, `feed`, and
`delay`.

## The Code Path

Each example follows the same route:

```text
load dataset
-> prepare fixed-grid train/validation/test arrays
-> build DenseMLP
-> train with train_mlp_regressor
-> save weights + preprocessing metadata
-> load checkpoint and predict
```

## Global 21-cm Signal: `T21`

`T21` is a one-axis emulator:

```text
[z, parameters] -> T21(z)
```

### Prepare Arrays

```python
# NumPy is used for host-side array manipulation during preprocessing.
import numpy as np

# T21-specific parameter filtering and emulator specification.
from emulators_21cmspace.t21.data import (
    prepare_twentyonecmspace_t21_parameters,
    t21_spec,
)

# Dataset loader for the raw 21cmSPACE global-signal files.
from emulators_21cmspace.twentyonecmspace import load_twentyonecmspace_t21

# Lower-level preprocessing utilities used to build the training arrays.
from jax_emu.data_preprocessing import (
    PreparedSplit,
    TargetScalingScalar,
    build_feature_scaler,
    build_fixed_axis_grid,
    flatten_resampled_rows,
    resample_targets_to_grid,
    shuffle_rows,
    split_simulations,
    transform_target,
    transformed_axis_configuration,
)

# Load the raw axes, parameter table, and T21 target array from disk.
product = load_twentyonecmspace_t21(dataset_root)

# Drop unused parameters and apply log10 transforms to selected columns.
prepared_parameters = prepare_twentyonecmspace_t21_parameters(product.parameters)

# Load the model contract: axes, parameter order, and target transform.
spec = t21_spec()

# Pull out the physical redshift grid read from the dataset.
z_axis = product.axes.z

# Pull out the raw T21 target grid with shape (n_simulations, n_z).
raw_t21_targets = product.target

# Store axes as a tuple because the generic preprocessing code supports many axes.
axes = (z_axis,)

# Store the axis preprocessing settings from the emulator specification.
axis_specs = spec.axes

# Define how each final input feature should be scaled before entering the MLP.
feature_scale_methods = {
    "z": "zscore",  # Redshift is continuous, so z-score scaling is used.
    "log10fstarII": "zscore",  # Logged continuous parameter.
    "log10fstarIII": "zscore",  # Logged continuous parameter.
    "log10Vc": "zscore",  # Logged continuous parameter.
    "log10fX": "zscore",  # Logged continuous parameter.
    "alpha": "minmax_zero_to_one",  # Discrete sampled parameter.
    "nu_0": "minmax_zero_to_one",  # Discrete sampled parameter.
    "tau": "zscore",  # Continuous parameter left in linear space.
    "log10fradio": "zscore",  # Logged continuous parameter.
    "pop": "minmax_zero_to_one",  # Discrete sampled parameter.
}

# T21 is trained in linear target space, so this leaves the target unchanged.
transformed_target = transform_target(
    raw_t21_targets,
    data_log=False,
    offset=None,
)

# Split simulations before fitting any preprocessing statistics.
(
    train_parameters,
    validation_parameters,
    test_parameters,
    train_target,
    validation_target,
    test_target,
) = split_simulations(
    prepared_parameters.values,  # Prepared 9-column parameter matrix.
    transformed_target,  # Target array in training target space.
    train_size=0.6,  # Fraction used for gradient updates.
    validation_size=0.2,  # Fraction used for validation loss.
    test_size=0.2,  # Fraction kept for final evaluation.
    random_state=42,  # Seed for reproducible simulation-level splitting.
)

# Transform the redshift axis and transform the requested interpolation limits.
transformed_axes, transformed_limits = transformed_axis_configuration(axes, axis_specs)

# Build the fixed redshift grid used for every simulation.
sampled_axes = build_fixed_axis_grid(transformed_axes, transformed_limits, axis_specs)

# Build the exact feature order: axis coordinates first, then prepared parameters.
feature_names = (
    *(axis.feature_name() for axis in axis_specs),
    *prepared_parameters.feature_names,
)

# Interpolate training targets onto the fixed redshift grid.
train_target_grid = resample_targets_to_grid(
    train_target,
    transformed_axes=transformed_axes,
    sampled_axes=sampled_axes,
)

# Interpolate validation targets onto the same fixed redshift grid.
validation_target_grid = resample_targets_to_grid(
    validation_target,
    transformed_axes=transformed_axes,
    sampled_axes=sampled_axes,
)

# Interpolate test targets onto the same fixed redshift grid.
test_target_grid = resample_targets_to_grid(
    test_target,
    transformed_axes=transformed_axes,
    sampled_axes=sampled_axes,
)

# Fit one global target standard deviation from the training targets only.
target_scaling = TargetScalingScalar.from_targets(train_target_grid)

# Scale training targets using the training-set target statistic.
train_target_grid = target_scaling.transform_grid(train_target_grid)

# Reuse the same target scaling for validation targets.
validation_target_grid = target_scaling.transform_grid(validation_target_grid)

# Reuse the same target scaling for test targets.
test_target_grid = target_scaling.transform_grid(test_target_grid)

# Flatten the training grids into scalar rows: [z, parameters] -> one T21 value.
train_features, train_targets = flatten_resampled_rows(
    train_parameters,
    train_target_grid,
    sampled_axes=sampled_axes,
)

# Flatten the validation grids in the same feature order.
validation_features, validation_targets = flatten_resampled_rows(
    validation_parameters,
    validation_target_grid,
    sampled_axes=sampled_axes,
)

# Flatten the test grids in the same feature order.
test_features, test_targets = flatten_resampled_rows(
    test_parameters,
    test_target_grid,
    sampled_axes=sampled_axes,
)

# Fit feature scaling statistics from the training rows only.
feature_scaler = build_feature_scaler(
    train_features,
    feature_names=feature_names,
    method_overrides=feature_scale_methods,
)

# Scale training inputs into the numerical space expected by the MLP.
train_features = feature_scaler.transform(train_features).astype(np.float32)

# Reuse the same feature scaler for validation inputs.
validation_features = feature_scaler.transform(validation_features).astype(np.float32)

# Reuse the same feature scaler for test inputs.
test_features = feature_scaler.transform(test_features).astype(np.float32)

# Store training targets as float32 arrays for JAX training.
train_targets = np.asarray(train_targets, dtype=np.float32)

# Store validation targets as float32 arrays for JAX evaluation.
validation_targets = np.asarray(validation_targets, dtype=np.float32)

# Store test targets as float32 arrays for final evaluation.
test_targets = np.asarray(test_targets, dtype=np.float32)

# Shuffle training rows so each mini-batch mixes simulations and redshifts.
train_features, train_targets = shuffle_rows(train_features, train_targets, seed=42)

# Shuffle validation rows in the same paired feature/target way.
validation_features, validation_targets = shuffle_rows(
    validation_features,
    validation_targets,
    seed=42,
)

# Shuffle test rows in the same paired feature/target way.
test_features, test_targets = shuffle_rows(test_features, test_targets, seed=42)

# Bundle arrays and metadata into the object consumed by the trainer.
prepared = PreparedSplit(
    feature_names=feature_names,  # Names and order of the MLP input columns.
    train_features=train_features,  # Scaled input rows used for gradient updates.
    train_targets=train_targets,  # Scaled target values used for gradient updates.
    validation_features=validation_features,  # Scaled input rows used for validation.
    validation_targets=validation_targets,  # Scaled target values used for validation.
    test_features=test_features,  # Scaled input rows kept for final testing.
    test_targets=test_targets,  # Scaled target values kept for final testing.
    feature_scaling=feature_scaler.scaling,  # Feature scaling metadata saved later.
    target_scaling=target_scaling,  # Target scaling metadata saved later.
)

# Inspect the final feature order that the trained model will expect.
print(prepared.feature_names)

# Inspect the final training input matrix shape.
print(prepared.train_features.shape)

# Inspect the final training target vector shape.
print(prepared.train_targets.shape)
```

This runs the data-loading and preprocessing workflow:

```text
raw T21 grids
-> drop failed simulations
-> transform parameters
-> split simulations
-> resample onto the fixed z grid
-> tile into scalar rows
-> scale features and targets
```

The returned `prepared` object is what the trainer consumes.

The convenience wrapper `prepare_twentyonecmspace_t21_training_split()` runs the
same code path above with the default `T21` settings.

### Train And Save

```python
# Convert dataclass configs into dictionaries for checkpoint metadata.
from dataclasses import asdict

# Use Path objects for checkpoint output paths.
from pathlib import Path

# JAX provides the random key used to initialize model weights.
import jax

# Flax NNX provides the module system and random stream wrapper.
from flax import nnx

# T21 spec is saved so inference can reconstruct the preprocessing contract.
from emulators_21cmspace.t21.data import t21_spec

# T21 config stores the default architecture and training settings.
from emulators_21cmspace.t21.model import t21_config

# DenseMLP is the neural network architecture used by the emulator.
from jax_emu.architectures import DenseMLP

# Shared trainer utilities update the model and evaluate the held-out test set.
from jax_emu.training import evaluate_mlp_regressor, train_mlp_regressor

# Checkpoint helpers save weights plus preprocessing metadata.
from jax_emu.utils import CheckpointMetadata, save

# Load the default T21 model, optimizer, and training settings.
config = t21_config()

# Build the MLP with an input width matching the prepared feature matrix.
model = DenseMLP(
    in_features=prepared.train_features.shape[1],  # Number of input columns.
    hidden_features=config.mlp.hidden_dim,  # Width of each hidden layer.
    hidden_layers=config.mlp.total_hidden_layers,  # Number of hidden layers.
    activation=config.mlp.activation,  # Non-linear activation after hidden layers.
    rngs=nnx.Rngs(jax.random.PRNGKey(42)),  # Random stream for weight initialization.
)

# Train the model and record training/validation loss curves.
model, history = train_mlp_regressor(
    model,  # Live NNX model to update.
    prepared.train_features,  # Training input rows.
    prepared.train_targets,  # Training target values.
    prepared.validation_features,  # Validation input rows.
    prepared.validation_targets,  # Validation target values.
    epochs=config.training.epochs,  # Maximum number of full passes over training data.
    batch_size=config.training.batch_size,  # Number of rows per mini-batch.
    prefetch_batches=config.training.prefetch_batches,  # Batches queued on device.
    learning_rate=config.optimizer.learning_rate,  # AdamW update step size.
    weight_decay=config.optimizer.weight_decay,  # AdamW L2-style regularisation.
    seed=42,  # Seed used for epoch-level row shuffling.
    early_stopping_patience=config.training.early_stopping_patience,  # Waiting time.
    early_stopping_min_delta=config.training.early_stopping_min_delta,  # Improvement size.
)

# Evaluate the trained model on the test split after training has finished.
test_loss = evaluate_mlp_regressor(
    model,  # Trained model to evaluate.
    prepared.test_features,  # Held-out test inputs.
    prepared.test_targets,  # Held-out test targets.
    batch_size=config.training.batch_size,  # Number of rows per evaluation batch.
    prefetch_batches=config.training.prefetch_batches,  # Batches queued on device.
)

# Package the non-weight information required to reuse the model at inference.
metadata = CheckpointMetadata(
    model_name="t21",  # Human-readable emulator name.
    package_version="0.1.0",  # Version label for the saved package format.
    emulator_spec=t21_spec(),  # Axis, parameter, and target-transform contract.
    input_scaling=prepared.feature_scaling,  # Feature scaling used before the MLP.
    target_scaling=prepared.target_scaling,  # Target scaling inverted after the MLP.
    training_config={  # Training settings kept with the checkpoint.
        "mlp": asdict(config.mlp),  # Network architecture settings.
        "optimizer": asdict(config.optimizer),  # Optimizer settings.
        "training": asdict(config.training),  # Epoch, batch, and stopping settings.
        "feature_names": list(prepared.feature_names),  # Required input column order.
    },
)

# Ensure the output directory exists before writing the checkpoint.
Path("outputs").mkdir(exist_ok=True)

# Save the trained weights, architecture settings, losses, and preprocessing metadata.
package_path = save(
    Path("outputs/t21_model.nenemu"),  # Output checkpoint path.
    model,  # Trained NNX model whose state will be saved.
    train_losses=history.train_losses,  # Training loss curve.
    val_losses=history.validation_losses,  # Validation loss curve.
    loss=config.training.loss_name,  # Loss name used during training.
    metadata=metadata,  # Preprocessing and training metadata.
    epochs=config.training.epochs,  # Maximum configured epochs.
    patience=config.training.early_stopping_patience,  # Early-stopping patience.
    learning_rate=config.optimizer.learning_rate,  # Optimizer learning rate.
    weight_decay=config.optimizer.weight_decay,  # Optimizer weight decay.
)

# Print the saved checkpoint path.
print(package_path)

# Print the final held-out test loss.
print(test_loss)
```

The checkpoint stores:

```text
DenseMLP architecture + trained weights + T21 preprocessing metadata
```

### Predict

```python
# JAX arrays keep inference inputs and outputs on the accelerator.
import jax.numpy as jnp

# T21 inference helpers load the checkpoint and apply the saved preprocessing contract.
from emulators_21cmspace.t21.infer import load_t21_package, predict_t21

# Load the trained model and its saved metadata.
package = load_t21_package("outputs/t21_model.nenemu")

# Provide one raw 12-column physical parameter row in the original dataset order.
physical_parameters = jnp.asarray(
    [
        [1e-2, 1e-3, 16.5, 1e2, 1.3, 500.0, 30.0, 0.055, 1e2, 232.0, 0.0, 0.0],
    ],
    dtype=jnp.float32,  # Match the float32 dtype used during training.
)

# Choose the redshift coordinates where the signal should be predicted.
z = jnp.linspace(6.0, 27.0, 200)

# Predict T21 and return an array with shape (n_sims, n_z).
t21 = predict_t21(package, physical_parameters, z)

# Inspect the prediction shape.
print(t21.shape)
```

The inference helper uses the metadata saved in the checkpoint to apply the
same parameter transforms and feature scaling used during training.

## 21-cm Power Spectrum: `Delta21`

`Delta21` is a two-axis emulator:

```text
[z, k, parameters] -> Delta21(z, k)
```

### Prepare Arrays

```python
# NumPy is used for host-side array manipulation during preprocessing.
import numpy as np

# Delta21-specific parameter filtering and emulator specification.
from emulators_21cmspace.delta21.data import (
    delta21_spec,
    prepare_twentyonecmspace_delta21_parameters,
)

# Dataset loader for the raw 21cmSPACE power-spectrum files.
from emulators_21cmspace.twentyonecmspace import load_twentyonecmspace_delta21

# Lower-level preprocessing utilities used to build the training arrays.
from jax_emu.data_preprocessing import (
    PreparedSplit,
    TargetScalingScalar,
    build_feature_scaler,
    build_fixed_axis_grid,
    flatten_resampled_rows,
    resample_targets_to_grid,
    shuffle_rows,
    split_simulations,
    transform_target,
    transformed_axis_configuration,
)

# Load the raw axes, parameter table, and Delta21 target array from disk.
product = load_twentyonecmspace_delta21(dataset_root)

# Drop unused parameters and apply log10 transforms to selected columns.
prepared_parameters = prepare_twentyonecmspace_delta21_parameters(product.parameters)

# Load the model contract: axes, parameter order, and target transform.
spec = delta21_spec()

# Pull out the physical redshift grid read from the dataset.
z_axis = product.axes.z

# Pull out the physical wavenumber grid read from the dataset.
k_axis = product.axes.k

# Pull out the raw Delta21 target grid with shape (n_simulations, n_z, n_k).
raw_delta21_targets = product.target

# Store axes as a tuple because Delta21 depends on redshift and wavenumber.
axes = (z_axis, k_axis)

# Store the axis preprocessing settings from the emulator specification.
axis_specs = spec.axes

# Define how each final input feature should be scaled before entering the MLP.
feature_scale_methods = {
    "z": "zscore",  # Redshift is continuous, so z-score scaling is used.
    "log10k": "zscore",  # Wavenumber is log-transformed then z-score scaled.
    "log10fstarII": "zscore",  # Logged continuous parameter.
    "log10fstarIII": "zscore",  # Logged continuous parameter.
    "log10Vc": "zscore",  # Logged continuous parameter.
    "log10fX": "zscore",  # Logged continuous parameter.
    "alpha": "minmax_zero_to_one",  # Discrete sampled parameter.
    "nu_0": "minmax_zero_to_one",  # Discrete sampled parameter.
    "tau": "zscore",  # Continuous parameter left in linear space.
    "log10fradio": "zscore",  # Logged continuous parameter.
    "pop": "minmax_zero_to_one",  # Discrete sampled parameter.
}

# Delta21 is trained in log10 target space using the configured positive offset.
transformed_target = transform_target(
    raw_delta21_targets,
    data_log=True,
    offset=1e-8,
)

# Split simulations before fitting any preprocessing statistics.
(
    train_parameters,
    validation_parameters,
    test_parameters,
    train_target,
    validation_target,
    test_target,
) = split_simulations(
    prepared_parameters.values,  # Prepared 9-column parameter matrix.
    transformed_target,  # Target array in training target space.
    train_size=0.6,  # Fraction used for gradient updates.
    validation_size=0.2,  # Fraction used for validation loss.
    test_size=0.2,  # Fraction kept for final evaluation.
    random_state=42,  # Seed for reproducible simulation-level splitting.
)

# Transform the z and k axes and transform the requested interpolation limits.
transformed_axes, transformed_limits = transformed_axis_configuration(axes, axis_specs)

# Build the fixed (z, k) grid used for every simulation.
sampled_axes = build_fixed_axis_grid(transformed_axes, transformed_limits, axis_specs)

# Build the exact feature order: axis coordinates first, then prepared parameters.
feature_names = (
    *(axis.feature_name() for axis in axis_specs),
    *prepared_parameters.feature_names,
)

# Interpolate training targets onto the fixed (z, k) grid.
train_target_grid = resample_targets_to_grid(
    train_target,
    transformed_axes=transformed_axes,
    sampled_axes=sampled_axes,
)

# Interpolate validation targets onto the same fixed (z, k) grid.
validation_target_grid = resample_targets_to_grid(
    validation_target,
    transformed_axes=transformed_axes,
    sampled_axes=sampled_axes,
)

# Interpolate test targets onto the same fixed (z, k) grid.
test_target_grid = resample_targets_to_grid(
    test_target,
    transformed_axes=transformed_axes,
    sampled_axes=sampled_axes,
)

# Fit one global target standard deviation from the training targets only.
target_scaling = TargetScalingScalar.from_targets(train_target_grid)

# Scale training targets using the training-set target statistic.
train_target_grid = target_scaling.transform_grid(train_target_grid)

# Reuse the same target scaling for validation targets.
validation_target_grid = target_scaling.transform_grid(validation_target_grid)

# Reuse the same target scaling for test targets.
test_target_grid = target_scaling.transform_grid(test_target_grid)

# Flatten the training grids into scalar rows: [z, k, parameters] -> one value.
train_features, train_targets = flatten_resampled_rows(
    train_parameters,
    train_target_grid,
    sampled_axes=sampled_axes,
)

# Flatten the validation grids in the same feature order.
validation_features, validation_targets = flatten_resampled_rows(
    validation_parameters,
    validation_target_grid,
    sampled_axes=sampled_axes,
)

# Flatten the test grids in the same feature order.
test_features, test_targets = flatten_resampled_rows(
    test_parameters,
    test_target_grid,
    sampled_axes=sampled_axes,
)

# Fit feature scaling statistics from the training rows only.
feature_scaler = build_feature_scaler(
    train_features,
    feature_names=feature_names,
    method_overrides=feature_scale_methods,
)

# Scale training inputs into the numerical space expected by the MLP.
train_features = feature_scaler.transform(train_features).astype(np.float32)

# Reuse the same feature scaler for validation inputs.
validation_features = feature_scaler.transform(validation_features).astype(np.float32)

# Reuse the same feature scaler for test inputs.
test_features = feature_scaler.transform(test_features).astype(np.float32)

# Store training targets as float32 arrays for JAX training.
train_targets = np.asarray(train_targets, dtype=np.float32)

# Store validation targets as float32 arrays for JAX evaluation.
validation_targets = np.asarray(validation_targets, dtype=np.float32)

# Store test targets as float32 arrays for final evaluation.
test_targets = np.asarray(test_targets, dtype=np.float32)

# Shuffle training rows so each mini-batch mixes simulations and grid points.
train_features, train_targets = shuffle_rows(train_features, train_targets, seed=42)

# Shuffle validation rows in the same paired feature/target way.
validation_features, validation_targets = shuffle_rows(
    validation_features,
    validation_targets,
    seed=42,
)

# Shuffle test rows in the same paired feature/target way.
test_features, test_targets = shuffle_rows(test_features, test_targets, seed=42)

# Bundle arrays and metadata into the object consumed by the trainer.
prepared = PreparedSplit(
    feature_names=feature_names,  # Names and order of the MLP input columns.
    train_features=train_features,  # Scaled input rows used for gradient updates.
    train_targets=train_targets,  # Scaled target values used for gradient updates.
    validation_features=validation_features,  # Scaled input rows used for validation.
    validation_targets=validation_targets,  # Scaled target values used for validation.
    test_features=test_features,  # Scaled input rows kept for final testing.
    test_targets=test_targets,  # Scaled target values kept for final testing.
    feature_scaling=feature_scaler.scaling,  # Feature scaling metadata saved later.
    target_scaling=target_scaling,  # Target scaling metadata saved later.
)

# Inspect the final feature order that the trained model will expect.
print(prepared.feature_names)

# Inspect the final training input matrix shape.
print(prepared.train_features.shape)

# Inspect the final training target vector shape.
print(prepared.train_targets.shape)
```

This runs the same generic preprocessing workflow, but with the `Delta21`
contract:

```text
raw Delta21 grids
-> drop failed simulations
-> transform parameters
-> log-transform k
-> log-transform Delta21 with offset 1e-8
-> split simulations
-> resample onto the fixed (z, k) grid
-> tile into scalar rows
-> scale features and targets
```

The convenience wrapper `prepare_twentyonecmspace_delta21_training_split()` runs
the same code path above with the default `Delta21` settings.

### Train And Save

```python
# Convert dataclass configs into dictionaries for checkpoint metadata.
from dataclasses import asdict

# Use Path objects for checkpoint output paths.
from pathlib import Path

# JAX provides the random key used to initialize model weights.
import jax

# Flax NNX provides the module system and random stream wrapper.
from flax import nnx

# Delta21 spec is saved so inference can reconstruct the preprocessing contract.
from emulators_21cmspace.delta21.data import delta21_spec

# Delta21 config stores the default architecture and training settings.
from emulators_21cmspace.delta21.model import delta21_config

# DenseMLP is the neural network architecture used by the emulator.
from jax_emu.architectures import DenseMLP

# Shared trainer utilities update the model and evaluate the held-out test set.
from jax_emu.training import evaluate_mlp_regressor, train_mlp_regressor

# Checkpoint helpers save weights plus preprocessing metadata.
from jax_emu.utils import CheckpointMetadata, save

# Load the default Delta21 model, optimizer, and training settings.
config = delta21_config()

# Build the MLP with an input width matching the prepared feature matrix.
model = DenseMLP(
    in_features=prepared.train_features.shape[1],  # Number of input columns.
    hidden_features=config.mlp.hidden_dim,  # Width of each hidden layer.
    hidden_layers=config.mlp.total_hidden_layers,  # Number of hidden layers.
    activation=config.mlp.activation,  # Non-linear activation after hidden layers.
    rngs=nnx.Rngs(jax.random.PRNGKey(42)),  # Random stream for weight initialization.
)

# Train the model and record training/validation loss curves.
model, history = train_mlp_regressor(
    model,  # Live NNX model to update.
    prepared.train_features,  # Training input rows.
    prepared.train_targets,  # Training target values.
    prepared.validation_features,  # Validation input rows.
    prepared.validation_targets,  # Validation target values.
    epochs=config.training.epochs,  # Maximum number of full passes over training data.
    batch_size=config.training.batch_size,  # Number of rows per mini-batch.
    prefetch_batches=config.training.prefetch_batches,  # Batches queued on device.
    learning_rate=config.optimizer.learning_rate,  # AdamW update step size.
    weight_decay=config.optimizer.weight_decay,  # AdamW L2-style regularisation.
    seed=42,  # Seed used for epoch-level row shuffling.
    early_stopping_patience=config.training.early_stopping_patience,  # Waiting time.
    early_stopping_min_delta=config.training.early_stopping_min_delta,  # Improvement size.
)

# Evaluate the trained model on the test split after training has finished.
test_loss = evaluate_mlp_regressor(
    model,  # Trained model to evaluate.
    prepared.test_features,  # Held-out test inputs.
    prepared.test_targets,  # Held-out test targets.
    batch_size=config.training.batch_size,  # Number of rows per evaluation batch.
    prefetch_batches=config.training.prefetch_batches,  # Batches queued on device.
)

# Package the non-weight information required to reuse the model at inference.
metadata = CheckpointMetadata(
    model_name="delta21",  # Human-readable emulator name.
    package_version="0.1.0",  # Version label for the saved package format.
    emulator_spec=delta21_spec(),  # Axis, parameter, and target-transform contract.
    input_scaling=prepared.feature_scaling,  # Feature scaling used before the MLP.
    target_scaling=prepared.target_scaling,  # Target scaling inverted after the MLP.
    training_config={  # Training settings kept with the checkpoint.
        "mlp": asdict(config.mlp),  # Network architecture settings.
        "optimizer": asdict(config.optimizer),  # Optimizer settings.
        "training": asdict(config.training),  # Epoch, batch, and stopping settings.
        "feature_names": list(prepared.feature_names),  # Required input column order.
    },
)

# Ensure the output directory exists before writing the checkpoint.
Path("outputs").mkdir(exist_ok=True)

# Save the trained weights, architecture settings, losses, and preprocessing metadata.
package_path = save(
    Path("outputs/delta21_model.nenemu"),  # Output checkpoint path.
    model,  # Trained NNX model whose state will be saved.
    train_losses=history.train_losses,  # Training loss curve.
    val_losses=history.validation_losses,  # Validation loss curve.
    loss=config.training.loss_name,  # Loss name used during training.
    metadata=metadata,  # Preprocessing and training metadata.
    epochs=config.training.epochs,  # Maximum configured epochs.
    patience=config.training.early_stopping_patience,  # Early-stopping patience.
    learning_rate=config.optimizer.learning_rate,  # Optimizer learning rate.
    weight_decay=config.optimizer.weight_decay,  # Optimizer weight decay.
)

# Print the saved checkpoint path.
print(package_path)

# Print the final held-out test loss.
print(test_loss)
```

The checkpoint stores:

```text
DenseMLP architecture + trained weights + Delta21 preprocessing metadata
```

### Predict

```python
# JAX arrays keep inference inputs and outputs on the accelerator.
import jax.numpy as jnp

# Delta21 inference helpers load the checkpoint and build a compiled predictor.
from emulators_21cmspace.delta21.infer import (
    build_delta21_predictor,
    load_delta21_package,
)

# Load the trained model and its saved metadata.
package = load_delta21_package("outputs/delta21_model.nenemu")

# Build the JIT-compiled numerical inference function once.
predictor = build_delta21_predictor(package)

# Provide one raw 12-column physical parameter row in the original dataset order.
physical_parameters = jnp.asarray(
    [
        [1e-2, 1e-3, 16.5, 1e2, 1.3, 500.0, 30.0, 0.055, 1e2, 232.0, 0.0, 0.0],
    ],
    dtype=jnp.float32,  # Match the float32 dtype used during training.
)

# Choose the redshift coordinates where the power spectrum should be predicted.
z = jnp.linspace(6.0, 27.0, 50)

# Choose the physical k coordinates using logarithmic spacing in k.
k = jnp.geomspace(3e-2 / 0.6704, 0.99 / 0.6704, 50, dtype=jnp.float32)

# Predict Delta21 and return an array with shape (n_sims, n_z, n_k).
delta21 = predictor(physical_parameters, z, k)

# Inspect the prediction shape.
print(delta21.shape)
```

The prediction has shape:

```text
(n_sims, n_z, n_k)
```

The inference helper keeps the same contract as training: `metadata.emulator_spec`
defines the axis and target transforms, `metadata.input_scaling` defines the
feature scaling, and `metadata.target_scaling` defines the inverse target
scaling.
