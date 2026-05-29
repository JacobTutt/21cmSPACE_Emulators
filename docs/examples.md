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
from emulators_21cmspace.t21.data import (
    prepare_twentyonecmspace_t21_parameters,
    t21_spec,
)
from emulators_21cmspace.twentyonecmspace import load_twentyonecmspace_t21
from jax_emu.data_preprocessing import prepare_fixed_grid_training_split

product = load_twentyonecmspace_t21(dataset_root)
prepared_parameters = prepare_twentyonecmspace_t21_parameters(product.parameters)
spec = t21_spec()

feature_scale_methods = {
    "z": "zscore",
    "log10fstarII": "zscore",
    "log10fstarIII": "zscore",
    "log10Vc": "zscore",
    "log10fX": "zscore",
    "alpha": "minmax_zero_to_one",
    "nu_0": "minmax_zero_to_one",
    "tau": "zscore",
    "log10fradio": "zscore",
    "pop": "minmax_zero_to_one",
}

prepared = prepare_fixed_grid_training_split(
    axes=(product.axes.z,),
    axis_specs=spec.axes,
    parameters=prepared_parameters,
    target=product.target,
    feature_scale_methods=feature_scale_methods,
    data_log=False,
    offset=None,
    train_size=0.6,
    validation_size=0.2,
    test_size=0.2,
    random_state=42,
    shuffle_seed=42,
    standardize_target=True,
)

print(prepared.feature_names)
print(prepared.train_features.shape)
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
from dataclasses import asdict
from pathlib import Path

import jax
from flax import nnx

from emulators_21cmspace.t21.data import t21_spec
from emulators_21cmspace.t21.model import t21_config
from jax_emu.architectures import DenseMLP
from jax_emu.training import evaluate_mlp_regressor, train_mlp_regressor
from jax_emu.utils import CheckpointMetadata, save

config = t21_config()

model = DenseMLP(
    in_features=prepared.train_features.shape[1],
    hidden_features=config.mlp.hidden_dim,
    hidden_layers=config.mlp.total_hidden_layers,
    activation=config.mlp.activation,
    rngs=nnx.Rngs(jax.random.PRNGKey(42)),
)

model, history = train_mlp_regressor(
    model,
    prepared.train_features,
    prepared.train_targets,
    prepared.validation_features,
    prepared.validation_targets,
    epochs=config.training.epochs,
    batch_size=config.training.batch_size,
    prefetch_batches=config.training.prefetch_batches,
    learning_rate=config.optimizer.learning_rate,
    weight_decay=config.optimizer.weight_decay,
    seed=42,
    early_stopping_patience=config.training.early_stopping_patience,
    early_stopping_min_delta=config.training.early_stopping_min_delta,
)

test_loss = evaluate_mlp_regressor(
    model,
    prepared.test_features,
    prepared.test_targets,
    batch_size=config.training.batch_size,
    prefetch_batches=config.training.prefetch_batches,
)

metadata = CheckpointMetadata(
    model_name="t21",
    package_version="0.1.0",
    emulator_spec=t21_spec(),
    input_scaling=prepared.feature_scaling,
    target_scaling=prepared.target_scaling,
    training_config={
        "mlp": asdict(config.mlp),
        "optimizer": asdict(config.optimizer),
        "training": asdict(config.training),
        "feature_names": list(prepared.feature_names),
    },
)

Path("outputs").mkdir(exist_ok=True)
package_path = save(
    Path("outputs/t21_model.nenemu"),
    model,
    train_losses=history.train_losses,
    val_losses=history.validation_losses,
    loss=config.training.loss_name,
    metadata=metadata,
    epochs=config.training.epochs,
    patience=config.training.early_stopping_patience,
    learning_rate=config.optimizer.learning_rate,
    weight_decay=config.optimizer.weight_decay,
)

print(package_path)
print(test_loss)
```

The checkpoint stores:

```text
DenseMLP architecture + trained weights + T21 preprocessing metadata
```

### Predict

```python
import jax.numpy as jnp

from emulators_21cmspace.t21.infer import load_t21_package, predict_t21

package = load_t21_package("outputs/t21_model.nenemu")

physical_parameters = jnp.asarray(
    [[1e-2, 1e-3, 16.5, 1e2, 1.3, 500.0, 30.0, 0.055, 1e2, 232.0, 0.0, 0.0]],
    dtype=jnp.float32,
)
z = jnp.linspace(6.0, 27.0, 200)

t21 = predict_t21(package, physical_parameters, z)
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
from emulators_21cmspace.delta21.data import (
    delta21_spec,
    prepare_twentyonecmspace_delta21_parameters,
)
from emulators_21cmspace.twentyonecmspace import load_twentyonecmspace_delta21
from jax_emu.data_preprocessing import prepare_fixed_grid_training_split

product = load_twentyonecmspace_delta21(dataset_root)
prepared_parameters = prepare_twentyonecmspace_delta21_parameters(product.parameters)
spec = delta21_spec()

feature_scale_methods = {
    "z": "zscore",
    "log10k": "zscore",
    "log10fstarII": "zscore",
    "log10fstarIII": "zscore",
    "log10Vc": "zscore",
    "log10fX": "zscore",
    "alpha": "minmax_zero_to_one",
    "nu_0": "minmax_zero_to_one",
    "tau": "zscore",
    "log10fradio": "zscore",
    "pop": "minmax_zero_to_one",
}

prepared = prepare_fixed_grid_training_split(
    axes=(product.axes.z, product.axes.k),
    axis_specs=spec.axes,
    parameters=prepared_parameters,
    target=product.target,
    feature_scale_methods=feature_scale_methods,
    data_log=True,
    offset=1e-8,
    train_size=0.6,
    validation_size=0.2,
    test_size=0.2,
    random_state=42,
    shuffle_seed=42,
    standardize_target=True,
)

print(prepared.feature_names)
print(prepared.train_features.shape)
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
from dataclasses import asdict
from pathlib import Path

import jax
from flax import nnx

from emulators_21cmspace.delta21.data import delta21_spec
from emulators_21cmspace.delta21.model import delta21_config
from jax_emu.architectures import DenseMLP
from jax_emu.training import evaluate_mlp_regressor, train_mlp_regressor
from jax_emu.utils import CheckpointMetadata, save

config = delta21_config()

model = DenseMLP(
    in_features=prepared.train_features.shape[1],
    hidden_features=config.mlp.hidden_dim,
    hidden_layers=config.mlp.total_hidden_layers,
    activation=config.mlp.activation,
    rngs=nnx.Rngs(jax.random.PRNGKey(42)),
)

model, history = train_mlp_regressor(
    model,
    prepared.train_features,
    prepared.train_targets,
    prepared.validation_features,
    prepared.validation_targets,
    epochs=config.training.epochs,
    batch_size=config.training.batch_size,
    prefetch_batches=config.training.prefetch_batches,
    learning_rate=config.optimizer.learning_rate,
    weight_decay=config.optimizer.weight_decay,
    seed=42,
    early_stopping_patience=config.training.early_stopping_patience,
    early_stopping_min_delta=config.training.early_stopping_min_delta,
)

test_loss = evaluate_mlp_regressor(
    model,
    prepared.test_features,
    prepared.test_targets,
    batch_size=config.training.batch_size,
    prefetch_batches=config.training.prefetch_batches,
)

metadata = CheckpointMetadata(
    model_name="delta21",
    package_version="0.1.0",
    emulator_spec=delta21_spec(),
    input_scaling=prepared.feature_scaling,
    target_scaling=prepared.target_scaling,
    training_config={
        "mlp": asdict(config.mlp),
        "optimizer": asdict(config.optimizer),
        "training": asdict(config.training),
        "feature_names": list(prepared.feature_names),
    },
)

Path("outputs").mkdir(exist_ok=True)
package_path = save(
    Path("outputs/delta21_model.nenemu"),
    model,
    train_losses=history.train_losses,
    val_losses=history.validation_losses,
    loss=config.training.loss_name,
    metadata=metadata,
    epochs=config.training.epochs,
    patience=config.training.early_stopping_patience,
    learning_rate=config.optimizer.learning_rate,
    weight_decay=config.optimizer.weight_decay,
)

print(package_path)
print(test_loss)
```

The checkpoint stores:

```text
DenseMLP architecture + trained weights + Delta21 preprocessing metadata
```

### Predict

```python
import jax.numpy as jnp
import numpy as np

from emulators_21cmspace.delta21.infer import load_delta21_package, predict_delta21

package = load_delta21_package("outputs/delta21_model.nenemu")

physical_parameters = jnp.asarray(
    [[1e-2, 1e-3, 16.5, 1e2, 1.3, 500.0, 30.0, 0.055, 1e2, 232.0, 0.0, 0.0]],
    dtype=jnp.float32,
)
z = jnp.linspace(6.0, 27.0, 50)
k = jnp.asarray(np.geomspace(3e-2 / 0.6704, 0.99 / 0.6704, 50), dtype=jnp.float32)

delta21 = predict_delta21(package, physical_parameters, z, k)
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
