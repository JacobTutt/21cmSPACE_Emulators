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
from emulators_21cmspace.t21.data import prepare_twentyonecmspace_t21_training_split

prepared = prepare_twentyonecmspace_t21_training_split(
    dataset_root,
    random_state=42,
    shuffle_seed=42,
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
    prepare_twentyonecmspace_delta21_training_split,
)

prepared = prepare_twentyonecmspace_delta21_training_split(
    dataset_root,
    random_state=42,
    shuffle_seed=42,
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

### Train And Save

For the standard 21cmSPACE power-spectrum workflow, the package provides a
complete Python helper around the same lower-level pieces:

```python
from emulators_21cmspace.delta21.train import train_delta21_from_dataset_root

summary = train_delta21_from_dataset_root(
    dataset_root,
    output_path="outputs/delta21_model.nenemu",
    epochs=10000,
    batch_size=10000,
    shuffle_seed=42,
)

print(summary["package_path"])
print(summary["test_loss"])
print(summary["feature_names"])
```

This helper prepares the arrays, builds `DenseMLP`, trains it with the shared
JAX trainer, evaluates the test split, and saves the checkpoint metadata.

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
