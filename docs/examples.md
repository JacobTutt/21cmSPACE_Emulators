# Examples

This page shows the installed command-line entrypoints and matching Python APIs
for the current 21cmSPACE emulator workflows:

- global 21-cm brightness temperature, `T21(z)`
- 21-cm power spectrum, `Delta21(z, k)`

## Dataset Root

Both workflows expect a 21cmSPACE dataset root containing MATLAB `.mat` files.
Use a local path placeholder such as:

```bash
DATASET_ROOT=/path/to/21cmspace-data
```

The loader reads these files from the root:

| File | MATLAB key | Used by |
| --- | --- | --- |
| `21cmspace_z_mat.mat` | `z21cm` | T21 and Delta21 |
| `21cmspace_k_mat.mat` | `ks` | T21 loader and Delta21 |
| `21cmspace_nu_mat.mat` | `nu_keV` | T21 loader and Delta21 loader |
| `21cmspace_parameters_mat.mat` | `parameters` | T21 and Delta21 |
| `21cmspace_T21_mat.mat` | `combined_T21s` | T21 |
| `21cmspace_Deltak_mat.mat` | `combined_Deltaks` | Delta21 |

The raw parameter table has 12 columns:

```text
fstarII, fstarIII, Vc, fX, alpha, nu_0, zeta, tau, fradio, pop, feed, delay
```

Inference accepts either that raw 12-column table or the prepared 9-column
feature table used by the model.

## Global 21-cm Signal: `T21`

The `T21` workflow trains and runs inference with the global brightness
temperature emulator.

### Smoke Test

Run a synthetic end-to-end check without any 21cmSPACE files:

```bash
21cmspace-t21-train --synthetic-smoke --epochs 2 --batch-size 64
```

This checks the pipeline. It does not produce a science-quality model.

### Training

Inspect the prepared array shapes first:

```bash
21cmspace-t21-train \
  --dataset-root /path/to/21cmspace-data \
  --prepare-only
```

Train a checkpoint from the dataset root:

```bash
21cmspace-t21-train \
  --dataset-root /path/to/21cmspace-data \
  --output outputs/t21_model.nenemu \
  --epochs 10000 \
  --batch-size 1000 \
  --shuffle-seed 42
```

If `--output` is omitted, the checkpoint is written to `t21_model.nenemu`.
Training also writes a JSON summary beside it.

### Inference

Describe the saved checkpoint:

```bash
21cmspace-t21-infer \
  --package outputs/t21_model.nenemu \
  --describe
```

Create small inference input files:

```bash
python - <<'PY'
import numpy as np

parameters = np.array([
    [1e-2, 1e-3, 16.5, 1e2, 1.3, 500.0, 30.0, 0.055, 1e2, 232.0, 0.0, 0.0],
])
z = np.linspace(6.0, 27.0, 200)

np.savetxt("example_t21_parameters.txt", parameters)
np.savetxt("example_t21_z.txt", z)
PY
```

Run prediction from the command line:

```bash
21cmspace-t21-infer \
  --package outputs/t21_model.nenemu \
  --parameters-file example_t21_parameters.txt \
  --z-file example_t21_z.txt \
  --output outputs/t21_prediction.npz
```

The output archive contains `t21`, `parameters`, and `z`. The prediction shape
is `(n_sims, n_z)`.

The same flow can be called from Python:

```python
import jax.numpy as jnp
import numpy as np

from emulators_21cmspace.t21.infer import describe_t21_package, predict_t21

package = "outputs/t21_model.nenemu"
print(describe_t21_package(package))

parameters = jnp.asarray(np.loadtxt("example_t21_parameters.txt"), dtype=jnp.float32)
z = jnp.linspace(6.0, 27.0, 200)
t21 = predict_t21(package, parameters, z)
```

### Defaults

Print the live defaults with:

```bash
21cmspace-t21-train --print-config
```

| Setting | Value |
| --- | --- |
| Input features | `z` plus 9 prepared parameters |
| Redshift axis | identity transform, limits `(6.0, 27.0)`, `nsample=200` |
| Target transform | identity |
| Target scaling | one global training-target standard deviation |
| MLP | input dim 10, 4 hidden layers, width 32, ReLU, output dim 1 |
| Training | `epochs=10000`, `batch_size=1000`, `prefetch_batches=2` |
| Early stopping | enabled, patience `50` |

## 21-cm Power Spectrum: `Delta21`

The `Delta21` workflow trains and runs inference with the 21-cm power-spectrum
emulator.

### Smoke Test

Run a synthetic end-to-end check without any 21cmSPACE files:

```bash
21cmspace-delta21-train --synthetic-smoke --epochs 2 --batch-size 64
```

This checks the pipeline and grid tiling. It does not produce a science-quality
model.

### Training

Inspect the prepared array shapes first:

```bash
21cmspace-delta21-train \
  --dataset-root /path/to/21cmspace-data \
  --prepare-only
```

Train a checkpoint from the dataset root:

```bash
21cmspace-delta21-train \
  --dataset-root /path/to/21cmspace-data \
  --output outputs/delta21_model.nenemu \
  --epochs 10000 \
  --batch-size 10000 \
  --shuffle-seed 42
```

If `--output` is omitted, the checkpoint is written to
`delta21_model.nenemu`. Training also writes a JSON summary beside it.

### Inference

Describe the saved checkpoint:

```bash
21cmspace-delta21-infer \
  --package outputs/delta21_model.nenemu \
  --describe
```

Create small inference input files:

```bash
python - <<'PY'
import numpy as np

little_h = 0.6704
parameters = np.array([
    [1e-2, 1e-3, 16.5, 1e2, 1.3, 500.0, 30.0, 0.055, 1e2, 232.0, 0.0, 0.0],
])
z = np.linspace(6.0, 27.0, 50)
k = np.geomspace(3e-2 / little_h, 0.99 / little_h, 50)

np.savetxt("example_delta21_parameters.txt", parameters)
np.savetxt("example_delta21_z.txt", z)
np.savetxt("example_delta21_k.txt", k)
PY
```

Run prediction from the command line:

```bash
21cmspace-delta21-infer \
  --package outputs/delta21_model.nenemu \
  --parameters-file example_delta21_parameters.txt \
  --z-file example_delta21_z.txt \
  --k-file example_delta21_k.txt \
  --output outputs/delta21_prediction.npz
```

The output archive contains `delta21`, `parameters`, `z`, and `k`. The
prediction shape is `(n_sims, n_z, n_k)`.

The same flow can be called from Python:

```python
import jax.numpy as jnp
import numpy as np

from emulators_21cmspace.delta21.infer import (
    describe_delta21_package,
    predict_delta21,
)

package = "outputs/delta21_model.nenemu"
print(describe_delta21_package(package))

parameters = jnp.asarray(np.loadtxt("example_delta21_parameters.txt"), dtype=jnp.float32)
z = jnp.linspace(6.0, 27.0, 50)
k = jnp.asarray(np.geomspace(3e-2 / 0.6704, 0.99 / 0.6704, 50), dtype=jnp.float32)
delta21 = predict_delta21(package, parameters, z, k)
```

### Defaults

Print the live defaults with:

```bash
21cmspace-delta21-train --print-config
```

| Setting | Value |
| --- | --- |
| Input features | `z`, `log10k`, plus 9 prepared parameters |
| Redshift axis | identity transform, limits `(6.0, 27.0)`, `nsample=50` |
| Wavenumber axis | log10 transform, limits `(3e-2 / 0.6704, 0.99 / 0.6704)`, `nsample=50` |
| Target transform | `log10(target + 1e-8)` |
| Target scaling | one global log-space training-target standard deviation |
| MLP | input dim 11, 4 hidden layers, width 100, tanh, output dim 1 |
| Training | `epochs=10000`, `batch_size=10000`, `prefetch_batches=2` |
| Early stopping | enabled, patience `50` |
