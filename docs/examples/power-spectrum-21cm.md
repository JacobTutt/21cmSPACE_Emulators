# Power-Spectrum 21-cm Example

This workflow trains and runs inference with the `Delta21` 21-cm power-spectrum
emulator.

## Smoke Test

Run a synthetic end-to-end training check without any 21cmSPACE files:

```bash
21cmspace-delta21-train --synthetic-smoke --epochs 2 --batch-size 64
```

The command prints final train and validation losses. It checks the pipeline
and grid tiling, but it does not produce a science-quality checkpoint.

## Real Dataset Training

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

If `--output` is omitted, the checkpoint is written to `delta21_model.nenemu`.
Training also writes a JSON summary beside it, for example
`outputs/delta21_model.summary.json`.

## Checkpoint Summary And Inference

Describe the saved checkpoint:

```bash
21cmspace-delta21-infer \
  --package outputs/delta21_model.nenemu \
  --describe
```

Create small inference input files. The parameter table below uses the raw
12-column 21cmSPACE order:

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

## Expected Dataset Files

The Delta21 loader reads the shared axes and the power-spectrum target from:

| File | MATLAB key |
| --- | --- |
| `21cmspace_z_mat.mat` | `z21cm` |
| `21cmspace_k_mat.mat` | `ks` |
| `21cmspace_nu_mat.mat` | `nu_keV` |
| `21cmspace_parameters_mat.mat` | `parameters` |
| `21cmspace_Deltak_mat.mat` | `combined_Deltaks` |

The loader divides stored `ks` values by `0.6704` before training or inference
examples use the emulator's physical `k` range. Rows containing NaNs in the
target array are dropped together with their matching parameter rows.

## Current Defaults

Print the live defaults with:

```bash
21cmspace-delta21-train --print-config
```

The current code defaults are:

| Setting | Value |
| --- | --- |
| Emulator name | `delta21` |
| Family | `power_spectrum` |
| Input features | `z`, `log10k`, plus 9 prepared parameters |
| Redshift axis | identity transform, limits `(6.0, 27.0)`, `nsample=50` |
| Wavenumber axis | log10 transform, limits `(3e-2 / 0.6704, 0.99 / 0.6704)`, `nsample=50` |
| Target transform | `log10(target + 1e-8)` |
| Train/validation/test split | `0.6 / 0.2 / 0.2` |
| Feature scaling | z-score for `z`, `log10k`, logged continuous parameters, and `tau`; min-max `[0, 1]` for `alpha`, `nu_0`, `pop` |
| Target scaling | one global log-space training-target standard deviation |
| MLP | input dim 11, 4 hidden layers, width 100, tanh, output dim 1 |
| Optimizer | AdamW, learning rate `1e-3`, weight decay `1e-4` |
| Training | `epochs=10000`, `batch_size=10000`, `prefetch_batches=2` |
| Early stopping | enabled, patience `50`, minimum delta `0.0` |
| Default output | `delta21_model.nenemu` |
