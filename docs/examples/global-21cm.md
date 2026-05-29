# Global 21-cm Signal Example

This workflow trains and runs inference with the `T21` global brightness
temperature emulator.

## Smoke Test

Run a synthetic end-to-end training check without any 21cmSPACE files:

```bash
21cmspace-t21-train --synthetic-smoke --epochs 2 --batch-size 64
```

The command prints final train and validation losses. It is a pipeline check,
not a science-quality model.

## Real Dataset Training

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
Training also writes a JSON summary beside it, for example
`outputs/t21_model.summary.json`.

## Checkpoint Summary And Inference

Describe the saved checkpoint:

```bash
21cmspace-t21-infer \
  --package outputs/t21_model.nenemu \
  --describe
```

Create small inference input files. The parameter table below uses the raw
12-column 21cmSPACE order:

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

## Expected Dataset Files

The T21 loader reads the shared axes and the global-signal target from:

| File | MATLAB key |
| --- | --- |
| `21cmspace_z_mat.mat` | `z21cm` |
| `21cmspace_k_mat.mat` | `ks` |
| `21cmspace_nu_mat.mat` | `nu_keV` |
| `21cmspace_parameters_mat.mat` | `parameters` |
| `21cmspace_T21_mat.mat` | `combined_T21s` |

Rows containing NaNs in the target array are dropped together with their
matching parameter rows.

## Current Defaults

Print the live defaults with:

```bash
21cmspace-t21-train --print-config
```

The current code defaults are:

| Setting | Value |
| --- | --- |
| Emulator name | `t21` |
| Family | `global_signal` |
| Input features | `z` plus 9 prepared parameters |
| Redshift axis | identity transform, limits `(6.0, 27.0)`, `nsample=200` |
| Target transform | identity |
| Train/validation/test split | `0.6 / 0.2 / 0.2` |
| Feature scaling | z-score for `z`, logged continuous parameters, and `tau`; min-max `[0, 1]` for `alpha`, `nu_0`, `pop` |
| Target scaling | one global training-target standard deviation |
| MLP | input dim 10, 4 hidden layers, width 32, ReLU, output dim 1 |
| Optimizer | AdamW, learning rate `1e-3`, weight decay `1e-4` |
| Training | `epochs=10000`, `batch_size=1000`, `prefetch_batches=2` |
| Early stopping | enabled, patience `50`, minimum delta `0.0` |
| Default output | `t21_model.nenemu` |
