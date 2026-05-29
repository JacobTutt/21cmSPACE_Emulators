# Preprocessing and Normalisation

The preprocessing layer turns raw 21cmSPACE simulation products into the scalar
rows consumed by the JAX emulators. It owns the deterministic contract between
training and inference: which axes are used, which quantities are log
transformed, how features are scaled, how targets are transformed, and how full
spectra are flattened into one-value regression examples.

![Preprocessing workflow](assets/preprocessing-flow.svg)

## Why Determinism Matters

Emulator inference is only meaningful when a trained checkpoint sees inputs in
the same coordinate system used during training. A checkpoint does not just
store network weights; it also depends on preprocessing metadata:

- feature names and column order
- axis transforms and fixed interpolation grids
- parameter transforms and discarded columns
- per-feature scaling statistics from the training split
- target transform and global target standard deviation

If any of these change between training and inference, the network can receive
numerically valid arrays that represent the wrong physical point. The
preprocessing code therefore builds one explicit, repeatable mapping from raw
simulation tables to model-ready arrays.

## Raw Inputs

The 21cmSPACE loaders read MATLAB files from a dataset directory:

```text
21cmspace_z_mat.mat
21cmspace_k_mat.mat
21cmspace_nu_mat.mat
21cmspace_parameters_mat.mat
21cmspace_Deltak_mat.mat
21cmspace_T21_mat.mat
```

The raw parameter table has 12 columns:

```text
fstarII, fstarIII, Vc, fX, alpha, nu_0, zeta, tau, fradio, pop, feed, delay
```

The current emulator workflows use nine parameter columns and drop `zeta`,
`feed`, and `delay`.

## Axis Transforms

Axis transforms define the coordinate system seen by the neural network before
feature scaling is applied.

| Workflow | Axis | Transform | Training feature | Current grid |
| --- | --- | --- | --- | --- |
| T21 | `z` | identity | `z` | 200 points from 6.0 to 27.0 |
| Delta21 | `z` | identity | `z` | 50 points from 6.0 to 27.0 |
| Delta21 | `k` | `log10` | `log10k` | 50 points in transformed k limits |

For Delta21, `k` is first loaded as `k / h` using `h = 0.6704`. The current
h-scaled bounds are `3e-2 / h` to `0.99 / h`; those limits are transformed to
`log10k` before the fixed grid is built. Redshift remains in physical `z` for
both emulators.

## Parameter Transforms

The shared parameter preparation step applies `log10` to selected
astrophysical parameters, keeps discrete parameters in physical values, and
records the discrete sets as metadata.

| Raw parameter | Prepared feature | Transform | Used by |
| --- | --- | --- | --- |
| `fstarII` | `log10fstarII` | `log10` | T21, Delta21 |
| `fstarIII` | `log10fstarIII` | `log10` | T21, Delta21 |
| `Vc` | `log10Vc` | `log10` | T21, Delta21 |
| `fX` | `log10fX` | `log10` | T21, Delta21 |
| `alpha` | `alpha` | identity, discrete | T21, Delta21 |
| `nu_0` | `nu_0` | identity, discrete | T21, Delta21 |
| `tau` | `tau` | identity | T21, Delta21 |
| `fradio` | `log10fradio` | `log10` | T21, Delta21 |
| `pop` | `pop` | identity, discrete | T21, Delta21 |

The discrete parameters are scaled with min-max scaling where used by the
current workflows:

```text
alpha, nu_0, pop
```

## Feature Scaling

Feature scaling statistics are computed from flattened training rows only and
then reused for validation, test, and inference. The supported scaling methods
are:

| Method | Operation | Typical use |
| --- | --- | --- |
| `zscore` | `(x - training_mean) / training_std` | continuous axes and continuous parameters |
| `minmax_zero_to_one` | `(x - training_min) / (training_max - training_min)` | discrete parameters |
| `identity` | no scaling | explicit opt-out |

The implementation also supports the legacy label `normalize` as `zscore` and
`standardize` as min-max scaling to `[-1, 1]`, but the current T21 and Delta21
defaults use `zscore` and `minmax_zero_to_one`.

## Target Transforms

Targets are transformed before train/validation/test splitting and before
fixed-grid interpolation.

| Workflow | Raw target | Target transform | Offset | Target scaling |
| --- | --- | --- | --- | --- |
| T21 | `T21` | identity | none | divide by one global training-target std |
| Delta21 | `Delta21` | `log10(Delta21 + offset)` | `1e-8` | divide by one global training-target std |

The global target scaling follows the `globalemu` convention: after the target
transform and fixed-grid interpolation, one standard deviation is computed
across every target value in the training split. All train, validation, and test
targets are divided by that scalar. The inverse operation multiplies by the
same scalar before undoing the target transform.

This is not a per-redshift or per-k normalisation. There is one target standard
deviation for the whole prepared training target grid.

## Fixed-Grid Interpolation

The training data are resampled onto one deterministic grid instead of using
per-simulation or random interpolation locations.

1. Load raw axes, raw parameters, and raw targets.
2. Drop simulations with NaN target values.
3. Prepare parameter features by dropping, logging, and recording discrete
   metadata.
4. Apply the target transform.
5. Split simulations into train, validation, and test sets.
6. Transform axes into their model coordinate system.
7. Build the fixed grid from each `AxisSpec`.
8. Linearly interpolate every simulation target onto that grid.
9. Compute and apply global target standard-deviation scaling.
10. Flatten every grid cell into one scalar target row.
11. Compute feature scaling from training rows and apply it to all splits.
12. Shuffle rows within each split with a fixed seed.

Flattening converts each full simulation grid into many scalar regression
examples:

```text
[axis coordinates, prepared parameters] -> one target value
```

For T21, each simulation contributes `200` rows:

```text
[z, log10fstarII, log10fstarIII, log10Vc, log10fX, alpha, nu_0, tau, log10fradio, pop] -> T21
```

For Delta21, each simulation contributes `50 * 50 = 2500` rows:

```text
[z, log10k, log10fstarII, log10fstarIII, log10Vc, log10fX, alpha, nu_0, tau, log10fradio, pop] -> Delta21
```

## Current Defaults

### T21

| Setting | Value |
| --- | --- |
| Spec name | `t21` |
| Family | `global_signal` |
| Axes | `z` |
| Axis transform | physical `z` |
| Fixed grid | 200 `z` points from 6.0 to 27.0 |
| Prepared parameters | 9 columns |
| Log parameters | `fstarII`, `fstarIII`, `Vc`, `fX`, `fradio` |
| Discrete min-max parameters | `alpha`, `nu_0`, `pop` |
| Continuous z-score features | `z`, log parameters, `tau` |
| Target transform | identity |
| Target scaling | global training-target std |
| Split fractions | 0.6 train, 0.2 validation, 0.2 test |
| Default seeds | `random_state=42`, `shuffle_seed=42` |

### Delta21

| Setting | Value |
| --- | --- |
| Spec name | `delta21` |
| Family | `power_spectrum` |
| Axes | `z`, `k` |
| Axis transforms | physical `z`, `log10k` |
| Fixed grid | 50 `z` points by 50 `log10k` points |
| k unit conversion | loaded `k / 0.6704` |
| k limits before log | `3e-2 / 0.6704` to `0.99 / 0.6704` |
| Prepared parameters | 9 columns |
| Log parameters | `fstarII`, `fstarIII`, `Vc`, `fX`, `fradio` |
| Discrete min-max parameters | `alpha`, `nu_0`, `pop` |
| Continuous z-score features | `z`, `log10k`, log parameters, `tau` |
| Target transform | `log10(Delta21 + 1e-8)` |
| Target scaling | global training-target std |
| Split fractions | 0.6 train, 0.2 validation, 0.2 test |
| Default seeds | `random_state=42`, `shuffle_seed=42` |

## Code Examples

Prepare the T21 training split:

```python
from emulators_21cmspace.t21.data import prepare_twentyonecmspace_t21_training_split

prepared = prepare_twentyonecmspace_t21_training_split(
    "/path/to/21cmSPACE",
    random_state=42,
    shuffle_seed=42,
)

print(prepared.feature_names)
print(prepared.train_features.shape)
print(prepared.train_targets.shape)
```

Prepare the Delta21 training split:

```python
from emulators_21cmspace.delta21.data import prepare_twentyonecmspace_delta21_training_split

prepared = prepare_twentyonecmspace_delta21_training_split(
    "/path/to/21cmSPACE",
    random_state=42,
    shuffle_seed=42,
)

print(prepared.feature_names)
print(prepared.target_scaling.to_dict())
```

Prepare only the raw parameter table if you already loaded the MATLAB arrays:

```python
from emulators_21cmspace.t21.data import prepare_twentyonecmspace_t21_parameters

parameter_features = prepare_twentyonecmspace_t21_parameters(raw_parameters)

print(parameter_features.feature_names)
print(parameter_features.discrete_values)
```

Use the shared fixed-grid preparation function directly for a custom workflow:

```python
from jax_emu.data_preprocessing.preparation import prepare_fixed_grid_training_split

prepared = prepare_fixed_grid_training_split(
    axes=(z_axis,),
    axis_specs=spec.axes,
    parameters=prepared_parameters,
    target=target_grid,
    feature_scale_methods={"z": "zscore"},
    data_log=False,
    offset=None,
)
```

The returned object contains the arrays needed by the trainer and the metadata
needed to reproduce preprocessing at inference time:

```python
prepared.train_features
prepared.train_targets
prepared.validation_features
prepared.validation_targets
prepared.test_features
prepared.test_targets
prepared.feature_scaling
prepared.target_scaling
```
