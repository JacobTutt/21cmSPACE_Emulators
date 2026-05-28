# Preprocessing

Preprocessing is the part of the pipeline that turns simulation files into
plain arrays that can be passed to the trainer.

The main modules are:

- [`emulators_21cmspace/twentyonecmspace.py`](../emulators_21cmspace/twentyonecmspace.py)
- [`data_preprocessing/specs.py`](../jax_emu/data_preprocessing/specs.py)
- [`data_preprocessing/transforms.py`](../jax_emu/data_preprocessing/transforms.py)
- [`data_preprocessing/scaling.py`](../jax_emu/data_preprocessing/scaling.py)
- [`data_preprocessing/tiling.py`](../jax_emu/data_preprocessing/tiling.py)
- [`data_preprocessing/parameters.py`](../jax_emu/data_preprocessing/parameters.py)
- [`data_preprocessing/preparation.py`](../jax_emu/data_preprocessing/preparation.py)
- [`emulators_21cmspace/t21/data.py`](../emulators_21cmspace/t21/data.py)
- [`emulators_21cmspace/delta21/data.py`](../emulators_21cmspace/delta21/data.py)

## Input Files

The 21cmSPACE loader expects a dataset directory containing files such as:

```text
21cmspace_z_mat.mat
21cmspace_k_mat.mat
21cmspace_nu_mat.mat
21cmspace_parameters_mat.mat
21cmspace_Deltak_mat.mat
21cmspace_T21_mat.mat
```

The loader returns:

- physical axes such as `z`, `k`, or `nu`
- the raw simulation parameter table
- the raw target array
- the target name
- simulation indices dropped because of invalid values

## Parameter Preparation

The raw 21cmSPACE parameter table has 12 columns:

```text
fstarII, fstarIII, Vc, fX, alpha, nu_0, zeta, tau, fradio, pop, feed, delay
```

The current emulator workflows use nine of them. They drop:

```text
zeta, feed, delay
```

and log-transform:

```text
fstarII, fstarIII, Vc, fX, fradio
```

The prepared parameter columns are:

```text
log10fstarII
log10fstarIII
log10Vc
log10fX
alpha
nu_0
tau
log10fradio
pop
```

This logic lives in `prepare_feature_matrix(...)` and is wrapped by the
emulator-specific helpers:

```python
prepare_twentyonecmspace_t21_parameters(raw_parameters)
prepare_twentyonecmspace_delta21_parameters(raw_parameters)
```

## Emulator Specs

Each emulator defines an `EmulatorSpec`. The spec records:

- emulator name and family
- axis names and transforms
- parameter names and transforms
- target transform and offset

For `T21`, each model input row is:

```text
z + 9 prepared parameters
```

For `Delta21`, each model input row is:

```text
z + log10k + 9 prepared parameters
```

The canonical feature order comes from:

```python
spec.input_feature_names()
```

## Target Transform

Targets are transformed before training.

`T21` uses:

```text
T21 -> T21
```

`Delta21` uses:

```text
Delta21 -> log10(Delta21 + 1)
```

The transform choice is stored in the emulator spec as `target_transform` and
`target_offset`, so inference can undo it later.

## Fixed-Grid Preparation

The main shared entry point is:

```python
prepare_fixed_grid_training_split(...)
```

It does the following:

1. apply the configured target transform
2. split simulations into train, validation, and test subsets
3. transform axes such as `k -> log10(k)`
4. build one fixed shared axis grid from the emulator spec
5. resample every simulation onto that grid
6. divide targets by one global training-label standard deviation
7. flatten grids into scalar training rows
8. compute feature scaling from training features
9. scale train, validation, and test features
10. shuffle rows within each split

The output is a `PreparedSplit` containing:

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

For the concrete `t21` and `delta21` workflows, `prepared.target_scaling`
stores one scalar standard deviation measured from the training targets. This
matches the old `globalemu` style and avoids per-redshift or per-k target
statistics.

At this point the science-specific preprocessing is done. The trainer receives
normal arrays and does not need to know where the simulations came from.
