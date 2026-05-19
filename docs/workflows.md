# Workflows

## Delta21

`Delta21` is a tiled scalar emulator for the 21-cm power spectrum.

### Input

Each training row contains:

- redshift `z`
- wave number `log10(k)`
- nine astrophysical parameters after workflow preprocessing

The resulting network input width is `11`.

### Target

Each row predicts one scalar power-spectrum value in transformed space:

- `log10(Delta21 + 1)`

### Preparation Logic

The current workflow uses the established scalar power-spectrum setup:

1. load HERA IDR4 `Delta21`, `z`, `k`, and parameter arrays
2. drop unused parameters and apply the workflow log transforms
3. split by simulation
4. draw random interpolation points over the allowed `z` and `k` ranges
5. flatten those interpolated values into scalar training rows
6. build a fixed-grid validation set
7. scale each feature with the configured workflow rules

### Model

The current `Delta21` network is:

- `11 -> 100 -> 100 -> 100 -> 100 -> 1`
- `ReLU`

## T21

`T21` is a tiled scalar emulator for the global 21-cm signal.

### Input

Each training row contains:

- redshift `z`
- nine astrophysical parameters after workflow preprocessing

The resulting network input width is `10`.

### Target

Each row predicts one scalar global-signal value in physical space:

- `T21(z)`

### Preparation Logic

The current workflow is intentionally different from `Delta21`.

It uses a fixed-grid global-signal approach:

1. load HERA IDR4 `T21`, `z`, and parameter arrays
2. drop unused parameters and apply the workflow log transforms
3. split by simulation
4. resample every signal onto one shared redshift grid
5. flatten that shared grid into scalar training rows
6. scale each feature with the configured workflow rules

### Model

The current `T21` network is:

- `10 -> 20 -> 20 -> 20 -> 20 -> 1`
- `tanh`

### Training Defaults

The current `T21` workflow uses a narrower network and early stopping because
the global-signal task is one-dimensional and does not need the same model
capacity as `Delta21`.
