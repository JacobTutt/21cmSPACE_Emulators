# Architecture

The emulator architecture is a shared dense MLP with workflow-specific
configuration. The same network class is used for both the global-signal
`T21` workflow and the power-spectrum `Delta21` workflow:

- [`jax_emu/architectures/mlp.py`](../jax_emu/architectures/mlp.py)
- [`emulators_21cmspace/t21/model.py`](../emulators_21cmspace/t21/model.py)
- [`emulators_21cmspace/delta21/model.py`](../emulators_21cmspace/delta21/model.py)

The main design choice is that the network is trained as a tiled scalar
regressor. It predicts one physical target value for one coordinate location
and one astrophysical parameter vector, rather than predicting a complete
signal vector in one forward pass.

![Tiled scalar-output emulator architecture](assets/network-tiling.svg)

## Tiled Scalar Regression

The tiling utilities are defined in
[`jax_emu/data_preprocessing/tiling.py`](../jax_emu/data_preprocessing/tiling.py).
They flatten each simulation output onto rows of the form:

```text
[axis coordinates, astrophysical parameters] -> scalar target
```

For `T21`, the axis coordinates are just redshift:

```text
[z, theta_1, ..., theta_9] -> T21(z; theta)
```

For `Delta21`, the axis coordinates are redshift and wavenumber:

```text
[z, log10(k), theta_1, ..., theta_9] -> Delta21(z, k; theta)
```

The feature order is canonical: axis values first, then astrophysical
parameters. After inference, the flat predictions are reshaped back onto the
original spectral grid with `reconstruct_spectra(...)`.

This is the GlobalEmu-style idea used here: the coordinates are part of the
input, so a single scalar-output MLP learns the continuous map over both
physical coordinate space and parameter space.

| Approach | Input row | Network output | Reconstruction | Main tradeoff |
| --- | --- | --- | --- | --- |
| Vector-output emulator | One parameter vector | Full signal vector or grid | Usually none | Output layer is tied to one fixed grid shape. |
| Tiled scalar-output emulator | Axis coordinate(s) plus one parameter vector | One scalar target value | Required reshape after prediction | More rows, but one architecture works across 1D and 2D spectral grids. |

## DenseMLP

`DenseMLP` is a Flax NNX module. It builds:

```text
input -> hidden layer(s) with activation -> linear readout
```

Each hidden layer is an `nnx.Linear` followed by the configured activation.
The readout layer is linear and does not apply an activation.

The constructor arguments are:

```python
model = DenseMLP(
    in_features=10,
    hidden_features=32,
    hidden_layers=4,
    out_features=1,
    activation="relu",
    rngs=nnx.Rngs(jax.random.PRNGKey(1)),
)
```

## MLPConfig Fields

Workflow configs use `MLPConfig` from
[`jax_emu/utils/config.py`](../jax_emu/utils/config.py). The config names are
slightly higher level than the `DenseMLP` constructor.

| Field | Meaning | Maps to `DenseMLP` |
| --- | --- | --- |
| `input_dim` | Number of input features after tiling and preprocessing. This is axis coordinates plus astrophysical parameters. | `in_features` when a fixed width is known, though training entrypoints usually use `prepared.train_features.shape[1]`. |
| `hidden_dim` | Width of every hidden layer. | `hidden_features` |
| `n_hidden_blocks` | Number of hidden linear blocks after the first hidden layer. | `hidden_layers = 1 + n_hidden_blocks` via `total_hidden_layers`. |
| `activation` | Non-linearity applied after each hidden linear layer. | `activation` |
| `output_dim` | Number of output values per input row. Current workflows use scalar regression, so this is `1`. | `out_features`; defaults to `1` in `DenseMLP`. |

The mapping is:

```python
config = t21_config()

model = DenseMLP(
    in_features=config.mlp.input_dim,
    hidden_features=config.mlp.hidden_dim,
    hidden_layers=config.mlp.total_hidden_layers,
    out_features=config.mlp.output_dim,
    activation=config.mlp.activation,
    rngs=nnx.Rngs(jax.random.PRNGKey(1)),
)
```

In the training entrypoints, the actual feature width is taken from the
prepared arrays:

```python
model = DenseMLP(
    in_features=prepared.train_features.shape[1],
    hidden_features=config.mlp.hidden_dim,
    hidden_layers=config.mlp.total_hidden_layers,
    activation=config.mlp.activation,
    rngs=nnx.Rngs(jax.random.PRNGKey(1)),
)
```

## Activation Functions

The supported activation names are defined in
[`jax_emu/architectures/mlp.py`](../jax_emu/architectures/mlp.py). The current
mapping is:

| Name | Implementation | Equation |
| --- | --- | --- |
| `relu` | `jax.nn.relu` | `relu(x) = max(0, x)` |
| `tanh` | `jax.numpy.tanh` | `tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))` |
| `gelu` | `jax.nn.gelu` | Conceptually `gelu(x) = x Phi(x)`, where `Phi` is the standard normal CDF. JAX uses its approximate form by default. |

`sigmoid` is not currently a supported `ActivationName` in `DenseMLP`. Adding it
would require extending the `ActivationName` literal and `_activation_fn(...)`
mapping.

## Current Workflow Defaults

| Workflow | Input features | Hidden width | Hidden layers | Activation | Output features |
| --- | ---: | ---: | ---: | --- | ---: |
| `T21` | `10` (`z` plus 9 parameters) | `32` | `4` | `relu` | `1` |
| `Delta21` | `11` (`z`, `log10(k)` plus 9 parameters) | `100` | `4` | `tanh` | `1` |

The defaults are stored in the workflow model modules:

```python
# emulators_21cmspace/t21/model.py
MLPConfig(
    input_dim=10,
    hidden_dim=32,
    n_hidden_blocks=3,
    activation="relu",
)
```

This gives:

```text
10 -> 32 -> 32 -> 32 -> 32 -> 1
```

```python
# emulators_21cmspace/delta21/model.py
MLPConfig(
    input_dim=11,
    hidden_dim=100,
    n_hidden_blocks=3,
    activation="tanh",
)
```

This gives:

```text
11 -> 100 -> 100 -> 100 -> 100 -> 1
```

The shared MLP class stays generic. The workflow modules own the default
scientific capacity and feature width, while preprocessing determines the
actual tiled feature matrix used for training and inference.
