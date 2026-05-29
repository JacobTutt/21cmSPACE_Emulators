# Architecture

The emulator architecture is a simple JAX implementation of the
scalar-regression idea used by
[GlobalEmu](https://github.com/htjb/globalemu)
([arXiv:2104.04336](https://arxiv.org/abs/2104.04336)) and
[AstroEmu](https://astroemu.readthedocs.io/en/latest/tutorial/). Instead of
training a network to emit a whole spectrum or grid in one pass, independent
coordinates are included in the input row and the network predicts one scalar
observable value.

![Tiled scalar-output emulator architecture](assets/network-tiling.svg)

The figure should be read from top to bottom. The first row shows the
traditional vector-output emulator, where physical variables map directly to a
full spectrum. The second row shows the key scalar-output idea: fold the
independent coordinate, such as redshift `z` or wavenumber `k`, into the input
and predict one value. The third row shows how vectorized calls over all
requested coordinates reconstruct the spectrum or grid.

## Scalar Regression Contract

The tiling utilities flatten simulation outputs onto rows of the form:

```text
[axis coordinates, astrophysical parameters] -> scalar target
```

For a one-dimensional observable:

```text
[z, theta_1, ..., theta_n] -> y(z; theta)
```

For a two-dimensional observable:

```text
[z, log10(k), theta_1, ..., theta_n] -> y(z, k; theta)
```

The feature order is canonical: axis values first, then astrophysical
parameters. After inference, the flat predictions are reshaped back onto the
original spectral grid with `reconstruct_spectra(...)`.

The important distinction is that the independent coordinates are part of the
input. A single scalar-output MLP learns the continuous map over coordinate
space and parameter space, while vectorized calls over the requested coordinate
grid reconstruct the observable.

| Approach | Input row | Network output | Reconstruction | Main tradeoff |
| --- | --- | --- | --- | --- |
| Vector-output emulator | One parameter vector | Full signal vector or grid | Usually none | Output layer is tied to one fixed grid shape. |
| Tiled scalar-output emulator | Axis coordinate(s) plus one parameter vector | One scalar target value | Vectorized call plus reshape | More rows, but one architecture works across 1D and 2D spectral grids. |

## DenseMLP and Configuration

`DenseMLP` is a Flax NNX module with hidden linear layers, a configured
activation, and a linear scalar readout. The readout does not apply an output
activation, so target-space constraints should be handled by preprocessing and
inverse transforms.

Workflow configs use `MLPConfig` from
[`jax_emu/utils/config.py`](../jax_emu/utils/config.py). The config describes
the feature width after tiling, the hidden-layer capacity, and the scalar output
width.

| Field | Meaning | Maps to `DenseMLP` |
| --- | --- | --- |
| `input_dim` | Number of input features after tiling and preprocessing. This is axis coordinates plus astrophysical parameters. | `in_features` when a fixed width is known, though training entrypoints usually use `prepared.train_features.shape[1]`. |
| `hidden_dim` | Width of every hidden layer. | `hidden_features` |
| `n_hidden_blocks` | Number of hidden linear blocks after the first hidden layer. | `hidden_layers = 1 + n_hidden_blocks` via `total_hidden_layers`. |
| `activation` | Non-linearity applied after each hidden linear layer. | `activation` |
| `output_dim` | Number of output values per input row. Current workflows use scalar regression, so this is `1`. | `out_features`; defaults to `1` in `DenseMLP`. |

In training entry points, the actual feature width is taken from the prepared
arrays, so the model follows the tiling contract rather than assuming a fixed
observable shape.

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

## Package Contract

Saved emulator packages keep the trained parameters together with preprocessing
metadata, target transforms, axis specifications, and reconstruction metadata.
That keeps inference tied to the same scaling and tiling assumptions used
during training.

The shared MLP class stays generic. Workflow modules own scientific defaults
and observable-specific metadata, while preprocessing determines the actual
tiled feature matrix used for training and inference.
