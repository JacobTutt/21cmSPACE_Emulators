# Network

The shared network is defined in:

- [`architectures/mlp.py`](../jax_emu/architectures/mlp.py)

The concrete workflow defaults are defined in:

- [`emulators_21cmspace/t21/model.py`](../emulators_21cmspace/t21/model.py)
- [`emulators_21cmspace/delta21/model.py`](../emulators_21cmspace/delta21/model.py)

## DenseMLP

`DenseMLP` is a Flax NNX module. It owns the network structure and the forward
pass.

The constructor takes:

- `in_features`: width of the emulator input row
- `hidden_features`: hidden-layer width
- `hidden_layers`: number of hidden layers
- `out_features`: output width, currently `1`
- `activation`: `relu`, `tanh`, or `gelu`
- `init_scale`: standard deviation for normal weight initialization
- `rngs`: the NNX random-number container used to initialize parameters

The model is called directly:

```python
predictions = model(features)
```

There is no separate `forward_mlp(...)` helper. The class `__call__` method is
the forward pass.

## Model Shape

The emulator is trained as a scalar regressor. Each row contains one axis
location plus one parameter vector, and the network predicts one scalar target
for that row.

For `T21`:

```text
input width = 1 redshift feature + 9 parameter features = 10
output width = 1
```

For `Delta21`:

```text
input width = 2 axis features + 9 parameter features = 11
output width = 1
```

The full signal is reconstructed later by reshaping the flat row predictions
back onto the requested axis grid.

## Workflow Defaults

`T21` currently uses a smaller tanh network:

```text
10 -> 20 -> 20 -> 20 -> 20 -> 1
activation = tanh
```

`Delta21` currently uses a wider ReLU network:

```text
11 -> 100 -> 100 -> 100 -> 100 -> 1
activation = relu
```

The defaults live in the workflow model files so the shared architecture stays
generic and the science workflow owns its chosen capacity.

## Saved Models

Checkpoint loading rebuilds the same `DenseMLP` architecture from saved
hyperparameters, then restores the saved NNX state with Orbax.

Conceptually:

```python
package = load("model.nenemu")
model = package["model"]
predictions = model(features)
```

The model shape is reconstructed from metadata, then Orbax restores the trained
NNX state and the state is merged into a live model object.
