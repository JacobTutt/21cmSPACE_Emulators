# Training

Training starts after preprocessing has produced plain feature and target
arrays. The trainer does not load simulation files, apply science-specific
parameter rules, or create the model. It receives an existing MLP and fits it.

The main modules are:

- [`training/trainer.py`](../jaxemu_21cmSPACE/training/trainer.py)
- [`utils/checkpointing.py`](../jaxemu_21cmSPACE/utils/checkpointing.py)
- [`emulators21/t21/train.py`](../jaxemu_21cmSPACE/emulators21/t21/train.py)
- [`emulators21/delta21/train.py`](../jaxemu_21cmSPACE/emulators21/delta21/train.py)
- [`emulators21/t21/infer.py`](../jaxemu_21cmSPACE/emulators21/t21/infer.py)
- [`emulators21/delta21/infer.py`](../jaxemu_21cmSPACE/emulators21/delta21/infer.py)

## Trainer Input

The shared training function is:

```python
train_mlp_regressor(
    model,
    train_features,
    train_targets,
    validation_features,
    validation_targets,
    ...
)
```

The model and arrays are already prepared:

- `model` is an initialised or loaded `DenseMLP`
- full feature and target arrays stay on the host
- features are in canonical emulator feature order
- feature scaling has already been applied
- targets have already been transformed into training space
- targets have already been divided by the global training-label std

## Training Loop

During training, the code:

1. creates a fresh Optax AdamW optimizer for the supplied model
2. shuffles training rows each epoch
3. forms mini-batches on the host
4. keeps a small queue of mini-batches prefetched on the JAX device
5. runs `model(batch_features)`
6. computes mean squared error
7. updates model parameters
8. evaluates validation loss
9. records train and validation losses once per epoch
10. restores the best model state if early stopping is enabled

The default `prefetch_batches=2` means the trainer queues the next mini-batch
with `jax.device_put` before the current one is consumed. This follows the JAX
training-cookbook pattern where host-side batch preparation can overlap with
jitted device computation.

The trainer returns:

```python
model, history
```

where `history` stores:

- `train_losses`
- `validation_losses`
- `best_epoch`
- `best_validation_loss`

## Workflow Training Entrypoints

The concrete workflow functions are:

```python
train_t21_from_dataset_root(dataset_root)
train_delta21_from_dataset_root(dataset_root)
```

Each function:

1. runs the emulator-specific preprocessing path
2. loads the workflow model/training defaults
3. initialises the workflow `DenseMLP`
4. trains the supplied MLP
5. evaluates the test split
6. writes a `.nenemu` checkpoint directory
7. writes a small JSON training summary beside the checkpoint

## Checkpoint Contents

A `.nenemu` path is a checkpoint directory containing:

```text
0/
  config/
  state/
```

`config/` stores Orbax-managed JSON metadata:

- package version
- model hyperparameters
- train and validation losses
- loss name
- checkpoint metadata

`state/` stores Orbax-managed arrays:

- the trained NNX model state written by Orbax

The checkpoint metadata stores:

- emulator spec
- feature scaling metadata
- target scaling metadata
- training configuration

## Loading A Saved MLP

Loading is handled by:

```python
package = load("model.nenemu")
model = package["model"]
```

Internally, loading:

1. reads the saved architecture settings
2. builds an abstract `DenseMLP` of the same shape
3. restores the latest Orbax checkpoint step by default
4. merges the restored state into a live model object

After loading, use the model directly:

```python
predictions = model(features)
```

## Inference Reconstruction

The inference helpers undo the training-space transforms in the correct order.

For both workflows:

```text
model output
-> undo global target scaling
-> undo physical target transform
-> reconstruct output grid
```

For `Delta21`, this means:

```text
network output
-> log10(Delta21 + 1)
-> Delta21
-> Delta21(z, k)
```

For `T21`, this means:

```text
network output
-> T21 in linear target space
-> T21
-> T21(z)
```

The feature scaling and target scaling metadata saved during training are what
make this inference path reproducible.
