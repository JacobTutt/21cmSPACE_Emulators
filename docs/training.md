# Training

Training starts after preprocessing has produced plain feature and target
arrays. The trainer does not load simulation files or apply science-specific
parameter rules. It receives arrays and fits an MLP.

The main modules are:

- [`training/trainer.py`](../src_jax/twentyonecmspace_emulators/training/trainer.py)
- [`utils/checkpointing.py`](../src_jax/twentyonecmspace_emulators/utils/checkpointing.py)
- [`utils/scaling.py`](../src_jax/twentyonecmspace_emulators/utils/scaling.py)
- [`emulators/t21/train.py`](../src_jax/twentyonecmspace_emulators/emulators/t21/train.py)
- [`emulators/delta21/train.py`](../src_jax/twentyonecmspace_emulators/emulators/delta21/train.py)
- [`emulators/t21/infer.py`](../src_jax/twentyonecmspace_emulators/emulators/t21/infer.py)
- [`emulators/delta21/infer.py`](../src_jax/twentyonecmspace_emulators/emulators/delta21/infer.py)

## Trainer Input

The shared training function is:

```python
train_mlp_regressor(
    train_features,
    train_targets,
    validation_features,
    validation_targets,
    ...
)
```

The arrays are already prepared:

- features are in canonical emulator feature order
- feature scaling has already been applied
- targets have already been transformed into training space
- target standardization has already been applied if enabled

## Training Loop

During training, the code:

1. initializes a `DenseMLP`
2. shuffles training rows each epoch
3. forms mini-batches
4. runs `model(batch_features)`
5. computes mean squared error
6. updates model parameters with Optax AdamW
7. evaluates validation loss
8. records train and validation losses
9. restores the best model state if early stopping is enabled

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
3. trains the shared MLP
4. evaluates the test split
5. writes a `.nenemu` package
6. writes a small JSON training summary beside the package

## Checkpoint Contents

A `.nenemu` file is a zip package containing:

```text
config.json
params.npz
```

`config.json` stores:

- package version
- model hyperparameters
- train and validation losses
- loss name
- checkpoint metadata

`params.npz` stores:

- the flattened trained NNX model state

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
2. constructs a `DenseMLP` of the same shape
3. reads the saved NNX state arrays
4. updates the new model object with the saved state

After loading, use the model directly:

```python
predictions = model(features)
```

## Inference Reconstruction

The inference helpers undo the training-space transforms in the correct order.

For both workflows:

```text
model output
-> undo target standardization
-> undo physical target transform
-> reconstruct output grid
```

For `Delta21`, this means:

```text
network output
-> unstandardized log10(Delta21 + 1)
-> Delta21
-> Delta21(z, k)
```

For `T21`, this means:

```text
network output
-> unstandardized T21
-> T21
-> T21(z)
```

The feature scaling and target scaling metadata saved during training are what
make this inference path reproducible.
