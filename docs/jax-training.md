# JAX Training

This page describes the low-level JAX training loop used by the emulator
workflows. It covers `jax_emu.training.trainer.train_mlp_regressor`, the batch
prefetcher, evaluation, and the checkpoint format used after training.

![Timeline showing host batch preparation, device_put prefetching, and device compute overlap.](assets/training-prefetch.svg)

## Training Entry Point

`train_mlp_regressor` trains an existing `DenseMLP` on already-prepared feature
and target arrays. It does not load simulation data, apply emulator-specific
preprocessing, choose parameter rules, or create the model. Workflow modules do
that work first, then pass the resulting arrays and initialized model into the
trainer.

The function receives:

- a live `DenseMLP`
- training feature and target arrays
- validation feature and target arrays
- optimizer settings: `learning_rate` and `weight_decay`
- loop settings: `batch_size`, `epochs`, `seed`, `prefetch_batches`, logging,
  and optional early stopping settings

It returns the trained model and a `TrainingHistory` containing epoch-level
training losses, validation losses, the best validation epoch, and the best
validation loss.

## Loss And Evaluation

Training uses mean squared error for scalar regression. For each mini-batch, the
model predicts `model(batch_features).squeeze(-1)`, subtracts the prepared target
values, squares the residuals, and averages over real examples in the batch.

The final batch of an epoch may be smaller than `batch_size`. The prefetcher pads
that batch to a fixed shape and creates a float mask with `1.0` for real examples
and `0.0` for padded rows. The loss multiplies squared errors by this mask, so
padded rows keep the compiled batch shape stable without changing the objective.

Validation is run after every training epoch. The validation step uses the same
prediction and squared-error calculation, but it does not call the optimizer. The
held-out test path uses `evaluate_mlp_regressor`, which runs the same
validation-style MSE calculation on an arbitrary prepared split.

Epoch losses are accumulated as sums of squared error and divided by the number
of real examples. This avoids giving the padded final batch extra weight.

## Optimizer

The trainer uses Optax through `flax.nnx.Optimizer`:

```python
optimizer = nnx.Optimizer(
    model,
    optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
    wrt=nnx.Param,
)
```

Optax defines the update rule. In the current trainer that rule is AdamW, with
the learning rate and weight decay supplied either directly or from the workflow
configuration. Flax NNX owns the live model state and applies the optimizer
updates to `nnx.Param` leaves. The compiled training step computes the MSE,
computes gradients with `nnx.value_and_grad`, and updates the model in place
through the NNX optimizer wrapper.

## Host And Device Model

Prepared datasets are normally NumPy arrays on the host. The trainer preserves
that layout: it slices host arrays into mini-batches and transfers only a small
number of batches to the JAX device at a time. This keeps large prepared arrays
out of device memory while still allowing device compute to run on fixed-shape
mini-batches.

If the caller passes JAX arrays, the trainer preserves them instead of copying
them back to NumPy. In that case batch indexing happens from the JAX arrays
provided by the caller. The usual workflow path is host-resident prepared arrays
with mini-batch transfer.

## Mini-Batch Prefetching

`_iter_device_batches` implements the prefetching used by training, validation,
and test evaluation. For each epoch it:

1. builds row indices for the split
2. optionally shuffles those indices for training
3. slices the next host mini-batch
4. pads the final batch and builds a mask when needed
5. calls `jax.device_put` for features, targets, and mask
6. stores the resulting device arrays in a small FIFO queue

`prefetch_batches` controls the maximum number of queued mini-batches. The
default is `2`, meaning the iterator tries to keep the current batch plus one
future batch available on device. When the training loop consumes one queued
batch, the iterator immediately slices and queues the next one if any remain.

JAX dispatch and `device_put` are asynchronous in the common accelerator path,
so this pattern can overlap host-side batch preparation and host-to-device
transfer with work already submitted to the device. The code does not create a
separate Python worker thread or guarantee full overlap in every backend and
data-size regime. It simply keeps a small queue of device batches ahead of the
consumer so the device is less likely to wait for host slicing and transfer.

## Early Stopping

Early stopping is controlled by `early_stopping_patience` and
`early_stopping_min_delta`. After each validation pass, the trainer compares the
new validation MSE with the best validation MSE seen so far. An improvement must
beat the previous best by at least `early_stopping_min_delta`.

When validation improves, the trainer deep-copies `nnx.state(model)` in memory.
If validation does not improve for `early_stopping_patience` consecutive epochs,
training stops. At the end of training, the best saved NNX state is restored into
the live model with `nnx.update`.

The trainer transfers scalar losses from device to CPU at epoch boundaries with
`jax.device_get`. That transfer is intentional and acceptable here because early
stopping, logging, and history storage are epoch-level decisions. It avoids
synchronizing every mini-batch just to print or branch on a Python float.

## Checkpointing

Training workflows save trained emulators with `jax_emu.utils.checkpointing.save`.
The saved package uses the `.nenemu` suffix and is an Orbax checkpoint manager
directory containing two checkpoint items:

- `state`: the Flax NNX model state, saved with Orbax `StandardSave`
- `config`: JSON metadata, saved with Orbax `JsonSave`

Before saving, checkpointing splits the live NNX model into graph structure and
state with `nnx.split(model)`. Only the state is stored as the weight checkpoint.
The JSON config stores architecture hyperparameters, training history, loss
name, package version, optimizer settings, and optional `CheckpointMetadata`.

`CheckpointMetadata` is where the reusable inference contract lives. It can
store the emulator spec, feature-scaling metadata, target-scaling metadata, and
the workflow training config. Loading reverses this process: the JSON config is
read first, a shape-compatible `DenseMLP` is rebuilt, Orbax restores the saved
NNX state into that structure, and `CheckpointMetadata` is reconstructed from
JSON if present.

## Starting From An Existing Model And Config

The trainer is intentionally small: create or load a model, prepare arrays, then
pass the workflow config values into `train_mlp_regressor`.

```python
import jax
from flax import nnx

from jax_emu.architectures.mlp import DenseMLP
from jax_emu.training.trainer import train_mlp_regressor

# prepared is produced by the relevant preprocessing workflow.
# config is a workflow config containing mlp, optimizer, and training sections.
model = DenseMLP(
    in_features=prepared.train_features.shape[1],
    hidden_features=config.mlp.hidden_dim,
    hidden_layers=config.mlp.total_hidden_layers,
    out_features=config.mlp.output_dim,
    activation=config.mlp.activation,
    rngs=nnx.Rngs(jax.random.PRNGKey(42)),
)

model, history = train_mlp_regressor(
    model,
    prepared.train_features,
    prepared.train_targets,
    prepared.validation_features,
    prepared.validation_targets,
    learning_rate=config.optimizer.learning_rate,
    weight_decay=config.optimizer.weight_decay,
    batch_size=config.training.batch_size,
    epochs=config.training.epochs,
    prefetch_batches=config.training.prefetch_batches,
    seed=42,
    early_stopping_patience=(
        config.training.early_stopping_patience
        if config.training.early_stop
        else None
    ),
    early_stopping_min_delta=config.training.early_stopping_min_delta,
)
```

Saving is normally handled by the workflow module after test evaluation, because
that layer has the emulator spec and scaling metadata needed to build a complete
`CheckpointMetadata` object.
