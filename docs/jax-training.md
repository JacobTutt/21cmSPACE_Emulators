# JAX Training

**Navigation:** [README](../README.md) · [Architecture](architecture.md) · [Preprocessing](preprocessing.md) · [JAX Training](jax-training.md) · [Checkpointing](checkpoint.md) · [Examples](examples.md)

Once a `DenseMLP` has been initialized, the next step is to train it against
prepared simulation data. Training updates the network parameters so that the
model predictions match the target emulator values as closely as possible.

The trainer uses a loss function to measure prediction error, automatic
differentiation to compute gradients of that loss, and an Optax optimizer to
update the model parameters.

## Training Parameters

`train_mlp_regressor` is the shared training entry point. It receives an
existing model, prepared train/validation arrays, and the main optimization
settings:

| Parameter | Definition |
| :--- | :--- |
| `model` | The initialized `DenseMLP` network to be trained. |
| `train_features` | 2D array of input features for the training set (coordinates + parameters). |
| `train_targets` | 1D array of ground-truth scalar values for the training set. |
| `validation_features` | 2D array of input features for the validation set. |
| `validation_targets` | 1D array of ground-truth scalar values for the validation set. |
| `epochs` | The number of complete passes through the training dataset. |
| `batch_size` | Number of rows used in each optimizer update. |
| `learning_rate` | Optimizer step size. |
| `weight_decay` | AdamW regularization term used to penalize large weights. |
| `learning_rate_schedule` | Rule used to change the learning rate during training. |
| `learning_rate_final_fraction` | Final learning-rate fraction for decay schedules. |
| `learning_rate_warmup_epochs` | Number of warmup epochs for `warmup_cosine`. |
| `seed` | Random seed for deterministic shuffling of the training data. |

## The Training Step

For each mini-batch, the trainer runs a compiled JAX/NNX training step:

1. **Forward pass**: Pass the mini-batch features through the network.
2. **Loss calculation**: Compute mean squared error between predictions and
   targets.
3. **Gradient calculation**: Use JAX automatic differentiation to compute
   gradients of the loss with respect to the trainable parameters.
4. **Optimizer update**: Apply the AdamW update to the live NNX model state.

## Code Example

```python
from jax_emu.training.trainer import train_mlp_regressor

model, history = train_mlp_regressor(
    model,
    train_features=train_features,
    train_targets=train_targets,
    validation_features=val_features,
    validation_targets=val_targets,
    epochs=1000,
    batch_size=1024,
    learning_rate=1e-3,
    weight_decay=1e-4,
    learning_rate_schedule="warmup_cosine",
    learning_rate_final_fraction=0.05,
    learning_rate_warmup_epochs=5,
    seed=42,
)
```

## Learning-Rate Schedules

A learning-rate schedule changes the optimizer step size during training. The
default is `constant`, which keeps the previous behaviour.

| Schedule | Behaviour |
| :--- | :--- |
| `constant` | Use the same learning rate for every optimizer step. |
| `cosine` | Smoothly decay from the initial learning rate to a final fraction. |
| `warmup_cosine` | Ramp up from zero, then apply cosine decay. |
| `exponential_decay` | Decay multiplicatively toward the final fraction. |

![Learning-rate scheduler curves](assets/learning-rate-schedules.svg)

Schedules are evaluated per mini-batch update, not only once per epoch:

```python
from jax_emu.training import build_learning_rate_schedule

schedule = build_learning_rate_schedule(
    learning_rate=1e-3,
    schedule_name="cosine",
    steps_per_epoch=100,
    epochs=1000,
    final_fraction=0.05,
)
```

The high-level training commands expose the same scheduler settings:

```bash
21cmspace-delta21-train \
  --dataset-root /path/to/21cmspace/data \
  --output outputs/delta21_model.nenemu \
  --learning-rate-schedule warmup_cosine \
  --learning-rate-final-fraction 0.05 \
  --learning-rate-warmup-epochs 5
```

## Evaluation Metrics

At the end of every epoch, after all training mini-batches have been used to
update the network, the trainer records two losses:

| Metric | Meaning |
| :--- | :--- |
| **Training loss** | Error on the data used to update the model parameters. |
| **Validation loss** | Error on held-out data that was not used for parameter updates. |

The training loss shows whether the network is fitting the training set. The
validation loss shows whether that fit generalises to unseen simulations.

## Loss Curve Analysis

The trainer returns a `TrainingHistory` object containing the training and
validation loss curves. After evaluating the held-out test set, these can be
plotted directly:

```python
from jax_emu.analysis import plot_training_history

plot_training_history(
    history,
    test_loss=test_loss,
    model_name="delta21",
    output_path="outputs/delta21_loss_curves.png",
)
```

The same curves are also saved inside the `.nenemu` package. The high-level
training workflows write an adjacent `.summary.json` file containing the final
test loss, so a saved run can be inspected later:

```python
from jax_emu.analysis import plot_package_losses

plot_package_losses(
    "outputs/delta21_model.nenemu",
    output_path="outputs/delta21_loss_curves.png",
)
```

The plot shows training loss, validation loss, the best validation epoch when
available, and the held-out test loss in the top-right corner.

## Efficient JAX Training

The training code is designed for prepared arrays that may be too large to keep
fully resident in GPU memory. The usual workflow is therefore host-to-device
streaming:

1. **Store arrays on the host**: Prepared feature and target arrays can remain
   in system memory.
2. **Train on mini-batches**: Only the current fixed-shape mini-batches are
   transferred to the JAX device.
3. **Prefetch future batches**: While the device trains on one batch, the next
   batches can be prepared and queued with `jax.device_put`.

### Training Pipeline Flow

The aim is to reduce idle accelerator time. Without prefetching, each iteration
waits for host preparation and device transfer before training can begin. With
prefetching, preparation and transfer for later batches overlap with the current
compiled training step.

![Mini-batch prefetching comparison](assets/training-prefetch.svg)

---

**Navigation:** [README](../README.md) · [Architecture](architecture.md) · [Preprocessing](preprocessing.md) · [JAX Training](jax-training.md) · [Checkpointing](checkpoint.md) · [Examples](examples.md)
