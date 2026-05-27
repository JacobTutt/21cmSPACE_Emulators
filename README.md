# Nenufar Emulators

`nenufar-emulators` is a JAX-first codebase for training and packaging 21-cm
emulators. The current focus is the HERA IDR4-style emulator workflow used for:

- `T21`: global 21-cm signal emulation
- `Delta21`: 21-cm power-spectrum emulation

The motivation for this repository is to keep the scientific emulator contract
explicit while moving the implementation toward a cleaner JAX interface. The
code should make it clear how simulation files become training arrays, how the
network is defined, how training is run, and how a saved emulator package can
be loaded later for inference.

## Repository Layout

```text
Nenufar_Emulators/
  pyproject.toml
  README.md
  docs/
    preprocessing.md
    network.md
    training.md
  src_jax/
    nenufar_emulators/
      data_preprocessing/
        hera_idr4.py
        parameters.py
        preparation.py
      architectures/
        mlp.py
      training/
        trainer.py
      utils/
        checkpointing.py
        config.py
        metrics.py
        scaling.py
        specs.py
        tiling.py
        transforms.py
      emulators/
        delta21/
          data.py
          model.py
          train.py
          infer.py
        t21/
          data.py
          model.py
          train.py
          infer.py
  tests/
```

The rough split is:

- `data_preprocessing/`: load simulation files, prepare parameters, resample
  targets, scale arrays, and produce train/validation/test arrays.
- `architectures/`: define reusable neural-network classes.
- `training/`: run optimization, validation, batching, and early stopping.
- `utils/`: shared metadata, transforms, scaling, tiling, metrics, configs, and
  checkpoint save/load code.
- `emulators/`: concrete `t21` and `delta21` workflows that combine the shared
  pieces into train and inference entry points.
- `tests/`: contract tests and end-to-end smoke tests.

## Workflow Walkthrough

The main path through the code is:

```text
simulation files
-> data loading
-> parameter preparation
-> target transform and fixed-grid resampling
-> feature and target scaling
-> MLP training
-> .nenemu checkpoint package
-> inference in physical units
```

The detailed walkthroughs are split by stage:

- [Preprocessing](docs/preprocessing.md): how raw arrays become emulator
  training features and targets.
- [Network](docs/network.md): how the MLP is defined and called.
- [Training](docs/training.md): how batches, optimization, checkpoint metadata,
  and inference reconstruction fit together.

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Run the tests:

```bash
python -m pytest -q
```

## Command-Line Entry Points

The package exposes four development commands:

```bash
nenufar-t21-train
nenufar-t21-infer
nenufar-delta21-train
nenufar-delta21-infer
```

Each training command can print its default spec/config, run a synthetic smoke
test, or train from a HERA IDR4 dataset root. Each inference command can inspect
a saved `.nenemu` package or generate predictions from a package plus input
parameter and axis files.

## Saved Emulator Packages

Training writes a `.nenemu` package. It stores:

- model architecture settings
- trained model state
- training and validation losses
- emulator spec
- feature scaling metadata
- target scaling metadata
- training configuration

This metadata is what lets inference rebuild the model inputs, undo target
standardization, undo physical target transforms, and return predictions in
physical units.
