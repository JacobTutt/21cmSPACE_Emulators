# 21cmSPACE Emulators

`21cmspace-emulators` is a JAX-first codebase for training and packaging 21-cm
emulators. The current focus is the 21cmSPACE-style emulator workflow used for:

- `T21`: global 21-cm signal emulation
- `Delta21`: 21-cm power-spectrum emulation

The motivation for this repository is to keep the scientific emulator contract
explicit while moving the implementation toward a cleaner JAX interface. The
code should make it clear how simulation files become training arrays, how the
network is defined, how training is run, and how a saved emulator package can
be loaded later for inference.

## Repository Layout

```text
21cmspace-emulators/
  pyproject.toml                         (package metadata and CLI entry points)
  README.md                              (project overview)
  docs/                                  (stage-by-stage workflow notes)
    preprocessing.md                     (data loading and array preparation)
    network.md                           (MLP architecture)
    training.md                          (optimization, checkpoints, inference)
  src_jax/                               (Python source root)
    twentyonecmspace_emulators/
      data_preprocessing/                (load simulations and build arrays)
        twentyonecmspace.py
        parameters.py
        preparation.py
      architectures/                     (shared network definitions)
        mlp.py
      training/                          (batching, optimization, validation)
        trainer.py
      utils/                             (specs, transforms, scaling, checkpointing)
        checkpointing.py
        config.py
        metrics.py
        scaling.py
        specs.py
        tiling.py
        transforms.py
      emulators/                         (concrete T21 and Delta21 workflows)
        delta21/                         (power-spectrum emulator)
          data.py
          model.py
          train.py
          infer.py
        t21/                             (global-signal emulator)
          data.py
          model.py
          train.py
          infer.py
  tests/                                 (contract and workflow smoke tests)
```

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
21cmspace-t21-train
21cmspace-t21-infer
21cmspace-delta21-train
21cmspace-delta21-infer
```

Each training command can print its default spec/config, run a synthetic smoke
test, or train from a 21cmSPACE dataset root. Each inference command can inspect
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
