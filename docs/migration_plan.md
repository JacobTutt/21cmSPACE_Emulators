# Migration Plan

## Objective

Build a new JAX-first emulator repository that:

- preserves the scientific contract of the old emulator code
- provides cleaner package boundaries
- supports both training and downstream inference
- stores sufficient metadata for reproducible checkpoints
- is testable without access to the full production datasets

## Scientific Scope

The first migration target is the shared emulator infrastructure used by:

- power-spectrum emulators
- global-signal emulators

The initial repository foundation will not depend on local science datasets.
Instead, it will define:

- package structure
- emulator specifications
- parameter transforms
- scaling metadata
- tiling rules
- checkpoint schema
- train/infer interfaces
- synthetic-data tests

## Old-Code Sources Of Truth

The old repository is used as reference, not as code to copy blindly.

Primary references:

- `Simon_OldCode/src/CosmicDawnSynergies/train_tools.py`
- `Simon_OldCode/src/CosmicDawnSynergies/models.py`
- `Simon_OldCode/src/CosmicDawnSynergies/loader_21cmSim.py`
- `Simon_OldCode/src/CosmicDawnSynergies/inference.py`
- `Simon_OldCode/scripts/train_torch_emu.py`
- `Simon_OldCode/scripts/train_globalemu.py`

Key responsibilities to preserve:

- parameter selection and log transforms
- discrete-parameter bookkeeping
- spectral tiling into scalar regression targets
- inference-time reconstruction from stored metadata
- emulator-specific axis definitions

## Astroemu Context

`astroemu` is a useful JAX design reference, especially for:

- dataset tiling
- composable normalization pipelines
- simple Optax training loops
- metadata-rich serialization

It should inform the architecture, but this repository should own its own code
and metadata model because the old NenuFAR workflows have stricter inference and
science-specific requirements.

## Target Architecture

```text
src_jax/
  nenufar_emulators/
    core/
      specs.py
      transforms.py
      scaling.py
      tiling.py
      batching.py
      metrics.py
      training.py
      checkpointing.py
    power_spectrum/
      data.py
      model.py
      train.py
      infer.py
    global_signal/
      data.py
      model.py
      train.py
      infer.py
```

## Delivery Stages

### Stage 1: Repository foundation

Deliverables:

- project metadata
- package scaffold
- migration documentation

Verification:

- package tree exists and imports resolve
- documentation clearly states scope and staged approach

### Stage 2: Shared emulator contracts

Deliverables:

- axis and parameter spec dataclasses
- parameter transform definitions
- scaling metadata schema
- checkpoint metadata schema

Verification:

- unit tests for config validation
- unit tests for transform round-trips
- unit tests for scaling metadata serialization

### Stage 3: Shared JAX utilities

Deliverables:

- tiling helpers for `[axes, params] -> scalar target`
- synthetic batch generation utilities
- MLP initialization and forward pass
- Optax training skeleton

Verification:

- unit tests for tiled shapes
- unit tests for output reconstruction shape
- smoke test for one short synthetic training run

### Stage 4: Power-spectrum emulator path

Deliverables:

- power-spectrum emulator spec
- power-spectrum train/infer entrypoints
- checkpoint save/load path

Verification:

- synthetic train/infer smoke test
- shape tests for `z-k` axes
- checkpoint round-trip test

### Stage 5: Global-signal emulator path

Deliverables:

- global-signal emulator spec
- global-signal preprocessing hooks
- global-signal train/infer entrypoints

Verification:

- synthetic train/infer smoke test
- shape tests for one-dimensional axes
- transform parity checks against expected preprocessing rules

### Stage 6: Data-backed parity

Deliverables:

- real dataset loaders
- reference configs for power and global models
- parity notebooks or scripts against old-code outputs

Verification:

- parameter transform parity against old code
- prediction agreement on fixed validation subsets
- checkpoint metadata sufficient for downstream likelihood use

## Verification Strategy Before Real Data Exists

Because the production data are not yet in this repository, verification will
focus on contract correctness rather than emulator accuracy.

This means testing:

- parameter ordering
- transform reversibility
- scaling behavior
- tiling semantics
- checkpoint completeness
- train/infer API stability

Once data arrive, parity tests replace placeholder smoke tests.

## Commit Rhythm

Commits should remain small and milestone-based.

Planned commit boundaries:

1. roadmap and project scaffold
2. shared core contracts
3. shared JAX utilities and tests
4. power-spectrum path
5. global-signal path

## Immediate Next Step

Implement Stage 1 and Stage 2 so the repository has a stable internal shape
before any dataset-specific code is added.
