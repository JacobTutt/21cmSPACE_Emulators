# Architecture

## Purpose

This repository is not meant to be a line-by-line transcription of the older
`CosmicDawnSynergies` project. Its purpose is to provide a cleaner and more
readable JAX codebase that preserves the important scientific behavior of the
legacy emulator workflows while organizing the code around the workflows that
people actually run.

The practical focus is currently narrow:

- `Delta21` on HERA IDR4
- `T21` on HERA IDR4

Other historical emulator branches from the old repository are useful context,
but they are not the primary product target of this codebase.

## Design Principles

- Keep the public package layout aligned with the real user workflow.
- Keep legacy scientific behavior explicit, not hidden in scattered helper code.
- Keep generic utilities separate from HERA-specific and emulator-specific logic.
- Prefer a small number of well-named modules over many thin wrappers.
- Use comments to explain intent and legacy rationale, not to narrate syntax.

## Target Structure

The package should read from top to bottom like the actual workflow:

1. data loading
2. legacy convention mapping
3. emulator-specific preparation
4. model training
5. archive save/load
6. inference

That leads to the following target organization:

- `data/`
  - load raw HERA IDR4 arrays
  - define legacy parameter and target conventions
  - prepare emulator training rows
- `models/`
  - define the shared Flax NNX MLP
  - define archive save/load behavior
- `training/`
  - own optimization and early stopping
- `emulators/`
  - define the two real emulator workflows: `Delta21` and `T21`
- `cli/`
  - expose user-facing training and inference commands
- `core/`
  - only generic math and data primitives that are truly reusable

## Legacy Behavior vs Legacy Structure

The repo should preserve legacy behavior where it affects the scientific
meaning of the training problem. Examples include:

- parameter ordering
- log transforms
- target transforms
- train/validation split behavior
- `Delta21` random interpolation workflow
- `T21` fixed-grid workflow

The repo should not preserve legacy structure merely because it existed before.
For example, broad families like "global signal" or "power spectrum" are less
useful than explicit workflow modules when the real supported scope is only
`T21` and `Delta21`.

## Documentation Style

The intended style for code in this repository is:

- every public module has a module docstring that explains its role
- every public function and class has a practical docstring
- inline comments appear every few lines around non-obvious logic
- comments explicitly explain legacy-sensitive decisions when relevant

The aim is for a scientifically literate reader to understand not only what a
function does, but why it exists in the workflow and how it relates to the
legacy code.
