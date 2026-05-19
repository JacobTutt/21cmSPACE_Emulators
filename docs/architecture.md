# Architecture

## Purpose

This repository is organized around the workflows people actually run. Its
purpose is to provide a clean and readable JAX codebase that keeps the
important scientific behavior explicit without carrying unnecessary structural
complexity.

The practical focus is currently narrow:

- `Delta21` on HERA IDR4
- `T21` on HERA IDR4

Other historical emulator branches are useful context,
but they are not the primary product target of this codebase.

## Design Principles

- Keep the public package layout aligned with the real user workflow.
- Keep scientific behavior explicit, not hidden in scattered helper code.
- Keep generic utilities separate from HERA-specific and emulator-specific logic.
- Prefer a small number of well-named modules over many thin wrappers.
- Use comments to explain intent and design rationale, not to narrate syntax.

## Target Structure

The package should read from top to bottom like the actual workflow:

1. data loading
2. workflow parameter and target preparation
3. emulator-specific preparation
4. model training
5. checkpoint package save/load
6. inference

That leads to the following target organization:

- `data/`
  - load raw HERA IDR4 arrays
  - define workflow parameter and target conventions
  - prepare emulator training rows
- `models/`
  - define the shared Flax NNX MLP
  - define checkpoint package save/load behavior
- `training/`
  - own optimization and early stopping
- `emulators/`
  - define the two real emulator workflows: `Delta21` and `T21`
- `cli/`
  - expose user-facing training and inference commands
- `core/`
  - only generic math and data primitives that are truly reusable

## Scientific Behavior vs Code Structure

The repo should preserve established behavior where it affects the scientific
meaning of the training problem. Examples include:

- parameter ordering
- log transforms
- target transforms
- train/validation split behavior
- `Delta21` random interpolation workflow
- `T21` fixed-grid workflow

The repo should not preserve earlier structure merely because it existed before.
For example, broad families like "global signal" or "power spectrum" are less
useful than explicit workflow modules when the real supported scope is only
`T21` and `Delta21`.

## Documentation Style

The intended style for code in this repository is:

- every public module has a module docstring that explains its role
- every public function and class has a practical docstring
- inline comments appear every few lines around non-obvious logic
- comments explicitly explain scientifically sensitive decisions when relevant

The aim is for a scientifically literate reader to understand not only what a
function does, but why it exists in the workflow and how it relates to the
rest of the repository.
