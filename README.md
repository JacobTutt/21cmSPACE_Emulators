# Nenufar_Emulators

`Nenufar_Emulators` is a JAX-native repository for training and packaging
21-cm emulators for inference work.

The current repository scope is intentionally narrow:

- `Delta21`: a tiled power-spectrum emulator trained on HERA IDR4 simulations
- `T21`: a tiled global-signal emulator trained on HERA IDR4 simulations

The code preserves selected legacy scientific conventions from the older
`CosmicDawnSynergies` repository where those conventions define the learning
problem, but it does not try to preserve the old repository structure. This
package is meant to stand on its own as a clear, documented training and
inference codebase.

## Design Goals

- keep the scientific contracts explicit
- keep the package layout close to the real workflow
- document legacy-derived choices clearly
- make training and checkpointing testable
- avoid migration-era scaffolding that is not part of the real product

## Current Workflow

For both supported emulators, the workflow is:

1. load HERA IDR4 files from disk
2. apply the legacy parameter and target conventions deliberately
3. prepare scalar training rows for the chosen emulator
4. train a Flax NNX MLP with Optax
5. save a metadata-rich emulator archive

The two emulators intentionally differ in how they prepare training rows:

- `Delta21` follows a `poweremu`-style random interpolation workflow
- `T21` follows a more `globalemu`-like fixed-grid workflow

## Documentation

- [docs/architecture.md](docs/architecture.md): package structure and design intent
- [docs/workflows.md](docs/workflows.md): end-to-end `Delta21` and `T21` workflows
- [docs/migration_plan.md](docs/migration_plan.md): migration notes and staged parity goals
- [docs/foundation_usage.md](docs/foundation_usage.md): setup and verification notes

## Status

The repository now has:

- HERA IDR4 data loading for `Delta21` and `T21`
- legacy-aligned parameter preparation
- JAX/Flax training paths for both emulators
- archive save/load support
- test coverage for preparation, training, and checkpoint contracts

Inference and production checkpoint-driven prediction APIs still need more
work before this should be treated as a finished scientific production
package.
