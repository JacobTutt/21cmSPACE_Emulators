# 21cmSPACE JAX Emulators

Welcome. `21cmspace-emulators` is a practical guide and codebase for building
JAX emulators for 21-cm cosmology simulations and multi-wavelength constraints.
It keeps the path from simulation products to saved emulator packages explicit:
prepare physical arrays, scale and tile inputs, train a JAX MLP, save a
`.nenemu` checkpoint, and use that package for inference.

The current implemented workflows target 21cmSPACE-style global-signal and
power-spectrum emulators. The same infrastructure is intended to support other
observables used in joint 21-cm and astrophysical inference.

## Repository Layout

```text
21cmspace-emulators/
  README.md                         (top-level guide)
  pyproject.toml                    (package metadata, dependencies, CLI entry points)
  docs/                             (detailed workflow documentation)
    architecture.md                 (model and package architecture)
    jax-training.md                 (JAX/Flax training workflow)
    preprocessing.md                (data contracts, transforms, scaling, tiling)
    examples.md                     (example index)
    examples/                       (worked emulator examples)
    references.md                   (science and software references)
  jax_emu/                          (reusable JAX emulator infrastructure)
    architectures/                  (shared MLP definitions)
    data_preprocessing/             (specs, parameter prep, transforms, scaling, tiling)
    training/                       (training and evaluation loops)
    utils/                          (configs, checkpoints, metrics)
  emulators_21cmspace/              (21cmSPACE-specific workflows)
    t21/                            (global 21-cm signal emulator)
    delta21/                        (21-cm power-spectrum emulator)
    twentyonecmspace.py             (shared 21cmSPACE constants/helpers)
  tests/                            (contract, workflow, and CLI smoke tests)
```

## What Can Be Emulated

The implemented entry points cover:

- global 21-cm signal, `T21`
- 21-cm power spectrum, `Delta21`

The emulator contract is also designed for related observables, including UV
luminosity functions, cosmic X-ray background and radio background constraints
(`CXB`/`CRB`), star-formation-rate density (`SFRD`), neutral fraction (`xHI`),
and thermal or radio histories.

## Model Approach

The default architecture is a dense MLP in JAX/Flax. Its role is deliberately
simple: map prepared cosmological/astrophysical parameters plus coordinate axes
such as redshift or wave number to one scalar target value, then reconstruct the
full observable grid after inference.

This follows the same broad emulator pattern used by GlobalEmu
([arXiv:2104.04336](https://arxiv.org/abs/2104.04336)): compact feed-forward
networks trained on simulation grids, with preprocessing and inverse transforms
treated as part of the saved emulator package.

## Documentation Map

Start here, then move into the stage-specific docs:

- [Architecture](docs/architecture.md): package layout, specs, checkpoints, and inference contracts.
- [JAX training](docs/jax-training.md): model initialization, batching, optimization, and validation.
- [Preprocessing](docs/preprocessing.md): parameter preparation, target transforms, scaling, and tiling.
- [Examples](docs/examples.md): index for worked workflows, including global 21-cm and power-spectrum examples.
- [References](docs/references.md): science background and citation list.

## Installation

The project metadata and CLI entry points live in `pyproject.toml`. From the
repository root, install in editable mode with development test dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

With `uv`, the equivalent workflow is:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Quick Smoke Tests

Run the Python test suite:

```bash
python -m pytest -q
```

Check the installed CLI entry points:

```bash
21cmspace-t21-train --print-spec
21cmspace-t21-train --synthetic-smoke --epochs 5 --batch-size 32
21cmspace-delta21-train --print-spec
21cmspace-delta21-train --synthetic-smoke --epochs 5 --batch-size 32
```

The inference commands operate on saved `.nenemu` packages:

```bash
21cmspace-t21-infer --package t21_model.nenemu --describe
21cmspace-delta21-infer --package delta21_model.nenemu --describe
```

Training from real 21cmSPACE data starts from a dataset root:

```bash
21cmspace-t21-train --dataset-root /path/to/21cmSPACE --output t21_model.nenemu
21cmspace-delta21-train --dataset-root /path/to/21cmSPACE --output delta21_model.nenemu
```

## References

See [docs/references.md](docs/references.md) for the curated reference list.
Key arXiv entries for this repository include:

- [2104.04336](https://arxiv.org/abs/2104.04336)
- [2312.08095](https://arxiv.org/abs/2312.08095)
- [2503.21687](https://arxiv.org/abs/2503.21687)
- [2508.13761](https://arxiv.org/abs/2508.13761)
