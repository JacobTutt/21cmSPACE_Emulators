# 21cmSPACE JAX Emulators

Welcome. `21cmspace-emulators` is a practical guide and codebase for building
JAX emulators for 21-cm cosmology simulations and multi-wavelength constraints.
It keeps the path from simulation products to saved emulator packages explicit:
prepare physical arrays, scale and tile inputs, train a JAX MLP, save a
`.nenemu` checkpoint, and use that package for inference.

The design follows the same scalar-regression emulator idea used by
[AstroEmu](https://astroemu.readthedocs.io/en/latest/tutorial/) and
[GlobalEmu](https://github.com/htjb/globalemu)
([arXiv:2104.04336](https://arxiv.org/abs/2104.04336)): combine physical
parameters with independent coordinates, predict scalar observable values, and
reconstruct spectra or grids after inference.

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
```

## What Can Be Emulated

The emulator contract is intended for observable families that can be expressed
as values on physical coordinate axes:

- global 21-cm brightness temperature, `T21(z)`, following the GlobalEmu
  precedent.
- 21-cm power spectra, `Delta21(z, k)`, as used in the 21cmSPACE
  multi-wavelength analyses cited in [references](docs/references.md),
  including the Pop III constraints paper
  ([arXiv:2312.08095](https://arxiv.org/abs/2312.08095)).
- UV luminosity functions, `UVLF` or `Phi(MUV, z)`, highlighted by the Jiten
  JWST/21-cm synergy paper
  ([arXiv:2503.21687](https://arxiv.org/abs/2503.21687)).
- diffuse backgrounds, including cosmic X-ray and radio backgrounds
  (`CXB`/`CRB`), used in the Jiten and Simon multi-wavelength studies listed in
  [references](docs/references.md).
- star-formation and IGM history summaries, including Pop II / Pop III `SFRD`,
  neutral fraction `xHI`, kinetic temperature `TK`, spin temperature `TS`, and
  radio background temperature `Trad`, following the Simon discovery-space
  paper ([arXiv:2508.13761](https://arxiv.org/abs/2508.13761)).
- compact global-signal summaries such as `T21,min`, trough redshift, and
  trough frequency, when the target is a scalar derived observable.

## Model Approach

The model is a simple JAX/Flax implementation of the GlobalEmu/AstroEmu idea:
a neural network maps cosmological and astrophysical parameters plus independent
coordinates, such as redshift `z` and/or wavenumber `k`, to scalar observable
values. Vectorized inference over the coordinate axes then reconstructs the
requested spectrum, surface, or grid.

## Documentation Map

Start here, then move into the stage-specific docs:

- [Architecture](docs/architecture.md): package layout, specs, checkpoints, and inference contracts.
- [JAX training](docs/jax-training.md): model initialization, batching, optimization, and validation.
- [Preprocessing](docs/preprocessing.md): parameter preparation, target transforms, scaling, and tiling.
- [Examples](docs/examples.md): index for worked workflows, including global 21-cm and power-spectrum examples.

## Installation

The project is managed with `uv` and includes a lock file. The main install
choice is the JAX backend: CPU-only JAX, CUDA 12 JAX, or CUDA 13 JAX. Choose
one backend extra for the machine you are using.

CPU-only development install:

```bash
uv sync --extra cpu --extra dev
source .venv/bin/activate
```

NVIDIA GPU install with CUDA 13 packages:

```bash
uv sync --extra cuda13 --extra dev
source .venv/bin/activate
```

NVIDIA GPU install with CUDA 12 packages:

```bash
uv sync --extra cuda12 --extra dev
source .venv/bin/activate
```

Use CUDA 13 when your driver stack supports it; use CUDA 12 for older CUDA 12
systems. The JAX project keeps the current backend compatibility notes at
[docs.jax.dev/en/latest/installation.html](https://docs.jax.dev/en/latest/installation.html).

The same extras can be installed with `pip` if you are not using `uv`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[cpu,dev]"
```

For GPU machines, replace `cpu` with `cuda12` or `cuda13`:

```bash
python -m pip install -e ".[cuda13,dev]"
```

Optional CLI checks:

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
