# 21cmSPACE JAX Emulators

The repository is designed to be a practical guide for building JAX emulators
from 21-cm cosmology simulations and other associated multi-wavelength probes
for use within JAX-accelerated Bayesian inference frameworks.

The model design follows the architecture ideas used by
[AstroEmu](https://astroemu.readthedocs.io/en/latest/tutorial/) and
[GlobalEmu](https://github.com/htjb/globalemu)
([arXiv:2104.04336](https://arxiv.org/abs/2104.04336)): combine physical
parameters with independent coordinates to allow the network to predict scalar
observable values, then reconstruct spectra or grids after inference.

## Repository Layout

```text
21cmspace-emulators/
  README.md
  docs/                             (detailed guides to emulator training)
  jax_emu/                          (reusable JAX emulator infrastructure)
    architectures/                  (shared MLP definitions)
    data_preprocessing/             (specs, parameter prep, transforms, scaling, tiling)
    training/                       (training and evaluation loops)
    utils/                          (configs, checkpoints, metrics)
  emulators_21cmspace/              (21cmSPACE-specific examples)
    t21/                            (global 21-cm signal emulator)
    delta21/                        (21-cm power-spectrum emulator)
    twentyonecmspace.py             (shared 21cmSPACE constants/helpers)
```

## Emulator Use Cases

The repository is designed to support a wide range of reionisation probes, so
emulators can be used in multi-wavelength joint analyses such as
[arXiv:2503.21687](https://arxiv.org/abs/2503.21687),
[arXiv:2312.08095](https://arxiv.org/abs/2312.08095), and
[arXiv:2508.13761](https://arxiv.org/abs/2508.13761). Emulator targets in this
style include:

- 21-cm global brightness temperature, `T21(z)`
- 21-cm power spectra, `Delta21(z, k)`
- UV luminosity functions, `UVLF`
- cosmic X-ray and radio backgrounds (`CXB`/`CRB`)
- star-formation-rate density, `SFRD(z)`
- ionisation and thermal histories, including `xHI(z)`, `TK(z)`, `TS(z)`, and
  `Trad(z)`

## AstroEmu Approach

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
