# 21cmSPACE JAX Emulators

**Navigation:** [README](README.md) · [Architecture](docs/architecture.md) · [Preprocessing](docs/preprocessing.md) · [JAX Training](docs/jax-training.md) · [Checkpointing](docs/checkpoint.md) · [Inference](docs/inference.md) · [Examples](docs/examples.md)

The repository is designed to be a practical guide for those wanting to build simple JAX emulators
from 21-cm cosmology simulations and other associated multi-wavelength probes
for use within JAX-accelerated Bayesian inference frameworks, for example
[BlackJAX](https://github.com/blackjax-devs/blackjax).

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
    inference/                      (priors, likelihoods, nested sampling)
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

- [Architecture](docs/architecture.md): tiled scalar-output networks and DenseMLP configuration.
- [Preprocessing](docs/preprocessing.md): parameter preparation, target transforms, scaling, and tiling.
- [JAX training](docs/jax-training.md): model initialization, batching, optimization, and validation.
- [Checkpointing](docs/checkpoint.md): saving model weights, losses, and preprocessing metadata.
- [Inference](docs/inference.md): priors, likelihoods, upper limits, and nested sampling.
- [Examples](docs/examples.md): worked global 21-cm and power-spectrum workflows.

## Installation

We recommend using `uv` for the fastest setup, but standard `pip` is also
supported.

### Choose Your Backend
The network architectures use JAX, which requires a backend matched to your
hardware (CPU or NVIDIA GPU).

Identify which machine:
- `cpu`: laptops or CPU clusters.
- `cuda12/cuda13`: GPU systems with NVIDIA Accelerators.


### Option A: Using uv (Recommended)

`uv` will manage the virtual environment and lock dependencies.

For CPU development:

```bash
uv sync --extra cpu --extra dev
source .venv/bin/activate
```

For GPU systems, choose the command that matches your driver stack:

```bash
# CUDA 12 (older systems)
uv sync --extra cuda12 --extra dev

# CUDA 13 (latest)
uv sync --extra cuda13 --extra dev

source .venv/bin/activate
```

### Option B: Using pip

If you prefer not to use `uv`, install the package manually into your own
virtual environment.

```bash
# 1. Create and enter the environment
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# 2. Install based on your hardware
# Replace [cpu] with [cuda12] or [cuda13] if using a GPU
python -m pip install -e ".[cpu,dev]"
```

---

**Navigation:** [README](README.md) · [Architecture](docs/architecture.md) · [Preprocessing](docs/preprocessing.md) · [JAX Training](docs/jax-training.md) · [Checkpointing](docs/checkpoint.md) · [Inference](docs/inference.md) · [Examples](docs/examples.md)
