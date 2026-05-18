# Foundation Usage

## Current State

The repository currently provides:

- emulator specifications for power-spectrum and global-signal families
- shared tiling utilities
- a generic JAX MLP
- a minimal Optax training loop
- synthetic smoke-run CLIs

It does not yet provide:

- production dataset loaders
- real training configs bound to local data files
- checkpoint save/load for learned weights
- inference against physical datasets

## Verification Commands

Create or refresh the local verification environment:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install '.[dev]'
```

Run the test suite:

```bash
.venv/bin/python -m pytest -q
```

Run the synthetic smoke trainers:

```bash
.venv/bin/nenufar-power-train --synthetic-smoke
.venv/bin/nenufar-global-train --synthetic-smoke
```

Print the baseline emulator specs:

```bash
.venv/bin/nenufar-power-train --print-spec
.venv/bin/nenufar-global-train --print-spec
```

## What These Smoke Runs Verify

- model input dimensionality is consistent with the emulator spec
- tiling from spectral targets to scalar regression samples works
- the shared JAX trainer can reduce loss on a controlled synthetic problem
- the repository entrypoints resolve correctly after installation

## What Comes Next

The next implementation stage is to connect real old-code parameter mappings and
real data loaders to these shared interfaces, then add checkpoint persistence
and parity checks against the legacy emulator outputs.
