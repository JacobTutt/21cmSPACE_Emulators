# Foundation Usage

## Current State

The repository currently provides:

- explicit `Delta21` and `T21` workflow modules
- shared tiling utilities
- a shared Flax NNX MLP
- a shared Optax training loop
- HERA IDR4 data preparation for `Delta21` and `T21`
- checkpoint package save/load support
- synthetic smoke-run CLIs

It does not yet provide:

- production inference against physical datasets
- a finished checkpoint-driven prediction API
- broader emulator coverage beyond `Delta21` and `T21`

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
.venv/bin/nenufar-delta21-train --synthetic-smoke
.venv/bin/nenufar-t21-train --synthetic-smoke
```

Print the baseline emulator specs:

```bash
.venv/bin/nenufar-delta21-train --print-spec
.venv/bin/nenufar-t21-train --print-spec
```

Print the workflow defaults:

```bash
.venv/bin/nenufar-delta21-train --print-config
.venv/bin/nenufar-t21-train --print-config
```

## What These Smoke Runs Verify

- model input dimensionality is consistent with the emulator spec
- tiling from spectral targets to scalar regression samples works
- the shared JAX trainer can reduce loss on a controlled synthetic problem
- the workflow-specific package layout resolves correctly after installation

## What Comes Next

The next implementation stage is to deepen the workflow-specific code paths
and improve inference surfaces around the real `Delta21` and `T21` workflows.
