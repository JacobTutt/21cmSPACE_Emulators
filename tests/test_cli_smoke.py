"""Smoke tests for lightweight CLI-facing helpers."""

from __future__ import annotations

from jaxemu_21cmSPACE.emulators21.delta21.train import run_synthetic_smoke as run_delta21_smoke
from jaxemu_21cmSPACE.emulators21.t21.train import run_synthetic_smoke as run_t21_smoke


def test_delta21_smoke_helper_runs() -> None:
    result = run_delta21_smoke(epochs=5, batch_size=32)
    assert result["final_validation_loss"] >= 0.0


def test_t21_smoke_helper_runs() -> None:
    result = run_t21_smoke(epochs=5, batch_size=32)
    assert result["final_validation_loss"] >= 0.0
