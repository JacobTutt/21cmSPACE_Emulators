"""Smoke tests for lightweight CLI-facing helpers."""

from __future__ import annotations

from nenufar_emulators.global_signal.train import run_synthetic_smoke as run_global_smoke
from nenufar_emulators.power_spectrum.train import run_synthetic_smoke as run_power_smoke


def test_power_smoke_helper_runs() -> None:
    result = run_power_smoke(epochs=5, batch_size=32)
    assert result["final_validation_loss"] >= 0.0


def test_global_smoke_helper_runs() -> None:
    result = run_global_smoke(epochs=5, batch_size=32)
    assert result["final_validation_loss"] >= 0.0
