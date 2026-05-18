"""Checkpoint metadata contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from nenufar_emulators.core.scaling import FeatureScaling
from nenufar_emulators.core.specs import EmulatorSpec


@dataclass(frozen=True)
class CheckpointMetadata:
    """Serializable metadata required for training and inference."""

    model_name: str
    package_version: str
    emulator_spec: EmulatorSpec
    input_scaling: tuple[FeatureScaling, ...]
    training_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert nested metadata into JSON-safe structures."""
        return {
            "model_name": self.model_name,
            "package_version": self.package_version,
            "emulator_spec": asdict(self.emulator_spec),
            "input_scaling": [feature.to_dict() for feature in self.input_scaling],
            "training_config": self.training_config,
        }
