"""Checkpoint metadata contracts.

The actual weight-serialization format is not implemented yet, but we already
know what surrounding context a usable emulator checkpoint must carry:
scientific spec, scaling metadata, and enough training configuration to
recreate the model contract later. This module defines that metadata shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from nenufar_emulators.core.scaling import FeatureScaling
from nenufar_emulators.core.specs import EmulatorSpec


@dataclass(frozen=True)
class CheckpointMetadata:
    """Serializable metadata required to make a trained model reusable.

    In practice a checkpoint is not just neural-network weights. We also need
    to know which emulator family the weights belong to, what input scaling was
    applied, and which version of the package wrote the file. Without that
    information, inference code cannot reliably reconstruct the original model
    contract.
    """

    model_name: str
    package_version: str
    emulator_spec: EmulatorSpec
    input_scaling: tuple[FeatureScaling, ...]
    training_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert nested metadata into plain Python structures for storage.

        The returned dictionary is designed to be safe to hand to JSON, YAML,
        or similar serializers without leaving dataclass objects embedded in
        the payload.
        """
        return {
            "model_name": self.model_name,
            "package_version": self.package_version,
            "emulator_spec": asdict(self.emulator_spec),
            "input_scaling": [feature.to_dict() for feature in self.input_scaling],
            "training_config": self.training_config,
        }
