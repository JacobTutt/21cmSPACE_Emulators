"""Serialization helpers for trained emulator packages.

This module turns a trained model and its supporting metadata into one
reusable ``.nenemu`` file. A saved package includes the network parameters,
training history, and metadata needed to prepare inference inputs in the same
way as training inputs.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from nenufar_emulators.architectures.mlp import DenseMLP, init_mlp
from nenufar_emulators.utils.scaling import FeatureScaling, TargetScalingSurface
from nenufar_emulators.utils.specs import AxisSpec, EmulatorSpec, ParameterSpec


def _package_version() -> str:
    """Return the installed package version or a sensible development fallback."""
    try:
        return version("nenufar-emulators")
    except PackageNotFoundError:
        return "0.1.0"


def _encode_state_path(path: tuple[Any, ...]) -> str:
    """Encode an NNX flat-state path into a string key for ``np.savez``."""
    encoded_parts = []
    for part in path:
        if isinstance(part, int):
            encoded_parts.append(f"[{part}]")
        else:
            encoded_parts.append(str(part))
    return "/".join(encoded_parts)


def _decode_state_path(path: str) -> tuple[Any, ...]:
    """Decode an ``np.savez`` key back into an NNX flat-state path."""
    decoded_parts: list[Any] = []
    for part in path.split("/"):
        if part.startswith("[") and part.endswith("]"):
            decoded_parts.append(int(part[1:-1]))
        else:
            decoded_parts.append(part)
    return tuple(decoded_parts)


def _flatten_state(state: nnx.State) -> dict[str, np.ndarray]:
    """Flatten an NNX state object into ``np.savez``-friendly entries."""
    return {
        _encode_state_path(tuple(path)): np.asarray(value)
        for path, value in nnx.to_flat_state(state)
    }


def _unflatten_state(flat: dict[str, np.ndarray]) -> nnx.State:
    """Reconstruct an NNX state object from ``np.savez`` entries."""
    return nnx.from_flat_state(
        [(_decode_state_path(path), jnp.asarray(value)) for path, value in flat.items()]
    )


def _axis_spec_from_dict(payload: dict[str, Any]) -> AxisSpec:
    """Reconstruct an :class:`AxisSpec` from serialized metadata."""
    limits = payload.get("limits")
    return AxisSpec(
        name=payload["name"],
        transform=payload.get("transform", "identity"),
        limits=None if limits is None else tuple(float(v) for v in limits),
        nsample=payload.get("nsample"),
    )


def _parameter_spec_from_dict(payload: dict[str, Any]) -> ParameterSpec:
    """Reconstruct a :class:`ParameterSpec` from serialized metadata."""
    discrete_values = payload.get("discrete_values")
    return ParameterSpec(
        name=payload["name"],
        transform=payload.get("transform", "identity"),
        discrete_values=(
            None
            if discrete_values is None
            else tuple(float(v) for v in discrete_values)
        ),
    )


def _emulator_spec_from_dict(payload: dict[str, Any]) -> EmulatorSpec:
    """Reconstruct an :class:`EmulatorSpec` from serialized metadata."""
    return EmulatorSpec(
        name=payload["name"],
        family=payload["family"],
        axes=tuple(_axis_spec_from_dict(axis) for axis in payload["axes"]),
        parameters=tuple(
            _parameter_spec_from_dict(parameter) for parameter in payload["parameters"]
        ),
        target_transform=payload.get("target_transform", "identity"),
        target_offset=float(payload.get("target_offset", 0.0)),
    )


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
    target_scaling: TargetScalingSurface | None
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
            "target_scaling": (
                None if self.target_scaling is None else self.target_scaling.to_dict()
            ),
            "training_config": self.training_config,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CheckpointMetadata":
        """Reconstruct metadata from the serialized dictionary form."""
        return cls(
            model_name=payload["model_name"],
            package_version=payload["package_version"],
            emulator_spec=_emulator_spec_from_dict(payload["emulator_spec"]),
            input_scaling=tuple(
                FeatureScaling(
                    name=feature["name"],
                    method=feature["method"],
                    minimum=float(feature["minimum"]),
                    maximum=float(feature["maximum"]),
                    mean=float(feature["mean"]),
                    std=float(feature["std"]),
                )
                for feature in payload["input_scaling"]
            ),
            target_scaling=(
                None
                if payload.get("target_scaling") is None
                else TargetScalingSurface.from_dict(payload["target_scaling"])
            ),
            training_config=payload["training_config"],
        )


def save(
    path: str | Path,
    model: DenseMLP,
    train_losses: list[float],
    val_losses: list[float],
    loss: str,
    *,
    metadata: CheckpointMetadata | None = None,
    epochs: int = 1000,
    patience: int | None = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Path:
    """Save a trained emulator to a ``.nenemu`` package.

    - ``config.json`` for hyperparameters, history, and metadata
    - ``params.npz`` for neural-network parameters
    """
    package_path = Path(path)
    if package_path.suffix != ".nenemu":
        package_path = package_path.with_suffix(".nenemu")

    config: dict[str, Any] = {
        "version": _package_version(),
        "hyperparams": {
            "in_features": model.in_features,
            "hidden_features": model.hidden_features,
            "hidden_layers": model.hidden_layers,
            "out_features": model.out_features,
            "activation": model.activation,
            "init_scale": model.init_scale,
            "epochs": epochs,
            "patience": patience,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        },
        "loss": loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    if metadata is not None:
        config["metadata"] = metadata.to_dict()

    flat_params = _flatten_state(nnx.state(model))

    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("config.json", json.dumps(config, indent=2))

        params_buffer = io.BytesIO()
        np.savez(params_buffer, **flat_params)
        zf.writestr("params.npz", params_buffer.getvalue())

    return package_path


def load(path: str | Path) -> dict[str, Any]:
    """Load a trained emulator from a ``.nenemu`` package.

    The returned dictionary includes a live NNX model object and the stored
    metadata needed for inference.
    """
    package_path = Path(path)
    with zipfile.ZipFile(package_path, "r") as zf:
        config = json.loads(zf.read("config.json"))

        params_buffer = io.BytesIO(zf.read("params.npz"))
        params_np = np.load(params_buffer)
        flat_params = {key: params_np[key] for key in params_np.files}
        state = _unflatten_state(flat_params)

        hyperparams = config["hyperparams"]
        model = init_mlp(
            key=jax.random.PRNGKey(0),
            in_features=int(hyperparams["in_features"]),
            hidden_features=int(hyperparams["hidden_features"]),
            out_features=int(hyperparams["out_features"]),
            hidden_layers=int(hyperparams["hidden_layers"]),
            activation=hyperparams["activation"],
            scale=float(hyperparams.get("init_scale", 1e-1)),
        )
        nnx.update(model, state)

        return {
            "model": model,
            "params": nnx.state(model),
            "hyperparams": hyperparams,
            "train_losses": config["train_losses"],
            "val_losses": config["val_losses"],
            "loss": config["loss"],
            "version": config["version"],
            "metadata": (
                None
                if "metadata" not in config
                else CheckpointMetadata.from_dict(config["metadata"])
            ),
        }
