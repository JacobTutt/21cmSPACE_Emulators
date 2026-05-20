"""Serialization helpers for trained emulator packages.

This module turns a trained model and its supporting metadata into one
reusable ``.nenemu`` file. A saved package includes the network parameters,
training history, preprocessing pipelines, and optional dataset snapshots that
make the model reproducible on another machine.
"""

from __future__ import annotations

import io
import json
import pickle
import zipfile
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from nenufar_emulators.core.datasets import SpectrumDataset
from nenufar_emulators.models import DenseMLP, init_mlp
from nenufar_emulators.core.scaling import FeatureScaling, TargetScalingSurface
from nenufar_emulators.core.specs import AxisSpec, EmulatorSpec, ParameterSpec


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


def _dataset_config(dataset: SpectrumDataset) -> dict[str, Any]:
    """Return the non-array configuration needed to rebuild a dataset split."""
    return {
        "axis_names": list(dataset.axis_names),
        "parameter_names": list(dataset.parameter_names),
        "tiling": dataset.tiling,
    }


def _write_dataset_arrays(npz_payload: dict[str, np.ndarray], split: str, dataset: SpectrumDataset) -> None:
    """Append one dataset split's arrays to an ``np.savez`` payload dict."""
    npz_payload[f"{split}/spectra"] = np.asarray(dataset.spectra)
    npz_payload[f"{split}/parameters"] = np.asarray(dataset.parameters)
    for idx, axis in enumerate(dataset.axes):
        npz_payload[f"{split}/axis_{idx}"] = np.asarray(axis)


def _reconstruct_dataset(
    split: str,
    config: dict[str, Any],
    arrays: dict[str, np.ndarray] | None,
    pipelines: dict[str, Any],
) -> SpectrumDataset | dict[str, Any]:
    """Rebuild a dataset split from stored arrays if they are present."""
    split_key = f"{split}_dataset"
    if split_key not in config:
        raise KeyError(f"Missing dataset config for split {split!r}.")
    dataset_config = config[split_key]
    if arrays is None:
        return dataset_config

    prefix = f"{split}/"
    required = [f"{prefix}spectra", f"{prefix}parameters"]
    if not all(key in arrays for key in required):
        return dataset_config

    axis_arrays = []
    axis_index = 0
    while f"{prefix}axis_{axis_index}" in arrays:
        axis_arrays.append(arrays[f"{prefix}axis_{axis_index}"])
        axis_index += 1
    if len(axis_arrays) != len(dataset_config["axis_names"]):
        return dataset_config

    return SpectrumDataset(
        spectra=arrays[f"{prefix}spectra"],
        axes=tuple(axis_arrays),
        parameters=arrays[f"{prefix}parameters"],
        axis_names=tuple(dataset_config["axis_names"]),
        parameter_names=tuple(dataset_config["parameter_names"]),
        forward_pipeline=pipelines.get(split) or None,
        tiling=dataset_config["tiling"],
    )


def save(
    path: str | Path,
    model: DenseMLP,
    train_losses: list[float],
    val_losses: list[float],
    loss: str,
    *,
    train_dataset: SpectrumDataset | None = None,
    val_dataset: SpectrumDataset | None = None,
    test_dataset: SpectrumDataset | None = None,
    metadata: CheckpointMetadata | None = None,
    epochs: int = 1000,
    patience: int | None = None,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Path:
    """Save a trained emulator to a ``.nenemu`` package.

    - ``config.json`` for hyperparameters, history, and dataset configs
    - ``params.npz`` for neural-network parameters
    - ``pipeline.pkl`` for preprocessing pipeline objects

    In addition, this repository stores ``datasets.npz`` when dataset objects
    are provided, because our current datasets are array-backed.
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

    datasets = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
    pipelines = {
        split: None if dataset is None else dataset.forward_pipeline
        for split, dataset in datasets.items()
    }
    for split, dataset in datasets.items():
        if dataset is not None:
            config[f"{split}_dataset"] = _dataset_config(dataset)

    flat_params = _flatten_state(nnx.state(model))

    dataset_arrays: dict[str, np.ndarray] = {}
    for split, dataset in datasets.items():
        if dataset is not None:
            _write_dataset_arrays(dataset_arrays, split, dataset)

    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("config.json", json.dumps(config, indent=2))

        params_buffer = io.BytesIO()
        np.savez(params_buffer, **flat_params)
        zf.writestr("params.npz", params_buffer.getvalue())

        zf.writestr("pipeline.pkl", pickle.dumps(pipelines))

        if dataset_arrays:
            dataset_buffer = io.BytesIO()
            np.savez(dataset_buffer, **dataset_arrays)
            zf.writestr("datasets.npz", dataset_buffer.getvalue())

    return package_path


def load(path: str | Path) -> dict[str, Any]:
    """Load a trained emulator from a ``.nenemu`` package.

    The returned dictionary includes a live NNX model object together with the
    stored metadata, preprocessing pipeline, and any packaged dataset splits.
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

        pipelines = pickle.loads(zf.read("pipeline.pkl"))

        dataset_arrays: dict[str, np.ndarray] | None = None
        if "datasets.npz" in zf.namelist():
            dataset_buffer = io.BytesIO(zf.read("datasets.npz"))
            datasets_np = np.load(dataset_buffer)
            dataset_arrays = {key: datasets_np[key] for key in datasets_np.files}

        result: dict[str, Any] = {
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
            "train_pipeline": pipelines["train"],
            "val_pipeline": pipelines["val"],
            "test_pipeline": pipelines["test"],
        }

        for split in ("train", "val", "test"):
            key = f"{split}_dataset"
            if key not in config:
                continue
            result[key] = _reconstruct_dataset(split, config, dataset_arrays, pipelines)

        return result
