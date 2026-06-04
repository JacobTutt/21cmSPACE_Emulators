"""
Serialization helpers for trained emulator checkpoints.

This module turns a trained model and its supporting metadata into one
reusable checkpoint directory. Orbax stores both the Flax NNX model state and
the JSON configuration needed to prepare inference inputs in the same way as
training inputs. This module handles:
- reconstruction of specs and metadata from JSON
- bundling of model weights with preprocessing and training configuration
- versioned model saving and loading for inference
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import jax
import orbax.checkpoint as ocp
from flax import nnx

from jax_emu.architectures.mlp import DenseMLP
from jax_emu.data_preprocessing.scaling import (
    FeatureScaling,
    TargetScalingScalar,
)
from jax_emu.data_preprocessing.specs import AxisSpec, EmulatorSpec, ParameterSpec


# Versioning
# ----------
# Track the package version for compatibility checks in checkpoints.

def _package_version() -> str:
    """
    Return the installed package version or a sensible development fallback.

    Returns
    -------
    str
        The version string (e.g. '0.1.0').
    """
    try:
        # Attempt to get the version of the installed package.
        return version("21cmspace-emulators")
    except PackageNotFoundError:
        # Fallback for development environments where the package is not installed.
        return "0.1.0"


# Spec Reconstruction
# -------------------
# Helpers for rebuilding specification objects from serialized JSON payloads.

def _axis_spec_from_dict(payload: dict[str, Any]) -> AxisSpec:
    """
    Reconstruct an AxisSpec from serialized metadata.

    Parameters
    ----------
    payload:
        A dictionary containing the serialized AxisSpec fields.

    Returns
    -------
    AxisSpec
        A reconstructed AxisSpec object.
    """
    # Limits are optional in the serialized metadata and may be None.
    limits = payload.get("limits")
    return AxisSpec(
        name=payload["name"],
        transform=payload.get("transform", "identity"),
        # Ensure limits are restored as a tuple of floats if they exist.
        limits=None if limits is None else tuple(float(v) for v in limits),
        nsample=payload.get("nsample"),
    )


def _parameter_spec_from_dict(payload: dict[str, Any]) -> ParameterSpec:
    """
    Reconstruct a ParameterSpec from serialized metadata.

    Parameters
    ----------
    payload:
        A dictionary containing the serialized ParameterSpec fields.

    Returns
    -------
    ParameterSpec
        A reconstructed ParameterSpec object.
    """
    # Discrete values are optional in the serialized metadata.
    discrete_values = payload.get("discrete_values")
    return ParameterSpec(
        name=payload["name"],
        transform=payload.get("transform", "identity"),
        # Ensure discrete values are restored as a tuple of floats if they exist.
        discrete_values=(
            None
            if discrete_values is None
            else tuple(float(v) for v in discrete_values)
        ),
    )


def _emulator_spec_from_dict(payload: dict[str, Any]) -> EmulatorSpec:
    """
    Reconstruct an EmulatorSpec from serialized metadata.

    Parameters
    ----------
    payload:
        A dictionary containing the serialized EmulatorSpec fields.

    Returns
    -------
    EmulatorSpec
        A reconstructed EmulatorSpec object.
    """
    # Rebuild the nested axis and parameter specs first, then the emulator spec.
    # This maintains the hierarchy of the original configuration.
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


def _target_scaling_from_dict(payload: dict[str, Any]) -> TargetScalingScalar:
    """
    Reconstruct target-scaling metadata from serialized checkpoint data.

    Parameters
    ----------
    payload:
        Serialized target-scaling metadata.

    Returns
    -------
    TargetScalingScalar
        The reconstructed scalar target-scaling object.
    """
    if payload.get("kind") != "global_std":
        raise ValueError("Unsupported target scaling metadata. Expected kind='global_std'.")
    return TargetScalingScalar.from_dict(payload)


# Checkpoint Metadata
# -------------------
# Stores the non-weight information needed to reuse a trained emulator.

@dataclass(frozen=True)
class CheckpointMetadata:
    """
    Storage utility for metadata required to make a trained model reusable.

    In practice a checkpoint is not just neural-network weights. We also need
    to know which emulator family the weights belong to, what input scaling was
    applied, and which version of the package wrote the file. Without that
    information, inference code cannot reliably reconstruct the original model
    contract.

    Parameters
    ----------
    model_name:
        User-defined name for the specific model instance.
    package_version:
        The version of the package used to train and save the model.
    emulator_spec:
        The full input/target contract (axes, parameters, transforms).
    input_scaling:
        Scaling metadata for every input feature.
    target_scaling:
        Scaling metadata for the target values (if applicable).
    training_config:
        A dictionary capturing the hyperparameter and training-loop settings.
    """

    model_name: str
    package_version: str
    emulator_spec: EmulatorSpec
    input_scaling: tuple[FeatureScaling, ...]
    target_scaling: TargetScalingScalar | None
    training_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """
        Convert nested metadata into plain Python structures for storage.

        The returned dictionary is designed to be safe to hand to JSON, YAML,
        or similar serializers without leaving dataclass objects embedded in
        the payload.

        Returns
        -------
        dict
            A JSON-safe dictionary representation of the metadata.
        """
        # Convert nested dataclasses and arrays into JSON-compatible values recursively.
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
        """
        Reconstruct metadata from the serialized dictionary form.

        Parameters
        ----------
        payload:
            The serialized metadata dictionary.

        Returns
        -------
        CheckpointMetadata
            A reconstructed metadata object.
        """
        # Rebuild the scaling and spec objects used by inference from the raw dictionary.
        return cls(
            model_name=payload["model_name"],
            package_version=payload["package_version"],
            emulator_spec=_emulator_spec_from_dict(payload["emulator_spec"]),
            # Reconstruct the sequence of feature scaling rules.
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
            # Reconstruct target scaling if it was provided in the checkpoint.
            target_scaling=(
                None
                if payload.get("target_scaling") is None
                else _target_scaling_from_dict(payload["target_scaling"])
            ),
            training_config=payload["training_config"],
        )


# Save / Load Logic
# ----------------
# Saves a trained DenseMLP with Orbax and restores it later for inference.

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
    learning_rate_schedule: str = "constant",
    learning_rate_final_fraction: float = 0.1,
    learning_rate_warmup_epochs: int = 0,
) -> Path:
    """
    Save a trained emulator to an Orbax checkpoint manager directory.

    The saved package directory follows the structure:
    - state/: The optimized weights (Flax NNX model state).
    - config/: JSON file containing hyperparameters, history, and metadata.

    Parameters
    ----------
    path:
        The directory path where the checkpoint will be saved.
    model:
        The live trained DenseMLP model.
    train_losses:
        A list of training losses recorded during the training process.
    val_losses:
        A list of validation losses recorded during the training process.
    loss:
        The name/type of the loss function used.
    metadata:
        Optional metadata object carrying preprocessing and inference specs.
    epochs:
        The total number of epochs trained.
    patience:
        The early stopping patience setting used.
    learning_rate:
        The initial or peak learning rate used for training.
    weight_decay:
        The weight decay (L2 regularization) parameter used.
    learning_rate_schedule:
        The learning-rate schedule used during training. Supported values are
        `constant`, `cosine`, `warmup_cosine`, and `exponential_decay`.
    learning_rate_final_fraction:
        The final learning-rate fraction used by `cosine`, `warmup_cosine`,
        and `exponential_decay`. For example, `0.05` means the final learning
        rate was `learning_rate * 0.05`. Ignored by `constant`.
    learning_rate_warmup_epochs:
        The number of epochs used to ramp from zero to `learning_rate` for
        `warmup_cosine`. Ignored by the other schedules.

    Returns
    -------
    Path
        The absolute path to the saved checkpoint directory.
    """
    # Use the .nenemu (Nenufar Emulator) suffix as the package convention.
    package_path = Path(path)
    if package_path.suffix != ".nenemu":
        package_path = package_path.with_suffix(".nenemu")

    # Match the old save behavior by replacing an existing checkpoint path if it exists.
    # This prevents directory-not-empty errors during successive training runs.
    if package_path.exists():
        if package_path.is_dir():
            shutil.rmtree(package_path)
        else:
            package_path.unlink()

    # Store model architecture, training history, and optimizer metadata as JSON.
    config: dict[str, Any] = {
        "version": _package_version(),
        "format": "orbax-nnx-composite",
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
            "learning_rate_schedule": learning_rate_schedule,
            "learning_rate_final_fraction": learning_rate_final_fraction,
            "learning_rate_warmup_epochs": learning_rate_warmup_epochs,
        },
        "loss": loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    # If full metadata is provided (including specs and scaling), bundle it into the config.
    if metadata is not None:
        config["metadata"] = metadata.to_dict()

    # Split the live NNX model into its static graph structure and its trainable weights (state).
    # Orbax saves the state directly as an optimized JAX tree.
    _, state = nnx.split(model)

    # Use CheckpointManager to coordinate the saving of both weights and JSON configuration.
    manager = ocp.CheckpointManager(
        package_path,
        options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True),
        # We define two items: 'state' (weights) and 'config' (JSON).
        item_names=("state", "config"),
    )
    manager.save(
        step=0,
        args=ocp.args.Composite(
            state=ocp.args.StandardSave(state),
            config=ocp.args.JsonSave(config),
        ),
    )
    # Block until the filesystem operations are complete.
    manager.wait_until_finished()

    return package_path


def load(path: str | Path, *, step: int | None = None) -> dict[str, Any]:
    """
    Load a trained emulator from an Orbax checkpoint manager directory.

    The returned dictionary includes a live NNX model object and the stored
    metadata needed for inference. If no step is given, the latest checkpoint
    step is restored.

    Parameters
    ----------
    path:
        The directory path of the checkpoint to load.
    step:
        Optional specific step index to restore. Defaults to the latest.

    Returns
    -------
    dict
        A dictionary containing the reconstructed 'model', its 'params',
        'hyperparams', and the associated 'metadata'.
    """
    package_path = Path(path)

    # Use CheckpointManager to read the contents.
    manager = ocp.CheckpointManager(
        package_path,
        item_names=("state", "config"),
    )
    # Determine which step index to restore from.
    restore_step = manager.latest_step() if step is None else step
    if restore_step is None:
        raise ValueError(f"No checkpoints found in {package_path}.")

    # Read the JSON config first so we know which architecture to rebuild before restoring weights.
    config_only = manager.restore(
        restore_step,
        args=ocp.args.Composite(config=ocp.args.JsonRestore()),
    )
    config = config_only["config"]
    hyperparams = config["hyperparams"]

    def build_model() -> DenseMLP:
        """
        Build a DenseMLP with the same architecture as the saved model.
        """
        return DenseMLP(
            in_features=int(hyperparams["in_features"]),
            hidden_features=int(hyperparams["hidden_features"]),
            out_features=int(hyperparams["out_features"]),
            hidden_layers=int(hyperparams["hidden_layers"]),
            activation=hyperparams["activation"],
            init_scale=float(hyperparams.get("init_scale", 1e-1)),
            # Use a dummy key for initialization; weights will be overwritten immediately.
            rngs=nnx.Rngs(jax.random.PRNGKey(0)),
        )

    # Build an abstract model (shape-only) so Orbax knows the exact NNX state structure to restore.
    abstract_model = nnx.eval_shape(build_model)
    graphdef, abstract_state = nnx.split(abstract_model)

    # Perform the full restoration of saved state and config.
    restored = manager.restore(
        restore_step,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_state),
            config=ocp.args.JsonRestore(),
        ),
    )
    state = restored["state"]
    config = restored["config"]

    # Merge the restored weights into the reconstructed graph structure to get a live model.
    model = nnx.merge(graphdef, state)

    # Return both the live model and all the training/metadata context needed by inference.
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
