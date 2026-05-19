"""T21 training helpers and CLI entrypoint."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from pprint import pprint
from typing import Any

import jax.numpy as jnp
import numpy as np

from nenufar_emulators.core.normalisation import StandardizationPipeline
from nenufar_emulators.t21.data import (
    build_t21_dataset,
    prepare_hera_idr4_t21_training_split,
    t21_spec,
)
from nenufar_emulators.t21.model import t21_config
from nenufar_emulators.serialization import CheckpointMetadata, save
from nenufar_emulators.trainer import train_mlp_dataset, train_mlp_regressor


def run_synthetic_smoke(
    *,
    epochs: int = 20,
    batch_size: int = 64,
) -> dict[str, float]:
    """Run a synthetic end-to-end smoke training exercise.

    The synthetic target is deliberately simple but still depends on both the
    redshift axis and the parameter vector so it exercises the same tiled-input
    path that a real global-signal emulator will use.
    """
    spec = t21_spec()
    config = t21_config()
    rng = np.random.default_rng(1)
    nsamples = 24
    z = np.linspace(6.0, 20.0, 20)
    parameters = np.column_stack(
        [
            10 ** rng.uniform(-3.0, -1.0, size=nsamples),  # fstarII
            10 ** rng.uniform(-4.0, -2.0, size=nsamples),  # fstarIII
            rng.uniform(4.0, 50.0, size=nsamples),  # Vc
            10 ** rng.uniform(1.0, 3.0, size=nsamples),  # fX
            rng.choice(np.array([1.0, 1.3, 1.5]), size=nsamples),  # alpha
            rng.choice(np.array([*range(100, 1600, 100), 2000, 3000], dtype=float), size=nsamples),  # nu_0
            rng.uniform(0.03, 0.09, size=nsamples),  # tau
            10 ** rng.uniform(1.0, 4.0, size=nsamples),  # fradio
            rng.choice(np.array([231.0, 232.0, 233.0]), size=nsamples),  # pop
        ]
    )

    targets = np.empty((nsamples, len(z)), dtype=float)
    for idx in range(nsamples):
        # The sinusoid gives the mock signal a recognisable one-dimensional
        # structure, while the parameter sum ensures the emulator must use the
        # non-axis inputs as well.
        targets[idx] = (
            np.sin(z / 4.0)
            + 0.05 * np.log10(parameters[idx, 0])
            + 0.03 * parameters[idx, 6]
        )

    split = int(0.8 * nsamples)
    base_train_dataset = build_t21_dataset(
        targets[:split],
        (z,),
        parameters[:split],
        spec=spec,
        tiling=False,
    )
    standardization = StandardizationPipeline.from_batch(
        base_train_dataset.as_batch(),
        standardize_axes=True,
        standardize_parameters=True,
    )
    train_dataset = build_t21_dataset(
        targets[:split],
        (z,),
        parameters[:split],
        spec=spec,
        forward_pipeline=[standardization],
        tiling=True,
    )
    validation_dataset = build_t21_dataset(
        targets[split:],
        (z,),
        parameters[split:],
        spec=spec,
        forward_pipeline=[standardization],
        tiling=True,
    )
    _, history = train_mlp_dataset(
        train_dataset,
        validation_dataset,
        hidden_features=config.mlp.hidden_dim,
        hidden_layers=config.mlp.total_hidden_layers,
        activation=config.mlp.activation,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        seed=1,
    )
    return {
        "final_train_loss": history.train_losses[-1],
        "final_validation_loss": history.validation_losses[-1],
    }


def _installed_package_version() -> str:
    """Return the installed package version or a development fallback."""
    try:
        return version("nenufar-emulators")
    except PackageNotFoundError:
        return "0.1.0"


def _write_training_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    """Write a human-readable JSON summary beside a saved model package."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))


def train_t21_from_dataset_root(
    dataset_root: str,
    *,
    output_path: str | Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    shuffle_seed: int = 42,
) -> dict[str, Any]:
    """Prepare, train, and save a T21 model package from HERA IDR4 data."""
    prepared = prepare_hera_idr4_t21_training_split(
        dataset_root,
        shuffle_seed=shuffle_seed,
    )
    config = t21_config()
    model, history = train_mlp_regressor(
        jnp.asarray(prepared.train_features),
        jnp.asarray(prepared.train_targets),
        jnp.asarray(prepared.validation_features),
        jnp.asarray(prepared.validation_targets),
        hidden_features=config.mlp.hidden_dim,
        hidden_layers=config.mlp.total_hidden_layers,
        activation=config.mlp.activation,
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        batch_size=config.training.batch_size if batch_size is None else batch_size,
        epochs=config.training.epochs if epochs is None else epochs,
        seed=shuffle_seed,
        early_stopping_patience=(
            config.training.early_stopping_patience if config.training.early_stop else None
        ),
        early_stopping_min_delta=config.training.early_stopping_min_delta,
    )

    output = Path("t21_model.nenemu") if output_path is None else Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    metadata = CheckpointMetadata(
        model_name="t21",
        package_version=_installed_package_version(),
        emulator_spec=t21_spec(),
        input_scaling=prepared.feature_scaling,
        training_config={
            "mlp": asdict(config.mlp),
            "optimizer": asdict(config.optimizer),
            "training": asdict(config.training),
            "feature_names": list(prepared.feature_names),
            "dataset_root": str(dataset_root),
            "shuffle_seed": shuffle_seed,
        },
    )
    package_path = save(
        output,
        model,
        train_losses=history.train_losses,
        val_losses=history.validation_losses,
        loss=config.training.loss_name,
        metadata=metadata,
        epochs=config.training.epochs if epochs is None else epochs,
        patience=config.training.early_stopping_patience if config.training.early_stop else None,
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )

    summary = {
        "package_path": str(package_path),
        "summary_path": str(package_path.with_suffix(".summary.json")),
        "feature_names": list(prepared.feature_names),
        "train_features_shape": list(prepared.train_features.shape),
        "train_targets_shape": list(prepared.train_targets.shape),
        "validation_features_shape": list(prepared.validation_features.shape),
        "validation_targets_shape": list(prepared.validation_targets.shape),
        "final_train_loss": history.train_losses[-1],
        "final_validation_loss": history.validation_losses[-1],
        "best_epoch": history.best_epoch,
        "best_validation_loss": history.best_validation_loss,
        "trained_model_type": type(model).__name__,
    }
    _write_training_summary(package_path.with_suffix(".summary.json"), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for T21 development tasks."""
    parser = argparse.ArgumentParser(description="T21 emulator entrypoint.")
    parser.add_argument("--print-spec", action="store_true", help="Print the default emulator spec.")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the current T21 model and training defaults.",
    )
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Run a synthetic smoke training job using generated data.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        help="Path to the HERA IDR4 dataset root for real T21 preparation/training.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare real HERA IDR4 arrays and print a summary without training.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to the output .nenemu package written after training.",
    )
    parser.add_argument("--epochs", type=int, help="Epoch count override for real training.")
    parser.add_argument("--batch-size", type=int, help="Batch size override for real training.")
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Seed used when shuffling the fixed-grid T21 training rows after the split.",
    )
    return parser


def main() -> None:
    """Run the T21 command-line workflow."""
    args = build_parser().parse_args()

    if args.print_spec:
        pprint(t21_spec())
        return
    if args.print_config:
        pprint(t21_config())
        return
    if args.synthetic_smoke:
        epochs = 20 if args.epochs is None else args.epochs
        batch_size = 64 if args.batch_size is None else args.batch_size
        result = run_synthetic_smoke(epochs=epochs, batch_size=batch_size)
        pprint(result)
        return
    if args.dataset_root:
        prepared = prepare_hera_idr4_t21_training_split(
            args.dataset_root,
            shuffle_seed=args.shuffle_seed,
        )
        summary = {
            "feature_names": prepared.feature_names,
            "train_features_shape": prepared.train_features.shape,
            "train_targets_shape": prepared.train_targets.shape,
            "validation_features_shape": prepared.validation_features.shape,
            "validation_targets_shape": prepared.validation_targets.shape,
        }
        if args.prepare_only:
            pprint(summary)
            return

        summary = train_t21_from_dataset_root(
            args.dataset_root,
            output_path=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            shuffle_seed=args.shuffle_seed,
        )
        pprint(summary)
        return

    raise SystemExit(
        "Real T21 dataset loading is available through --dataset-root. "
        "Use --prepare-only to inspect prepared arrays, or --synthetic-smoke for the mock path."
    )
