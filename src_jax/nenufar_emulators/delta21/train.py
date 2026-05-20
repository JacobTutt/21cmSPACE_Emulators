"""Delta21 training helpers and CLI entrypoint."""

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
from nenufar_emulators.delta21.data import (
    build_delta21_dataset,
    delta21_spec,
    prepare_hera_idr4_delta21_training_split,
)
from nenufar_emulators.delta21.model import delta21_config
from nenufar_emulators.serialization import CheckpointMetadata, save
from nenufar_emulators.trainer import (
    evaluate_mlp_regressor,
    train_mlp_dataset,
    train_mlp_regressor,
)


def run_synthetic_smoke(
    *,
    epochs: int = 20,
    batch_size: int = 64,
) -> dict[str, float]:
    """Run a small synthetic end-to-end smoke training exercise.

    This does not attempt to mimic the real science signal faithfully. Its job
    is to verify that the Delta21 spec, tiling logic, and workflow config are
    internally consistent.
    """
    spec = delta21_spec()
    config = delta21_config()
    rng = np.random.default_rng(0)
    nsamples = 24
    z = np.linspace(6.0, 16.0, 5)
    k = np.geomspace(0.05, 0.5, 6)
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

    # Build a positive target in physical space. The spec pipeline attached by
    # the dataset will later apply the configured log10(target + 1) transform.
    zz, kk = np.meshgrid(z, k, indexing="ij")
    base_signal = (zz + 1.0) * (kk + 0.5)
    targets = np.empty((nsamples, len(z), len(k)), dtype=float)
    for idx in range(nsamples):
        targets[idx] = base_signal + 0.02 * np.log10(parameters[idx, 0]) + 0.03 * parameters[idx, 6]

    split = int(0.8 * nsamples)
    base_train_dataset = build_delta21_dataset(
        targets[:split],
        (z, k),
        parameters[:split],
        spec=spec,
        tiling=False,
    )
    standardization = StandardizationPipeline.from_batch(
        base_train_dataset.as_batch(),
        standardize_axes=True,
        standardize_parameters=True,
    )
    train_dataset = build_delta21_dataset(
        targets[:split],
        (z, k),
        parameters[:split],
        spec=spec,
        forward_pipeline=[standardization],
        tiling=True,
    )
    validation_dataset = build_delta21_dataset(
        targets[split:],
        (z, k),
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
        seed=0,
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


def train_delta21_from_dataset_root(
    dataset_root: str,
    *,
    output_path: str | Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    shuffle_seed: int = 42,
    log_every: int | None = 1,
) -> dict[str, Any]:
    """Prepare, train, and save a Delta21 model package from HERA IDR4 data."""
    prepared = prepare_hera_idr4_delta21_training_split(
        dataset_root,
        shuffle_seed=shuffle_seed,
    )
    config = delta21_config()
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
        log_every=log_every,
        log_prefix="delta21",
    )
    test_loss = evaluate_mlp_regressor(
        model,
        jnp.asarray(prepared.test_features),
        jnp.asarray(prepared.test_targets),
        batch_size=config.training.batch_size if batch_size is None else batch_size,
    )

    output = Path("delta21_model.nenemu") if output_path is None else Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    metadata = CheckpointMetadata(
        model_name="delta21",
        package_version=_installed_package_version(),
        emulator_spec=delta21_spec(),
        input_scaling=prepared.feature_scaling,
        target_scaling=prepared.target_scaling,
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
        "test_features_shape": list(prepared.test_features.shape),
        "test_targets_shape": list(prepared.test_targets.shape),
        "final_train_loss": history.train_losses[-1],
        "final_validation_loss": history.validation_losses[-1],
        "best_epoch": history.best_epoch,
        "best_validation_loss": history.best_validation_loss,
        "test_loss": test_loss,
        "trained_model_type": type(model).__name__,
    }
    _write_training_summary(package_path.with_suffix(".summary.json"), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for Delta21 development tasks.

    The CLI is intentionally narrow for now. It exists to expose inspection and
    verification tasks while the training path continues to be refined.
    """
    parser = argparse.ArgumentParser(description="Delta21 emulator entrypoint.")
    parser.add_argument("--print-spec", action="store_true", help="Print the default emulator spec.")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the current Delta21 model and training defaults.",
    )
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Run a synthetic smoke training job using generated data.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        help="Path to the HERA IDR4 dataset root for real Delta21 preparation/training.",
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
        "--log-every",
        type=int,
        default=1,
        help="Print training and validation losses every N epochs.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="Seed used when shuffling the fixed-grid Delta21 training rows after the split.",
    )
    return parser


def main() -> None:
    """Run the Delta21 command-line workflow."""
    args = build_parser().parse_args()

    if args.print_spec:
        pprint(delta21_spec())
        return
    if args.print_config:
        pprint(delta21_config())
        return
    if args.synthetic_smoke:
        epochs = 20 if args.epochs is None else args.epochs
        batch_size = 64 if args.batch_size is None else args.batch_size
        result = run_synthetic_smoke(epochs=epochs, batch_size=batch_size)
        pprint(result)
        return
    if args.dataset_root:
        prepared = prepare_hera_idr4_delta21_training_split(
            args.dataset_root,
            shuffle_seed=args.shuffle_seed,
        )
        summary = {
            "feature_names": prepared.feature_names,
            "train_features_shape": prepared.train_features.shape,
            "train_targets_shape": prepared.train_targets.shape,
            "validation_features_shape": prepared.validation_features.shape,
            "validation_targets_shape": prepared.validation_targets.shape,
            "test_features_shape": prepared.test_features.shape,
            "test_targets_shape": prepared.test_targets.shape,
        }
        if args.prepare_only:
            pprint(summary)
            return

        summary = train_delta21_from_dataset_root(
            args.dataset_root,
            output_path=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            shuffle_seed=args.shuffle_seed,
            log_every=args.log_every,
        )
        pprint(summary)
        return

    raise SystemExit(
        "Real Delta21 dataset loading is available through --dataset-root. "
        "Use --prepare-only to inspect prepared arrays, or --synthetic-smoke for the mock path."
    )
