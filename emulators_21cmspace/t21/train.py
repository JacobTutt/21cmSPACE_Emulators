"""
T21 training helpers and CLI entrypoint.

This module provides the high-level orchestration for training a T21 emulator.
It includes utilities for running synthetic smoke tests, preparing real
21cmSPACE data, executing the training loop, and saving versioned model
checkpoints. It also defines the command-line interface for the T21 workflow.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from pprint import pprint
from typing import Any

import jax
import numpy as np
from flax import nnx

from jax_emu.data_preprocessing.preparation import prepare_fixed_grid_training_split
from jax_emu.architectures.mlp import DenseMLP
from emulators_21cmspace.t21.data import (
    prepare_twentyonecmspace_t21_parameters,
    prepare_twentyonecmspace_t21_training_split,
    t21_spec,
)
from emulators_21cmspace.t21.model import t21_config
from jax_emu.utils.checkpointing import CheckpointMetadata, save
from jax_emu.training.trainer import (
    evaluate_mlp_regressor,
    train_mlp_regressor,
)


# Synthetic Validation
# --------------------
# Tools for verifying the pipeline integrity with generated data.

def run_synthetic_smoke(
    *,
    epochs: int = 20,
    batch_size: int = 64,
    prefetch_batches: int = 2,
) -> dict[str, float]:
    """
    Run a synthetic end-to-end smoke training exercise.

    The synthetic target is deliberately simple but still depends on both the
    redshift axis and the parameter vector so it exercises the same tiled-input
    path that a real global-signal emulator will use.

    Parameters
    ----------
    epochs:
        Number of training epochs for the smoke test.
    batch_size:
        Number of samples per mini-batch.
    prefetch_batches:
        Number of batches to queue on the JAX device.

    Returns
    -------
    dict
        A dictionary containing the final train and validation losses.
    """
    # Load default specs and configuration.
    spec = t21_spec()
    config = t21_config()
    rng = np.random.default_rng(1)

    # Generate mock parameters and coordinates.
    nsamples = 24
    z = np.linspace(6.0, 27.0, 30)
    raw_parameters = np.column_stack(
        [
            10 ** rng.uniform(-3.0, -1.0, size=nsamples),  # fstarII
            10 ** rng.uniform(-4.0, -2.0, size=nsamples),  # fstarIII
            rng.uniform(4.0, 50.0, size=nsamples),  # Vc
            10 ** rng.uniform(1.0, 3.0, size=nsamples),  # fX
            rng.choice(np.array([1.0, 1.3, 1.5]), size=nsamples),  # alpha
            rng.choice(np.array([*range(100, 1600, 100), 2000, 3000], dtype=float), size=nsamples),  # nu_0
            rng.uniform(10.0, 60.0, size=nsamples),  # zeta, discarded
            rng.uniform(0.03, 0.09, size=nsamples),  # tau
            10 ** rng.uniform(1.0, 4.0, size=nsamples),  # fradio
            rng.choice(np.array([231.0, 232.0, 233.0]), size=nsamples),  # pop
            np.zeros(nsamples),  # feed, discarded
            np.zeros(nsamples),  # delay, discarded
        ]
    )

    # Create mock signals based on a sinusoidal function plus parameter dependence.
    targets = np.empty((nsamples, len(z)), dtype=float)
    for idx in range(nsamples):
        # The sinusoid gives the mock signal a recognisable one-dimensional
        # structure, while the parameter sum ensures the emulator must use the
        # non-axis inputs as well.
        targets[idx] = (
            np.sin(z / 4.0)
            + 0.05 * np.log10(raw_parameters[idx, 0])
            + 0.03 * raw_parameters[idx, 7]
        )

    # Run the preparation pipeline on mock data.
    prepared = prepare_fixed_grid_training_split(
        axes=(z,),
        axis_specs=spec.axes,
        parameters=prepare_twentyonecmspace_t21_parameters(raw_parameters),
        target=targets,
        feature_scale_methods={
            "z": "zscore",
            "log10fstarII": "zscore",
            "log10fstarIII": "zscore",
            "log10Vc": "zscore",
            "log10fX": "zscore",
            "alpha": "minmax_zero_to_one",
            "nu_0": "minmax_zero_to_one",
            "tau": "zscore",
            "log10fradio": "zscore",
            "pop": "minmax_zero_to_one",
        },
        data_log=False,
        offset=None,
        random_state=1,
        shuffle_seed=1,
        # Divide linear T21 targets by one global training-label std.
        standardize_target=True,
    )

    # Initialise the MLP model.
    model = DenseMLP(
        in_features=prepared.train_features.shape[1],
        hidden_features=config.mlp.hidden_dim,
        hidden_layers=config.mlp.total_hidden_layers,
        activation=config.mlp.activation,
        rngs=nnx.Rngs(jax.random.PRNGKey(1)),
    )

    # Run a short training loop.
    _, history = train_mlp_regressor(
        model,
        prepared.train_features,
        prepared.train_targets,
        prepared.validation_features,
        prepared.validation_targets,
        epochs=epochs,
        batch_size=batch_size,
        prefetch_batches=prefetch_batches,
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        seed=1,
    )

    return {
        "final_train_loss": history.train_losses[-1],
        "final_validation_loss": history.validation_losses[-1],
    }


# Training Workflows
# ------------------
# Orchestration for training real T21 models from simulation data.

def train_t21_from_dataset_root(
    dataset_root: str,
    *,
    output_path: str | Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    prefetch_batches: int | None = None,
    shuffle_seed: int = 42,
    log_every: int | None = 1,
) -> dict[str, Any]:
    """
    Prepare, train, and save a T21 model package from 21cmSPACE data.

    Parameters
    ----------
    dataset_root:
        Path to the 21cmSPACE dataset.
    output_path:
        Optional custom path for the saved .nenemu package.
    epochs, batch_size, prefetch_batches:
        Overrides for training hyperparameters.
    shuffle_seed:
        Random seed for repeatability.
    log_every:
        Frequency for printing progress logs.

    Returns
    -------
    dict
        A summary of the training run, including loss metrics and file paths.
    """
    # 1. Prepare the datasets (splitting, resampling, scaling, flattening).
    prepared = prepare_twentyonecmspace_t21_training_split(
        dataset_root,
        shuffle_seed=shuffle_seed,
    )
    config = t21_config()

    # 2. Build the neural network architecture.
    model = DenseMLP(
        in_features=prepared.train_features.shape[1],
        hidden_features=config.mlp.hidden_dim,
        hidden_layers=config.mlp.total_hidden_layers,
        activation=config.mlp.activation,
        rngs=nnx.Rngs(jax.random.PRNGKey(shuffle_seed)),
    )

    # 3. Execute the training loop.
    model, history = train_mlp_regressor(
        model,
        prepared.train_features,
        prepared.train_targets,
        prepared.validation_features,
        prepared.validation_targets,
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        batch_size=config.training.batch_size if batch_size is None else batch_size,
        epochs=config.training.epochs if epochs is None else epochs,
        prefetch_batches=(
            config.training.prefetch_batches
            if prefetch_batches is None
            else prefetch_batches
        ),
        seed=shuffle_seed,
        early_stopping_patience=(
            config.training.early_stopping_patience if config.training.early_stop else None
        ),
        early_stopping_min_delta=config.training.early_stopping_min_delta,
        log_every=log_every,
        log_prefix="t21",
    )

    # 4. Evaluate performance on the held-out test set.
    test_loss = evaluate_mlp_regressor(
        model,
        prepared.test_features,
        prepared.test_targets,
        batch_size=config.training.batch_size if batch_size is None else batch_size,
        prefetch_batches=(
            config.training.prefetch_batches
            if prefetch_batches is None
            else prefetch_batches
        ),
    )

    # 5. Prepare and save the versioned model package (.nenemu).
    output = Path("t21_model.nenemu") if output_path is None else Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Create the metadata object required for future inference.
    metadata = CheckpointMetadata(
        model_name="t21",
        package_version=_installed_package_version(),
        emulator_spec=t21_spec(),
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
    # Write the weights and config to disk.
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

    # 6. Generate and return a training summary.
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


# CLI Entrypoint
# -------------
# Logic for parsing arguments and executing requested T21 tasks.

def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line interface for T21 development tasks.
    """
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
        help="Path to the 21cmSPACE dataset root for real T21 preparation/training.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare real 21cmSPACE arrays and print a summary without training.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to the output .nenemu checkpoint directory written after training.",
    )
    parser.add_argument("--epochs", type=int, help="Epoch count override for real training.")
    parser.add_argument("--batch-size", type=int, help="Batch size override for real training.")
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        help="Number of mini-batches to keep queued on the JAX device.",
    )
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
        help="Seed used when shuffling the fixed-grid T21 training rows after the split.",
    )
    return parser


def main() -> None:
    """
    Run the T21 command-line workflow.
    """
    args = build_parser().parse_args()

    # Route based on the provided CLI flags.
    if args.print_spec:
        pprint(t21_spec())
        return
    if args.print_config:
        pprint(t21_config())
        return
    if args.synthetic_smoke:
        epochs = 20 if args.epochs is None else args.epochs
        batch_size = 64 if args.batch_size is None else args.batch_size
        prefetch_batches = (
            t21_config().training.prefetch_batches
            if args.prefetch_batches is None
            else args.prefetch_batches
        )
        result = run_synthetic_smoke(
            epochs=epochs,
            batch_size=batch_size,
            prefetch_batches=prefetch_batches,
        )
        pprint(result)
        return
    if args.dataset_root:
        # Prepare the real dataset.
        prepared = prepare_twentyonecmspace_t21_training_split(
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
            # Print preparation summary and exit.
            pprint(summary)
            return

        # Execute training on the prepared data.
        summary = train_t21_from_dataset_root(
            args.dataset_root,
            output_path=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            prefetch_batches=args.prefetch_batches,
            shuffle_seed=args.shuffle_seed,
            log_every=args.log_every,
        )
        pprint(summary)
        return

    raise SystemExit(
        "Real T21 dataset loading is available through --dataset-root. "
        "Use --prepare-only to inspect prepared arrays, or --synthetic-smoke for the mock path."
    )


# Internal Helpers
# ----------------
# Lower-level utilities for versioning and file writing.

def _installed_package_version() -> str:
    """Return the installed package version or a development fallback."""
    try:
        return version("21cmspace-emulators")
    except PackageNotFoundError:
        return "0.1.0"


def _write_training_summary(summary_path: Path, summary: dict[str, Any]) -> None:
    """Write a human-readable JSON summary beside a saved model checkpoint."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
