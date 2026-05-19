"""Global-signal training entrypoints.

Like the power-spectrum CLI, this module is currently focused on verification
and transparency rather than on pretending the production training workflow is
already complete.
"""

from __future__ import annotations

import argparse
from pprint import pprint

import jax.numpy as jnp
import numpy as np

from nenufar_emulators.core.normalisation import StandardizationPipeline
from nenufar_emulators.core.training import train_mlp_dataset, train_mlp_regressor
from nenufar_emulators.global_signal.data import (
    build_global_signal_dataset,
    default_global_signal_spec,
    prepare_hera_idr4_t21_training_split,
)
from nenufar_emulators.global_signal.model import t21_frad_legacy_bundle


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
    spec = default_global_signal_spec()
    bundle = t21_frad_legacy_bundle()
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
    base_train_dataset = build_global_signal_dataset(
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
    train_dataset = build_global_signal_dataset(
        targets[:split],
        (z,),
        parameters[:split],
        spec=spec,
        forward_pipeline=[standardization],
        tiling=True,
    )
    validation_dataset = build_global_signal_dataset(
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
        hidden_features=bundle.mlp.hidden_dim,
        hidden_layers=bundle.mlp.total_hidden_layers,
        activation=bundle.mlp.activation,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=bundle.optimizer.learning_rate,
        weight_decay=bundle.optimizer.weight_decay,
        seed=1,
    )
    return {
        "final_train_loss": history.train_losses[-1],
        "final_validation_loss": history.validation_losses[-1],
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for global-signal development tasks."""
    parser = argparse.ArgumentParser(description="Global-signal emulator entrypoint.")
    parser.add_argument("--print-spec", action="store_true", help="Print the default emulator spec.")
    parser.add_argument(
        "--print-legacy-config",
        action="store_true",
        help="Print the legacy-aligned model and training defaults.",
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
    """Run the current global-signal CLI.

    At this stage the command is mainly for visibility and verification: it can
    print contracts, print legacy defaults, and run a smoke test, but it does
    not yet train on the real science datasets.
    """
    args = build_parser().parse_args()

    if args.print_spec:
        pprint(default_global_signal_spec())
        return
    if args.print_legacy_config:
        pprint(t21_frad_legacy_bundle())
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

        bundle = t21_frad_legacy_bundle()
        model, history = train_mlp_regressor(
            jnp.asarray(prepared.train_features),
            jnp.asarray(prepared.train_targets),
            jnp.asarray(prepared.validation_features),
            jnp.asarray(prepared.validation_targets),
            hidden_features=bundle.mlp.hidden_dim,
            hidden_layers=bundle.mlp.total_hidden_layers,
            activation=bundle.mlp.activation,
            learning_rate=bundle.optimizer.learning_rate,
            weight_decay=bundle.optimizer.weight_decay,
            batch_size=bundle.training.batch_size if args.batch_size is None else args.batch_size,
            epochs=bundle.training.epochs if args.epochs is None else args.epochs,
            seed=args.shuffle_seed,
            early_stopping_patience=(
                bundle.training.early_stopping_patience if bundle.training.early_stop else None
            ),
            early_stopping_min_delta=bundle.training.early_stopping_min_delta,
        )
        pprint(
            {
                **summary,
                "final_train_loss": history.train_losses[-1],
                "final_validation_loss": history.validation_losses[-1],
                "trained_model_type": type(model).__name__,
            }
        )
        return

    raise SystemExit(
        "Real global-signal dataset loading is available through --dataset-root. "
        "Use --prepare-only to inspect prepared arrays, or --synthetic-smoke for the mock path."
    )
