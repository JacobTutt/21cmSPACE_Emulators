"""Power-spectrum training entrypoints.

At the moment these entrypoints are intentionally modest: they expose the
specifications and synthetic smoke runs needed to verify the repository before
real datasets are wired in.
"""

from __future__ import annotations

import argparse
from pprint import pprint

import jax.numpy as jnp
import numpy as np

from nenufar_emulators.core.normalisation import StandardizationPipeline
from nenufar_emulators.core.training import train_mlp_dataset, train_mlp_regressor
from nenufar_emulators.power_spectrum.data import (
    build_power_spectrum_dataset,
    default_power_spectrum_spec,
    prepare_hera_idr4_delta21_training_split,
)
from nenufar_emulators.power_spectrum.model import delta21_frad_legacy_bundle


def run_synthetic_smoke(
    *,
    epochs: int = 20,
    batch_size: int = 64,
) -> dict[str, float]:
    """Run a small synthetic end-to-end smoke training exercise.

    This does not attempt to mimic the real science signal faithfully. Its job
    is to verify that the power-spectrum spec, tiling logic, and legacy-aligned
    architecture bundle are internally consistent.
    """
    spec = default_power_spectrum_spec()
    bundle = delta21_frad_legacy_bundle()
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
    # the dataset will later apply the legacy log10(target + 1) transform.
    zz, kk = np.meshgrid(z, k, indexing="ij")
    base_signal = (zz + 1.0) * (kk + 0.5)
    targets = np.empty((nsamples, len(z), len(k)), dtype=float)
    for idx in range(nsamples):
        targets[idx] = base_signal + 0.02 * np.log10(parameters[idx, 0]) + 0.03 * parameters[idx, 6]

    split = int(0.8 * nsamples)
    base_train_dataset = build_power_spectrum_dataset(
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
    train_dataset = build_power_spectrum_dataset(
        targets[:split],
        (z, k),
        parameters[:split],
        spec=spec,
        forward_pipeline=[standardization],
        tiling=True,
    )
    validation_dataset = build_power_spectrum_dataset(
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
        hidden_features=bundle.mlp.hidden_dim,
        hidden_layers=bundle.mlp.total_hidden_layers,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=bundle.optimizer.learning_rate,
        weight_decay=bundle.optimizer.weight_decay,
        seed=0,
    )
    return {
        "final_train_loss": history.train_losses[-1],
        "final_validation_loss": history.validation_losses[-1],
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for power-spectrum development tasks.

    The CLI is intentionally narrow for now. It exists to expose inspection and
    verification tasks while the real dataset-driven training path is still
    being migrated.
    """
    parser = argparse.ArgumentParser(description="Power-spectrum emulator entrypoint.")
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
        help="Path to the HERA IDR4 dataset root for real Delta21 preparation/training.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare real HERA IDR4 arrays and print a summary without training.",
    )
    parser.add_argument("--epochs", type=int, help="Epoch count override for real training.")
    parser.add_argument("--batch-size", type=int, help="Batch size override for real training.")
    parser.add_argument(
        "--interpolation-seed",
        type=int,
        default=0,
        help="Seed used when sampling interpolation points for the old-style training rows.",
    )
    return parser


def main() -> None:
    """CLI entrypoint.

    The hard failure on the default path is deliberate. Until real dataset
    loading exists, the command should say plainly what it can do today rather
    than pretending to support a production workflow.
    """
    args = build_parser().parse_args()

    if args.print_spec:
        pprint(default_power_spectrum_spec())
        return
    if args.print_legacy_config:
        pprint(delta21_frad_legacy_bundle())
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
            interpolation_seed=args.interpolation_seed,
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

        bundle = delta21_frad_legacy_bundle()
        model, history = train_mlp_regressor(
            jnp.asarray(prepared.train_features),
            jnp.asarray(prepared.train_targets),
            jnp.asarray(prepared.validation_features),
            jnp.asarray(prepared.validation_targets),
            hidden_features=bundle.mlp.hidden_dim,
            hidden_layers=bundle.mlp.total_hidden_layers,
            learning_rate=bundle.optimizer.learning_rate,
            weight_decay=bundle.optimizer.weight_decay,
            batch_size=bundle.training.batch_size if args.batch_size is None else args.batch_size,
            epochs=bundle.training.epochs if args.epochs is None else args.epochs,
            seed=args.interpolation_seed,
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
        "Real power-spectrum dataset loading is available through --dataset-root. "
        "Use --prepare-only to inspect prepared arrays, or --synthetic-smoke for the mock path."
    )
