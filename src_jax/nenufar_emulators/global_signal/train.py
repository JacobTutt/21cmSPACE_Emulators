"""Global-signal training entrypoints."""

from __future__ import annotations

import argparse
from pprint import pprint

import jax.numpy as jnp
import numpy as np

from nenufar_emulators.core.tiling import tile_spectra
from nenufar_emulators.core.training import train_mlp_regressor
from nenufar_emulators.global_signal.data import default_global_signal_spec


def run_synthetic_smoke(
    *,
    epochs: int = 20,
    batch_size: int = 64,
) -> dict[str, float]:
    """Run a synthetic end-to-end smoke training exercise."""
    spec = default_global_signal_spec()
    rng = np.random.default_rng(1)
    nsamples = 24
    z = np.linspace(6.0, 20.0, 20)
    parameters = rng.normal(size=(nsamples, len(spec.parameters)))

    targets = np.empty((nsamples, len(z)), dtype=float)
    for idx in range(nsamples):
        targets[idx] = np.sin(z / 4.0) + 0.05 * parameters[idx].sum()

    features, flat_targets, _ = tile_spectra(parameters, (z,), targets)
    split = int(0.8 * len(features))
    _, history = train_mlp_regressor(
        jnp.asarray(features[:split]),
        jnp.asarray(flat_targets[:split]),
        jnp.asarray(features[split:]),
        jnp.asarray(flat_targets[split:]),
        hidden_features=32,
        hidden_layers=2,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=5e-3,
        weight_decay=0.0,
        seed=1,
    )
    return {
        "final_train_loss": history.train_losses[-1],
        "final_validation_loss": history.validation_losses[-1],
    }


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Global-signal emulator entrypoint.")
    parser.add_argument("--print-spec", action="store_true", help="Print the default emulator spec.")
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Run a synthetic smoke training job using generated data.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for synthetic smoke.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for synthetic smoke.")
    return parser


def main() -> None:
    """CLI entrypoint."""
    args = build_parser().parse_args()
    spec = default_global_signal_spec()

    if args.print_spec:
        pprint(spec)
        return
    if args.synthetic_smoke:
        result = run_synthetic_smoke(epochs=args.epochs, batch_size=args.batch_size)
        pprint(result)
        return

    raise SystemExit(
        "Real global-signal dataset loading is not implemented yet. "
        "Use --print-spec or --synthetic-smoke for now."
    )
