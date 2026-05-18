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

from nenufar_emulators.core.tiling import tile_spectra
from nenufar_emulators.core.training import train_mlp_regressor
from nenufar_emulators.global_signal.data import default_global_signal_spec
from nenufar_emulators.global_signal.model import t21_arad_legacy_bundle


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
    bundle = t21_arad_legacy_bundle()
    rng = np.random.default_rng(1)
    nsamples = 24
    z = np.linspace(6.0, 20.0, 20)
    parameters = rng.normal(size=(nsamples, len(spec.parameters)))

    targets = np.empty((nsamples, len(z)), dtype=float)
    for idx in range(nsamples):
        # The sinusoid gives the mock signal a recognisable one-dimensional
        # structure, while the parameter sum ensures the emulator must use the
        # non-axis inputs as well.
        targets[idx] = np.sin(z / 4.0) + 0.05 * parameters[idx].sum()

    features, flat_targets, _ = tile_spectra(parameters, (z,), targets)
    split = int(0.8 * len(features))
    _, history = train_mlp_regressor(
        jnp.asarray(features[:split]),
        jnp.asarray(flat_targets[:split]),
        jnp.asarray(features[split:]),
        jnp.asarray(flat_targets[split:]),
        hidden_features=bundle.mlp.hidden_dim,
        hidden_layers=bundle.mlp.total_hidden_layers,
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
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for synthetic smoke.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for synthetic smoke.")
    return parser


def main() -> None:
    """Run the current global-signal CLI.

    At this stage the command is mainly for visibility and verification: it can
    print contracts, print legacy defaults, and run a smoke test, but it does
    not yet train on the real science datasets.
    """
    args = build_parser().parse_args()
    spec = default_global_signal_spec()

    if args.print_spec:
        pprint(spec)
        return
    if args.print_legacy_config:
        pprint(t21_arad_legacy_bundle())
        return
    if args.synthetic_smoke:
        result = run_synthetic_smoke(epochs=args.epochs, batch_size=args.batch_size)
        pprint(result)
        return

    raise SystemExit(
        "Real global-signal dataset loading is not implemented yet. "
        "Use --print-spec or --synthetic-smoke for now."
    )
