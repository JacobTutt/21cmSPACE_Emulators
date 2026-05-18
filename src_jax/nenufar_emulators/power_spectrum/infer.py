"""Power-spectrum inference entrypoints.

Inference support is intentionally not over-promised yet. This placeholder CLI
exists so the repository already has a visible place where real checkpoint
loading and spectrum prediction will eventually live.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Power-spectrum inference entrypoint.")
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print the current inference implementation status.",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    args = build_parser().parse_args()
    if args.describe:
        print(
            "Inference foundation exists at the metadata/spec level, but no "
            "production checkpoint loader or physical-data prediction path is wired yet."
        )
        return
    raise SystemExit("Use --describe. Production inference wiring is not implemented yet.")
