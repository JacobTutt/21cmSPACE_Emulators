"""Global-signal inference entrypoints."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Global-signal inference entrypoint.")
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
