"""Model architectures used by emulator workflows."""

from twentyonecmspace_emulators.architectures.mlp import DenseMLP, init_mlp

__all__ = [
    "DenseMLP",
    "init_mlp",
]
