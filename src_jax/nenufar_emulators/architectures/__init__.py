"""Model architectures used by emulator workflows."""

from nenufar_emulators.architectures.mlp import DenseMLP, init_mlp

__all__ = [
    "DenseMLP",
    "init_mlp",
]
