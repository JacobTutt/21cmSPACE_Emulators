"""Generic primitives shared across emulator workflows.

`core` is intentionally small. It is the place for reusable data structures
and numerical helpers that do not belong specifically to HERA loading,
workflow conventions, model definition, or emulator-specific workflows.
"""

from nenufar_emulators.core.datasets import (
    NormalisationPipeline,
    SpectrumBatch,
    SpectrumDataset,
    TiledBatch,
)
from nenufar_emulators.core.scaling import FeatureScaler, FeatureScaling
from nenufar_emulators.core.specs import AxisSpec, EmulatorSpec, ParameterSpec
from nenufar_emulators.core.tiling import reconstruct_spectra, tile_spectra

__all__ = [
    "AxisSpec",
    "EmulatorSpec",
    "FeatureScaler",
    "FeatureScaling",
    "NormalisationPipeline",
    "ParameterSpec",
    "SpectrumBatch",
    "SpectrumDataset",
    "TiledBatch",
    "reconstruct_spectra",
    "tile_spectra",
]
