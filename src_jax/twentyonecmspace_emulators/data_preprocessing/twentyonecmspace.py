"""21cmSPACE raw-data loaders.

This module centralizes the file names, MATLAB keys, and shared loading logic
for the 21cmSPACE emulator dataset so the rest of the package can work with
clean, typed data objects instead of repeating path fragments and MATLAB keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

DIMENSIONLESS_HUBBLE_PARAMETER = 0.6704

TWENTYONECMSPACE_COLUMNS = (
    "fstarII",
    "fstarIII",
    "Vc",
    "fX",
    "alpha",
    "nu_0",
    "zeta",
    "tau",
    "fradio",
    "pop",
    "feed",
    "delay",
)


@dataclass(frozen=True)
class TwentyOneCmSpaceAxes:
    """Shared physical axes stored in the 21cmSPACE dataset."""

    z: np.ndarray
    k: np.ndarray
    nu_keV: np.ndarray


@dataclass(frozen=True)
class TwentyOneCmSpaceProduct:
    """One loaded science target plus the shared axes and parameter table."""

    axes: TwentyOneCmSpaceAxes
    parameters: np.ndarray
    target: np.ndarray
    target_name: str
    nan_simulation_indices: np.ndarray


def load_twentyonecmspace_axes(
    dataset_root: str | Path,
    *,
    little_h: float = DIMENSIONLESS_HUBBLE_PARAMETER,
) -> TwentyOneCmSpaceAxes:
    """Load the shared 21cmSPACE axis arrays.

    The Delta21 workflow trains on ``k / h`` rather than on the stored raw
    ``k`` values, so the conversion is applied here once and then reused by
    the rest of the package.
    """
    root = Path(dataset_root)
    return TwentyOneCmSpaceAxes(
        z=_load_mat_vector(root, "21cmspace_z_mat.mat", "z21cm"),
        k=_load_mat_vector(root, "21cmspace_k_mat.mat", "ks") / little_h,
        nu_keV=_load_mat_vector(root, "21cmspace_nu_mat.mat", "nu_keV"),
    )


def load_twentyonecmspace_delta21(
    dataset_root: str | Path,
    *,
    little_h: float = DIMENSIONLESS_HUBBLE_PARAMETER,
) -> TwentyOneCmSpaceProduct:
    """Load the raw 21cmSPACE `Delta21` training inputs."""
    axes = load_twentyonecmspace_axes(dataset_root, little_h=little_h)
    parameters = _load_mat_matrix(Path(dataset_root), "21cmspace_parameters_mat.mat", "parameters")
    target = _load_mat_matrix(Path(dataset_root), "21cmspace_Deltak_mat.mat", "combined_Deltaks")
    return _package_product(axes, parameters, target, target_name="Delta21")


def load_twentyonecmspace_t21(dataset_root: str | Path) -> TwentyOneCmSpaceProduct:
    """Load the raw 21cmSPACE `T21` training inputs."""
    axes = load_twentyonecmspace_axes(dataset_root)
    parameters = _load_mat_matrix(Path(dataset_root), "21cmspace_parameters_mat.mat", "parameters")
    target = _load_mat_matrix(Path(dataset_root), "21cmspace_T21_mat.mat", "combined_T21s")
    return _package_product(axes, parameters, target, target_name="T21")


def _package_product(
    axes: TwentyOneCmSpaceAxes,
    parameters: np.ndarray,
    target: np.ndarray,
    *,
    target_name: str,
) -> TwentyOneCmSpaceProduct:
    """Drop NaN simulations and return a self-describing loaded product."""
    nan_indices = nan_simulation_indices(target)
    if len(nan_indices) > 0:
        parameters = np.delete(parameters, nan_indices, axis=0)
        target = np.delete(target, nan_indices, axis=0)
    return TwentyOneCmSpaceProduct(
        axes=axes,
        parameters=np.asarray(parameters, dtype=float),
        target=np.asarray(target, dtype=float),
        target_name=target_name,
        nan_simulation_indices=nan_indices,
    )


def nan_simulation_indices(target: np.ndarray) -> np.ndarray:
    """Return the simulation indices containing NaNs in a target array."""
    arr = np.asarray(target, dtype=float)
    if not np.isnan(arr).any():
        return np.array([], dtype=int)
    return np.unique(np.argwhere(np.isnan(arr))[:, 0]).astype(int)


def _load_mat_vector(dataset_root: Path, filename: str, key: str) -> np.ndarray:
    """Load one one-dimensional MATLAB array and flatten it to shape `(n,)`."""
    values = loadmat(dataset_root / filename)[key]
    return np.asarray(values, dtype=float).ravel()


def _load_mat_matrix(dataset_root: Path, filename: str, key: str) -> np.ndarray:
    """Load one MATLAB array and cast it to floating point."""
    values = loadmat(dataset_root / filename)[key]
    return np.asarray(values, dtype=float)
