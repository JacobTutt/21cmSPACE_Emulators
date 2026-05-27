"""HERA IDR4 raw-data loaders.

This module centralizes the file names, MATLAB keys, and shared loading logic
for the HERA IDR4 emulator dataset so the rest of the package can work with
clean, typed data objects instead of repeating path fragments and MATLAB keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERA_LITTLE_H = 0.6704

HERA_IDR4_COLUMNS = (
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
class HeraIdr4Axes:
    """Shared physical axes stored in the HERA IDR4 dataset."""

    z: np.ndarray
    k: np.ndarray
    nu_keV: np.ndarray


@dataclass(frozen=True)
class HeraIdr4Product:
    """One loaded science target plus the shared axes and parameter table."""

    axes: HeraIdr4Axes
    parameters: np.ndarray
    target: np.ndarray
    target_name: str
    nan_simulation_indices: np.ndarray


def load_hera_idr4_axes(dataset_root: str | Path, *, little_h: float = HERA_LITTLE_H) -> HeraIdr4Axes:
    """Load the shared HERA IDR4 axis arrays.

    The Delta21 workflow trains on ``k / h`` rather than on the stored raw
    ``k`` values, so the conversion is applied here once and then reused by
    the rest of the package.
    """
    root = Path(dataset_root)
    return HeraIdr4Axes(
        z=_load_mat_vector(root, "hera_z_mat.mat", "z21cm"),
        k=_load_mat_vector(root, "hera_k_mat.mat", "ks") / little_h,
        nu_keV=_load_mat_vector(root, "hera_nu_mat.mat", "nu_keV"),
    )


def load_hera_idr4_delta21(
    dataset_root: str | Path,
    *,
    little_h: float = HERA_LITTLE_H,
) -> HeraIdr4Product:
    """Load the raw HERA IDR4 `Delta21` training inputs."""
    axes = load_hera_idr4_axes(dataset_root, little_h=little_h)
    parameters = _load_mat_matrix(Path(dataset_root), "hera_parameters_mat.mat", "parameters")
    target = _load_mat_matrix(Path(dataset_root), "hera_Deltak_mat.mat", "combined_Deltaks")
    return _package_product(axes, parameters, target, target_name="Delta21")


def load_hera_idr4_t21(dataset_root: str | Path) -> HeraIdr4Product:
    """Load the raw HERA IDR4 `T21` training inputs."""
    axes = load_hera_idr4_axes(dataset_root)
    parameters = _load_mat_matrix(Path(dataset_root), "hera_parameters_mat.mat", "parameters")
    target = _load_mat_matrix(Path(dataset_root), "hera_T21_mat.mat", "combined_T21s")
    return _package_product(axes, parameters, target, target_name="T21")


def _package_product(
    axes: HeraIdr4Axes,
    parameters: np.ndarray,
    target: np.ndarray,
    *,
    target_name: str,
) -> HeraIdr4Product:
    """Drop NaN simulations and return a self-describing loaded product."""
    nan_indices = nan_simulation_indices(target)
    if len(nan_indices) > 0:
        parameters = np.delete(parameters, nan_indices, axis=0)
        target = np.delete(target, nan_indices, axis=0)
    return HeraIdr4Product(
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
