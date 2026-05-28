"""
21cmSPACE raw-data loaders.

This module centralizes the file names, MATLAB keys, and shared loading logic
for the 21cmSPACE emulator dataset so the rest of the package can work with
clean, typed data objects instead of repeating path fragments and MATLAB keys.
This module handles:
- loading redshift (z), wavenumber (k), and frequency axes
- loading simulation parameter tables and signal targets (T21, Delta21)
- filtering out failed simulations (NaN rows) from the training set
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


# Constants
# ---------
# Physics and dataset-specific constants used during loading.

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


# Data Containers
# ---------------
# Structured objects for holding the loaded 21cmSPACE data.

@dataclass(frozen=True)
class TwentyOneCmSpaceAxes:
    """
    Shared physical axes stored in the 21cmSPACE dataset.

    Parameters
    ----------
    z:
        Redshift coordinates.
    k:
        Wavenumber coordinates (usually converted to units of h/Mpc).
    nu_keV:
        X-ray frequency coordinates in keV.
    """

    z: np.ndarray
    k: np.ndarray
    nu_keV: np.ndarray


@dataclass(frozen=True)
class TwentyOneCmSpaceProduct:
    """
    One loaded science target plus the shared axes and parameter table.

    Parameters
    ----------
    axes:
        The physical coordinates matching the target grid.
    parameters:
        The table of simulation inputs with shape (n_sims, n_params).
    target:
        The simulation signal grids with shape (n_sims, *grid_shape).
    target_name:
        Human-readable label for the signal (e.g. 'T21').
    nan_simulation_indices:
        Indices of simulations that were dropped because they contained NaNs.
    """

    axes: TwentyOneCmSpaceAxes
    parameters: np.ndarray
    target: np.ndarray
    target_name: str
    nan_simulation_indices: np.ndarray


# Loading Workflows
# -----------------
# High-level functions for retrieving specific datasets from disk.

def load_twentyonecmspace_axes(
    dataset_root: str | Path,
    *,
    little_h: float = DIMENSIONLESS_HUBBLE_PARAMETER,
) -> TwentyOneCmSpaceAxes:
    """
    Load the shared 21cmSPACE axis arrays.

    The Delta21 workflow trains on ``k / h`` rather than on the stored raw
    ``k`` values, so the conversion is applied here once and then reused by
    the rest of the package.

    Parameters
    ----------
    dataset_root:
        The base directory containing the .mat files.
    little_h:
        The dimensionless Hubble parameter used for k-axis unit conversion.

    Returns
    -------
    TwentyOneCmSpaceAxes
        The loaded and converted axis coordinates.
    """
    root = Path(dataset_root)
    # Load each axis from its respective MATLAB file and apply necessary unit conversions.
    return TwentyOneCmSpaceAxes(
        z=_load_mat_vector(root, "21cmspace_z_mat.mat", "z21cm"),
        # Convert k from physical to h-scaled units as expected by the emulators.
        k=_load_mat_vector(root, "21cmspace_k_mat.mat", "ks") / little_h,
        nu_keV=_load_mat_vector(root, "21cmspace_nu_mat.mat", "nu_keV"),
    )


def load_twentyonecmspace_delta21(
    dataset_root: str | Path,
    *,
    little_h: float = DIMENSIONLESS_HUBBLE_PARAMETER,
) -> TwentyOneCmSpaceProduct:
    """
    Load the raw 21cmSPACE `Delta21` training inputs.

    Parameters
    ----------
    dataset_root:
        The base directory containing the .mat files.
    little_h:
        The dimensionless Hubble parameter for k-scaling.

    Returns
    -------
    TwentyOneCmSpaceProduct
        The bundled Delta21 dataset.
    """
    # Load shared axes first.
    axes = load_twentyonecmspace_axes(dataset_root, little_h=little_h)
    # Load the simulation parameter table (features).
    parameters = _load_mat_matrix(Path(dataset_root), "21cmspace_parameters_mat.mat", "parameters")
    # Load the power-spectrum grids (targets).
    target = _load_mat_matrix(Path(dataset_root), "21cmspace_Deltak_mat.mat", "combined_Deltaks")
    # Package into a clean product, filtering out any failed (NaN) simulations.
    return _package_product(axes, parameters, target, target_name="Delta21")


def load_twentyonecmspace_t21(dataset_root: str | Path) -> TwentyOneCmSpaceProduct:
    """
    Load the raw 21cmSPACE `T21` training inputs.

    Parameters
    ----------
    dataset_root:
        The base directory containing the .mat files.

    Returns
    -------
    TwentyOneCmSpaceProduct
        The bundled T21 dataset.
    """
    # Load shared axes.
    axes = load_twentyonecmspace_axes(dataset_root)
    # Load simulation parameters.
    parameters = _load_mat_matrix(Path(dataset_root), "21cmspace_parameters_mat.mat", "parameters")
    # Load the brightness temperature grids (targets).
    target = _load_mat_matrix(Path(dataset_root), "21cmspace_T21_mat.mat", "combined_T21s")
    # Package into a clean product.
    return _package_product(axes, parameters, target, target_name="T21")


# Internal Helpers
# ----------------
# Lower-level utilities for data cleaning and file I/O.

def _package_product(
    axes: TwentyOneCmSpaceAxes,
    parameters: np.ndarray,
    target: np.ndarray,
    *,
    target_name: str,
) -> TwentyOneCmSpaceProduct:
    """
    Drop NaN simulations and return a self-describing loaded product.
    """
    # Identify which simulations failed (contain NaNs) so they don't break training.
    nan_indices = nan_simulation_indices(target)
    if len(nan_indices) > 0:
        # Remove the offending rows from both parameters and targets.
        parameters = np.delete(parameters, nan_indices, axis=0)
        target = np.delete(target, nan_indices, axis=0)

    # Return the clean, bundled dataset product.
    return TwentyOneCmSpaceProduct(
        axes=axes,
        parameters=np.asarray(parameters, dtype=float),
        target=np.asarray(target, dtype=float),
        target_name=target_name,
        nan_simulation_indices=nan_indices,
    )


def nan_simulation_indices(target: np.ndarray) -> np.ndarray:
    """
    Return the simulation indices containing NaNs in a target array.

    Parameters
    ----------
    target:
        The target array to check.

    Returns
    -------
    np.ndarray
        A 1D array of integer row indices containing at least one NaN.
    """
    arr = np.asarray(target, dtype=float)
    # If no NaNs exist at all, return an empty array immediately.
    if not np.isnan(arr).any():
        return np.array([], dtype=int)
    # Find all rows that contain at least one NaN value.
    return np.unique(np.argwhere(np.isnan(arr))[:, 0]).astype(int)


def _load_mat_vector(dataset_root: Path, filename: str, key: str) -> np.ndarray:
    """
    Load one one-dimensional MATLAB array and flatten it to shape (n,).
    """
    # Access the MATLAB dictionary by key and ensure it is returned as a flat 1D float array.
    values = loadmat(dataset_root / filename)[key]
    return np.asarray(values, dtype=float).ravel()


def _load_mat_matrix(dataset_root: Path, filename: str, key: str) -> np.ndarray:
    """
    Load one MATLAB array and cast it to floating point.
    """
    # Access the MATLAB matrix and ensure it is treated as float64.
    values = loadmat(dataset_root / filename)[key]
    return np.asarray(values, dtype=float)
