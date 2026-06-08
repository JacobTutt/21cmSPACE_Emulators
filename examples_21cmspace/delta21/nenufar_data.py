"""
NenuFAR power-spectrum data helpers.

This module loads published NenuFAR power-spectrum upper limits into the same
`PowerSpectrumData` container used by the generic likelihood code. The current
Table 4 data product gives spherical `(z, k)` points, residual power estimates,
and 2-sigma upper limits, but it does not publish a window matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jax_emu.inference.likelihood import PowerSpectrumData


DEFAULT_NENUFAR_TABLE4_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "nenufar" / "munshi_2025_table4.csv"
)


@dataclass(frozen=True)
class NenuFARPowerSpectrumDataset:
    """
    Bundled NenuFAR power-spectrum likelihood inputs.

    Parameters
    ----------
    power_data:
        Generic likelihood data. The `upper_limit` field stores the residual
        power estimate used as the one-sided likelihood threshold.
    reported_upper_limit:
        Published 2-sigma upper limit from Table 4.
    source:
        Source table path or description.
    """

    power_data: PowerSpectrumData
    reported_upper_limit: np.ndarray
    source: str


def load_nenufar_table4_dataset(
    path: str | Path = DEFAULT_NENUFAR_TABLE4_PATH,
    *,
    use_reported_upper_limit_as_threshold: bool = False,
) -> NenuFARPowerSpectrumDataset:
    """
    Load the NenuFAR Table 4 power-spectrum points.

    Parameters
    ----------
    path:
        CSV table containing `z`, `k`, residual power, and 2-sigma upper limit.
    use_reported_upper_limit_as_threshold:
        If false, use the residual power estimate as the one-sided likelihood
        threshold and derive `sigma` from the reported 2-sigma upper limit. If
        true, use the reported 2-sigma upper limit itself as the threshold.

    Returns
    -------
    NenuFARPowerSpectrumDataset
        Dataset ready for `PowerSpectrumUpperLimitLikelihood`.
    """
    table_path = Path(path)
    table = np.genfromtxt(table_path, delimiter=",", names=True, dtype=float)
    if table.ndim == 0:
        table = table.reshape(1)

    coordinates = np.column_stack(
        [
            table["z"].astype(np.float32),
            table["k_h_cMpc"].astype(np.float32),
        ]
    )
    residual_power = table["delta21_mK2"].astype(np.float32)
    reported_upper_limit = table["delta21_upper_limit_2sigma_mK2"].astype(np.float32)

    sigma = 0.5 * (reported_upper_limit - residual_power)
    if np.any(sigma <= 0.0):
        raise ValueError("NenuFAR Table 4 implies non-positive uncertainties.")

    threshold = reported_upper_limit if use_reported_upper_limit_as_threshold else residual_power
    power_data = PowerSpectrumData(
        coordinates=coordinates,
        upper_limit=threshold,
        sigma=sigma,
        window_matrix=None,
    )
    return NenuFARPowerSpectrumDataset(
        power_data=power_data,
        reported_upper_limit=reported_upper_limit,
        source=str(table_path),
    )


def nenufar_dataset_summary(dataset: NenuFARPowerSpectrumDataset) -> dict[str, Any]:
    """
    Return a compact summary of the NenuFAR likelihood arrays.
    """
    power_data = dataset.power_data
    coordinates = np.asarray(power_data.coordinates)
    return {
        "n_data_bins": int(power_data.upper_limit.shape[0]),
        "redshifts": sorted(float(value) for value in np.unique(coordinates[:, 0])),
        "k_min": float(np.min(coordinates[:, 1])),
        "k_max": float(np.max(coordinates[:, 1])),
        "source": dataset.source,
        "has_window_matrix": power_data.window_matrix is not None,
    }
