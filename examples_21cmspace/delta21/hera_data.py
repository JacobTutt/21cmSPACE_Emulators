"""
HERA power-spectrum data helpers.

This module bridges HERA power-spectrum data products to the generic
`PowerSpectrumData` likelihood container. The old analysis code extracted each
band/field separately, applied the HERA window matrix, and evaluated a one-sided
upper-limit likelihood. The helpers here keep the same data contract while
returning JAX-ready arrays for the new inference layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from jax_emu.inference.likelihood import PowerSpectrumData


DEFAULT_HERA_IDR2_ROOT = (
    Path(__file__).resolve().parents[1] / "data" / "hera" / "observations_H1C_IDR2"
)


@dataclass(frozen=True)
class HERADataSelection:
    """
    Storage utility for one HERA band/field extraction.

    Parameters
    ----------
    path:
        HERA HDF5 file path or format string. For IDR2-style field files this
        may be `".../pspec_h1c_idr2_field{}.h5"`.
    band:
        HERA spectral window band. The IDR2 public products use bands 1 and 2.
    field:
        Field identifier used when `path` is a format string.
    kstart:
        k value used to choose the first retained bin before decimation.
    decimation_factor:
        Keep every Nth k-bin after the initial bin. The old HERA runs used 2.
    kstart_modulo:
        If true, walk backwards from `kstart` in decimation steps while the
        uncertainty remains non-zero. This matches the old likelihood helper.
    set_negative_to_zero:
        Replace negative measured power-spectrum estimates with zero before
        the upper-limit likelihood is evaluated.
    """

    path: str | Path
    band: int = 1
    field: str | int = "1"
    kstart: float = 0.0
    decimation_factor: int | None = 2
    kstart_modulo: bool = True
    set_negative_to_zero: bool = True


@dataclass(frozen=True)
class HERAObservation:
    """
    Storage utility for one extracted HERA observation.

    The model points and data bins are separate because the HERA window matrix
    maps model-side k values into data-bin space.
    """

    z: float
    k_model: np.ndarray
    upper_limit: np.ndarray
    sigma: np.ndarray
    window_matrix: np.ndarray
    k_data: np.ndarray
    source_file: str
    band: int
    field: str

    @property
    def coordinates(self) -> np.ndarray:
        """
        Return explicit `(z, k)` model coordinates for this observation.
        """
        z_column = np.full_like(self.k_model, self.z, dtype=np.float32)
        return np.column_stack([z_column, self.k_model.astype(np.float32)])


@dataclass(frozen=True)
class HERAPowerSpectrumDataset:
    """
    Bundled HERA power-spectrum likelihood inputs.
    """

    power_data: PowerSpectrumData
    observations: tuple[HERAObservation, ...]


def default_h1c_idr2_selections(
    root: str | Path = DEFAULT_HERA_IDR2_ROOT,
    *,
    field: str | int = "1",
) -> tuple[HERADataSelection, ...]:
    """
    Return the H1C IDR2 selections used by the old HERA-only examples.

    The old runs used field 1, band 1 with `kstart=0.256`, and band 2 with
    `kstart=0.192`, both decimated by a factor of two.
    """
    path = Path(root) / "pspec_h1c_idr2_field{}.h5"
    return (
        HERADataSelection(path=path, band=1, field=field, kstart=0.256),
        HERADataSelection(path=path, band=2, field=field, kstart=0.192),
    )


def load_hera_power_spectrum_dataset(
    selections: Iterable[HERADataSelection],
) -> HERAPowerSpectrumDataset:
    """
    Load and combine HERA observations into one `PowerSpectrumData` object.
    """
    observations = tuple(extract_hera_observation(selection) for selection in selections)
    return combine_hera_observations(observations)


def extract_hera_observation(selection: HERADataSelection) -> HERAObservation:
    """
    Extract one HERA observation from an HDF5 data product.

    This uses `hera_pspec` when available. That keeps the data extraction
    aligned with the old analysis code and avoids reimplementing the HERA
    cosmology and k-bin helpers by hand.
    """
    try:
        import hera_pspec as hp
    except ImportError as exc:
        raise ImportError(
            "Reading HERA HDF5 products requires `hera_pspec`. Either install "
            "hera_pspec or use `load_hera_power_spectrum_npz(...)` with an "
            "already extracted cache."
        ) from exc

    path = _format_selection_path(selection.path, selection.field)
    uvp = hp.UVPSpec()
    uvp.read_hdf5(path)

    # H1C IDR2 files store both bands in each field file. The old code selected
    # the band key by index and used the matching spectral-window index.
    band_key = uvp.get_all_keys()[selection.band - 1]
    spw_index = uvp.spw_array[selection.band - 1]

    # Convert the spectral-window frequency range into a central redshift.
    spw_frequencies = uvp.get_spw_ranges()[spw_index][:2]
    z = float(uvp.cosmo.f2z(np.mean(spw_frequencies)))

    # Pull data, diagonal errors, model k bins, and window matrix.
    k_data = np.asarray(uvp.get_kparas(spw_index), dtype=float)
    upper_limit = np.asarray(uvp.get_data(band_key)[0].real.copy(), dtype=float)
    sigma = np.sqrt(np.asarray(uvp.get_cov(band_key)[0].diagonal().real.copy(), dtype=float))
    window_matrix = np.asarray(uvp.get_window_function(band_key)[0], dtype=float)

    if selection.set_negative_to_zero:
        upper_limit = upper_limit.copy()
        upper_limit[upper_limit < 0.0] = 0.0

    # Match the old k-range cut before optional decimation.
    mask = (k_data < 1.47) & (k_data > 0.045)
    k_data = k_data[mask]
    upper_limit = upper_limit[mask]
    sigma = sigma[mask]
    window_matrix = window_matrix[mask][:, mask]

    if selection.decimation_factor is not None:
        k_data, upper_limit, sigma, window_matrix = _decimate_observation(
            k_data,
            upper_limit,
            sigma,
            window_matrix,
            kstart=selection.kstart,
            decimation_factor=selection.decimation_factor,
            kstart_modulo=selection.kstart_modulo,
        )

    # For the IDR2 workflow, model k bins are the retained data k bins after
    # masking/decimation. Later HERA products can provide a different model grid
    # through the NPZ cache route.
    return HERAObservation(
        z=z,
        k_model=k_data.astype(np.float32),
        upper_limit=upper_limit.astype(np.float32),
        sigma=sigma.astype(np.float32),
        window_matrix=window_matrix.astype(np.float32),
        k_data=k_data.astype(np.float32),
        source_file=str(path),
        band=int(selection.band),
        field=str(selection.field),
    )


def combine_hera_observations(
    observations: Iterable[HERAObservation],
) -> HERAPowerSpectrumDataset:
    """
    Combine several HERA observations into one block-window likelihood input.
    """
    observation_tuple = tuple(observations)
    if not observation_tuple:
        raise ValueError("At least one HERA observation is required.")

    coordinates = np.concatenate([obs.coordinates for obs in observation_tuple], axis=0)
    upper_limit = np.concatenate([obs.upper_limit for obs in observation_tuple], axis=0)
    sigma = np.concatenate([obs.sigma for obs in observation_tuple], axis=0)
    window_matrix = _block_diag([obs.window_matrix for obs in observation_tuple])

    power_data = PowerSpectrumData(
        coordinates=coordinates,
        upper_limit=upper_limit,
        sigma=sigma,
        window_matrix=window_matrix,
    )
    return HERAPowerSpectrumDataset(power_data=power_data, observations=observation_tuple)


def save_hera_power_spectrum_npz(
    dataset: HERAPowerSpectrumDataset,
    path: str | Path,
) -> Path:
    """
    Save extracted HERA likelihood arrays to a portable NPZ cache.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    power_data = dataset.power_data
    np.savez(
        output_path,
        coordinates=np.asarray(power_data.coordinates),
        upper_limit=np.asarray(power_data.upper_limit),
        sigma=np.asarray(power_data.sigma),
        window_matrix=np.asarray(power_data.window_matrix),
        observation_z=np.asarray([obs.z for obs in dataset.observations], dtype=np.float32),
        observation_band=np.asarray([obs.band for obs in dataset.observations], dtype=np.int32),
        observation_field=np.asarray([obs.field for obs in dataset.observations]),
        observation_source=np.asarray([obs.source_file for obs in dataset.observations]),
    )
    return output_path


def load_hera_power_spectrum_npz(path: str | Path) -> HERAPowerSpectrumDataset:
    """
    Load a portable HERA likelihood cache produced by this package.
    """
    payload = np.load(path, allow_pickle=True)
    required = ("coordinates", "upper_limit", "sigma", "window_matrix")
    missing = [name for name in required if name not in payload.files]
    if missing:
        raise ValueError(f"HERA NPZ cache is missing arrays: {missing}.")

    power_data = PowerSpectrumData(
        coordinates=payload["coordinates"],
        upper_limit=payload["upper_limit"],
        sigma=payload["sigma"],
        window_matrix=payload["window_matrix"],
    )
    return HERAPowerSpectrumDataset(power_data=power_data, observations=())


def _format_selection_path(path: str | Path, field: str | int) -> Path:
    """
    Resolve a selection path, including IDR2-style field format strings.
    """
    path_string = str(path)
    if "{" in path_string:
        return Path(path_string.format(field))
    return Path(path_string)


def _decimate_observation(
    k_data: np.ndarray,
    upper_limit: np.ndarray,
    sigma: np.ndarray,
    window_matrix: np.ndarray,
    *,
    kstart: float,
    decimation_factor: int,
    kstart_modulo: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Decimate HERA k bins in the same way as the old likelihood helper.
    """
    initial_index = int(np.argmin(np.abs(k_data - kstart)))
    if kstart_modulo:
        if sigma[initial_index] == 0:
            raise ValueError("Chosen kstart has zero uncertainty.")
        while (
            initial_index >= decimation_factor
            and sigma[initial_index - decimation_factor] != 0
        ):
            initial_index -= decimation_factor

    keep = slice(initial_index, None, decimation_factor)
    return (
        k_data[keep],
        upper_limit[keep],
        sigma[keep],
        window_matrix[keep][:, keep],
    )


def _block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    """
    Build a block-diagonal matrix without adding another dependency.
    """
    row_count = sum(block.shape[0] for block in blocks)
    column_count = sum(block.shape[1] for block in blocks)
    output = np.zeros((row_count, column_count), dtype=np.float32)

    row_start = 0
    column_start = 0
    for block in blocks:
        rows, columns = block.shape
        output[row_start : row_start + rows, column_start : column_start + columns] = block
        row_start += rows
        column_start += columns
    return output


def hera_dataset_summary(dataset: HERAPowerSpectrumDataset) -> dict[str, Any]:
    """
    Return a compact summary of the HERA likelihood arrays.
    """
    power_data = dataset.power_data
    return {
        "n_model_points": int(power_data.coordinates.shape[0]),
        "n_data_bins": int(power_data.upper_limit.shape[0]),
        "window_shape": list(np.asarray(power_data.window_matrix).shape),
        "redshifts": sorted(
            float(value) for value in np.unique(np.asarray(power_data.coordinates)[:, 0])
        ),
        "k_min": float(np.min(np.asarray(power_data.coordinates)[:, 1])),
        "k_max": float(np.max(np.asarray(power_data.coordinates)[:, 1])),
    }
