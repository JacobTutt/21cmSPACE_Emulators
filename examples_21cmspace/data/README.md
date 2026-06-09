# Example Data Products

This folder contains small observation data products used by the
`examples_21cmspace` inference workflows. These files are example inputs for
the telescope-specific likelihoods, not training simulations.

## HERA H1C IDR2

```text
hera/observations_H1C_IDR2/
  pspec_h1c_idr2_field1.h5
  pspec_h1c_idr2_field2.h5
  pspec_h1c_idr2_field3.h5
```

These are HERA Phase I H1C IDR2 power-spectrum HDF5 files. They are used by:

```text
examples_21cmspace/delta21/hera_data.py
examples_21cmspace/delta21/hera_inference.py
```

The loader reads the HDF5 arrays directly, extracts the selected field and
band, applies the same k-bin selection and decimation used by the older
HERA-only workflow, and returns:

- explicit model coordinates, `(z, k)`
- upper-limit data values
- diagonal uncertainties
- the HERA window matrix

The HERA likelihood then evaluates the emulator at the model-side coordinates
and applies the window matrix before computing the one-sided upper-limit
likelihood.

Citation/provenance:

- HERA Phase I limits data-access note:
  <https://reionization.org/manual_uploads/Accessing_HERA_PhaseI_Limits_External.html>
- HERA H1C IDR2 power-spectrum memo:
  <https://reionization.org/manual_uploads/HERA086_H1C_IDR2_power_spectrum_notes.html>

## NenuFAR Table 4

```text
nenufar/
  munshi_2025_table4.csv
```

This CSV stores the Table 4 power-spectrum points from Munshi et al. (2025),
`arXiv:2507.10533`. The columns are:

```text
z,k_h_cMpc,delta21_mK2,delta21_upper_limit_2sigma_mK2
```

They are used by:

```text
examples_21cmspace/delta21/nenufar_data.py
examples_21cmspace/delta21/nenufar_inference.py
```

The table provides spherical `(z, k)` points, residual power estimates, and
reported 2-sigma upper limits. The paper does not publish a window matrix with
Table 4, so the example likelihood evaluates the emulator directly at the
tabulated `(z, k)` coordinates.

Citation/provenance:

- Munshi et al. (2025), *Improved upper limits on the 21-cm signal power
  spectrum at z=17.0 and z=20.3 from an optimal field observed with NenuFAR*:
  <https://arxiv.org/abs/2507.10533>
