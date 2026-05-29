# Examples

These examples show the installed command-line entrypoints and the matching
Python APIs for the 21cmSPACE emulator workflows.

## Available Workflows

- [Global 21-cm signal (`T21`)](examples/global-21cm.md)
- [21-cm power spectrum (`Delta21`)](examples/power-spectrum-21cm.md)

## Dataset Root

Both workflows expect a 21cmSPACE dataset root containing MATLAB `.mat` files.
Use a local path placeholder such as:

```bash
DATASET_ROOT=/path/to/21cmspace-data
```

The loader reads these files from the root:

| File | MATLAB key | Used by |
| --- | --- | --- |
| `21cmspace_z_mat.mat` | `z21cm` | T21 and Delta21 |
| `21cmspace_k_mat.mat` | `ks` | T21 loader and Delta21 |
| `21cmspace_nu_mat.mat` | `nu_keV` | T21 loader and Delta21 loader |
| `21cmspace_parameters_mat.mat` | `parameters` | T21 and Delta21 |
| `21cmspace_T21_mat.mat` | `combined_T21s` | T21 |
| `21cmspace_Deltak_mat.mat` | `combined_Deltaks` | Delta21 |

The raw parameter table has 12 columns:

```text
fstarII, fstarIII, Vc, fX, alpha, nu_0, zeta, tau, fradio, pop, feed, delay
```

Inference accepts either that raw 12-column table or the prepared 9-column
feature table used by the model.
