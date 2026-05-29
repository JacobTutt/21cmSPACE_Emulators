# Research References

This repository implements JAX-native emulator infrastructure for the
21cmSPACE-style workflows currently exposed as `T21` global-signal emulation
and `Delta21` 21-cm power-spectrum emulation. The papers below are the main
research context for those observable contracts and for likely future
multi-wavelength emulator families.

## Papers

| Paper | arXiv | Contributed / used | Relevance to this repo |
| --- | --- | --- | --- |
| GLOBALEMU: A novel and robust approach for emulating the sky-averaged 21-cm signal from the cosmic dawn and epoch of reionisation | [2104.04336](https://arxiv.org/abs/2104.04336) | Introduced `globalemu`, a neural emulator for the sky-averaged 21-cm brightness temperature using redshift as an input feature together with astrophysical parameters. Also discusses emulation of neutral-fraction histories. | Direct precedent for the scalar-regression design used by `T21`: tiled redshift rows, astrophysical parameters as inputs, physical preprocessing, and one global target standard deviation. |
| Constraining the properties of Population III galaxies with multi-wavelength observations | [2312.08095](https://arxiv.org/abs/2312.08095) | Used 21cmSPACE in a Bayesian multi-wavelength analysis with HERA 21-cm power-spectrum upper limits, SARAS 3 global-signal constraints, cosmic X-ray background (CXB), and cosmic radio background (CRB). Included separate Pop II / Pop III star-formation prescriptions and emulators for 21-cm and background observables. | Matches the 21cmSPACE parameter and observable naming used here: `T21`, `Delta21`, Pop II/III star-formation parameters, X-ray efficiency, radio efficiency, and background observables. |
| Exploiting synergies between JWST and cosmic 21-cm observations to uncover star formation in the early Universe | [2503.21687](https://arxiv.org/abs/2503.21687) | Extended 21cmSPACE inference with HST/JWST UV luminosity functions (UVLFs), SARAS 3, HERA, CXB, and CRB. Added UVLFs as a 21cmSPACE output and trained emulators for global signal, power spectrum, CXB, CRB, and UVLFs. | Provides the closest research target for future multi-observable support beyond the current `T21` and `Delta21` workflows, especially UVLF and diffuse-background emulator contracts. |
| Narrowing the discovery space of the cosmological 21-cm signal using multi-wavelength constraints | [2508.13761](https://arxiv.org/abs/2508.13761) | Used multi-wavelength constraints to infer IGM properties and the allowed 21-cm discovery space. Key derived quantities include `T21,min`, `Delta21`, kinetic temperature `TK`, radio background temperature `Trad`, spin temperature `TS`, Pop II SFRD, and neutral fraction `xHI`. | Useful for naming and prioritising derived-observable outputs that are not yet implemented in this repo but are natural products of the same 21cmSPACE inference pipeline. |

## Emulator-Observable Families

- Implemented now:
  - Global 21-cm brightness temperature: `T21(z)`.
  - 21-cm dimensionless power spectrum: `Delta21(z, k)` or
    `Delta_21^2(k, z)`.

- Used in the referenced 21cmSPACE multi-wavelength analyses:
  - UV luminosity functions: `UVLF`, usually `Phi(MUV, z)`.
  - Cosmic X-ray background: `CXB`, integrated or band-limited X-ray flux.
  - Cosmic radio background: `CRB`, radio background temperature or flux.
  - Star-formation history summaries: Pop II / Pop III SFRD and SFE.
  - IGM history summaries: neutral fraction `xHI` or ionized fraction `xe`,
    kinetic temperature `TK`, spin temperature `TS`, and radio background
    temperature `Trad`.
  - Global-signal summary statistics: absorption-trough depth `T21,min`,
    trough redshift `zmin`, and corresponding observing frequency `nu_min`.

The current code should be read as the `T21` and `Delta21` subset of this
larger observable graph. Future observables should keep the same pattern:
explicit axis specs, explicit parameter transforms, target-space transforms
stored in checkpoint metadata, and inference helpers that return physical
units.
