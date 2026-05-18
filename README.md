# Nenufar_Emulators

Clean JAX-native emulators for NenuFAR-era 21-cm inference work.

The repository is being built as a structured replacement for the older
`CosmicDawnSynergies` codebase, with two primary targets:

- a global-signal emulator family
- a power-spectrum emulator family

The immediate goal is to reproduce the scientifically relevant behavior of the
old code while improving:

- package structure
- documentation
- testability
- checkpoint metadata
- inference readiness
- training efficiency

The implementation plan lives in [docs/migration_plan.md](docs/migration_plan.md).
