# Inference

**Navigation:** [README](../README.md) · [Architecture](architecture.md) · [Preprocessing](preprocessing.md) · [JAX Training](jax-training.md) · [Checkpointing](checkpoint.md) · [Inference](inference.md) · [Examples](examples.md)

The inference layer turns a trained emulator into a likelihood evaluator. The
important point is that the emulator should be initialized on the same
coordinates as the data before sampling starts.

```text
observed redshift / k points
  -> fixed emulator
  -> likelihood(parameters)
  -> nested sampler
```

This avoids interpolation inside the sampler and keeps repeated likelihood
calls as small as possible.

The likelihood classes store the data arrays and build a jitted log-likelihood
function during initialization. After construction, each call should only pass
new astrophysical parameters through the emulator and likelihood math.

## Priors

Nested samplers usually explore a unit cube. `PriorSpec` maps that unit cube to
physical astrophysical parameters:

```python
from jax_emu.inference import DiscretePrior, LogUniformPrior, PriorSpec, UniformPrior

prior = PriorSpec([
    LogUniformPrior("fstarII", 1e-3, 0.5),
    LogUniformPrior("fstarIII", 1e-3, 0.5),
    LogUniformPrior("Vc", 4.2, 100.0),
    LogUniformPrior("fX", 1e-3, 1e3),
    DiscretePrior("alpha", [1.0, 1.3, 1.5]),
    DiscretePrior("nu_0", [100.0, 200.0, 300.0, 500.0, 1000.0, 2000.0, 3000.0]),
    UniformPrior("tau", 0.02, 0.10),
    LogUniformPrior("fradio", 1e-1, 1e5),
    DiscretePrior("pop", [231.0, 232.0, 233.0]),
])

theta = prior.transform(unit_cube)
```

For joint analyses with nuisance parameters, use grouped priors:

```python
prior = PriorSpec({
    "astro": [
        LogUniformPrior("fstarII", 1e-3, 0.5),
        LogUniformPrior("fstarIII", 1e-3, 0.5),
        LogUniformPrior("Vc", 4.2, 100.0),
        LogUniformPrior("fX", 1e-3, 1e3),
        DiscretePrior("alpha", [1.0, 1.3, 1.5]),
        DiscretePrior("nu_0", [100.0, 200.0, 300.0, 500.0, 1000.0, 2000.0, 3000.0]),
        UniformPrior("tau", 0.02, 0.10),
        LogUniformPrior("fradio", 1e-1, 1e5),
        DiscretePrior("pop", [231.0, 232.0, 233.0]),
    ],
    "foreground": [
        UniformPrior("a0", 3.54, 3.55),
        UniformPrior("a1", -0.23, -0.21),
        UniformPrior("a2", 0.0, 0.01),
        UniformPrior("a3", -0.01, 0.0),
        UniformPrior("a4", 0.0, 0.01),
        UniformPrior("a5", -0.01, 0.01),
        UniformPrior("a6", -0.01, 0.01),
    ],
    "noise": [
        LogUniformPrior("sigma", 0.01, 1.0),
    ],
})

theta = prior.transform(unit_cube)
```

This returns:

```text
theta["astro"]       -> emulator parameters
theta["foreground"]  -> foreground polynomial coefficients
theta["noise"]       -> noise nuisance parameter
```

## Global Signal

For a measured global signal, use a Gaussian likelihood:

```python
from examples_21cmspace.t21.emulator import build_t21_fixed_coordinate_emulator
from jax_emu.inference import GlobalSignalLikelihood

emulator = build_t21_fixed_coordinate_emulator(package, z_data)

likelihood = GlobalSignalLikelihood(
    emulator=emulator,
    data=t21_data,
    sigma=t21_sigma,
)

loglike = likelihood(theta)
```

The noise term must be defined in one of two ways:

```python
# Fixed noise or observational uncertainty.
likelihood = GlobalSignalLikelihood(
    emulator=emulator,
    data=t21_data,
    sigma=t21_sigma,
)

# Or sampled noise nuisance parameter.
likelihood = GlobalSignalLikelihood(
    emulator=emulator,
    data=t21_data,
)

theta["noise"] = noise_parameter
```

If neither fixed `sigma` nor `theta["noise"]` is available, the likelihood will
raise an error.

For data with a smooth foreground, use the foreground likelihood. The
foreground is a polynomial in reduced log-frequency:

```text
foreground = 10 ** sum(a_i * x**i)
```

where `x` is reduced log-frequency on `[-1, 1]`.

```python
from jax_emu.inference import GlobalSignalForegroundLikelihood

likelihood = GlobalSignalForegroundLikelihood(
    emulator=emulator,
    data=temperature_data,
    frequency=frequency_mhz,
    signal_scale=1e-3,
)

loglike = likelihood(theta)
```

The likelihood uses `theta["astro"]` for the emulator, `theta["foreground"]`
for the polynomial coefficients, and `theta["noise"]` for the noise standard
deviation, unless a fixed `sigma` is passed at initialization. Other likelihoods
in a joint run can ignore the nuisance groups and use only `theta["astro"]`.

## Power Spectrum

HERA power-spectrum data are usually upper limits. The likelihood
therefore uses a one-sided Gaussian CDF rather than a two-sided Gaussian
residual:

```text
theory below limit -> little penalty
theory above limit -> rapidly decreasing likelihood
```

If the observation supplies a window matrix, initialize the emulator on the
model coordinate points and let the likelihood apply the window. The coordinate
array is a list of explicit `(z, k)` pairs, not a rectangular grid.

For the bundled H1C IDR2 example, the loader follows the older HERA-only setup:
field 1, bands 1 and 2, the same `kstart` values, the same decimation, and the
same block window matrix used before the upper-limit likelihood is evaluated.

```python
from examples_21cmspace.delta21.emulator import build_delta21_fixed_point_emulator
from examples_21cmspace.delta21.hera_data import (
    default_h1c_idr2_selections,
    load_hera_power_spectrum_dataset,
)
from jax_emu.inference import PowerSpectrumUpperLimitLikelihood

# This reads the HERA HDF5 products and returns coordinates, limits, errors,
# and the block window matrix in one validated container.
hera_data = load_hera_power_spectrum_dataset(
    default_h1c_idr2_selections(field="1")
).power_data

emulator = build_delta21_fixed_point_emulator(
    package,
    hera_data.coordinates,
)

likelihood = PowerSpectrumUpperLimitLikelihood(
    emulator=emulator,
    upper_limit=hera_data.upper_limit,
    sigma=hera_data.sigma,
    window_matrix=hera_data.window_matrix,
    theory_fractional_error=0.2,
)

loglike = likelihood(theta)
```

Direct HDF5 extraction uses `h5py`. The loader reads the stored power-spectrum
values, covariance, spectral-window metadata, cosmology, and window matrices,
then applies the same band and k-bin selections used by the older HERA-only
workflow. You can still extract once and save a portable cache:

```bash
21cmspace-hera-infer \
  --package outputs/delta21_model.nenemu \
  --summary-only \
  --write-hera-cache outputs/hera_h1c_idr2_field1.npz
```

Then later runs can use the cache directly:

```bash
21cmspace-hera-infer \
  --package outputs/delta21_model.nenemu \
  --hera-npz outputs/hera_h1c_idr2_field1.npz \
  --output-dir outputs/hera_nested_sampling
```

For mock detections or future symmetric measurements, use
`PowerSpectrumGaussianLikelihood` instead.

## Joint Constraints

Independent datasets are combined by summing their log likelihoods:

```python
from jax_emu.inference import JointLikelihood

joint_likelihood = JointLikelihood([
    global_signal_likelihood,
    power_spectrum_likelihood,
])

loglike = joint_likelihood(theta)
terms = joint_likelihood.contributions(theta)
```

## Nested Sampling

The BlackJAX adapter keeps the sampler in unit-cube space and applies the prior
transform before each likelihood call. The main sampler sizes can be defined as
fractions or multiples of the number of sampled dimensions:

```python
import jax
from jax_emu.inference import NestedSamplingConfig, run_nested_sampling

config = NestedSamplingConfig(
    n_live_scale=25,           # n_live = n_dim * 25
    num_delete_fraction=0.2,   # replace 20% of live points per step
    num_inner_steps_scale=5,   # inner MCMC steps = n_dim * 5
    logz_live_threshold=-3.0,  # stop when live evidence is negligible
    output_dir="outputs/hera_nested_sampling",
)

result = run_nested_sampling(
    prior=prior,
    likelihood=joint_likelihood,
    key=jax.random.PRNGKey(0),
    config=config,
)
```

If exact values are needed, set `n_live`, `num_delete`, or `num_inner_steps`
directly in the config. Explicit values override the dimension-scaled versions.

During the run, BlackJAX collects dead points until:

```text
logZ_live - logZ < logz_live_threshold
```

The final dead and live points are then combined and written in an
anesthetic-friendly format:

```text
outputs/hera_nested_sampling/
  nested_sampling_results.csv
  parameter_names.json
  sampler_config.json
  test_stats.txt
```

`nested_sampling_results.csv` contains the physical parameter columns, `logL`,
and `logL_birth`. This can be read with `anesthetic.NestedSamples`.

The likelihood and emulator remain separate from the sampler, so the same
likelihood can be tested directly before launching a full nested-sampling run.

---

**Navigation:** [README](../README.md) · [Architecture](architecture.md) · [Preprocessing](preprocessing.md) · [JAX Training](jax-training.md) · [Checkpointing](checkpoint.md) · [Inference](inference.md) · [Examples](examples.md)
