from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat

from emulators_21cmspace.delta21.data import delta21_low_z_sampled_axes, delta21_spec
from emulators_21cmspace.delta21.infer import (
    build_delta21_emulator,
    build_delta21_fixed_grid_emulator,
    load_delta21_package,
)
from emulators_21cmspace.t21.infer import (
    build_t21_emulator,
    build_t21_fixed_grid_emulator,
    load_t21_package,
)


BASE = Path("/rds/user/jlt67/hpc-work/PhD/NenuFAR")
DATA = BASE / "HERA_IDR4_Emulator_Data_21cmspace_names"
OUT = BASE / "runs/inference_benchmarks"

T21_MODEL = BASE / "runs/t21_cubic_gelu64_exponential_lr_2000_continue/t21_model.nenemu"
DELTA21_MODEL = BASE / "runs/delta21_cubic_gelu160_exponential_lr_1000/delta21_model.nenemu"


def timer() -> float:
    return time.perf_counter()


def timed_call(fn: Callable[[], jax.Array]) -> tuple[float, jax.Array]:
    start = timer()
    result = fn()
    result.block_until_ready()
    return timer() - start, result


def repeated_eval(
    fn: Callable[[], jax.Array],
    *,
    repeats: int,
) -> tuple[float, float]:
    times = []
    for _ in range(repeats):
        seconds, _ = timed_call(fn)
        times.append(seconds)
    return float(np.mean(times)), float(np.std(times))


def load_raw_parameters(n: int) -> jax.Array:
    parameters = loadmat(DATA / "21cmspace_parameters_mat.mat")["parameters"][:n]
    return jnp.asarray(parameters, dtype=jnp.float32)


def delta21_axes() -> tuple[jax.Array, jax.Array]:
    z_grid, logk_grid = delta21_low_z_sampled_axes(delta21_spec().axes)
    return (
        jnp.asarray(z_grid, dtype=jnp.float32),
        jnp.asarray(np.power(10.0, logk_grid), dtype=jnp.float32),
    )


def benchmark_t21(batch_size: int) -> dict[str, object]:
    parameters = load_raw_parameters(batch_size)
    z = jnp.linspace(6.0, 27.0, 200, dtype=jnp.float32)
    package = load_t21_package(T21_MODEL)

    dynamic = build_t21_emulator(package)
    dynamic_first, dynamic_output = timed_call(lambda: dynamic.forward_model(parameters, z))
    dynamic_mean, dynamic_std = repeated_eval(
        lambda: dynamic.forward_model(parameters, z),
        repeats=10,
    )

    fixed_init_start = timer()
    fixed = build_t21_fixed_grid_emulator(package, z)
    fixed_init_seconds = timer() - fixed_init_start
    fixed_first, fixed_output = timed_call(lambda: fixed.emulate(parameters))
    fixed_mean, fixed_std = repeated_eval(lambda: fixed.emulate(parameters), repeats=10)

    max_abs_diff = float(
        jnp.max(jnp.abs(dynamic_output - fixed_output)).block_until_ready()
    )

    return {
        "model": "t21_cubic_gelu64_exponential_lr_2000_continue",
        "batch_size": batch_size,
        "grid_shape": [int(z.shape[0])],
        "dynamic_first_call_seconds": dynamic_first,
        "dynamic_eval_mean_seconds": dynamic_mean,
        "dynamic_eval_std_seconds": dynamic_std,
        "fixed_grid_init_seconds": fixed_init_seconds,
        "fixed_first_call_seconds": fixed_first,
        "fixed_eval_mean_seconds": fixed_mean,
        "fixed_eval_std_seconds": fixed_std,
        "max_abs_difference_dynamic_vs_fixed": max_abs_diff,
    }


def benchmark_delta21(batch_size: int) -> dict[str, object]:
    parameters = load_raw_parameters(batch_size)
    z, k = delta21_axes()
    package = load_delta21_package(DELTA21_MODEL)

    dynamic = build_delta21_emulator(package)
    dynamic_first, dynamic_output = timed_call(lambda: dynamic.forward_model(parameters, z, k))
    dynamic_mean, dynamic_std = repeated_eval(
        lambda: dynamic.forward_model(parameters, z, k),
        repeats=5,
    )

    fixed_init_start = timer()
    fixed = build_delta21_fixed_grid_emulator(package, z, k)
    fixed_init_seconds = timer() - fixed_init_start
    fixed_first, fixed_output = timed_call(lambda: fixed.emulate(parameters))
    fixed_mean, fixed_std = repeated_eval(lambda: fixed.emulate(parameters), repeats=5)

    max_abs_diff = float(
        jnp.max(jnp.abs(dynamic_output - fixed_output)).block_until_ready()
    )

    return {
        "model": "delta21_cubic_gelu160_exponential_lr_1000",
        "batch_size": batch_size,
        "grid_shape": [int(z.shape[0]), int(k.shape[0])],
        "dynamic_first_call_seconds": dynamic_first,
        "dynamic_eval_mean_seconds": dynamic_mean,
        "dynamic_eval_std_seconds": dynamic_std,
        "fixed_grid_init_seconds": fixed_init_seconds,
        "fixed_first_call_seconds": fixed_first,
        "fixed_eval_mean_seconds": fixed_mean,
        "fixed_eval_std_seconds": fixed_std,
        "max_abs_difference_dynamic_vs_fixed": max_abs_diff,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("JAX devices:", jax.devices(), flush=True)

    results = {
        "t21": [benchmark_t21(batch_size) for batch_size in (1, 128)],
        "delta21": [benchmark_delta21(batch_size) for batch_size in (1, 64)],
    }

    output_path = OUT / "gelu_fixed_grid_inference_benchmark.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    print(f"Wrote benchmark to {output_path}", flush=True)


if __name__ == "__main__":
    main()
