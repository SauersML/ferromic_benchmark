from __future__ import annotations

import atexit
import math
import os
import subprocess
import sys
import tempfile
import timeit
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar


T = TypeVar("T")


@dataclass
class BenchmarkRecord:
    """Simple container for benchmark metadata."""

    category: str
    size_label: str
    seconds: float
    details: str | None = None
    skipped: bool = False


BENCHMARK_RESULTS: list[BenchmarkRecord] = []
BENCHMARK_RESULTS_PRINTED = False
SKIPPED_BENCHMARKS: set[tuple[str, str]] = set()


LARGE_SCALE_LABELS = ("x1000", "x1e6", "LARGEST")
LARGEST_SCALE_LABEL = "LARGEST"
LARGEST_ENV_VAR = "RUN_LARGEST_SCALE"
LARGEST_SCALE_REASON = (
    "requires enabling RUN_LARGEST_SCALE=1 and significant CPU/RAM/disk resources"
)
MEMMAP_ROOT = Path(tempfile.gettempdir()) / "allel_docs_examples_large"


BASE_WEIR_VARIANTS = 5
BASE_WEIR_SAMPLES = 4
BASE_HAP_VARIANTS = 8
BASE_HAP_SAMPLES = 4
BASE_SEQDIVERSITY_VARIANTS = 9
BASE_SEQDIVERSITY_SAMPLES = 2
BASE_PCA_VARIANTS = 4
BASE_PCA_SAMPLES = 3


def _build_even_subpops(n_samples: int, n_subpops: int = 2) -> list[list[int]]:
    """Evenly partition sample indices into subpopulations."""

    if n_samples < n_subpops:
        raise ValueError("Number of samples must be >= number of subpopulations")

    base_size, remainder = divmod(n_samples, n_subpops)
    subpops: list[list[int]] = []
    start = 0
    for pop in range(n_subpops):
        pop_size = base_size + (1 if pop < remainder else 0)
        stop = start + pop_size
        subpops.append(list(range(start, stop)))
        start = stop
    return subpops


def _is_scale_enabled(scale_label: str) -> bool:
    if scale_label != LARGEST_SCALE_LABEL:
        return True
    value = os.environ.get(LARGEST_ENV_VAR, "").strip().lower()
    if not value:
        return True
    return value in {"1", "true", "yes", "on"}


def _ensure_scale_enabled(scale_label: str) -> None:
    if not _is_scale_enabled(scale_label):
        raise RuntimeError(
            f"Scale '{scale_label}' is disabled. {LARGEST_SCALE_REASON}."
        )


def _record_benchmark_skip(category: str, label: str, reason: str) -> None:
    key = (category, label)
    if key in SKIPPED_BENCHMARKS:
        return
    SKIPPED_BENCHMARKS.add(key)
    BENCHMARK_RESULTS.append(
        BenchmarkRecord(
            category=category,
            size_label=label,
            seconds=math.nan,
            details=f"skipped: {reason}",
            skipped=True,
        )
    )


def _memmap_path(name: str, shape: Sequence[int], dtype_str: str) -> Path:
    MEMMAP_ROOT.mkdir(parents=True, exist_ok=True)
    dims = "x".join(str(dim) for dim in shape)
    filename = f"{name}_{dims}_{dtype_str}.dat"
    return MEMMAP_ROOT / filename


def _load_or_create_memmap(
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
    initializer: Callable[["np.memmap"], None],
) -> "np.memmap":
    import numpy as np

    path = _memmap_path(name, shape, np.dtype(dtype).str)
    if not path.exists():
        mm = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
        try:
            initializer(mm)
            mm.flush()
        finally:
            del mm
    return np.memmap(path, mode="r+", dtype=dtype, shape=shape)


def _initialize_weir_large(mm: "np.memmap", *, seed: int) -> None:
    import numpy as np

    n_variants, n_samples, ploidy = mm.shape
    subpops = _build_even_subpops(n_samples, n_subpops=2)
    subpop_labels = np.empty(n_samples, dtype=np.int32)
    for label, indices in enumerate(subpops):
        subpop_labels[indices] = label

    rng = np.random.default_rng(seed)
    allele_lookup = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.int8)
    chunk_variants = max(1, min(48, n_variants))

    for start in range(0, n_variants, chunk_variants):
        stop = min(start + chunk_variants, n_variants)
        chunk = mm[start:stop]
        size = stop - start
        block_size = min(512, max(32, size))
        base_block_freqs = rng.beta(0.7, 0.7, size=(math.ceil(size / block_size),))
        base_freqs = np.repeat(base_block_freqs, block_size)[:size]
        base_freqs = np.clip(base_freqs, 0.01, 0.99)

        divergence = rng.uniform(30.0, 110.0, size=(size, len(subpops)))
        alpha = np.clip(base_freqs[:, None] * divergence, 0.5, None)
        beta = np.clip((1.0 - base_freqs)[:, None] * divergence, 0.5, None)
        pop_freqs = rng.beta(alpha, beta)

        sweep_mask = rng.random(size=(size, len(subpops))) < 0.025
        sweep_shift = rng.uniform(0.15, 0.35, size=(size, len(subpops)))
        pop_freqs = np.where(
            sweep_mask,
            np.clip(
                np.where(pop_freqs >= 0.5, pop_freqs + sweep_shift, pop_freqs - sweep_shift),
                0.005,
                0.995,
            ),
            pop_freqs,
        )

        sample_freqs = pop_freqs[:, subpop_labels]
        genotype_counts = rng.binomial(ploidy, sample_freqs)
        chunk[:] = allele_lookup[genotype_counts]
        missing_rate = rng.uniform(0.008, 0.015)
        missing_mask = rng.random((size, n_samples)) < missing_rate
        if np.any(missing_mask):
            chunk[missing_mask] = -1


def _initialize_haplotype_large(
    mm: "np.memmap", *, seed: int, n_samples_per_pop: int, max_allele: int
) -> None:
    import numpy as np

    rng = np.random.default_rng(seed)
    n_variants_total, total_samples = mm.shape
    chunk_variants = max(1, min(64, n_variants_total))
    allele_values = np.arange(max_allele, dtype=np.int8)
    n_pops = max(1, total_samples // n_samples_per_pop)

    if n_samples_per_pop * n_pops != total_samples:
        raise ValueError(
            "Total haplotype samples must be divisible by n_samples_per_pop for LARGEST scale"
        )

    def _sample_dirichlet(alpha: "np.ndarray") -> "np.ndarray":
        gammas = rng.gamma(alpha, 1.0)
        denom = np.sum(gammas, axis=-1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        return gammas / denom

    for start in range(0, n_variants_total, chunk_variants):
        stop = min(start + chunk_variants, n_variants_total)
        chunk = mm[start:stop]
        size = stop - start
        block_size = min(256, max(32, size))
        base_blocks = rng.dirichlet(np.linspace(1.5, 0.4, max_allele), size=(math.ceil(size / block_size),))
        base_weights = np.repeat(base_blocks, block_size, axis=0)[:size]

        divergence = rng.uniform(6.0, 22.0, size=(size, n_pops))
        mutational_bias = rng.beta(0.6, 0.9, size=(size, max_allele))
        mutational_bias = np.clip(mutational_bias, 0.02, 0.95)

        for pop_index in range(n_pops):
            start_col = pop_index * n_samples_per_pop
            stop_col = start_col + n_samples_per_pop
            alpha = base_weights * divergence[:, pop_index][:, None]
            alpha = np.clip(alpha + mutational_bias, 0.2, None)
            probs = _sample_dirichlet(alpha)

            cdf = np.cumsum(probs, axis=-1)
            random_values = rng.random((size, n_samples_per_pop, 1))
            indices = np.sum(random_values > cdf[:, None, :-1], axis=-1)
            chunk[:, start_col:stop_col] = allele_values[indices]
        migration_mask = rng.random((size, total_samples)) < 0.002
        if np.any(migration_mask):
            migrants = rng.integers(0, max_allele, size=np.count_nonzero(migration_mask), dtype=np.int8)
            chunk[migration_mask] = migrants
        missing_mask = rng.random((size, total_samples)) < 0.012

        if np.any(missing_mask):
            chunk[missing_mask] = -1


def _initialize_sequence_large(mm: "np.memmap", *, seed: int) -> None:
    import numpy as np

    n_variants, n_samples, ploidy = mm.shape
    rng = np.random.default_rng(seed)
    allele_lookup = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.int8)
    chunk_variants = max(1, min(64, n_variants))
    n_subpops = min(3, n_samples)
    subpops = _build_even_subpops(n_samples, n_subpops=n_subpops)
    subpop_labels = np.empty(n_samples, dtype=np.int32)
    for label, indices in enumerate(subpops):
        subpop_labels[indices] = label

    for start in range(0, n_variants, chunk_variants):
        stop = min(start + chunk_variants, n_variants)
        chunk = mm[start:stop]
        size = stop - start
        block_size = min(512, max(64, size))
        base_blocks = rng.beta(0.75, 0.75, size=(math.ceil(size / block_size),))
        base_freqs = np.repeat(base_blocks, block_size)[:size]
        base_freqs = np.clip(base_freqs, 0.005, 0.995)

        concentration = rng.uniform(35.0, 120.0, size=(size, n_subpops))
        alpha = np.clip(base_freqs[:, None] * concentration, 0.5, None)
        beta = np.clip((1.0 - base_freqs)[:, None] * concentration, 0.5, None)
        subpop_freqs = rng.beta(alpha, beta)

        adaptation_mask = rng.random((size, n_subpops)) < 0.04
        adaptation_shift = rng.uniform(0.1, 0.3, size=(size, n_subpops))
        adapted = np.where(
            subpop_freqs >= 0.5,
            np.minimum(0.999, subpop_freqs + adaptation_shift),
            np.maximum(0.001, subpop_freqs - adaptation_shift),
        )
        subpop_freqs = np.where(adaptation_mask, adapted, subpop_freqs)

        sample_freqs = subpop_freqs[:, subpop_labels]
        genotype_counts = rng.binomial(ploidy, sample_freqs)
        chunk[:] = allele_lookup[genotype_counts]
        missing_rate = rng.uniform(0.01, 0.02)
        missing_mask = rng.random((size, n_samples)) < missing_rate
        if np.any(missing_mask):
            chunk[missing_mask] = -1


def _initialize_pca_large(mm: "np.memmap", *, seed: int) -> None:
    import numpy as np

    n_variants, n_samples = mm.shape
    rng = np.random.default_rng(seed)
    chunk_variants = max(1, min(64, n_variants))

    for start in range(0, n_variants, chunk_variants):
        stop = min(start + chunk_variants, n_variants)
        chunk = mm[start:stop]
        size = stop - start
        minor_allele_freqs = rng.uniform(0.01, 0.5, size=size)
        genotype_counts = rng.binomial(2, minor_allele_freqs[:, None], size=(size, n_samples))
        chunk[:] = genotype_counts.astype(np.float32)
@lru_cache(maxsize=None)
def _simulate_weir_genotypes(scale_label: str) -> tuple[Any, list[list[int]]]:
    """Generate realistic genotype data for Weir & Cockerham Fst examples."""

    import numpy as np

    configs = {
        "x1000": (200, 100, 1000),
        "x1e6": (4000, 5000, 1_000_000),
        # Keep the "LARGEST" dataset dramatically larger than the million-scale run
        # while staying inside the CI disk limits (~44M diploid genotypes ≈ 0.08 GiB).
        LARGEST_SCALE_LABEL: (5_500, 8_000, None),


    }

    if scale_label not in configs:
        raise ValueError(f"Unknown scale label: {scale_label}")

    n_variants, n_samples, required_factor = configs[scale_label]
    if scale_label == LARGEST_SCALE_LABEL:
        _ensure_scale_enabled(scale_label)

    base_total = BASE_WEIR_VARIANTS * BASE_WEIR_SAMPLES
    simulated_total = n_variants * n_samples
    if required_factor is not None:
        expected = base_total * required_factor
        if simulated_total != expected:
            raise AssertionError("Simulated dataset does not match required scale factor")
    else:
        smaller_scale_total = configs["x1e6"][0] * configs["x1e6"][1]
        if simulated_total <= smaller_scale_total:
            raise AssertionError("Largest-scale dataset must exceed million-scale size")

    ploidy = 2
    subpops = _build_even_subpops(n_samples, n_subpops=2)

    if scale_label == LARGEST_SCALE_LABEL:
        g = _load_or_create_memmap(
            name="weir_genotypes_largest",
            shape=(n_variants, n_samples, ploidy),
            dtype=np.int8,
            initializer=lambda mm: _initialize_weir_large(mm, seed=424242),
        )
        return g, subpops

    rng = np.random.default_rng(42 if scale_label == "x1000" else 4242)
    subpop_labels = np.empty(n_samples, dtype=np.int8)
    for label, indices in enumerate(subpops):
        subpop_labels[indices] = label

    allele_lookup = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.int8)
    g = np.empty((n_variants, n_samples, ploidy), dtype=np.int8)
    missing_mask = rng.random((n_variants, n_samples)) < 0.01

    pop_freqs = rng.beta(0.8, 0.8, size=(n_variants, len(subpops)))
    pop_freqs = np.clip(pop_freqs, 0.02, 0.98)

    for variant_index in range(n_variants):
        sample_freqs = pop_freqs[variant_index, subpop_labels]
        genotype_counts = rng.binomial(ploidy, sample_freqs)
        variant_genotypes = allele_lookup[genotype_counts]
        variant_missing = missing_mask[variant_index]
        if np.any(variant_missing):
            variant_genotypes = variant_genotypes.copy()
            variant_genotypes[variant_missing, :] = -1
        g[variant_index] = variant_genotypes

    return g, subpops


@lru_cache(maxsize=None)
def _simulate_haplotype_array(scale_label: str, *, include_missing_row: bool = False) -> tuple[Any, Any]:
    """Generate haplotype data with optional missing row for divergence tests."""

    import numpy as np
    import allel

    configs = {
        "x1000": (100, None, 1000),
        "x1e6": (5000, None, 1_000_000),
        # Scale the "LARGEST" haplotype data well past the million-scale scenario
        # while remaining within CI storage budgets (~73M haplotypes ≈ 0.07 GiB).
        LARGEST_SCALE_LABEL: (5_500, 6_656, None),


    }

    if scale_label not in configs:
        raise ValueError(f"Unknown scale label: {scale_label}")

    n_samples_per_pop, explicit_variants, required_factor = configs[scale_label]
    if scale_label == LARGEST_SCALE_LABEL:
        _ensure_scale_enabled(scale_label)

    base_variants = BASE_HAP_VARIANTS + (1 if include_missing_row else 0)
    base_total = base_variants * BASE_HAP_SAMPLES

    total_samples = n_samples_per_pop * 2
    if required_factor is not None:
        required_total = base_total * required_factor
        if required_total % total_samples != 0:
            raise AssertionError("Haplotype configuration does not divide evenly")
        n_variants_total = required_total // total_samples
    else:
        if explicit_variants is None:
            raise AssertionError("Explicit variant count required for largest scale")
        n_variants_total = explicit_variants
        smaller_required_total = base_total * configs["x1e6"][2]
        if n_variants_total * total_samples <= smaller_required_total:
            raise AssertionError("Largest-scale haplotypes must exceed million-scale size")

    if scale_label == LARGEST_SCALE_LABEL:
        dtype = np.int8
        shape = (n_variants_total, total_samples)
        haplotypes = _load_or_create_memmap(
            name="haplotypes_largest",
            shape=shape,
            dtype=dtype,
            initializer=lambda mm: _initialize_haplotype_large(
                mm,
                seed=123_000,
                n_samples_per_pop=n_samples_per_pop,
                max_allele=4,
            ),
        )
        if include_missing_row:
            haplotypes[-1] = -1
        pos_step = 10
        pos = np.arange(2, 2 + pos_step * n_variants_total, pos_step, dtype=np.int32)
        return allel.HaplotypeArray(haplotypes, copy=False), pos

    rng = np.random.default_rng(123 if scale_label == "x1000" else 123_456)
    max_allele = 4
    haplotypes = np.empty((n_variants_total, total_samples), dtype=np.int8)

    limit = n_variants_total - (1 if include_missing_row else 0)
    for variant_index in range(limit):
        pop_freqs = rng.dirichlet(np.full(max_allele, 1.0), size=2)
        for pop_index in range(2):
            start = pop_index * n_samples_per_pop
            stop = start + n_samples_per_pop
            haplotypes[variant_index, start:stop] = rng.choice(
                np.arange(max_allele, dtype=np.int8),
                size=n_samples_per_pop,
                p=pop_freqs[pop_index],
            )

    if include_missing_row:
        haplotypes[-1] = -1

    missing_mask = rng.random(haplotypes.shape) < 0.01
    haplotypes[missing_mask] = -1

    pos_step = 10
    pos = np.arange(2, 2 + pos_step * n_variants_total, pos_step, dtype=np.int32)

    return allel.HaplotypeArray(haplotypes), pos


@lru_cache(maxsize=None)
def _simulate_sequence_genotypes(scale_label: str) -> tuple[Any, Any]:
    """Simulate genotype data for sequence diversity and Watterson's theta."""

    import numpy as np
    import allel

    configs = {
        "x1000": (360, 50, 1000),
        "x1e6": (6000, 3000, 1_000_000),
        # Expand the "LARGEST" sequence dataset far beyond the million-scale baseline
        # without recreating the original disk usage issue (~44M diploid calls ≈ 0.08 GiB).
        LARGEST_SCALE_LABEL: (8_000, 5_500, None),

    }

    if scale_label not in configs:
        raise ValueError(f"Unknown scale label: {scale_label}")

    n_variants, n_samples, required_factor = configs[scale_label]
    if scale_label == LARGEST_SCALE_LABEL:
        _ensure_scale_enabled(scale_label)

    base_total = BASE_SEQDIVERSITY_VARIANTS * BASE_SEQDIVERSITY_SAMPLES
    simulated_total = n_variants * n_samples
    if required_factor is not None:
        expected_total = base_total * required_factor
        if simulated_total != expected_total:
            raise AssertionError(
                "Simulated sequence dataset does not match required scale factor"
            )
    else:
        smaller_scale_total = configs["x1e6"][0] * configs["x1e6"][1]
        if simulated_total <= smaller_scale_total:
            raise AssertionError("Largest-scale sequence data must exceed million scale")

    ploidy = 2

    if scale_label == LARGEST_SCALE_LABEL:
        genotype = _load_or_create_memmap(
            name="sequence_genotypes_largest",
            shape=(n_variants, n_samples, ploidy),
            dtype=np.int8,
            initializer=lambda mm: _initialize_sequence_large(mm, seed=2025),
        )
        pos = np.arange(2, 2 + 5 * n_variants, 5, dtype=np.int32)
        return allel.GenotypeArray(genotype, copy=False), pos

    rng = np.random.default_rng(2023 if scale_label == "x1000" else 2024)
    genotype = np.empty((n_variants, n_samples, ploidy), dtype=np.int8)

    allele_lookup = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.int8)
    allele_freqs = rng.beta(0.9, 0.9, size=n_variants)
    allele_freqs = np.clip(allele_freqs, 0.01, 0.99)
    for variant_index in range(n_variants):
        genotype_counts = rng.binomial(ploidy, allele_freqs[variant_index], size=n_samples)
        genotype_variant = allele_lookup[genotype_counts]
        missing_mask = rng.random(n_samples) < 0.01
        if np.any(missing_mask):
            genotype_variant = genotype_variant.copy()
            genotype_variant[missing_mask, :] = -1
        genotype[variant_index] = genotype_variant

    pos = np.arange(2, 2 + 5 * n_variants, 5, dtype=np.int32)
    return allel.GenotypeArray(genotype), pos


@lru_cache(maxsize=None)
def _simulate_pca_matrix(scale_label: str) -> Any:
    """Simulate genotype matrices for PCA demonstrations."""

    import numpy as np

    configs = {
        "x1000": (120, 100, 1000),
        "x1e6": (4000, 3000, 1_000_000),
        # Increase the "LARGEST" PCA matrix substantially beyond the million-scale case
        # while keeping the generated matrix manageable for CI runners (~44M floats ≈ 0.17 GiB).
        LARGEST_SCALE_LABEL: (8_000, 5_500, None),

    }

    if scale_label not in configs:
        raise ValueError(f"Unknown scale label: {scale_label}")

    n_variants, n_samples, required_factor = configs[scale_label]
    if scale_label == LARGEST_SCALE_LABEL:
        _ensure_scale_enabled(scale_label)

    base_total = BASE_PCA_VARIANTS * BASE_PCA_SAMPLES
    simulated_total = n_variants * n_samples
    if required_factor is not None:
        expected = base_total * required_factor
        if simulated_total != expected:
            raise AssertionError("Simulated PCA dataset does not match required scale factor")
    else:
        smaller_total = configs["x1e6"][0] * configs["x1e6"][1]
        if simulated_total <= smaller_total:
            raise AssertionError("Largest-scale PCA matrix must exceed million scale")

    if scale_label == LARGEST_SCALE_LABEL:
        matrix = _load_or_create_memmap(
            name="pca_matrix_largest",
            shape=(n_variants, n_samples),
            dtype=np.float32,
            initializer=lambda mm: _initialize_pca_large(mm, seed=707),
        )
        return matrix

    rng = np.random.default_rng(7 if scale_label == "x1000" else 77)
    minor_allele_freqs = rng.uniform(0.01, 0.5, size=n_variants)
    matrix = np.empty((n_variants, n_samples), dtype=np.float64)
    for variant_index, maf in enumerate(minor_allele_freqs):
        geno_counts = rng.binomial(2, maf, size=n_samples)
        matrix[variant_index] = geno_counts.astype(np.float64)

    return matrix


@lru_cache(maxsize=None)
def _weir_results_cached(scale_label: str) -> tuple[Any, Any, Any]:
    import allel

    g, subpops = _simulate_weir_genotypes(scale_label)
    details = f"{g.shape[0]}x{g.shape[1]}x{g.shape[2]}"
    return benchmark_call(
        "allel.weir_cockerham_fst",
        scale_label,
        lambda: allel.weir_cockerham_fst(g, subpops),
        details=details,
    )


@lru_cache(maxsize=None)
def _hudson_results_cached(scale_label: str) -> tuple[Any, Any]:
    import allel

    g, subpops = _simulate_weir_genotypes(scale_label)
    g_array = allel.GenotypeArray(g)
    ac1 = g_array.count_alleles(subpop=subpops[0])
    ac2 = g_array.count_alleles(subpop=subpops[1])
    details = f"{ac1.shape[0]}x{ac1.shape[1]}"
    return benchmark_call(
        "allel.hudson_fst",
        scale_label,
        lambda: allel.hudson_fst(ac1, ac2),
        details=details,
    )


@lru_cache(maxsize=None)
def _mean_pairwise_difference_cached(scale_label: str) -> Any:
    import allel

    haplotypes, _ = _simulate_haplotype_array(scale_label)
    ac = haplotypes.count_alleles()
    details = f"{ac.shape[0]}x{ac.shape[1]}"
    return benchmark_call(
        "allel.mean_pairwise_difference",
        scale_label,
        lambda: allel.mean_pairwise_difference(ac),
        details=details,
    )


@lru_cache(maxsize=None)
def _mean_pairwise_difference_between_cached(scale_label: str) -> Any:
    import allel

    haplotypes, _ = _simulate_haplotype_array(scale_label)
    total_samples = haplotypes.shape[1]
    half = total_samples // 2
    ac1 = haplotypes.count_alleles(subpop=list(range(0, half)))
    ac2 = haplotypes.count_alleles(subpop=list(range(half, total_samples)))
    details = f"{ac1.shape[0]}x{ac1.shape[1]}"
    return benchmark_call(
        "allel.mean_pairwise_difference_between",
        scale_label,
        lambda: allel.mean_pairwise_difference_between(ac1, ac2),
        details=details,
    )


@lru_cache(maxsize=None)
def _sequence_divergence_cached(scale_label: str) -> float:
    import allel

    haplotypes, pos = _simulate_haplotype_array(scale_label, include_missing_row=True)
    total_samples = haplotypes.shape[1]
    half = total_samples // 2
    ac1 = haplotypes.count_alleles(subpop=list(range(0, half)))
    ac2 = haplotypes.count_alleles(subpop=list(range(half, total_samples)))
    start = 1
    stop = pos[-1] + 5
    details = f"{ac1.shape[0]}x{ac1.shape[1]}"
    return benchmark_call(
        "allel.sequence_divergence",
        scale_label,
        lambda: allel.sequence_divergence(pos, ac1, ac2, start=start, stop=stop),
        details=details,
    )


@lru_cache(maxsize=None)
def _sequence_diversity_cached(scale_label: str) -> tuple[float, float]:
    import allel

    genotype, pos = _simulate_sequence_genotypes(scale_label)
    ac = genotype.count_alleles()
    start = 1
    stop = pos[-1] + 5
    details = f"{ac.shape[0]}x{ac.shape[1]}"

    pi = benchmark_call(
        "allel.sequence_diversity",
        scale_label,
        lambda: allel.sequence_diversity(pos, ac, start=start, stop=stop),
        details=details,
    )
    theta = benchmark_call(
        "allel.watterson_theta",
        scale_label,
        lambda: allel.watterson_theta(pos, ac, start=start, stop=stop),
        details=details,
    )
    return pi, theta


@lru_cache(maxsize=None)
def _pca_results_cached(scale_label: str) -> tuple[Any, Any]:
    import allel

    gn = _simulate_pca_matrix(scale_label)
    details = f"{gn.shape[0]}x{gn.shape[1]}"
    return benchmark_call(
        "allel.pca",
        scale_label,
        lambda: allel.pca(gn, n_components=3),
        details=details,
    )


@lru_cache(maxsize=None)
def _randomized_pca_results_cached(scale_label: str) -> tuple[Any, Any]:
    import allel

    gn = _simulate_pca_matrix(scale_label)
    details = f"{gn.shape[0]}x{gn.shape[1]}"
    return benchmark_call(
        "allel.randomized_pca",
        scale_label,
        lambda: allel.randomized_pca(gn, n_components=3, random_state=0),
        details=details,
    )


def ensure_dependencies_installed() -> None:
    """Install runtime dependencies required for the documentation tests."""

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "scikit-allel",
            "scipy",
            "scikit-learn",
            "pytest",
        ],
        check=True,
    )


def benchmark_call(
    category: str,
    size_label: str,
    func: Callable[[], T],
    *,
    repeat: int = 1,
    number: int = 1,
    details: str | None = None,
) -> T:
    """Measure execution time for ``func`` and record the result."""

    if repeat < 1 or number < 1:
        raise ValueError("repeat and number must be positive integers")

    result: dict[str, T] = {}

    def runner() -> None:
        result["value"] = func()

    timer = timeit.Timer(runner)
    times = timer.repeat(repeat=repeat, number=number)
    best = min(times) / number
    BENCHMARK_RESULTS.append(
        BenchmarkRecord(category=category, size_label=size_label, seconds=best, details=details)
    )
    return result["value"]


def _print_benchmarks() -> None:
    global BENCHMARK_RESULTS_PRINTED

    if BENCHMARK_RESULTS_PRINTED or not BENCHMARK_RESULTS:
        return

    BENCHMARK_RESULTS_PRINTED = True
    print("\nBenchmark results (best of single run):")
    size_order = {label: index for index, label in enumerate(LARGE_SCALE_LABELS)}
    size_order["original"] = -1

    def _sort_key(record: BenchmarkRecord) -> tuple[str, int, str]:
        return (
            record.category,
            size_order.get(record.size_label, len(size_order)),
            record.size_label,
        )

    for record in sorted(BENCHMARK_RESULTS, key=_sort_key):
        if record.skipped:
            detail = f" ({record.details})" if record.details else ""
            print(f"{record.category} [{record.size_label}]: skipped{detail}")
            continue
        detail = f" ({record.details})" if record.details else ""
        print(f"{record.category} [{record.size_label}]: {record.seconds:.6f}s{detail}")


atexit.register(_print_benchmarks)


def _build_weir_cockerham_inputs():
    import numpy as np

    g = np.array(
        [
            [[0, 0], [0, 0], [1, 1], [1, 1]],
            [[0, 1], [0, 1], [0, 1], [0, 1]],
            [[0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 1], [1, 2], [1, 1], [2, 2]],
            [[0, 0], [1, 1], [0, 1], [-1, -1]],
        ]
    )
    subpops: Sequence[Sequence[int]] = [[0, 1], [2, 3]]
    return g, subpops


def _build_haplotype_array():
    import numpy as np
    import allel

    h = allel.HaplotypeArray(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 1, 2],
            [0, 1, 1, 2],
            [0, 1, -1, -1],
        ]
    )
    pos = np.array([2, 4, 7, 14, 15, 18, 19, 25, 27])
    return h, pos


@lru_cache(maxsize=1)
def _weir_original_results_cached() -> tuple[Any, Any, Any]:
    import allel

    g, subpops = _build_weir_cockerham_inputs()
    details = f"{g.shape[0]}x{g.shape[1]}x{g.shape[2]}"
    return benchmark_call(
        "allel.weir_cockerham_fst",
        "original",
        lambda: allel.weir_cockerham_fst(g, subpops),
        details=details,
    )


def test_weir_cockerham_fst_components():
    import numpy as np
    import allel

    g, subpops = _build_weir_cockerham_inputs()
    a, b, c = allel.weir_cockerham_fst(g, subpops)
    g, _ = _build_weir_cockerham_inputs()
    a, b, c = _weir_original_results_cached()

    expected_a = np.array(
        [
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -0.125, -0.125],
            [-0.375, -0.375, 0.0],
        ]
    )
    expected_b = np.array(
        [
            [0.0, 0.0, 0.0],
            [-0.25, -0.25, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.125, 0.25],
            [0.41666667, 0.41666667, 0.0],
        ]
    )
    expected_c = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.0, 0.0],
            [0.125, 0.25, 0.125],
            [0.16666667, 0.16666667, 0.0],
        ]
    )

    np.testing.assert_allclose(a, expected_a, rtol=0, atol=1e-8)
    np.testing.assert_allclose(b, expected_b, rtol=0, atol=1e-8)
    np.testing.assert_allclose(c, expected_c, rtol=0, atol=1e-8)

    for label in LARGE_SCALE_LABELS:
        if not _is_scale_enabled(label):
            _record_benchmark_skip(
                "allel.weir_cockerham_fst",
                label,
                LARGEST_SCALE_REASON,
            )
            continue
        g_large, subpops_large = _simulate_weir_genotypes(label)
        a_large, b_large, c_large = _weir_results_cached(label)
        assert a_large.shape[0] == g_large.shape[0]
        assert b_large.shape == a_large.shape
        assert c_large.shape == a_large.shape
        assert a_large.shape[1] >= 2
        combined = a_large + b_large + c_large
        finite_mask = np.isfinite(combined)
        assert np.count_nonzero(finite_mask) > 0
        assert any(len(pop) > 0 for pop in subpops_large)


def test_weir_cockerham_fst_variants_and_overall():
    import numpy as np
    import allel

    g, subpops = _build_weir_cockerham_inputs()
    a, b, c = allel.weir_cockerham_fst(g, subpops)
    a, b, c = _weir_original_results_cached()

    with np.errstate(divide="ignore", invalid="ignore"):
        fst = a / (a + b + c)

    expected_fst = np.array(
        [
            [1.0, 1.0, np.nan],
            [0.0, 0.0, np.nan],
            [np.nan, np.nan, np.nan],
            [0.0, -0.5, -0.5],
            [-1.8, -1.8, np.nan],
        ]
    )
    np.testing.assert_allclose(fst, expected_fst, rtol=0, atol=1e-8, equal_nan=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        fst_variant = np.sum(a, axis=1) / (np.sum(a, axis=1) + np.sum(b, axis=1) + np.sum(c, axis=1))

    expected_fst_variant = np.array([1.0, 0.0, np.nan, -0.4, -1.8])
    np.testing.assert_allclose(fst_variant, expected_fst_variant, rtol=0, atol=1e-8, equal_nan=True)

    fst_overall = np.sum(a) / (np.sum(a) + np.sum(b) + np.sum(c))
    np.testing.assert_allclose(fst_overall, -4.36809058868914e-17, rtol=0, atol=1e-24)

    for label in LARGE_SCALE_LABELS:
        if not _is_scale_enabled(label):
            _record_benchmark_skip(
                "allel.weir_cockerham_fst",
                label,
                LARGEST_SCALE_REASON,
            )
            continue
        a_large, b_large, c_large = _weir_results_cached(label)
        with np.errstate(divide="ignore", invalid="ignore"):
            fst_large = a_large / (a_large + b_large + c_large)
        assert fst_large.shape == a_large.shape
        finite_mask = np.isfinite(fst_large)
        assert np.count_nonzero(finite_mask) > 0

        with np.errstate(divide="ignore", invalid="ignore"):
            fst_variant_large = np.sum(a_large, axis=1) / (
                np.sum(a_large, axis=1) + np.sum(b_large, axis=1) + np.sum(c_large, axis=1)
            )
        assert fst_variant_large.shape[0] == a_large.shape[0]
        finite_variant = np.isfinite(fst_variant_large)
        assert np.count_nonzero(finite_variant) > 0

        denom = np.sum(a_large) + np.sum(b_large) + np.sum(c_large)
        if denom != 0:
            fst_overall_large = np.sum(a_large) / denom
            assert np.isfinite(fst_overall_large)


def test_hudson_fst_examples():
    import numpy as np
    import allel

    g = allel.GenotypeArray(
        [
            [[0, 0], [0, 0], [1, 1], [1, 1]],
            [[0, 1], [0, 1], [0, 1], [0, 1]],
            [[0, 0], [0, 0], [0, 0], [0, 0]],
            [[0, 1], [1, 2], [1, 1], [2, 2]],
            [[0, 0], [1, 1], [0, 1], [-1, -1]],
        ]
    )
    subpops = [[0, 1], [2, 3]]
    ac1 = g.count_alleles(subpop=subpops[0])
    ac2 = g.count_alleles(subpop=subpops[1])

    num, den = allel.hudson_fst(ac1, ac2)
    details = f"{ac1.shape[0]}x{ac1.shape[1]}"
    num, den = benchmark_call(
        "allel.hudson_fst",
        "original",
        lambda: allel.hudson_fst(ac1, ac2),
        details=details,
    )

    expected_num = np.array([1.0, -0.16666667, 0.0, -0.125, -0.33333333])
    expected_den = np.array([1.0, 0.5, 0.0, 0.625, 0.5])
    np.testing.assert_allclose(num, expected_num, rtol=0, atol=1e-8)
    np.testing.assert_allclose(den, expected_den, rtol=0, atol=1e-8)

    with np.errstate(divide="ignore", invalid="ignore"):
        fst = num / den

    expected_fst = np.array([1.0, -0.33333333, np.nan, -0.2, -0.66666667])
    np.testing.assert_allclose(fst, expected_fst, rtol=0, atol=1e-8, equal_nan=True)

    fst_average = np.sum(num) / np.sum(den)
    np.testing.assert_allclose(fst_average, 0.1428571428571429, rtol=0, atol=1e-12)

    for label in LARGE_SCALE_LABELS:
        if not _is_scale_enabled(label):
            _record_benchmark_skip(
                "allel.hudson_fst",
                label,
                LARGEST_SCALE_REASON,
            )
            continue
        num_large, den_large = _hudson_results_cached(label)
        assert num_large.shape == den_large.shape
        valid_mask = den_large > 0
        if np.any(valid_mask):
            fst_large = num_large[valid_mask] / den_large[valid_mask]
            assert np.all(np.isfinite(fst_large))


def test_mean_pairwise_difference():
    import numpy as np
    import allel

    h, _ = _build_haplotype_array()
    ac = h.count_alleles()
    mpd = allel.mean_pairwise_difference(ac)
    details = f"{ac.shape[0]}x{ac.shape[1]}"
    mpd = benchmark_call(
        "allel.mean_pairwise_difference",
        "original",
        lambda: allel.mean_pairwise_difference(ac),
        details=details,
    )
    expected_mpd = np.array([0.0, 0.5, 0.66666667, 0.5, 0.0, 0.83333333, 0.83333333, 1.0])
    np.testing.assert_allclose(mpd, expected_mpd, rtol=0, atol=1e-8)

    for label in LARGE_SCALE_LABELS:
        if not _is_scale_enabled(label):
            _record_benchmark_skip(
                "allel.mean_pairwise_difference",
                label,
                LARGEST_SCALE_REASON,
            )
            continue
        mpd_large = _mean_pairwise_difference_cached(label)
        assert mpd_large.ndim == 1
        assert mpd_large.size > 0
        finite = mpd_large[np.isfinite(mpd_large)]
        assert np.all(finite >= 0)


def test_sequence_diversity_and_watterson_theta():
    import numpy as np
    import allel

    g = allel.GenotypeArray(
        [
            [[0, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[0, 1], [1, 1]],
            [[1, 1], [1, 1]],
            [[0, 0], [1, 2]],
            [[0, 1], [1, 2]],
            [[0, 1], [-1, -1]],
            [[-1, -1], [-1, -1]],
        ]
    )
    ac = g.count_alleles()
    pos = np.array([2, 4, 7, 14, 15, 18, 19, 25, 27])

    pi = allel.sequence_diversity(pos, ac, start=1, stop=31)
    details = f"{ac.shape[0]}x{ac.shape[1]}"
    pi = benchmark_call(
        "allel.sequence_diversity",
        "original",
        lambda: allel.sequence_diversity(pos, ac, start=1, stop=31),
        details=details,
    )
    np.testing.assert_allclose(pi, 0.13978494623655915, rtol=0, atol=1e-12)

    theta_hat_w = allel.watterson_theta(pos, ac, start=1, stop=31)
    theta_hat_w = benchmark_call(
        "allel.watterson_theta",
        "original",
        lambda: allel.watterson_theta(pos, ac, start=1, stop=31),
        details=details,
    )
    np.testing.assert_allclose(theta_hat_w, 0.10557184750733138, rtol=0, atol=1e-12)

    for label in LARGE_SCALE_LABELS:
        if not _is_scale_enabled(label):
            _record_benchmark_skip(
                "allel.sequence_diversity",
                label,
                LARGEST_SCALE_REASON,
            )
            _record_benchmark_skip(
                "allel.watterson_theta",
                label,
                LARGEST_SCALE_REASON,
            )
            continue
        pi_large, theta_large = _sequence_diversity_cached(label)
        assert pi_large >= 0
        assert theta_large >= 0


def test_mean_pairwise_difference_between_and_sequence_divergence():
    import numpy as np
    import allel

    h, pos = _build_haplotype_array()
    ac1 = h.count_alleles(subpop=[0, 1])
    ac2 = h.count_alleles(subpop=[2, 3])

    mpd_between = allel.mean_pairwise_difference_between(ac1, ac2)
    details = f"{ac1.shape[0]}x{ac1.shape[1]}"
    mpd_between = benchmark_call(
        "allel.mean_pairwise_difference_between",
        "original",
        lambda: allel.mean_pairwise_difference_between(ac1, ac2),
        details=details,
    )
    expected_mpd_between = np.array([0.0, 0.5, 1.0, 0.5, 0.0, 1.0, 0.75, np.nan])
    np.testing.assert_allclose(mpd_between, expected_mpd_between, rtol=0, atol=1e-8, equal_nan=True)

    h_div = allel.HaplotypeArray(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 1, 2],
            [0, 1, 1, 2],
            [0, 1, -1, -1],
            [-1, -1, -1, -1],
        ]
    )
    ac1_div = h_div.count_alleles(subpop=[0, 1])
    ac2_div = h_div.count_alleles(subpop=[2, 3])

    dxy = allel.sequence_divergence(pos, ac1_div, ac2_div, start=1, stop=31)
    dxy = benchmark_call(
        "allel.sequence_divergence",
        "original",
        lambda: allel.sequence_divergence(pos, ac1_div, ac2_div, start=1, stop=31),
        details=details,
    )
    np.testing.assert_allclose(dxy, 0.12096774193548387, rtol=0, atol=1e-12)

    for label in LARGE_SCALE_LABELS:
        if not _is_scale_enabled(label):
            _record_benchmark_skip(
                "allel.mean_pairwise_difference_between",
                label,
                LARGEST_SCALE_REASON,
            )
            _record_benchmark_skip(
                "allel.sequence_divergence",
                label,
                LARGEST_SCALE_REASON,
            )
            continue
        mpd_between_large = _mean_pairwise_difference_between_cached(label)
        assert mpd_between_large.ndim == 1
        assert mpd_between_large.size > 0

        dxy_large = _sequence_divergence_cached(label)
        assert dxy_large >= 0


def _build_pca_input():
    import numpy as np

    # Simple, deterministic genotype matrix for PCA demonstrations.
    return np.array(
        [
            [0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 2.0, 1.0],
        ]
    )


def _assert_componentwise_allclose(actual, expected, *, rtol: float = 0.0, atol: float = 1e-8) -> None:
    import numpy as np

    actual_arr = np.asarray(actual)
    expected_arr = np.asarray(expected)

    if actual_arr.shape != expected_arr.shape:
        raise AssertionError(f"Shape mismatch: {actual_arr.shape} != {expected_arr.shape}")

    for idx in range(actual_arr.shape[1]):
        col = actual_arr[:, idx]
        target = expected_arr[:, idx]
        if np.allclose(col, target, rtol=rtol, atol=atol):
            continue
        if np.allclose(col, -target, rtol=rtol, atol=atol):
            continue
        raise AssertionError(f"Component {idx} differs beyond tolerances")


def test_pca_example():
    import numpy as np
    import allel

    gn = _build_pca_input()
    coords, model = allel.pca(gn, n_components=2)
    coords, model = benchmark_call(
        "allel.pca",
        "original",
        lambda: allel.pca(gn, n_components=2),
        details=f"{gn.shape[0]}x{gn.shape[1]}",
    )

    expected_coords = np.array(
        [
            [3.43125381, -0.47591733],
            [-0.91940169, 1.77614767],
            [-2.51185212, -1.30023033],
        ]
    )
    _assert_componentwise_allclose(coords, expected_coords, atol=1e-8)

    expected_variance_ratio = np.array([0.78867513, 0.21132487])
    np.testing.assert_allclose(model.explained_variance_ratio_, expected_variance_ratio, rtol=0, atol=1e-8)

    for label in LARGE_SCALE_LABELS:
        if not _is_scale_enabled(label):
            _record_benchmark_skip(
                "allel.pca",
                label,
                LARGEST_SCALE_REASON,
            )
            continue
        coords_large, model_large = _pca_results_cached(label)
        assert coords_large.shape[1] == 3
        assert model_large.explained_variance_ratio_.shape[0] == 3
        assert np.all(model_large.explained_variance_ratio_ >= 0)


def test_randomized_pca_example():
    import numpy as np
    import allel

    gn = _build_pca_input()
    coords, model = allel.randomized_pca(gn, n_components=2, random_state=0)
    coords, model = benchmark_call(
        "allel.randomized_pca",
        "original",
        lambda: allel.randomized_pca(gn, n_components=2, random_state=0),
        details=f"{gn.shape[0]}x{gn.shape[1]}",
    )

    expected_coords = np.array(
        [
            [3.43125381, -0.47591733],
            [-0.91940169, 1.77614767],
            [-2.51185212, -1.30023033],
        ]
    )
    _assert_componentwise_allclose(coords, expected_coords, atol=1e-6)

    expected_variance_ratio = np.array([0.78867513, 0.21132487])
    np.testing.assert_allclose(model.explained_variance_ratio_, expected_variance_ratio, rtol=0, atol=1e-6)

    for label in LARGE_SCALE_LABELS:
        if not _is_scale_enabled(label):
            _record_benchmark_skip(
                "allel.randomized_pca",
                label,
                LARGEST_SCALE_REASON,
            )
            continue
        coords_large, model_large = _randomized_pca_results_cached(label)
        assert coords_large.shape[1] == 3
        assert model_large.explained_variance_ratio_.shape[0] == 3
        assert np.all(model_large.explained_variance_ratio_ >= 0)


def main() -> int:
    ensure_dependencies_installed()

    import pytest  # type: ignore

    exit_code = pytest.main(["-s", __file__])
    _print_benchmarks()
    return exit_code


def pytest_sessionfinish(session, exitstatus):  # type: ignore[unused-argument]
    _print_benchmarks()


if __name__ == "__main__":
    raise SystemExit(main())
