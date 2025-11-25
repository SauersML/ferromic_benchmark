from __future__ import annotations

import argparse
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

    library: str
    category: str
    size_label: str
    seconds: float
    details: str | None = None
    skipped: bool = False
    result_repr: str | None = None


BENCHMARK_RESULTS: list[BenchmarkRecord] = []
BENCHMARK_RESULTS_PRINTED = False

TSV_PATHS = {
    "scikit-allel": Path("benchmark-results-scikit-allel.tsv"),
    "ferromic": Path("benchmark-results-ferromic.tsv"),
}
LIBRARY_ALIASES = {
    "allel": "scikit-allel",
    "scikit-allel": "scikit-allel",
    "ferromic": "ferromic",
}
DEFAULT_LIBRARIES: tuple[str, ...] = tuple(TSV_PATHS.keys())
ENABLED_LIBRARIES: set[str] = set(DEFAULT_LIBRARIES)
_TSV_HEADER = (
    "library",
    "benchmark",
    "size_label",
    "seconds",
    "result",
    "details",
    "skipped",
)


def _stringify_benchmark_result(value: Any) -> str:
    """Create a stable string representation of a benchmark result."""

    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - defensive: numpy may be unavailable
        np = None  # type: ignore

    def _convert(obj: Any) -> str:
        if np is not None:
            if isinstance(obj, np.ndarray):  # type: ignore[attr-defined]
                return repr(obj.tolist())
            if isinstance(obj, np.generic):  # type: ignore[attr-defined]
                return repr(obj.item())
        if isinstance(obj, dict):
            items = []
            for key, val in sorted(obj.items(), key=lambda item: repr(item[0])):
                items.append(f"{_convert(key)}: {_convert(val)}")
            return "{" + ", ".join(items) + "}"
        if isinstance(obj, list):
            return "[" + ", ".join(_convert(item) for item in obj) + "]"
        if isinstance(obj, tuple):
            inner = ", ".join(_convert(item) for item in obj)
            if len(obj) == 1:
                inner += ","
            return "(" + inner + ")"
        if isinstance(obj, (set, frozenset)):
            parts = sorted(_convert(item) for item in obj)
            return "{" + ", ".join(parts) + "}"
        return repr(obj)

    return _convert(value)


def _sanitize_tsv_field(text: str) -> str:
    """Escape control characters that conflict with TSV formatting."""

    return text.replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")


def _sorted_benchmark_records() -> list[BenchmarkRecord]:
    size_order = {label: index for index, label in enumerate(LARGE_SCALE_LABELS)}
    size_order["original"] = -1

    def _sort_key(record: BenchmarkRecord) -> tuple[str, str, int, str]:
        return (
            record.library,
            record.category,
            size_order.get(record.size_label, len(size_order)),
            record.size_label,
        )

    return sorted(BENCHMARK_RESULTS, key=_sort_key)


def _tsv_path_for_library(library: str) -> Path:
    normalized = _normalize_library_name(library)
    return TSV_PATHS.get(normalized, Path(f"benchmark-results-{normalized}.tsv"))


def _normalize_library_name(library: str) -> str:
    return LIBRARY_ALIASES.get(library, library)


def set_enabled_libraries(libraries: Sequence[str] | None) -> None:
    """Configure which libraries should run during this benchmark session."""

    global ENABLED_LIBRARIES

    if not libraries:
        ENABLED_LIBRARIES = set(DEFAULT_LIBRARIES)
        return

    normalized: set[str] = set()
    for entry in libraries:
        normalized.add(_normalize_library_name(entry))

    unknown = sorted(name for name in normalized if name not in TSV_PATHS)
    if unknown:
        raise ValueError(f"Unknown libraries requested: {', '.join(unknown)}")

    ENABLED_LIBRARIES = normalized


def is_library_enabled(library: str) -> bool:
    return _normalize_library_name(library) in ENABLED_LIBRARIES


def _write_benchmark_results_tsv() -> None:
    """Persist benchmark metadata and outputs for CI artifact collection."""

    records_by_path: dict[Path, list[str]] = {}

    if BENCHMARK_RESULTS:
        for record in _sorted_benchmark_records():
            seconds_str = ""
            if not record.skipped and not math.isnan(record.seconds):
                seconds_str = f"{record.seconds:.9f}"

            result_field = record.result_repr or ""
            details_field = record.details or ""
            skipped_field = "true" if record.skipped else "false"

            line_values = [
                record.library,
                str(record.category),
                str(record.size_label),
                seconds_str,
                result_field,
                details_field,
                skipped_field,
            ]
            sanitized = [_sanitize_tsv_field(value) for value in line_values]

            path = _tsv_path_for_library(record.library)
            records_by_path.setdefault(path, ["\t".join(_TSV_HEADER)]).append(
                "\t".join(sanitized)
            )

    all_paths = {path for path in TSV_PATHS.values()}
    all_paths.update(records_by_path.keys())

    for path in all_paths:
        lines = records_by_path.get(path, ["\t".join(_TSV_HEADER)])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


LARGEST_SCALE_LABEL = "LARGEST"
LARGEST_WIDE_SCALE_LABEL = f"{LARGEST_SCALE_LABEL}-wide"
LARGE_SCALE_LABELS = (
    "medium",
    "big",
    LARGEST_SCALE_LABEL,
    LARGEST_WIDE_SCALE_LABEL,
)
LARGEST_SCALE_ENV_VAR = "RUN_LARGEST_SCALE"
LARGEST_SCALE_REASON = (
    "requires significant CPU/RAM/disk resources; set "
    f"{LARGEST_SCALE_ENV_VAR}=1 to enable"
)
FERROMIC_BIG_ENV_VAR = "RUN_FERROMIC_BIG_SCALE"
FERROMIC_BIG_REASON = "requires enabling RUN_FERROMIC_BIG_SCALE=1 due to ferromic runtime"
MEMMAP_ROOT = Path(tempfile.gettempdir()) / "allel_docs_examples_large"


BASE_WEIR_VARIANTS = 5
BASE_WEIR_SAMPLES = 4
BASE_HAP_VARIANTS = 8
BASE_HAP_SAMPLES = 4
BASE_SEQDIVERSITY_VARIANTS = 9
BASE_SEQDIVERSITY_SAMPLES = 2
BASE_PCA_VARIANTS = 4
BASE_PCA_SAMPLES = 3


AVERAGE_FST_BLOCK_LENGTHS: dict[str, int] = {
    "medium": 10,
    "big": 64,
    LARGEST_SCALE_LABEL: 96,
    LARGEST_WIDE_SCALE_LABEL: 112,
}


def _scale_family(scale_label: str) -> str:
    """Return the base family for the provided ``scale_label``."""

    if scale_label.startswith("big-"):
        return "big"
    if scale_label.startswith(f"{LARGEST_SCALE_LABEL}-"):
        return LARGEST_SCALE_LABEL
    return scale_label


def _requires_ferromic_opt_in(scale_label: str) -> bool:
    return _scale_family(scale_label) == "big"


def _is_ferromic_big_scale_enabled() -> bool:
    value = os.environ.get(FERROMIC_BIG_ENV_VAR, "").strip().lower()
    if not value:
        return False
    return value in {"1", "true", "yes", "on"}


def _scale_skip_reason(scale_label: str) -> str:
    family = _scale_family(scale_label)
    if family == "big":
        return FERROMIC_BIG_REASON
    if family == LARGEST_SCALE_LABEL:
        return LARGEST_SCALE_REASON
    return ""


def _block_length_for_average_fst(scale_label: str) -> int:
    try:
        return AVERAGE_FST_BLOCK_LENGTHS[scale_label]
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise ValueError(f"Unknown scale label for block length: {scale_label}") from exc


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
    family = _scale_family(scale_label)
    if family != LARGEST_SCALE_LABEL:
        return True

    value = os.environ.get(LARGEST_SCALE_ENV_VAR, "").strip().lower()
    if not value:
        return False

    return value in {"1", "true", "yes", "on"}


def _ensure_scale_enabled(scale_label: str) -> None:
    if not _is_scale_enabled(scale_label):
        raise RuntimeError(
            f"Scale '{scale_label}' is disabled. {LARGEST_SCALE_REASON}."
        )


def _enabled_scale_labels() -> tuple[str, ...]:
    return tuple(label for label in LARGE_SCALE_LABELS if _is_scale_enabled(label))


def _is_library_scale_enabled(library: str, scale_label: str) -> bool:
    if not _is_scale_enabled(scale_label):
        return False
    normalized = _normalize_library_name(library)
    if _requires_ferromic_opt_in(scale_label):
        if normalized == "ferromic":
            return _is_ferromic_big_scale_enabled()
        if normalized == "scikit-allel" and "ferromic" in ENABLED_LIBRARIES:
            return _is_ferromic_big_scale_enabled()
    return True


def _enabled_scale_labels_for_library(library: str) -> tuple[str, ...]:
    normalized = _normalize_library_name(library)
    return tuple(
        label
        for label in _enabled_scale_labels()
        if _is_library_scale_enabled(normalized, label)
    )


def _record_scale_skip(
    *,
    library: str,
    category: str,
    size_label: str,
    details: str | None = None,
    reason: str | None = None,
) -> None:
    normalized = _normalize_library_name(library)
    combined_details: str | None = None
    if details or reason:
        parts = [part for part in (details, reason) if part]
        combined_details = "; ".join(parts)
    BENCHMARK_RESULTS.append(
        BenchmarkRecord(
            library=normalized,
            category=category,
            size_label=size_label,
            seconds=math.nan,
            details=combined_details,
            skipped=True,
        )
    )


def _maybe_call(value: Any) -> Any:
    """Return ``value()`` when callable, otherwise ``value``."""

    return value() if callable(value) else value


def _ferromic_sample_names(population: Any) -> Sequence[str]:
    """Access sample names from ``ferromic`` populations across API versions."""

    sample_names = _maybe_call(getattr(population, "sample_names"))
    if not hasattr(sample_names, "__len__"):
        sample_names = list(sample_names)
    return sample_names


def _ferromic_haplotypes(population: Any) -> Sequence[Any]:
    """Access haplotypes from ``ferromic`` populations across API versions."""

    haplotypes = _maybe_call(getattr(population, "haplotypes"))
    if not hasattr(haplotypes, "__len__"):
        haplotypes = list(haplotypes)
    return haplotypes


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
        "medium": (100, 100, 500),
        "big": (240, 800, 9_600),
        # Right-size the "LARGEST" datasets to stay comfortably beyond the
        # million-scale configuration while remaining well below the memory
        # limits of resource-constrained CI runners.
        LARGEST_SCALE_LABEL: (320, 1_000, None),
        LARGEST_WIDE_SCALE_LABEL: (288, 1_056, None),
    }

    if scale_label not in configs:
        raise ValueError(f"Unknown scale label: {scale_label}")

    n_variants, n_samples, required_factor = configs[scale_label]
    family = _scale_family(scale_label)
    if family == LARGEST_SCALE_LABEL:
        _ensure_scale_enabled(scale_label)

    base_total = BASE_WEIR_VARIANTS * BASE_WEIR_SAMPLES
    simulated_total = n_variants * n_samples
    if required_factor is not None:
        expected = base_total * required_factor
        if simulated_total != expected:
            raise AssertionError("Simulated dataset does not match required scale factor")
    else:
        smaller_totals = [
            configs[label][0] * configs[label][1]
            for label in configs
            if _scale_family(label) == "big"
        ]
        smaller_scale_total = max(smaller_totals)
        if simulated_total <= smaller_scale_total:
            raise AssertionError("Largest-scale dataset must exceed million-scale size")

    ploidy = 2
    subpops = _build_even_subpops(n_samples, n_subpops=2)

    memmap_label = scale_label.lower().replace("-", "_")
    if family == LARGEST_SCALE_LABEL:
        g = _load_or_create_memmap(
            name=f"weir_genotypes_{memmap_label}",
            shape=(n_variants, n_samples, ploidy),
            dtype=np.int8,
            initializer=lambda mm: _initialize_weir_large(
                mm,
                seed=424_242 + sum(ord(ch) for ch in scale_label),
            ),
        )
        return g, subpops

    seeds = {
        "medium": 42,
        "big": 4_242,
    }
    rng = np.random.default_rng(seeds.get(scale_label, 4242))
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

    configs = {
        "medium": (100, None, 500),
        "big": (600, None, 9_600),
        # Scale the "LARGEST" haplotype data to remain beyond the million-scale
        # scenario while keeping the overall Python object count small enough to
        # avoid exhausting CI memory limits.
        LARGEST_SCALE_LABEL: (800, 320, None),
        LARGEST_WIDE_SCALE_LABEL: (720, 360, None),
    }

    if scale_label not in configs:
        raise ValueError(f"Unknown scale label: {scale_label}")

    n_samples_per_pop, explicit_variants, required_factor = configs[scale_label]
    family = _scale_family(scale_label)
    if family == LARGEST_SCALE_LABEL:
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
        smaller_required_total = base_total * configs["big"][2]
        minimum_variants = smaller_required_total // total_samples + 1
        n_variants_total = max(explicit_variants, minimum_variants)

    memmap_label = scale_label.lower().replace("-", "_")
    if family == LARGEST_SCALE_LABEL:
        dtype = np.int8
        shape = (n_variants_total, total_samples)
        haplotypes = _load_or_create_memmap(
            name=f"haplotypes_{memmap_label}",
            shape=shape,
            dtype=dtype,
            initializer=lambda mm: _initialize_haplotype_large(
                mm,
                seed=123_000 + sum(ord(ch) for ch in scale_label),
                n_samples_per_pop=n_samples_per_pop,
                max_allele=4,
            ),
        )
        if include_missing_row:
            haplotypes[-1] = -1
        pos_step = 10
        pos = np.arange(2, 2 + pos_step * n_variants_total, pos_step, dtype=np.int32)
        return haplotypes, pos

    seeds = {
        "medium": 123,
        "big": 123_456,
    }
    rng = np.random.default_rng(seeds.get(scale_label, 123_456))
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

    return haplotypes, pos


@lru_cache(maxsize=None)
def _simulate_sequence_genotypes(scale_label: str) -> tuple[Any, Any]:
    """Simulate genotype data for sequence diversity and Watterson's theta."""

    import numpy as np

    configs = {
        "medium": (180, 50, 500),
        "big": (360, 600, 12_000),
        # Expand the "LARGEST" sequence dataset just beyond the million-scale
        # baseline while keeping the underlying arrays compact enough for
        # low-memory CI environments.
        LARGEST_SCALE_LABEL: (420, 640, None),
        LARGEST_WIDE_SCALE_LABEL: (384, 672, None),

    }

    if scale_label not in configs:
        raise ValueError(f"Unknown scale label: {scale_label}")

    n_variants, n_samples, required_factor = configs[scale_label]
    family = _scale_family(scale_label)
    if family == LARGEST_SCALE_LABEL:
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
        smaller_totals = [
            configs[label][0] * configs[label][1]
            for label in configs
            if _scale_family(label) == "big"
        ]
        smaller_scale_total = max(smaller_totals)
        if simulated_total <= smaller_scale_total:
            raise AssertionError("Largest-scale sequence data must exceed million scale")

    ploidy = 2

    memmap_label = scale_label.lower().replace("-", "_")
    if family == LARGEST_SCALE_LABEL:
        genotype = _load_or_create_memmap(
            name=f"sequence_genotypes_{memmap_label}",
            shape=(n_variants, n_samples, ploidy),
            dtype=np.int8,
            initializer=lambda mm: _initialize_sequence_large(
                mm,
                seed=2_025 + sum(ord(ch) for ch in scale_label),
            ),
        )
        pos = np.arange(2, 2 + 5 * n_variants, 5, dtype=np.int32)
        return genotype, pos

    seeds = {
        "medium": 2_023,
        "big": 2_024,
    }
    rng = np.random.default_rng(seeds.get(scale_label, 2_024))
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
    return genotype, pos


@lru_cache(maxsize=None)
def _simulate_pca_matrix(scale_label: str) -> Any:
    """Simulate genotype matrices for PCA demonstrations."""

    import numpy as np

    configs = {
        "medium": (60, 100, 500),
        "big": (240, 600, 12_000),
        # Grow the "LARGEST" PCA matrices modestly beyond the million-scale
        # configuration while ensuring the float32 arrays remain lightweight for
        # constrained CI systems.
        LARGEST_SCALE_LABEL: (288, 672, None),
        LARGEST_WIDE_SCALE_LABEL: (256, 720, None),

    }

    if scale_label not in configs:
        raise ValueError(f"Unknown scale label: {scale_label}")

    n_variants, n_samples, required_factor = configs[scale_label]
    family = _scale_family(scale_label)
    if family == LARGEST_SCALE_LABEL:
        _ensure_scale_enabled(scale_label)

    base_total = BASE_PCA_VARIANTS * BASE_PCA_SAMPLES
    simulated_total = n_variants * n_samples
    if required_factor is not None:
        expected = base_total * required_factor
        if simulated_total != expected:
            raise AssertionError("Simulated PCA dataset does not match required scale factor")
    else:
        smaller_total = max(
            configs[label][0] * configs[label][1]
            for label in configs
            if _scale_family(label) == "big"
        )
        if simulated_total <= smaller_total:
            raise AssertionError("Largest-scale PCA matrix must exceed million scale")

    memmap_label = scale_label.lower().replace("-", "_")
    if family == LARGEST_SCALE_LABEL:
        matrix = _load_or_create_memmap(
            name=f"pca_matrix_{memmap_label}",
            shape=(n_variants, n_samples),
            dtype=np.float32,
            initializer=lambda mm: _initialize_pca_large(
                mm,
                seed=707 + sum(ord(ch) for ch in scale_label),
            ),
        )
        return matrix

    seeds = {
        "medium": 7,
        "big": 77,
    }
    rng = np.random.default_rng(seeds.get(scale_label, 77))
    minor_allele_freqs = rng.uniform(0.01, 0.5, size=n_variants)
    matrix = np.empty((n_variants, n_samples), dtype=np.float64)
    for variant_index, maf in enumerate(minor_allele_freqs):
        geno_counts = rng.binomial(2, maf, size=n_samples)
        matrix[variant_index] = geno_counts.astype(np.float64)

    return matrix


FERROMIC_WEIR_INPUTS: dict[str, dict[str, Any]] = {}
FERROMIC_HAPLOTYPE_INPUTS: dict[tuple[str, bool], dict[str, Any]] = {}
FERROMIC_SEQUENCE_INPUTS: dict[str, dict[str, Any]] = {}
FERROMIC_PCA_INPUTS: dict[str, dict[str, Any]] = {}


def _build_sample_names(n_samples: int) -> list[str]:
    return [f"sample_{index}" for index in range(n_samples)]


def _build_ferromic_diploid_variants(genotypes: Any, positions: Any) -> list[dict[str, Any]]:
    return [
        {"position": int(pos), "genotypes": _diploid_variant_genotypes(genotypes[idx])}
        for idx, pos in enumerate(positions)
    ]


def _build_ferromic_haploid_variants(haplotypes: Any, positions: Any) -> list[dict[str, Any]]:
    return [
        {"position": int(pos), "genotypes": _haploid_variant_genotypes(haplotypes[idx])}
        for idx, pos in enumerate(positions)
    ]


def _diploid_variant_genotypes(genotype_slice: "np.ndarray") -> list[Any]:
    import numpy as np

    missing_mask = np.any(genotype_slice < 0, axis=1)
    clipped = np.clip(genotype_slice, 0, None).astype(np.uint8, copy=False)

    entries = np.empty(genotype_slice.shape[0], dtype=object)
    entries[missing_mask] = None
    if entries.size:
        non_missing = clipped[~missing_mask]
        if non_missing.size:
            converted = np.empty(non_missing.shape[0], dtype=object)
            for idx, row in enumerate(non_missing):
                converted[idx] = (int(row[0]), int(row[1]))
            entries[~missing_mask] = converted

    return entries.tolist()


def _haploid_variant_genotypes(haplotype_slice: "np.ndarray") -> list[Any]:
    import numpy as np

    missing_mask = haplotype_slice < 0
    alleles = np.clip(haplotype_slice, 0, None).astype(np.uint8, copy=False)

    entries = alleles.astype(object)
    entries[missing_mask] = None

    return entries.tolist()




def _build_sample_to_group(
    sample_names: Sequence[str],
    subpops: Sequence[Sequence[int]],
    *,
    ploidy: int,
) -> dict[str, tuple[int, int]]:
    mapping: dict[str, tuple[int, int]] = {}
    haplotypes_per_sample = max(ploidy, 1)
    for group_index, indices in enumerate(subpops):
        for sample_index in indices:
            haplotype_index = sample_index % haplotypes_per_sample
            mapping[sample_names[sample_index]] = (group_index, haplotype_index)
    return mapping


def _ferromic_weir_inputs(scale_label: str) -> dict[str, Any]:
    import numpy as np

    import ferromic

    if scale_label not in FERROMIC_WEIR_INPUTS:
        g, subpops = _simulate_weir_genotypes(scale_label)
        positions = np.arange(g.shape[0], dtype=np.int64)

        haplotype_lists = [
            [
                (sample_index, allele_index)
                for sample_index in indices
                for allele_index in range(g.shape[2])
            ]
            for indices in subpops
        ]

        sample_names = _build_sample_names(g.shape[1])
        sample_to_group = _build_sample_to_group(
            sample_names, subpops, ploidy=g.shape[2]
        )
        sequence_length = int(positions[-1] + 1) if positions.size else 0
        haplotypes_all = [
            (sample_index, allele_index)
            for sample_index in range(g.shape[1])
            for allele_index in range(g.shape[2])
        ]
        population_all = ferromic.Population.from_numpy(
            "all_samples",
            g,
            positions,
            haplotypes_all,
            sequence_length,
            sample_names=sample_names,
        )
        populations = [
            population_all.with_haplotypes(f"pop{group_index}", haplotypes)
            for group_index, haplotypes in enumerate(haplotype_lists)
        ]
        region = (
            int(positions[0]) if positions.size else 0,
            int(positions[-1]) if positions.size else 0,
        )
        payload = {
            "genotypes": g,
            "subpops": subpops,
            "positions": positions,
            "sample_names": sample_names,
            "sample_to_group": sample_to_group,
            "sequence_length": sequence_length,
            "populations": populations,
            "region": region,
            "population": population_all,
            "build_variants": lambda gen=g, pos=positions: _build_ferromic_diploid_variants(
                gen, pos
            ),
        }
        FERROMIC_WEIR_INPUTS[scale_label] = payload

    return FERROMIC_WEIR_INPUTS[scale_label]


def _ferromic_haplotype_inputs(
    scale_label: str, *, include_missing_row: bool = False
) -> dict[str, Any]:
    import numpy as np

    import ferromic

    cache_key = (scale_label, include_missing_row)
    if cache_key not in FERROMIC_HAPLOTYPE_INPUTS:
        haplotypes, pos = _simulate_haplotype_array(
            scale_label, include_missing_row=include_missing_row
        )
        hap_array = np.asarray(haplotypes)
        pos_array = np.asarray(pos)

        pos_array = pos_array[: hap_array.shape[0]]
        sample_count = hap_array.shape[1]
        sample_names = _build_sample_names(sample_count)
        sequence_length = int(pos_array[-1] + 1) if pos_array.size else 0
        haplotypes_all = [(index, 0) for index in range(sample_count)]
        hap_cube = hap_array[:, :, np.newaxis]
        population = ferromic.Population.from_numpy(
            "haplotypes",
            hap_cube,
            pos_array,
            haplotypes_all,
            sequence_length,
            sample_names=sample_names,
        )
        payload = {
            "haplotypes_array": hap_array,
            "positions": pos_array,
            "sample_names": sample_names,
            "sequence_length": sequence_length,
            "population": population,
            "build_variants": lambda hap=hap_array, pos=pos_array: _build_ferromic_haploid_variants(
                hap, pos
            ),
        }
        FERROMIC_HAPLOTYPE_INPUTS[cache_key] = payload

    return FERROMIC_HAPLOTYPE_INPUTS[cache_key]


def _ferromic_sequence_inputs(scale_label: str) -> dict[str, Any]:
    import numpy as np

    import ferromic

    if scale_label not in FERROMIC_SEQUENCE_INPUTS:
        genotype, pos = _simulate_sequence_genotypes(scale_label)
        g_array = np.asarray(genotype)
        pos_array = np.asarray(pos)
        pos_array = pos_array[: g_array.shape[0]]
        sample_names = _build_sample_names(g_array.shape[1])
        sequence_length = int(pos_array[-1] + 1) if pos_array.size else 0
        haplotypes_all = [
            (sample_index, allele_index)
            for sample_index in range(g_array.shape[1])
            for allele_index in range(g_array.shape[2])
        ]
        population = ferromic.Population.from_numpy(
            "sequence",
            g_array,
            pos_array,
            haplotypes_all,
            sequence_length,
            sample_names=sample_names,
        )
        payload = {
            "genotype_array": g_array,
            "positions": pos_array,
            "sample_names": sample_names,
            "sequence_length": sequence_length,
            "population": population,
        }
        FERROMIC_SEQUENCE_INPUTS[scale_label] = payload

    return FERROMIC_SEQUENCE_INPUTS[scale_label]


def _ferromic_pca_inputs(scale_label: str) -> dict[str, Any]:
    import numpy as np

    if scale_label not in FERROMIC_PCA_INPUTS:
        matrix = _simulate_pca_matrix(scale_label)
        data = np.asarray(matrix, dtype=np.float64)

        sample_names = _build_sample_names(data.shape[1])
        allele_lookup = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.int16)
        rounded = np.clip(np.rint(data), 0, 2).astype(np.intp)
        genotype_cube = allele_lookup[rounded]
        mask = np.isnan(data)
        if mask.any():
            genotype_cube = genotype_cube.copy()
            genotype_cube[mask] = -1
        genotype_cube = genotype_cube.astype(np.int16, copy=False)
        positions = np.arange(data.shape[0], dtype=np.int64)
        dense_variants = {"genotypes": genotype_cube, "positions": positions}

        payload = {
            "matrix": data,
            "dense_variants": dense_variants,
            "sample_names": sample_names,
        }
        FERROMIC_PCA_INPUTS[scale_label] = payload

    return FERROMIC_PCA_INPUTS[scale_label]


@lru_cache(maxsize=None)
def _weir_results_cached(scale_label: str) -> tuple[Any, Any, Any]:
    g, subpops = _simulate_weir_genotypes(scale_label)
    details = f"{g.shape[0]}x{g.shape[1]}x{g.shape[2]}"
    result: tuple[Any, Any, Any] | None = None

    if is_library_enabled("scikit-allel") and _is_library_scale_enabled(
        "scikit-allel", scale_label
    ):
        import allel

        result = benchmark_call(
            "allel.weir_cockerham_fst",
            scale_label,
            lambda: allel.weir_cockerham_fst(g, subpops),
            details=details,
        )

    if is_library_enabled("ferromic"):
        ferromic_details = details
        if _is_library_scale_enabled("ferromic", scale_label):
            import ferromic

            ferromic_inputs = _ferromic_weir_inputs(scale_label)
            variants = ferromic_inputs["build_variants"]()
            benchmark_call(
                "ferromic.wc_fst",
                scale_label,
                lambda: ferromic.wc_fst(
                    variants,
                    ferromic_inputs["sample_names"],
                    ferromic_inputs["sample_to_group"],
                    ferromic_inputs["region"],
                ),
                details=ferromic_details,
                library="ferromic",
            )
        else:
            reason = _scale_skip_reason(scale_label)
            _record_scale_skip(
                library="ferromic",
                category="ferromic.wc_fst",
                size_label=scale_label,
                details=ferromic_details,
                reason=reason,
            )

    return result  # type: ignore[return-value]


@lru_cache(maxsize=None)
def _hudson_results_cached(scale_label: str) -> tuple[Any, Any]:
    g, subpops = _simulate_weir_genotypes(scale_label)
    result: tuple[Any, Any] | None = None
    details: str | None = None

    if is_library_enabled("scikit-allel") and _is_library_scale_enabled(
        "scikit-allel", scale_label
    ):
        import allel

        g_array = allel.GenotypeArray(g)
        ac1 = g_array.count_alleles(subpop=subpops[0])
        ac2 = g_array.count_alleles(subpop=subpops[1])
        details = f"{ac1.shape[0]}x{ac1.shape[1]}"
        result = benchmark_call(
            "allel.hudson_fst",
            scale_label,
            lambda: allel.hudson_fst(ac1, ac2),
            details=details,
        )

    if is_library_enabled("ferromic"):
        ferromic_details = details or f"{g.shape[0]}x{g.shape[2]}"
        if _is_library_scale_enabled("ferromic", scale_label):
            import ferromic

            ferromic_inputs = _ferromic_weir_inputs(scale_label)
            populations = ferromic_inputs["populations"]
            benchmark_call(
                "ferromic.hudson_fst",
                scale_label,
                lambda: ferromic.hudson_fst(populations[0], populations[1]),
                details=ferromic_details,
                library="ferromic",
            )
        else:
            reason = _scale_skip_reason(scale_label)
            _record_scale_skip(
                library="ferromic",
                category="ferromic.hudson_fst",
                size_label=scale_label,
                details=ferromic_details,
                reason=reason,
            )

    return result  # type: ignore[return-value]


@lru_cache(maxsize=None)
def _average_weir_results_cached(scale_label: str) -> tuple[float, float, Any, Any]:
    g, subpops = _simulate_weir_genotypes(scale_label)
    blen = _block_length_for_average_fst(scale_label)
    details = f"{g.shape[0]}x{g.shape[1]}x{g.shape[2]} (blen={blen})"
    result: tuple[float, float, Any, Any] | None = None

    if is_library_enabled("scikit-allel") and _is_library_scale_enabled(
        "scikit-allel", scale_label
    ):
        import allel

        result = benchmark_call(
            "allel.average_weir_cockerham_fst",
            scale_label,
            lambda: allel.average_weir_cockerham_fst(g, subpops, blen=blen),
            details=details,
        )

    if is_library_enabled("ferromic"):
        ferromic_details = details
        if _is_library_scale_enabled("ferromic", scale_label):
            import ferromic

            ferromic_inputs = _ferromic_weir_inputs(scale_label)
            variants = ferromic_inputs["build_variants"]()
            benchmark_call(
                "ferromic.wc_fst_average",
                scale_label,
                lambda: ferromic.wc_fst(
                    variants,
                    ferromic_inputs["sample_names"],
                    ferromic_inputs["sample_to_group"],
                    ferromic_inputs["region"],
                ),
                details=ferromic_details,
                library="ferromic",
            )
        else:
            reason = _scale_skip_reason(scale_label)
            _record_scale_skip(
                library="ferromic",
                category="ferromic.wc_fst_average",
                size_label=scale_label,
                details=ferromic_details,
                reason=reason,
            )

    return result  # type: ignore[return-value]


@lru_cache(maxsize=None)
def _average_hudson_results_cached(scale_label: str) -> tuple[float, float, Any, Any]:
    g, subpops = _simulate_weir_genotypes(scale_label)
    blen = _block_length_for_average_fst(scale_label)
    result: tuple[float, float, Any, Any] | None = None
    details: str | None = None

    if is_library_enabled("scikit-allel") and _is_library_scale_enabled(
        "scikit-allel", scale_label
    ):
        import allel

        g_array = allel.GenotypeArray(g)
        ac1 = g_array.count_alleles(subpop=subpops[0])
        ac2 = g_array.count_alleles(subpop=subpops[1])
        details = f"{ac1.shape[0]}x{ac1.shape[1]} (blen={blen})"
        result = benchmark_call(
            "allel.average_hudson_fst",
            scale_label,
            lambda: allel.average_hudson_fst(ac1, ac2, blen=blen),
            details=details,
        )

    if is_library_enabled("ferromic"):
        ferromic_details = details or f"{g.shape[0]}x{g.shape[2]} (blen={blen})"
        if _is_library_scale_enabled("ferromic", scale_label):
            import ferromic

            ferromic_inputs = _ferromic_weir_inputs(scale_label)
            populations = ferromic_inputs["populations"]
            region = ferromic_inputs["region"]
            benchmark_call(
                "ferromic.hudson_fst_average",
                scale_label,
                lambda: ferromic.hudson_fst_with_sites(populations[0], populations[1], region),
                details=ferromic_details,
                library="ferromic",
            )
        else:
            reason = _scale_skip_reason(scale_label)
            _record_scale_skip(
                library="ferromic",
                category="ferromic.hudson_fst_average",
                size_label=scale_label,
                details=ferromic_details,
                reason=reason,
            )

    return result  # type: ignore[return-value]


@lru_cache(maxsize=None)
def _mean_pairwise_difference_cached(scale_label: str) -> Any:
    haplotypes, _ = _simulate_haplotype_array(scale_label)
    import numpy as np

    hap_array = np.asarray(haplotypes)
    details = f"{hap_array.shape[0]}x{hap_array.shape[1]}"
    result: Any = None

    if is_library_enabled("scikit-allel") and _is_library_scale_enabled(
        "scikit-allel", scale_label
    ):
        import allel

        hap_object = allel.HaplotypeArray(hap_array)
        ac = hap_object.count_alleles()
        details = f"{ac.shape[0]}x{ac.shape[1]}"
        result = benchmark_call(
            "allel.mean_pairwise_difference",
            scale_label,
            lambda: allel.mean_pairwise_difference(ac),
            details=details,
        )

    if is_library_enabled("ferromic"):
        ferromic_details = details
        if _is_library_scale_enabled("ferromic", scale_label):
            import ferromic

            ferromic_inputs = _ferromic_haplotype_inputs(scale_label)
            variants = ferromic_inputs["build_variants"]()
            sample_count = len(ferromic_inputs["sample_names"])
            benchmark_call(
                "ferromic.pairwise_differences",
                scale_label,
                lambda: ferromic.pairwise_differences(
                    variants, sample_count, ferromic_inputs["sequence_length"]
                ),
                details=ferromic_details,
                library="ferromic",
            )
        else:
            reason = _scale_skip_reason(scale_label)
            _record_scale_skip(
                library="ferromic",
                category="ferromic.pairwise_differences",
                size_label=scale_label,
                details=ferromic_details,
                reason=reason,
            )

    return result


@lru_cache(maxsize=None)
def _mean_pairwise_difference_between_cached(scale_label: str) -> Any:
    haplotypes, _ = _simulate_haplotype_array(scale_label)
    import numpy as np

    hap_array = np.asarray(haplotypes)
    total_samples = hap_array.shape[1]
    details = f"{hap_array.shape[0]}x{hap_array.shape[1]}"
    result: Any = None

    if is_library_enabled("scikit-allel") and _is_library_scale_enabled(
        "scikit-allel", scale_label
    ):
        import allel

        hap_object = allel.HaplotypeArray(hap_array)
        half = total_samples // 2
        ac1 = hap_object.count_alleles(subpop=list(range(0, half)))
        ac2 = hap_object.count_alleles(subpop=list(range(half, total_samples)))
        details = f"{ac1.shape[0]}x{ac1.shape[1]}"
        result = benchmark_call(
            "allel.mean_pairwise_difference_between",
            scale_label,
            lambda: allel.mean_pairwise_difference_between(ac1, ac2),
            details=details,
        )

    if is_library_enabled("ferromic"):
        ferromic_details = details
        if _is_library_scale_enabled("ferromic", scale_label):
            import ferromic

            ferromic_inputs = _ferromic_haplotype_inputs(scale_label)
            variants = ferromic_inputs["build_variants"]()
            sample_count = len(ferromic_inputs["sample_names"])
            pop1_indices = set(range(0, sample_count // 2))
            pop2_indices = set(range(sample_count // 2, sample_count))

            def _ferromic_pairwise_between() -> float:
                comparisons = ferromic.pairwise_differences(
                    variants, sample_count, ferromic_inputs["sequence_length"]
                )
                diff_total = 0.0
                comparable_total = 0
                for comparison in comparisons:
                    if (
                        comparison.sample_i in pop1_indices
                        and comparison.sample_j in pop2_indices
                    ) or (
                        comparison.sample_i in pop2_indices
                        and comparison.sample_j in pop1_indices
                    ):
                        diff_total += float(comparison.differences)
                        comparable_total += int(comparison.comparable_sites)
                if comparable_total == 0:
                    return float("nan")
                return diff_total / comparable_total

            benchmark_call(
                "ferromic.pairwise_differences_between",
                scale_label,
                _ferromic_pairwise_between,
                details=ferromic_details,
                library="ferromic",
            )
        else:
            reason = _scale_skip_reason(scale_label)
            _record_scale_skip(
                library="ferromic",
                category="ferromic.pairwise_differences_between",
                size_label=scale_label,
                details=ferromic_details,
                reason=reason,
            )

    return result


@lru_cache(maxsize=None)
def _sequence_divergence_cached(scale_label: str) -> float:
    haplotypes, pos = _simulate_haplotype_array(scale_label, include_missing_row=True)
    import numpy as np

    hap_array = np.asarray(haplotypes)
    total_samples = hap_array.shape[1]
    details: str | None = f"{hap_array.shape[0]}x{hap_array.shape[1]}"
    result: float | None = None

    if is_library_enabled("scikit-allel") and _is_library_scale_enabled(
        "scikit-allel", scale_label
    ):
        import allel

        hap_object = allel.HaplotypeArray(hap_array)
        half = total_samples // 2
        ac1 = hap_object.count_alleles(subpop=list(range(0, half)))
        ac2 = hap_object.count_alleles(subpop=list(range(half, total_samples)))
        start = 1
        stop = pos[-1] + 5
        details = f"{ac1.shape[0]}x{ac1.shape[1]}"
        result = benchmark_call(
            "allel.sequence_divergence",
            scale_label,
            lambda: allel.sequence_divergence(pos, ac1, ac2, start=start, stop=stop),
            details=details,
        )

    if is_library_enabled("ferromic"):
        ferromic_details = details
        if _is_library_scale_enabled("ferromic", scale_label):
            import ferromic

            ferromic_inputs = _ferromic_haplotype_inputs(
                scale_label, include_missing_row=True
            )
            base_population = ferromic_inputs["population"]
            sample_total = len(_ferromic_sample_names(base_population))
            mid = sample_total // 2
            pop1 = base_population.with_haplotypes(
                "pop1", [(index, 0) for index in range(0, mid)]
            )
            pop2 = base_population.with_haplotypes(
                "pop2", [(index, 0) for index in range(mid, sample_total)]
            )

            benchmark_call(
                "ferromic.hudson_dxy",
                scale_label,
                lambda: ferromic.hudson_dxy(pop1, pop2),
                details=ferromic_details,
                library="ferromic",
            )
        else:
            reason = _scale_skip_reason(scale_label)
            _record_scale_skip(
                library="ferromic",
                category="ferromic.hudson_dxy",
                size_label=scale_label,
                details=ferromic_details,
                reason=reason,
            )

    return result  # type: ignore[return-value]


@lru_cache(maxsize=None)
def _sequence_diversity_cached(scale_label: str) -> tuple[float, float]:
    genotype, pos = _simulate_sequence_genotypes(scale_label)
    import numpy as np

    geno_array = np.asarray(genotype)
    details: str | None = f"{geno_array.shape[0]}x{geno_array.shape[2]}"
    start = 1
    stop = pos[-1] + 5
    pi: float | None = None
    theta: float | None = None

    if is_library_enabled("scikit-allel") and _is_library_scale_enabled(
        "scikit-allel", scale_label
    ):
        import allel

        g_object = allel.GenotypeArray(geno_array)
        ac = g_object.count_alleles()
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

    if is_library_enabled("ferromic"):
        ferromic_details = details
        if _is_library_scale_enabled("ferromic", scale_label):
            import ferromic

            ferromic_inputs = _ferromic_sequence_inputs(scale_label)
            population = ferromic_inputs["population"]

            benchmark_call(
                "ferromic.nucleotide_diversity",
                scale_label,
                lambda: population.nucleotide_diversity(),
                details=ferromic_details,
                library="ferromic",
            )

            def _ferromic_theta() -> float:
                seg_sites = population.segregating_sites()
                sample_count = len(_ferromic_haplotypes(population))
                return ferromic.watterson_theta(
                    seg_sites, sample_count, population.sequence_length
                )

            benchmark_call(
                "ferromic.watterson_theta",
                scale_label,
                _ferromic_theta,
                details=ferromic_details,
                library="ferromic",
            )
        else:
            reason = _scale_skip_reason(scale_label)
            _record_scale_skip(
                library="ferromic",
                category="ferromic.nucleotide_diversity",
                size_label=scale_label,
                details=ferromic_details,
                reason=reason,
            )
            _record_scale_skip(
                library="ferromic",
                category="ferromic.watterson_theta",
                size_label=scale_label,
                details=ferromic_details,
                reason=reason,
            )

    return pi, theta  # type: ignore[return-value]


@lru_cache(maxsize=None)
def _pca_results_cached(scale_label: str) -> tuple[Any, Any]:
    gn = _simulate_pca_matrix(scale_label)
    details = f"{gn.shape[0]}x{gn.shape[1]}"
    result: tuple[Any, Any] | None = None

    if is_library_enabled("scikit-allel") and _is_library_scale_enabled(
        "scikit-allel", scale_label
    ):
        import allel

        result = benchmark_call(
            "allel.pca",
            scale_label,
            lambda: allel.pca(gn, n_components=3),
            details=details,
        )

    if is_library_enabled("ferromic"):
        ferromic_details = details
        if _is_library_scale_enabled("ferromic", scale_label):
            import ferromic

            ferromic_inputs = _ferromic_pca_inputs(scale_label)
            benchmark_call(
                "ferromic.chromosome_pca",
                scale_label,
                lambda: ferromic.chromosome_pca(
                    ferromic_inputs["dense_variants"],
                    ferromic_inputs["sample_names"],
                    n_components=3,
                ),
                details=ferromic_details,
                library="ferromic",
            )
        else:
            reason = _scale_skip_reason(scale_label)
            _record_scale_skip(
                library="ferromic",
                category="ferromic.chromosome_pca",
                size_label=scale_label,
                details=ferromic_details,
                reason=reason,
            )

    return result  # type: ignore[return-value]


@lru_cache(maxsize=None)
def _randomized_pca_results_cached(scale_label: str) -> tuple[Any, Any]:
    gn = _simulate_pca_matrix(scale_label)
    details = f"{gn.shape[0]}x{gn.shape[1]}"
    result: tuple[Any, Any] | None = None

    if is_library_enabled("scikit-allel") and _is_library_scale_enabled(
        "scikit-allel", scale_label
    ):
        import allel

        result = benchmark_call(
            "allel.randomized_pca",
            scale_label,
            lambda: allel.randomized_pca(gn, n_components=3, random_state=0),
            details=details,
        )

    if is_library_enabled("ferromic"):
        ferromic_details = details
        if _is_library_scale_enabled("ferromic", scale_label):
            import ferromic

            ferromic_inputs = _ferromic_pca_inputs(scale_label)
            benchmark_call(
                "ferromic.chromosome_pca_randomized",
                scale_label,
                lambda: ferromic.chromosome_pca(
                    ferromic_inputs["dense_variants"],
                    ferromic_inputs["sample_names"],
                    n_components=3,
                ),
                details=ferromic_details,
                library="ferromic",
            )
        else:
            reason = _scale_skip_reason(scale_label)
            _record_scale_skip(
                library="ferromic",
                category="ferromic.chromosome_pca_randomized",
                size_label=scale_label,
                details=ferromic_details,
                reason=reason,
            )

    return result  # type: ignore[return-value]


def ensure_dependencies_installed(libraries: Sequence[str]) -> None:
    """Install runtime dependencies required for the documentation tests."""

    requested = {_normalize_library_name(name) for name in libraries}
    packages = ["scipy", "scikit-learn", "pytest"]

    if "ferromic" in requested:
        packages.append("ferromic")
    if "scikit-allel" in requested:
        packages.append("scikit-allel")

    if packages:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", *packages],
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
    library: str = "scikit-allel",
) -> T:
    """Measure execution time for ``func`` and record the result."""

    if repeat < 1 or number < 1:
        raise ValueError("repeat and number must be positive integers")

    normalized_library = _normalize_library_name(library)

    if not is_library_enabled(normalized_library):
        BENCHMARK_RESULTS.append(
            BenchmarkRecord(
                library=normalized_library,
                category=category,
                size_label=size_label,
                seconds=math.nan,
                details=details,
                skipped=True,
            )
        )
        return None  # type: ignore[return-value]

    result: dict[str, T] = {}

    def runner() -> None:
        result["value"] = func()

    timer = timeit.Timer(runner)
    times = timer.repeat(repeat=repeat, number=number)
    best = min(times) / number
    value = result["value"]
    BENCHMARK_RESULTS.append(
        BenchmarkRecord(
            library=normalized_library,
            category=category,
            size_label=size_label,
            seconds=best,
            details=details,
            result_repr=_stringify_benchmark_result(value),
        )
    )
    return value


def _print_benchmarks() -> None:
    global BENCHMARK_RESULTS_PRINTED

    if BENCHMARK_RESULTS_PRINTED:
        return

    BENCHMARK_RESULTS_PRINTED = True
    _write_benchmark_results_tsv()

    if not BENCHMARK_RESULTS:
        return

    print("\nBenchmark results (best of single run):")

    for record in _sorted_benchmark_records():
        if record.skipped:
            detail = f" ({record.details})" if record.details else ""
            print(
                f"{record.library}::{record.category}"
                f" [{record.size_label}]: skipped{detail}"
            )
            continue
        detail = f" ({record.details})" if record.details else ""
        print(
            f"{record.library}::{record.category}"
            f" [{record.size_label}]: {record.seconds:.6f}s{detail}"
        )


atexit.register(_print_benchmarks)


def _validate_paired_benchmarks() -> None:
    required_libraries = {"scikit-allel", "ferromic"}
    if not required_libraries.issubset(ENABLED_LIBRARIES):
        return

    scale_labels = set(_enabled_scale_labels())
    if not scale_labels:
        return

    executed: dict[str, set[str]] = {}
    for record in BENCHMARK_RESULTS:
        if record.skipped or record.size_label not in scale_labels:
            continue
        executed.setdefault(record.size_label, set()).add(record.library)

    missing: dict[str, set[str]] = {}
    for label in scale_labels:
        libs_present = executed.get(label, set())
        expected_libs = {
            library
            for library in required_libraries
            if _is_library_scale_enabled(library, label)
        }
        missing_libs = expected_libs - libs_present
        if missing_libs:
            missing[label] = missing_libs

    if missing:
        details = ", ".join(
            f"{label} (missing {', '.join(sorted(libs))})"
            for label, libs in sorted(missing.items())
        )
        raise AssertionError(
            "Missing paired benchmark results for scale labels: " + details
        )


def _skip_if_library_disabled(library: str) -> None:
    if is_library_enabled(library):
        return

    import pytest  # type: ignore

    pytest.skip(f"Library '{library}' benchmarks disabled via CLI flag")


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
    if not is_library_enabled("scikit-allel"):
        return None  # type: ignore[return-value]

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
    _skip_if_library_disabled("scikit-allel")

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

    for label in _enabled_scale_labels_for_library("scikit-allel"):
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
    _skip_if_library_disabled("scikit-allel")

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

    for label in _enabled_scale_labels_for_library("scikit-allel"):
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


def test_average_weir_cockerham_fst_block_jackknife():
    _skip_if_library_disabled("scikit-allel")

    import numpy as np
    import allel

    g, subpops = _build_weir_cockerham_inputs()

    details = f"{g.shape[0]}x{g.shape[1]}x{g.shape[2]} (blen=2)"
    fst, se, vb, vj = benchmark_call(
        "allel.average_weir_cockerham_fst",
        "original",
        lambda: allel.average_weir_cockerham_fst(g, subpops, blen=2),
        details=details,
    )

    expected_fst = -4.36809058868914e-17
    expected_se = 8.0 / 15.0
    expected_vb = np.array([2.0 / 3.0, -0.4])
    expected_vj = np.array([-0.4, 2.0 / 3.0])

    np.testing.assert_allclose(fst, expected_fst, rtol=0, atol=1e-12)
    np.testing.assert_allclose(se, expected_se, rtol=0, atol=1e-12)
    np.testing.assert_allclose(vb, expected_vb, rtol=0, atol=1e-12)
    np.testing.assert_allclose(vj, expected_vj, rtol=0, atol=1e-12)

    for label in _enabled_scale_labels_for_library("scikit-allel"):
        g_large, _ = _simulate_weir_genotypes(label)
        fst_large, se_large, vb_large, vj_large = _average_weir_results_cached(label)
        vb_large = np.asarray(vb_large)
        vj_large = np.asarray(vj_large)

        blen = _block_length_for_average_fst(label)
        expected_blocks = max(1, math.ceil((g_large.shape[0] - blen + 1) / blen))

        assert np.isfinite(fst_large)
        assert np.isfinite(se_large)
        assert se_large >= 0
        assert vb_large.ndim == 1
        assert vj_large.ndim == 1
        assert vb_large.shape == (expected_blocks,)
        assert vj_large.shape == (expected_blocks,)
        assert np.all(np.isfinite(vb_large))
        assert np.all(np.isfinite(vj_large))


def test_hudson_fst_examples():
    _skip_if_library_disabled("scikit-allel")

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

    for label in _enabled_scale_labels_for_library("scikit-allel"):
        num_large, den_large = _hudson_results_cached(label)
        assert num_large.shape == den_large.shape
        valid_mask = den_large > 0
        if np.any(valid_mask):
            fst_large = num_large[valid_mask] / den_large[valid_mask]
            assert np.all(np.isfinite(fst_large))


def test_average_hudson_fst_block_jackknife():
    _skip_if_library_disabled("scikit-allel")

    import numpy as np
    import allel

    g, subpops = _build_weir_cockerham_inputs()
    g_array = allel.GenotypeArray(g)
    ac1 = g_array.count_alleles(subpop=subpops[0])
    ac2 = g_array.count_alleles(subpop=subpops[1])

    details = f"{ac1.shape[0]}x{ac1.shape[1]} (blen=2)"
    fst, se, vb, vj = benchmark_call(
        "allel.average_hudson_fst",
        "original",
        lambda: allel.average_hudson_fst(ac1, ac2, blen=2),
        details=details,
    )

    expected_fst = 1.0 / 7.0
    expected_se = 17.0 / 45.0
    expected_vb = np.array([5.0 / 9.0, -0.2])
    expected_vj = np.array([-0.2, 5.0 / 9.0])

    np.testing.assert_allclose(fst, expected_fst, rtol=0, atol=1e-12)
    np.testing.assert_allclose(se, expected_se, rtol=0, atol=1e-12)
    np.testing.assert_allclose(vb, expected_vb, rtol=0, atol=1e-12)
    np.testing.assert_allclose(vj, expected_vj, rtol=0, atol=1e-12)

    for label in _enabled_scale_labels_for_library("scikit-allel"):
        g_large, _ = _simulate_weir_genotypes(label)
        fst_large, se_large, vb_large, vj_large = _average_hudson_results_cached(label)
        vb_large = np.asarray(vb_large)
        vj_large = np.asarray(vj_large)

        blen = _block_length_for_average_fst(label)
        expected_blocks = max(1, math.ceil((g_large.shape[0] - blen + 1) / blen))

        assert np.isfinite(fst_large)
        assert np.isfinite(se_large)
        assert se_large >= 0
        assert vb_large.ndim == 1
        assert vj_large.ndim == 1
        assert vb_large.shape == (expected_blocks,)
        assert vj_large.shape == (expected_blocks,)
        assert np.all(np.isfinite(vb_large))
        assert np.all(np.isfinite(vj_large))


def test_mean_pairwise_difference():
    _skip_if_library_disabled("scikit-allel")

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

    for label in _enabled_scale_labels_for_library("scikit-allel"):
        mpd_large = _mean_pairwise_difference_cached(label)
        assert mpd_large.ndim == 1
        assert mpd_large.size > 0
        finite = mpd_large[np.isfinite(mpd_large)]
        assert np.all(finite >= 0)


def test_sequence_diversity_and_watterson_theta():
    _skip_if_library_disabled("scikit-allel")

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

    for label in _enabled_scale_labels_for_library("scikit-allel"):
        pi_large, theta_large = _sequence_diversity_cached(label)
        assert pi_large >= 0
        assert theta_large >= 0


def test_mean_pairwise_difference_between_and_sequence_divergence():
    _skip_if_library_disabled("scikit-allel")

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

    for label in _enabled_scale_labels_for_library("scikit-allel"):
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
    _skip_if_library_disabled("scikit-allel")

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

    for label in _enabled_scale_labels_for_library("scikit-allel"):
        coords_large, model_large = _pca_results_cached(label)
        assert coords_large.shape[1] == 3
        assert model_large.explained_variance_ratio_.shape[0] == 3
        assert np.all(model_large.explained_variance_ratio_ >= 0)


def test_randomized_pca_example():
    _skip_if_library_disabled("scikit-allel")

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

    for label in _enabled_scale_labels_for_library("scikit-allel"):
        coords_large, model_large = _randomized_pca_results_cached(label)
        assert coords_large.shape[1] == 3
        assert model_large.explained_variance_ratio_.shape[0] == 3
        assert np.all(model_large.explained_variance_ratio_ >= 0)


def test_ferromic_weir_benchmarks():
    _skip_if_library_disabled("ferromic")

    for label in _enabled_scale_labels():
        _weir_results_cached(label)
        _average_weir_results_cached(label)


def test_ferromic_hudson_benchmarks():
    _skip_if_library_disabled("ferromic")

    for label in _enabled_scale_labels():
        _hudson_results_cached(label)
        _average_hudson_results_cached(label)


def test_ferromic_sequence_benchmarks():
    _skip_if_library_disabled("ferromic")

    for label in _enabled_scale_labels():
        _sequence_divergence_cached(label)
        _sequence_diversity_cached(label)


def test_ferromic_pairwise_benchmarks():
    _skip_if_library_disabled("ferromic")

    for label in _enabled_scale_labels():
        _mean_pairwise_difference_cached(label)
        _mean_pairwise_difference_between_cached(label)


def test_ferromic_pca_benchmarks():
    _skip_if_library_disabled("ferromic")

    for label in _enabled_scale_labels():
        _pca_results_cached(label)
        _randomized_pca_results_cached(label)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark suites for ferromic and scikit-allel")
    parser.add_argument(
        "--ferromic",
        action="store_true",
        help="Run only ferromic benchmarks",
    )
    parser.add_argument(
        "--allel",
        action="store_true",
        help="Run only scikit-allel benchmarks",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    requested: list[str] = []
    if args.ferromic:
        requested.append("ferromic")
    if args.allel:
        requested.append("scikit-allel")

    set_enabled_libraries(requested or None)
    libraries = sorted(ENABLED_LIBRARIES)

    ensure_dependencies_installed(libraries)

    import pytest  # type: ignore

    exit_code = pytest.main(["-s", __file__])
    _validate_paired_benchmarks()
    _print_benchmarks()
    return exit_code


def pytest_sessionfinish(session, exitstatus):  # type: ignore[unused-argument]
    _validate_paired_benchmarks()
    _print_benchmarks()


if __name__ == "__main__":
    sys.modules.setdefault("runs", sys.modules[__name__])
    raise SystemExit(main(sys.argv[1:]))
