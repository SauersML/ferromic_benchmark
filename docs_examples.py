#!/usr/bin/env python3
"""Execute scikit-allel documentation examples via pytest assertions."""

from __future__ import annotations

import subprocess
import sys
from typing import Sequence


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


def test_weir_cockerham_fst_components():
    import numpy as np
    import allel

    g, subpops = _build_weir_cockerham_inputs()
    a, b, c = allel.weir_cockerham_fst(g, subpops)

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


def test_weir_cockerham_fst_variants_and_overall():
    import numpy as np
    import allel

    g, subpops = _build_weir_cockerham_inputs()
    a, b, c = allel.weir_cockerham_fst(g, subpops)

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


def test_mean_pairwise_difference():
    import numpy as np
    import allel

    h, _ = _build_haplotype_array()
    ac = h.count_alleles()
    mpd = allel.mean_pairwise_difference(ac)
    expected_mpd = np.array([0.0, 0.5, 0.66666667, 0.5, 0.0, 0.83333333, 0.83333333, 1.0])
    np.testing.assert_allclose(mpd, expected_mpd, rtol=0, atol=1e-8)


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
    np.testing.assert_allclose(pi, 0.13978494623655915, rtol=0, atol=1e-12)

    theta_hat_w = allel.watterson_theta(pos, ac, start=1, stop=31)
    np.testing.assert_allclose(theta_hat_w, 0.10557184750733138, rtol=0, atol=1e-12)


def test_mean_pairwise_difference_between_and_sequence_divergence():
    import numpy as np
    import allel

    h, pos = _build_haplotype_array()
    ac1 = h.count_alleles(subpop=[0, 1])
    ac2 = h.count_alleles(subpop=[2, 3])

    mpd_between = allel.mean_pairwise_difference_between(ac1, ac2)
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
    np.testing.assert_allclose(dxy, 0.12096774193548387, rtol=0, atol=1e-12)


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


def test_randomized_pca_example():
    import numpy as np
    import allel

    gn = _build_pca_input()
    coords, model = allel.randomized_pca(gn, n_components=2, random_state=0)

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


def main() -> int:
    ensure_dependencies_installed()

    import pytest  # type: ignore

    return pytest.main([__file__])


if __name__ == "__main__":
    raise SystemExit(main())

