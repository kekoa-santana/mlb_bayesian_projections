"""Pin numerical behavior of compute_coverage_levels.

Ensures the multi-level coverage helper stays numerically equivalent to
compute_coverage_from_sd called per level, and guards against silent
regressions from future refactors.
"""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import (
    compute_coverage_from_sd,
    compute_coverage_levels,
)


def test_keys_are_integer_percent_strings():
    rng = np.random.default_rng(0)
    actual = rng.normal(5.0, 2.0, 200)
    mean = np.full(200, 5.0)
    sd = np.full(200, 2.0)

    got = compute_coverage_levels(actual, mean, sd)

    assert set(got.keys()) == {"50", "80", "90"}


def test_custom_levels_roundtrip():
    rng = np.random.default_rng(1)
    actual = rng.normal(0.22, 0.05, 500)
    mean = np.full(500, 0.22)
    sd = np.full(500, 0.05)

    got = compute_coverage_levels(actual, mean, sd, levels=(0.5, 0.95))

    assert set(got.keys()) == {"50", "95"}


def test_matches_per_level_primitive():
    """compute_coverage_levels must equal per-level compute_coverage_from_sd."""
    rng = np.random.default_rng(2)
    n = 1000
    actual = rng.normal(4.5, 1.5, n)
    mean = rng.normal(4.5, 0.3, n)
    sd = rng.uniform(1.0, 2.0, n)

    got = compute_coverage_levels(actual, mean, sd, levels=(0.50, 0.80, 0.90))

    for level, key in [(0.50, "50"), (0.80, "80"), (0.90, "90")]:
        want = compute_coverage_from_sd(actual, mean, sd, level=level)
        assert got[key] == want, f"mismatch at {level}: {got[key]} vs {want}"


def test_known_normal_fixture():
    """On N(0,1) samples with mean=0, sd=1, coverage ≈ nominal level."""
    rng = np.random.default_rng(3)
    n = 50_000
    actual = rng.standard_normal(n)
    mean = np.zeros(n)
    sd = np.ones(n)

    got = compute_coverage_levels(actual, mean, sd, levels=(0.50, 0.80, 0.90))

    # With 50k draws, sampling error on coverage is ~sqrt(p(1-p)/n) ≈ 0.002.
    assert abs(got["50"] - 0.50) < 0.01
    assert abs(got["80"] - 0.80) < 0.01
    assert abs(got["90"] - 0.90) < 0.01
