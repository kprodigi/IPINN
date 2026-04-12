"""Smoke tests for publication / CI (no data files required)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_d_common_positive(m):
    assert m.D_COMMON > 0


def test_classifier_features_use_ea_common(m):
    assert "EA_common" in m.CLASSIFIER_FEATURES
    assert "Angle" in m.CLASSIFIER_FEATURES


def test_ea_at_common_lc1_full_stroke_matches_tabular_ea(m):
    row = pd.Series(
        {
            "disp_end": 80.0,
            "EA": 123.4,
            "F_mean": 9.99,
            "LC": "LC1",
            "Angle": 55.0,
        }
    )
    assert abs(m.ea_at_common_from_row(row, None) - 123.4) < 1e-9


def test_runtime_helpers_exist(m):
    assert callable(m.log_runtime_environment)
    assert callable(m.check_publication_dependencies)
    assert callable(m.write_statistical_testing_policy)


def test_pareto_dominance_vectorized_matches_reference():
    """Regression: vectorised max-EA / min-IPF dominance must match O(n^2) reference."""
    rng = np.random.default_rng(42)
    n = 40
    ea = rng.uniform(1.0, 20.0, size=n)
    ipf = rng.uniform(0.5, 15.0, size=n)
    is_dom = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if ea[j] >= ea[i] and ipf[j] <= ipf[i] and (ea[j] > ea[i] or ipf[j] < ipf[i]):
                is_dom[i] = True
                break
    ge_ea = ea[:, None] >= ea[None, :]
    le_ipf = ipf[:, None] <= ipf[None, :]
    strict = (ea[:, None] > ea[None, :]) | (ipf[:, None] < ipf[None, :])
    dom = ge_ea & le_ipf & strict
    np.fill_diagonal(dom, False)
    is_dom_v = np.any(dom, axis=0)
    assert np.array_equal(is_dom, is_dom_v)
