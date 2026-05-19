"""Tests for auxiliary figures, manifest, and dry-run config (no full training)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_generate_output_manifest_creates_csv(tmp_path, m):
    (tmp_path / "Fig_dummy.png").write_bytes(b"x")
    (tmp_path / "Table_dummy.csv").write_text("a\n1\n")
    m.generate_output_manifest(str(tmp_path), m.setup_logging(str(tmp_path)))
    mf = tmp_path / "MANIFEST_outputs.csv"
    assert mf.is_file()
    df = pd.read_csv(mf)
    assert "Fig_dummy.png" in set(df["filename"])


def test_fig_landscape_ensemble_disagreement_smoke(tmp_path, m):
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({
        "angle": np.linspace(45, 70, n),
        "lc": ["LC1"] * (n // 2) + ["LC2"] * (n - n // 2),
        "EA_std": rng.uniform(0.01, 0.5, size=n),
        "IPF_std": rng.uniform(0.01, 0.3, size=n),
    })
    log = m.setup_logging(str(tmp_path))
    m.fig_landscape_ensemble_disagreement(df, str(tmp_path), log)
    assert (tmp_path / "Fig_landscape_ensemble_disagreement.png").is_file()


def test_fig_lc_classifier_diagnostics_pr_panel(tmp_path, m):
    diag = {
        "cv_method": "test",
        "cv_y": np.array([0, 0, 1, 1, 0, 1, 1, 0]),
        "cv_pred": np.array([0, 1, 1, 1, 0, 0, 1, 1]),
        "cv_prob_lc2": np.array([0.1, 0.4, 0.7, 0.8, 0.2, 0.35, 0.75, 0.45]),
    }
    log = m.setup_logging(str(tmp_path))
    m.fig_lc_classifier_diagnostics(diag, str(tmp_path), log)
    assert (tmp_path / "Fig_lc_classifier_cv_diagnostics.png").is_file()
    assert (tmp_path / "Table_lc_classifier_cv_predictions.csv").is_file()


def test_dry_run_caps_epochs_in_get_model_config(m):
    prev = m.CFG.dry_run
    try:
        m.CFG.dry_run = True
        cfg = m.get_model_config("hard", protocol="random")
        assert int(cfg["epochs"]) <= 15
    finally:
        m.CFG.dry_run = prev
