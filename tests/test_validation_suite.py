"""Tests for the Phase-1/2 held-out-design validation suite and mechanics analysis.

Every function here backs a reviewer-facing claim, so the tests pin the exact
semantics: level/shape decomposition, skill scores, jackknife+ honesty,
signature extraction on synthetic curves with known properties, and the
kinematic regression recovering a planted trend.
"""

import logging

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def quiet():
    lg = logging.getLogger("t_valsuite")
    lg.setLevel(logging.CRITICAL)
    lg.addHandler(logging.NullHandler())
    return lg


class TestCurveErrorMetrics:
    def test_pure_level_shift_kills_raw_r2_but_not_shape_r2(self, m):
        d = np.linspace(0, 80, 200)
        y = 0.1 * d + np.sin(d / 5.0)          # rich shape
        y_shifted = y + 5.0                     # pure level offset
        met = m.curve_error_metrics(y, y_shifted)
        assert met["R2_raw"] < 0.0              # raw R² destroyed by the offset
        assert met["R2_shape"] > 0.999          # shape is perfect
        assert met["bias"] == pytest.approx(5.0, rel=1e-6)
        assert met["pearson_r"] == pytest.approx(1.0, abs=1e-9)

    def test_perfect_prediction(self, m):
        y = np.linspace(0, 10, 50)
        met = m.curve_error_metrics(y, y.copy())
        assert met["R2_raw"] == pytest.approx(1.0)
        assert met["NRMSE_range"] == pytest.approx(0.0, abs=1e-12)


class TestSkillScore:
    def test_semantics(self, m):
        assert m.skill_score(5.0, 10.0) == pytest.approx(0.5)   # halves the floor
        assert m.skill_score(10.0, 10.0) == pytest.approx(0.0)  # matches it
        assert m.skill_score(20.0, 10.0) == pytest.approx(-1.0) # doubles it
        assert np.isnan(m.skill_score(5.0, 0.0))                # degenerate floor


class TestJackknifePlus:
    def test_never_uses_own_residual_and_coverage_is_honest(self, m):
        rng = np.random.default_rng(0)
        truths = rng.normal(0.0, 1.0, 8)
        preds = truths + rng.normal(0.0, 0.1, 8)
        tbl, cov = m.jackknife_plus_intervals(preds, truths, alpha=0.1)
        assert len(tbl) == 8 and 0.0 <= cov <= 1.0
        # An extreme outlier fold must NOT inflate its own interval:
        preds2 = preds.copy()
        preds2[3] += 100.0
        tbl2, cov2 = m.jackknife_plus_intervals(preds2, truths, alpha=0.1)
        row = tbl2.iloc[3]
        assert not row["covered"]               # honest failure at the outlier
        assert row["half_width"] < 1.0          # width came from OTHER folds


class TestCrushSignatures:
    def _synthetic_curve(self, densify=True):
        d = np.linspace(0, 100, 800)
        F = np.where(d < 4, 0.15 * d, 0.5)      # peak-ish rise then plateau 0.5
        if densify:
            F = F + np.where(d > 70, 0.08 * (d - 70), 0.0)   # densification rise
        E = np.concatenate([[0.0], np.cumsum(0.5 * (F[1:] + F[:-1]) * np.diff(d))])
        return d, F, E

    def test_onset_detected_on_densifying_curve(self, m):
        d, F, E = self._synthetic_curve(densify=True)
        sig = m.extract_crush_signatures(d, F, E)
        # F reaches 1.5×plateau (0.75) at d = 70 + 0.25/0.08 ≈ 73.1 mm
        assert 70.0 < sig["d_dens_mm"] < 78.0
        assert sig["F_plateau_kN"] == pytest.approx(0.5, rel=0.05)

    def test_no_onset_on_flat_plateau(self, m):
        d, F, E = self._synthetic_curve(densify=False)
        sig = m.extract_crush_signatures(d, F, E)
        assert np.isnan(sig["d_dens_mm"])
        assert sig["CFE"] > 0.0

    def test_signature_table_has_all_cells(self, m, quiet):
        rows = []
        for lc in ("LC1", "LC2"):
            for ang in (45.0, 60.0):
                d, F, E = self._synthetic_curve()
                rows.append(pd.DataFrame({"disp_mm": d, "load_kN": F, "energy_J": E,
                                          "Angle": ang, "LC": lc}))
        df_all = pd.concat(rows, ignore_index=True)
        sig = m.compute_mode_signature_table(df_all, quiet)
        assert len(sig) == 4
        assert set(sig.columns) >= {"LC", "Angle_deg", "IPF_kN", "CFE", "d_dens_mm"}


class TestKinematics:
    def test_regression_recovers_planted_trend(self, m, quiet):
        theta = np.array([45.0, 50.0, 55.0, 60.0, 65.0, 70.0])
        f = m.THETA_KINEMATIC_CANDIDATES["cos_theta"](theta)
        rows = []
        for lc in ("LC1",):
            for th, fv in zip(theta, f):
                rows.append({"LC": lc, "Angle_deg": th,
                             "d_dens_mm": 10.0 + 40.0 * fv,   # exact planted law
                             "F_plateau_kN": 0.5, "k0_kN_per_mm": 0.05,
                             "IPF_kN": 0.6, "CFE": 0.7,
                             "oscillation_norm": 0.1, "d_max_mm": 80.0,
                             "IPF_loc_mm": 2.0})
        kin = m.fit_densification_kinematics(pd.DataFrame(rows), quiet)
        best = kin[(kin["Quantity"] == "d_dens_mm")].sort_values("R2_fit").iloc[-1]
        assert best["candidate"] == "cos_theta"
        assert best["R2_fit"] > 0.9999

    def test_mechanics_trend_baseline_extrapolates_planted_law(self, m):
        theta = np.array([45.0, 50.0, 55.0, 60.0, 65.0])
        y = 2.0 + 3.0 * m.THETA_KINEMATIC_CANDIDATES["sin_theta"](theta)
        pred, fname = m.mechanics_trend_baseline(theta, y, 70.0)
        truth = 2.0 + 3.0 * float(m.THETA_KINEMATIC_CANDIDATES["sin_theta"](np.array([70.0]))[0])
        assert fname == "sin_theta"
        assert pred == pytest.approx(truth, rel=1e-6)


class TestMechanicsFeatureMap:
    def test_mechanics_features_change_embedding_only(self, m, quiet):
        df = pd.DataFrame({"disp_mm": np.linspace(0, 80, 40),
                           "load_kN": np.linspace(0, 8, 40),
                           "energy_J": np.linspace(0, 300, 40),
                           "Angle": 55.0, "LC": "LC1"})
        df = pd.concat([df, df.assign(LC="LC2")], ignore_index=True)
        sd, so, enc, params = m.create_preprocessors(df, quiet)
        try:
            m.CFG.theta_feature_map = ""
            X_default = m.build_features(df, sd, enc)
            m.CFG.theta_feature_map = "mechanics"
            X_mech = m.build_features(df, sd, enc)
        finally:
            m.CFG.theta_feature_map = ""
        assert X_default.shape == X_mech.shape          # drop-in dimensionality
        assert np.allclose(X_default[:, 0], X_mech[:, 0])   # d untouched
        assert np.allclose(X_default[:, 1], X_mech[:, 1])   # sinθ shared
        assert not np.allclose(X_default[:, 2], X_mech[:, 2])  # cosθ -> (1+cosθ)/2
        th = np.deg2rad(55.0)
        assert X_mech[0, 2] == pytest.approx((1 + np.cos(th)) / 2, rel=1e-5)
