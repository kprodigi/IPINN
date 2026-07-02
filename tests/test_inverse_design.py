"""Tests for the inverse-design machinery and data-QA paths.

These cover the previously untested core of the inverse demonstration:
objective construction, coarse-grid/GP-BO minimisation, target provenance,
the LC-plausibility penalty, the physically-stated multiplicity threshold,
inference-time plausibility diagnostics, curve-level bootstrap, and the
committed-dataset ingestion path (xlsx + QA).
"""

import logging
import os

import numpy as np
import pandas as pd
import pytest

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@pytest.fixture()
def quiet_logger():
    lg = logging.getLogger("test_inverse_design")
    lg.setLevel(logging.CRITICAL)
    lg.addHandler(logging.NullHandler())
    return lg


def _synthetic_df_metrics():
    """12 designs (6 angles x 2 LCs) with LC-separable EA/IPF, mirroring reality."""
    rows = []
    for i, ang in enumerate([45.0, 50.0, 55.0, 60.0, 65.0, 70.0]):
        rows.append({"LC": "LC1", "Angle": ang, "EA": 800 + 40 * i, "IPF": 5.2 + 0.3 * i,
                     "F_mean": 10.0, "CFE": 1.5, "disp_end": 80.0,
                     "EA_common": 25.0 + 1.5 * i})
        rows.append({"LC": "LC2", "Angle": ang, "EA": 550 + 30 * i, "IPF": 5.65 + 0.35 * i,
                     "F_mean": 8.0, "CFE": 1.2, "disp_end": 130.0,
                     "EA_common": 33.0 + 1.2 * i})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data ingestion + QA on the committed dataset
# ---------------------------------------------------------------------------
class TestCommittedDataIngestion:
    @pytest.mark.skipif(not os.path.isdir(DATA_DIR), reason="data/ not present")
    def test_load_data_xlsx_path_and_qa(self, m, quiet_logger):
        df = m.load_data(DATA_DIR, quiet_logger)
        # Both LCs, the true 45-70 degree grid, and the corrupted LC2 theta=70
        # logger row auto-dropped (10,500 raw rows -> 10,499 clean rows).
        assert set(df["LC"].unique()) == {"LC1", "LC2"}
        assert sorted(df["Angle"].unique()) == [45.0, 50.0, 55.0, 60.0, 65.0, 70.0]
        assert len(df) == 10499
        # Every curve is monotone in displacement after QA.
        for (_, _), g in df.groupby(["LC", "Angle"]):
            assert np.all(np.diff(g["disp_mm"].values) >= -1e-9)
        # Provenance fingerprint is stamped for bundle cross-checks.
        assert isinstance(m.CFG.data_fingerprint, str) and len(m.CFG.data_fingerprint) >= 8


# ---------------------------------------------------------------------------
# Bootstrap unit
# ---------------------------------------------------------------------------
class TestBootstrapIndices:
    def test_row_mode_preserves_size(self, m):
        df = _synthetic_df_metrics().rename(columns={})
        rng = np.random.default_rng(0)
        m.CFG.bootstrap_unit = "row"
        idx = m.bootstrap_indices(df, rng)
        assert len(idx) == len(df)
        assert idx.min() >= 0 and idx.max() < len(df)

    def test_curve_mode_resamples_whole_curves(self, m):
        df = pd.concat([_synthetic_df_metrics()] * 3, ignore_index=True)
        rng = np.random.default_rng(0)
        m.CFG.bootstrap_unit = "curve"
        try:
            idx = m.bootstrap_indices(df, rng)
            picked = df.iloc[idx]
            # Every picked (Angle, LC) curve must be included with ALL its rows.
            counts = df.groupby(["Angle", "LC"]).size()
            for (ang, lc), grp in picked.groupby(["Angle", "LC"]):
                assert len(grp) % counts[(ang, lc)] == 0
        finally:
            m.CFG.bootstrap_unit = "row"


# ---------------------------------------------------------------------------
# IPF fallback provenance
# ---------------------------------------------------------------------------
class TestIpfFallbackFlag:
    def test_prominent_peak_not_fallback(self, m):
        d = np.linspace(0, 80, 400)
        f = np.where(d < 5, d, 5.0 - 0.5 * (d - 5) / 75.0)  # sharp early peak
        ipf, loc, info = m.compute_ipf_robust(d, f, return_info=True)
        assert not info["fallback"]
        assert ipf == pytest.approx(f.max(), rel=0.05)

    def test_monotone_curve_uses_fallback(self, m):
        d = np.linspace(0, 80, 400)
        f = 0.2 * d  # no peak at all
        ipf, loc, info = m.compute_ipf_robust(d, f, return_info=True)
        assert info["fallback"]
        # Fallback = max load in the first 25% of stroke.
        assert loc <= 0.25 * d[-1] + 1e-9

    def test_two_tuple_compatibility(self, m):
        d = np.linspace(0, 80, 200)
        f = 0.2 * d
        out = m.compute_ipf_robust(d, f)
        assert len(out) == 2


# ---------------------------------------------------------------------------
# Inverse objective + minimisers
# ---------------------------------------------------------------------------
class TestInverseMinimisers:
    def test_coarse_grid_fallback_recovers_argmin(self, m, quiet_logger):
        true_theta = {"LC1": 52.0, "LC2": 66.0}
        objective_funcs = {
            lc: (lambda t, c=true_theta[lc]: (t - c) ** 2) for lc in ("LC1", "LC2")
        }
        best_lc, res = m._inverse_coarse_grid_fallback(
            objective_funcs, (45.0, 70.0), ["LC1", "LC2"], quiet_logger)
        assert best_lc in ("LC1", "LC2")
        assert abs(res["x_best"] - true_theta[best_lc]) < 1.0

    @pytest.mark.skipif(not hasattr(pytest, "importorskip"), reason="never")
    def test_gp_bo_minimize_quadratic(self, m, quiet_logger):
        if not m.HAS_SKLEARN_GP:
            pytest.skip("sklearn GP not available")
        bo = m.BOConfig()
        bo.n_calls_total, bo.n_init, bo.seed = 15, 5, 7
        res = m.gp_bo_minimize(lambda t: (t - 57.0) ** 2, (45.0, 70.0), bo, quiet_logger)
        assert abs(res["x_best"] - 57.0) < 2.0

    def test_objective_rejects_nonpositive_ea(self, m):
        # _make_objective must steer away from physically invalid candidates.
        class _FakeModel:  # never called: we monkeypatch the ensemble below
            pass
        calls = {}
        def fake_ensemble(*a, **k):
            return {"EA": -5.0, "IPF": 1.0}
        orig = m.compute_ea_ipf_ensemble
        m.compute_ea_ipf_ensemble = fake_ensemble
        try:
            obj = m._make_objective([], "hard", "LC1", None, None, None,
                                    target_ea=30.0, target_ipf=0.5,
                                    w_ea=1.0, w_ipf=1.0)
            assert obj(55.0) >= 1e6
        finally:
            m.compute_ea_ipf_ensemble = orig


# ---------------------------------------------------------------------------
# Multiplicity threshold in physical units
# ---------------------------------------------------------------------------
class TestSolutionLandscape:
    def test_threshold_counts_only_near_optimal_minima(self, m, quiet_logger):
        # Two basins: J=0 at 50, J=0.0003 at 65 (~1.7% combined rel error),
        # and a shallow decoy at J=0.02 (14%) that must NOT count at 2% tol.
        def obj(t):
            return min((t - 50.0) ** 2 * 1e-4,
                       (t - 65.0) ** 2 * 1e-4 + 0.0003,
                       (t - 58.0) ** 2 * 1e-4 + 0.02)
        res = m.compute_solution_landscape({"LC1": obj}, (45.0, 70.0), 0.0,
                                           quiet_logger, n_grid=251, rel_tol=0.02)
        thetas = [round(mm["theta"]) for mm in res["local_minima_LC1"]]
        assert 50 in thetas and 65 in thetas
        assert 58 not in thetas
        assert res["multiplicity_index"] == 2
        assert res["multiplicity_rel_tol"] == 0.02


# ---------------------------------------------------------------------------
# Target provenance + LC plausibility penalty
# ---------------------------------------------------------------------------
class TestTargetsAndClassifier:
    def test_feasible_targets_carry_source_design(self, m, quiet_logger):
        dm = _synthetic_df_metrics()
        targets = m.generate_feasible_targets(dm, quiet_logger)
        assert len(targets) == 5
        for t in targets:
            assert t["source_angle"] in dm["Angle"].values
            assert t["source_lc"] in ("LC1", "LC2")
            # Target (EA, IPF) must be exactly the source row's values.
            row = dm[(dm["Angle"] == t["source_angle"]) & (dm["LC"] == t["source_lc"])]
            assert t["EA"] == pytest.approx(float(row["EA_common"].iloc[0]))

    def test_lc_penalty_zero_weight_and_monotone_in_weight(self, m, quiet_logger):
        pytest.importorskip("sklearn")
        dm = _synthetic_df_metrics()
        cal_ens, feat_scaler, diag = m.train_lc_plausibility_classifier(dm, quiet_logger)
        # n=12 < 20 -> calibration must be honestly recorded as uncalibrated.
        assert diag["calibration_mode"] == "uncalibrated"
        pen0, p0 = m.compute_lc_penalty(cal_ens, feat_scaler, 30.0, 6.0, "LC1",
                                        prob_weight=0.0, angle_deg=55.0)
        assert pen0 == pytest.approx(0.0)
        pen1, p1 = m.compute_lc_penalty(cal_ens, feat_scaler, 30.0, 6.0, "LC1",
                                        prob_weight=0.5, angle_deg=55.0)
        pen2, p2 = m.compute_lc_penalty(cal_ens, feat_scaler, 30.0, 6.0, "LC1",
                                        prob_weight=1.0, angle_deg=55.0)
        assert p0 == pytest.approx(p1) == pytest.approx(p2)  # p_LC independent of weight
        assert 0.0 <= pen1 <= pen2  # penalty scales with weight


# ---------------------------------------------------------------------------
# Plausibility diagnostics from compute_ea_ipf_ensemble
# ---------------------------------------------------------------------------
class TestPlausibilityDiagnostics:
    def test_diagnostics_present_and_sane(self, m, quiet_logger):
        torch = pytest.importorskip("torch")
        # Tiny real models: structure check only (random weights are fine).
        train_df = pd.DataFrame({
            "disp_mm": np.tile(np.linspace(0, 80, 30), 4),
            "load_kN": np.tile(np.linspace(0, 10, 30), 4),
            "energy_J": np.tile(np.linspace(0, 400, 30), 4),
            "Angle": np.repeat([45.0, 55.0, 65.0, 70.0], 30),
            "LC": ["LC1", "LC2"] * 60,
        })
        scaler_disp, scaler_out, enc, params = m.create_preprocessors(train_df, quiet_logger)
        in_d = m.build_features(train_df.head(1), scaler_disp, enc).shape[1]
        net = m.HardEnergyNet(in_d, [8], 0.0, 10.0)
        net.configure_zero_bc(params, enabled=False)
        out = m.compute_ea_ipf_ensemble([net, net], "hard", 55.0, "LC1",
                                        scaler_disp, enc, params, d_eval=80.0)
        pl = out["plausibility"]
        for k in ("neg_force_frac", "nonmono_energy_frac", "neg_ea_frac",
                  "ipf_fallback_frac", "all_plausible"):
            assert k in pl
        for k in ("neg_force_frac", "nonmono_energy_frac", "neg_ea_frac", "ipf_fallback_frac"):
            assert 0.0 <= pl[k] <= 1.0
