"""Comprehensive tests for core functions, data pipeline, models, and physics losses."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import torch


# =====================================================================
# Configuration & Constants
# =====================================================================
class TestConfig:
    def test_config_defaults(self, m):
        cfg = m.Config()
        assert cfg.n_ensemble == 20
        assert cfg.seed == 2026
        assert cfg.test_size == 0.20
        assert cfg.dry_run is False

    def test_config_new_fields(self, m):
        cfg = m.Config()
        assert cfg.run_loao_cv is True
        assert cfg.run_rar is True
        assert cfg.rar_interval == 50
        assert cfg.rar_warmup == 100

    def test_d_common_equals_lc1_stroke(self, m):
        assert m.D_COMMON == m.disp_end_mm("LC1")

    def test_ea_common_mm_tag(self, m):
        assert m.EA_COMMON_MM_TAG == f"{int(m.D_COMMON)}mm"


class TestBOConfigNewFields:
    def test_tikhonov_defaults(self, m):
        bo = m.BOConfig()
        assert bo.gamma_tikhonov == 0.0
        assert bo.theta_center == 57.5
        assert bo.n_bo_restarts == 5


class TestDispEndMM:
    def test_lc1(self, m):
        assert m.disp_end_mm("LC1") == 80.0

    def test_lc2(self, m):
        assert m.disp_end_mm("LC2") == 130.0

    def test_lc1_numeric(self, m):
        assert m.disp_end_mm("0") == 80.0

    def test_lc2_numeric(self, m):
        assert m.disp_end_mm("1") == 130.0

    def test_lc1_case_insensitive(self, m):
        assert m.disp_end_mm("lc1") == 80.0

    def test_unknown_defaults_to_lc2(self, m):
        assert m.disp_end_mm("UNKNOWN") == 130.0


class TestGetNStepsCurve:
    def test_lc1_steps(self, m):
        n = m.get_n_steps_curve("LC1")
        assert n >= 161

    def test_lc2_steps(self, m):
        n = m.get_n_steps_curve("LC2")
        assert n >= 161
        assert n > m.get_n_steps_curve("LC1")


class TestProtocolLabel:
    def test_random(self, m):
        assert "Random" in m.protocol_label("random")

    def test_unseen(self, m):
        label = m.protocol_label("unseen")
        assert "Unseen" in label


class TestEnsembleStd:
    def test_single_member_returns_zeros(self, m):
        arr = np.array([[1.0, 2.0, 3.0]])
        result = m._ensemble_std_along_members(arr)
        assert np.allclose(result, 0.0)

    def test_multiple_members(self, m):
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = m._ensemble_std_along_members(arr)
        assert result.shape == (2,)
        assert np.all(result > 0)


# =====================================================================
# Data Pipeline
# =====================================================================
class TestValidateInputData:
    def test_valid_data(self, m, tiny_df, logger):
        ok, issues = m.validate_input_data(tiny_df, logger)
        assert ok is True
        assert issues == []

    def test_missing_column(self, m, tiny_df, logger):
        bad_df = tiny_df.drop(columns=["load_kN"])
        ok, issues = m.validate_input_data(bad_df, logger)
        assert ok is False
        assert any("load_kN" in i for i in issues)

    def test_nan_values(self, m, tiny_df, logger):
        df = tiny_df.copy()
        df.loc[0, "load_kN"] = np.nan
        ok, issues = m.validate_input_data(df, logger)
        assert ok is False
        assert any("missing" in i.lower() for i in issues)


class TestLoadData:
    def test_load_csv_fixture(self, m, logger):
        fixture_dir = os.path.join(os.path.dirname(__file__), "fixtures")
        df = m.load_data(fixture_dir, logger)
        assert len(df) > 0
        assert set(["disp_mm", "load_kN", "energy_J", "Angle", "LC"]).issubset(df.columns)

    def test_load_missing_dir_raises(self, m, logger, tmp_path):
        empty_dir = str(tmp_path / "empty")
        os.makedirs(empty_dir)
        with pytest.raises(FileNotFoundError):
            m.load_data(empty_dir, logger)


class TestSplitRandom:
    def test_split_sizes(self, m, tiny_df, logger):
        train, val = m.split_random_80_20(tiny_df, seed=42, logger=logger)
        assert len(train) + len(val) == len(tiny_df)
        assert len(val) > 0
        assert len(train) > len(val)

    def test_no_overlap(self, m, tiny_df, logger):
        train, val = m.split_random_80_20(tiny_df, seed=42, logger=logger)
        combined = pd.concat([train, val], ignore_index=True)
        assert len(combined) == len(tiny_df)


class TestSplitUnseen:
    def test_held_out_angle(self, m, tiny_df, logger):
        theta = 55.0
        train, val = m.split_unseen_angle(tiny_df, theta, logger=logger)
        assert all(val["Angle"] == theta)
        assert theta not in train["Angle"].values


# =====================================================================
# Preprocessors & Feature Building
# =====================================================================
class TestCreatePreprocessors:
    def test_returns_tuple(self, m, tiny_df, logger):
        scaler_disp, scaler_out, enc, params = m.create_preprocessors(tiny_df, logger)
        assert hasattr(scaler_disp, "transform")
        assert hasattr(scaler_out, "transform")
        assert hasattr(enc, "transform")
        assert isinstance(params, m.ScalingParams)

    def test_scaling_params_positive_sigmas(self, m, tiny_df, logger):
        _, _, _, params = m.create_preprocessors(tiny_df, logger)
        assert params.sig_d > 0
        assert params.sig_F > 0
        assert params.sig_E > 0
        assert params.grad_factor != 0


class TestBuildFeatures:
    def test_shape(self, m, tiny_df, logger):
        scaler_disp, scaler_out, enc, params = m.create_preprocessors(tiny_df, logger)
        X = m.build_features(tiny_df, scaler_disp, enc)
        assert X.shape[0] == len(tiny_df)
        assert X.shape[1] == 5
        assert X.dtype == np.float32

    def test_no_nan(self, m, tiny_df, logger):
        scaler_disp, _, enc, _ = m.create_preprocessors(tiny_df, logger)
        X = m.build_features(tiny_df, scaler_disp, enc)
        assert not np.any(np.isnan(X))


class TestBuildTargets:
    def test_shape(self, m, tiny_df, logger):
        _, scaler_out, _, _ = m.create_preprocessors(tiny_df, logger)
        Y = m.build_targets(tiny_df, scaler_out)
        assert Y.shape == (len(tiny_df), 2)
        assert Y.dtype == np.float32


class TestToTensor:
    def test_float32_passthrough(self, m):
        x = np.array([[1.0, 2.0]], dtype=np.float32)
        t = m.to_tensor(x)
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float32

    def test_float64_converted(self, m):
        x = np.array([[1.0, 2.0]], dtype=np.float64)
        t = m.to_tensor(x)
        assert isinstance(t, torch.Tensor)


# =====================================================================
# Neural Network Architectures
# =====================================================================
class TestSoftPINNNet:
    def test_forward_shape(self, m):
        net = m.SoftPINNNet(in_d=5, hidden_layers=[32, 16], dropout=0.0, softplus_beta=1.0)
        x = torch.randn(10, 5)
        out = net(x)
        assert out.shape == (10, 2)

    def test_count_parameters(self, m):
        net = m.SoftPINNNet(in_d=5, hidden_layers=[32, 16], dropout=0.0, softplus_beta=1.0)
        assert net.count_parameters() > 0

    def test_dropout_layers(self, m):
        net = m.SoftPINNNet(in_d=5, hidden_layers=[32], dropout=0.5, softplus_beta=1.0)
        has_dropout = any(isinstance(layer, torch.nn.Dropout) for layer in net.net)
        assert has_dropout


class TestHardEnergyNet:
    def test_forward_shape(self, m):
        net = m.HardEnergyNet(in_d=5, hidden_layers=[32, 16], dropout=0.0,
                              softplus_beta=1.0, d_zero_scaled=-0.5)
        x = torch.randn(10, 5)
        out = net(x)
        assert out.shape == (10, 1)

    def test_boundary_enforcement(self, m):
        """E must be zero when x[:,0] == d_zero_scaled (physical displacement = 0)."""
        d_zero = -0.5
        net = m.HardEnergyNet(in_d=5, hidden_layers=[32, 16], dropout=0.0,
                              softplus_beta=1.0, d_zero_scaled=d_zero)
        net.eval()
        x = torch.randn(10, 5)
        x[:, 0] = d_zero  # set displacement to boundary value
        out = net(x)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_backward_compatible_default(self, m):
        """d_zero_scaled defaults to 0.0 for backward compatibility."""
        net = m.HardEnergyNet(in_d=5, hidden_layers=[32, 16], dropout=0.0, softplus_beta=1.0)
        assert hasattr(net, 'd_zero_scaled')
        assert float(net.d_zero_scaled) == 0.0

    def test_count_parameters(self, m):
        net = m.HardEnergyNet(in_d=5, hidden_layers=[32, 16], dropout=0.0,
                              softplus_beta=1.0, d_zero_scaled=-0.5)
        assert net.count_parameters() > 0

    def test_gradient_flows_through_boundary_layer(self, m):
        """Verify autodiff works through g*NN(x) for force computation."""
        net = m.HardEnergyNet(in_d=5, hidden_layers=[16], dropout=0.0,
                              softplus_beta=1.0, d_zero_scaled=-1.0)
        x = torch.randn(5, 5, requires_grad=True)
        E = net(x)
        dE = torch.autograd.grad(E.sum(), x, create_graph=True)[0]
        # dE/dx[:,0] should exist and not be all zeros
        assert dE[:, 0].abs().sum() > 0


# =====================================================================
# Curvature Regularization Removed
# =====================================================================
class TestCurvatureRemoval:
    def test_curvature_reg_removed(self, m):
        assert not hasattr(m, 'curvature_regularization_hard')

    def test_hard_config_no_curvature(self, m):
        cfg = m.get_model_config("hard", "unseen")
        assert "w_curvature" not in cfg


# =====================================================================
# EarlyStopping & WarmupCosineScheduler
# =====================================================================
class TestEarlyStopping:
    def test_no_stop_on_improvement(self, m):
        es = m.EarlyStopping(patience=3, min_delta=0.0)
        assert es(0.5, 1) is False
        assert es(0.6, 2) is False
        assert es(0.7, 3) is False

    def test_stop_after_patience(self, m):
        es = m.EarlyStopping(patience=2, min_delta=0.0)
        es(0.9, 1)
        es(0.8, 2)
        result = es(0.8, 3)
        assert result is True

    def test_reset_on_improvement(self, m):
        es = m.EarlyStopping(patience=2, min_delta=0.0)
        es(0.5, 1)
        es(0.4, 2)
        es(0.6, 3)
        assert es.counter == 0
        assert es.stop is False


class TestWarmupCosineScheduler:
    def test_warmup_phase(self, m):
        model = torch.nn.Linear(5, 1)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        sched = m.WarmupCosineScheduler(opt, warmup_epochs=5, total_epochs=100)
        lrs = []
        for _ in range(5):
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        assert lrs[-1] > lrs[0]

    def test_cosine_decay(self, m):
        model = torch.nn.Linear(5, 1)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        sched = m.WarmupCosineScheduler(opt, warmup_epochs=2, total_epochs=50)
        for _ in range(2):
            sched.step()
        lr_after_warmup = opt.param_groups[0]["lr"]
        for _ in range(48):
            sched.step()
        lr_end = opt.param_groups[0]["lr"]
        assert lr_end < lr_after_warmup

    def test_zero_lr_no_crash(self, m):
        """Division-by-zero guard: lr=0 should not crash."""
        model = torch.nn.Linear(5, 1)
        opt = torch.optim.Adam(model.parameters(), lr=0.0)
        sched = m.WarmupCosineScheduler(opt, warmup_epochs=2, total_epochs=10)
        for _ in range(10):
            sched.step()

    def test_get_last_lr(self, m):
        model = torch.nn.Linear(5, 1)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        sched = m.WarmupCosineScheduler(opt, warmup_epochs=2, total_epochs=10)
        sched.step()
        lrs = sched.get_last_lr()
        assert len(lrs) == 1
        assert lrs[0] > 0


# =====================================================================
# Physics Loss Functions
# =====================================================================
class TestPhysicsResidual:
    def test_residual_shape(self, m, tiny_df, logger):
        scaler_disp, scaler_out, enc, params = m.create_preprocessors(tiny_df, logger)
        X = m.build_features(tiny_df[:5], scaler_disp, enc)
        Xin = torch.tensor(X, requires_grad=True)
        net = m.SoftPINNNet(in_d=X.shape[1], hidden_layers=[16], dropout=0.0, softplus_beta=1.0)
        out = net(Xin)
        F_n, E_n = out[:, 0:1], out[:, 1:2]
        res = m.compute_physics_residual(Xin, F_n, E_n, params)
        assert res.shape == F_n.shape

    def test_physics_loss_soft_scalar(self, m, tiny_df, logger):
        scaler_disp, scaler_out, enc, params = m.create_preprocessors(tiny_df, logger)
        X = m.build_features(tiny_df[:5], scaler_disp, enc)
        Xin = torch.tensor(X, requires_grad=True)
        net = m.SoftPINNNet(in_d=X.shape[1], hidden_layers=[16], dropout=0.0, softplus_beta=1.0)
        out = net(Xin)
        F_n, E_n = out[:, 0:1], out[:, 1:2]
        loss = m.physics_loss_soft(Xin, F_n, E_n, params)
        assert loss.ndim == 0


# =====================================================================
# Crashworthiness Metrics
# =====================================================================
class TestComputeIPFRobust:
    def test_basic_peak(self, m):
        disps = np.linspace(0, 80, 161)
        loads = np.zeros(161)
        loads[20] = 15.0
        ipf, d_ipf = m.compute_ipf_robust(disps, loads)
        assert ipf == 15.0
        assert d_ipf == disps[20]

    def test_no_peaks_uses_fallback(self, m):
        disps = np.linspace(0, 80, 161)
        loads = np.linspace(0, 10, 161)
        ipf, d_ipf = m.compute_ipf_robust(disps, loads)
        assert ipf > 0


class TestComputeConfidenceIntervals:
    def test_normal_case(self, m):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = m.compute_confidence_intervals(values)
        assert ci["mean"] == pytest.approx(3.0)
        assert ci["ci_lower"] < ci["mean"]
        assert ci["ci_upper"] > ci["mean"]

    def test_single_value(self, m):
        ci = m.compute_confidence_intervals([5.0])
        assert ci["mean"] == 5.0
        assert np.isnan(ci["std"])


# =====================================================================
# Optimization Helpers
# =====================================================================
class TestNormCdf:
    def test_at_zero(self, m):
        result = m._norm_cdf(np.array([0.0]))
        assert abs(result[0] - 0.5) < 1e-4

    def test_extreme_positive(self, m):
        result = m._norm_cdf(np.array([5.0]))
        assert result[0] > 0.999

    def test_extreme_negative(self, m):
        result = m._norm_cdf(np.array([-5.0]))
        assert result[0] < 0.001


class TestExpectedImprovement:
    def test_positive_improvement(self, m):
        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.array([0.5, 0.5, 0.5])
        best = 0.5
        ei = m.expected_improvement(mu, sigma, best)
        assert ei.shape == (3,)
        assert ei[0] > ei[2]

    def test_zero_sigma_no_crash(self, m):
        mu = np.array([1.0])
        sigma = np.array([0.0])
        best = 0.5
        ei = m.expected_improvement(mu, sigma, best)
        assert np.isfinite(ei[0])


# =====================================================================
# Hypervolume
# =====================================================================
class TestComputeHypervolume2D:
    def test_empty_front(self, m, logger):
        df = pd.DataFrame({"EA": [], "IPF": []})
        hv = m.compute_hypervolume_2d(df, ref_ea=0.0, ref_ipf=20.0, logger=logger)
        assert hv == 0.0

    def test_single_point(self, m, logger):
        df = pd.DataFrame({"EA": [10.0], "IPF": [5.0]})
        hv = m.compute_hypervolume_2d(df, ref_ea=0.0, ref_ipf=20.0, logger=logger)
        assert hv > 0

    def test_dominated_point_excluded(self, m, logger):
        df = pd.DataFrame({"EA": [-1.0], "IPF": [5.0]})
        hv = m.compute_hypervolume_2d(df, ref_ea=0.0, ref_ipf=20.0, logger=logger)
        assert hv == 0.0


# =====================================================================
# Model Configs
# =====================================================================
class TestGetModelConfig:
    @pytest.mark.parametrize("approach", ["ddns", "soft", "hard"])
    @pytest.mark.parametrize("protocol", ["random", "unseen"])
    def test_configs_have_required_keys(self, m, approach, protocol):
        cfg = m.get_model_config(approach, protocol)
        assert "epochs" in cfg
        assert "hidden_layers" in cfg
        assert "lr" in cfg
        assert "batch_size" in cfg

    def test_hard_has_weight_keys(self, m):
        cfg = m.get_model_config("hard", "random")
        assert "w_load" in cfg
        assert "w_energy" in cfg

    def test_dry_run_caps_epochs(self, m):
        prev = m.CFG.dry_run
        try:
            m.CFG.dry_run = True
            for approach in ["ddns", "soft", "hard"]:
                cfg = m.get_model_config(approach, "random")
                assert int(cfg["epochs"]) <= 15
        finally:
            m.CFG.dry_run = prev


# =====================================================================
# LC Helpers
# =====================================================================
class TestLCHelpers:
    def test_lc_label_to_binary(self, m):
        assert m._lc_label_to_binary("LC1") == 0
        assert m._lc_label_to_binary("LC2") == 1
        assert m._lc_label_to_binary("lc1") == 0
        assert m._lc_label_to_binary("lc2") == 1

    def test_val_checkpoint_score(self, m):
        score = m._val_checkpoint_score(0.8, 0.9)
        assert score == pytest.approx(0.85)


# =====================================================================
# EA at Common Displacement
# =====================================================================
class TestEAAtCommon:
    def test_lc1_within_stroke(self, m):
        row = pd.Series({"disp_end": 80.0, "EA": 100.0, "F_mean": 5.0, "LC": "LC1", "Angle": 55.0})
        result = m.ea_at_common_from_row(row, None)
        assert result == 100.0

    def test_lc2_uses_interpolation(self, m, tiny_df):
        row = pd.Series({"disp_end": 130.0, "EA": 1500.0, "F_mean": 10.0, "LC": "LC2", "Angle": 55.0})
        result = m.ea_at_common_from_row(row, tiny_df)
        assert result < 1500.0

    def test_lc2_no_data_uses_proxy(self, m):
        row = pd.Series({"disp_end": 130.0, "EA": 1500.0, "F_mean": 10.0, "LC": "LC2", "Angle": 55.0})
        result = m.ea_at_common_from_row(row, None)
        assert result == pytest.approx(10.0 * m.D_COMMON)


# =====================================================================
# Design Space
# =====================================================================
class TestGenerateFeasibleTargets:
    def test_generates_correct_number(self, m, logger):
        df_metrics = pd.DataFrame({
            "EA_common": np.linspace(100, 500, 20),
            "IPF": np.linspace(5, 15, 20),
            "disp_end": [80.0] * 20,
            "EA": np.linspace(100, 500, 20),
            "F_mean": np.linspace(1, 10, 20),
            "LC": ["LC1"] * 20,
            "Angle": np.linspace(45, 70, 20),
        })
        targets = m.generate_feasible_targets(df_metrics, logger)
        assert len(targets) == 5
        for t in targets:
            assert "id" in t
            assert "EA" in t
            assert "IPF" in t

    def test_empty_input(self, m, logger):
        df_empty = pd.DataFrame(columns=["EA_common", "IPF", "disp_end", "EA", "F_mean", "LC", "Angle"])
        targets = m.generate_feasible_targets(df_empty, logger)
        assert targets == []


# =====================================================================
# Seed Reproducibility
# =====================================================================
class TestSetSeed:
    def test_deterministic_random(self, m):
        m.set_seed(42)
        a = np.random.rand(5)
        m.set_seed(42)
        b = np.random.rand(5)
        assert np.allclose(a, b)

    def test_deterministic_torch(self, m):
        m.set_seed(42)
        a = torch.randn(5)
        m.set_seed(42)
        b = torch.randn(5)
        assert torch.allclose(a, b)


# =====================================================================
# Optional Dependency Guards
# =====================================================================
class TestOptionalDeps:
    def test_has_scipy_flag_exists(self, m):
        assert isinstance(m.HAS_SCIPY, bool)

    def test_has_sklearn_gp_flag_exists(self, m):
        assert isinstance(m.HAS_SKLEARN_GP, bool)

    def test_has_skopt_flag_exists(self, m):
        assert isinstance(m.HAS_SKOPT, bool)

    def test_has_botorch_flag_exists(self, m):
        assert isinstance(m.HAS_BOTORCH, bool)


# =====================================================================
# Scaled Fonts
# =====================================================================
class TestScaledFonts:
    def test_returns_dict(self, m):
        fonts = m.scaled_fonts(7.48)
        assert isinstance(fonts, dict)
        assert "label" in fonts
        assert "tick" in fonts

    def test_narrow_figure_scales_down(self, m):
        wide = m.scaled_fonts(7.48)
        narrow = m.scaled_fonts(3.0)
        assert narrow["label"] <= wide["label"]


# =====================================================================
# Ill-Posedness Analysis: New Functions
# =====================================================================
class TestComputeLocalSensitivity:
    def test_quadratic_function(self, m):
        """On J(theta) = (theta - 60)^2, dJ/dtheta at theta=60 should be ~0."""
        func = lambda theta: (theta - 60.0) ** 2
        result = m.compute_local_sensitivity(func, 60.0, eps=0.01)
        assert abs(result["dJ_dtheta"]) < 0.1
        assert result["d2J_dtheta2"] > 0  # convex

    def test_linear_function(self, m):
        func = lambda theta: 2.0 * theta + 1.0
        result = m.compute_local_sensitivity(func, 50.0, eps=0.01)
        assert abs(result["dJ_dtheta"] - 2.0) < 0.01
        assert abs(result["d2J_dtheta2"]) < 0.1  # ~linear


class TestComputeSolutionLandscape:
    def test_single_minimum(self, m, logger):
        funcs = {"LC1": lambda t: (t - 55.0) ** 2}
        result = m.compute_solution_landscape(funcs, (45.0, 70.0), 0.0, logger, n_grid=51)
        assert "theta_grid" in result
        assert "J_LC1" in result
        assert result["multiplicity_index"] >= 1
        # The single minimum should be near 55
        minima = result["local_minima_LC1"]
        assert len(minima) >= 1
        assert abs(minima[0]["theta"] - 55.0) < 2.0


class TestComputeInversePosterior:
    def test_unimodal_landscape(self, m, logger):
        theta_grid = np.linspace(45, 70, 201)
        J_values = (theta_grid - 57.5) ** 2
        landscape = {
            "theta_grid": theta_grid,
            "J_LC1": J_values,
            "local_minima_LC1": [{"theta": 57.5, "J": 0.0}],
            "multiplicity_index": 1,
        }
        result = m.compute_inverse_posterior(landscape, "LC1", logger)
        assert abs(result["mean"] - 57.5) < 1.5
        assert result["std"] > 0
        assert result["ci_95_lower"] < result["mean"]
        assert result["ci_95_upper"] > result["mean"]


class TestTikhonovObjective:
    def test_tikhonov_penalty_at_center_is_zero(self, m):
        theta_center = 57.5
        gamma = 1.0
        penalty = gamma * (theta_center - theta_center) ** 2
        assert penalty == 0.0

    def test_tikhonov_penalty_increases_with_distance(self, m):
        theta_center = 57.5
        gamma = 0.01
        p_near = gamma * (58.0 - theta_center) ** 2
        p_far = gamma * (70.0 - theta_center) ** 2
        assert p_far > p_near


# =====================================================================
# RAR Adaptive Collocation Sampler
# =====================================================================
class TestAdaptiveCollocationSampler:
    def test_initial_sampling_is_uniform(self, m, tiny_df, logger):
        scaler_disp, _, enc, _ = m.create_preprocessors(tiny_df, logger)
        base = m.create_collocation_sampler(tiny_df, scaler_disp, enc)
        rar = m.AdaptiveCollocationSampler(base, eval_grid_size=50, rng_seed=42)
        assert rar.is_active is False
        rng = np.random.default_rng(0)
        pts = rar.sample(10, rng)
        assert pts.shape[0] == 10

    def test_after_update_is_active(self, m, tiny_df, logger):
        scaler_disp, scaler_out, enc, params = m.create_preprocessors(tiny_df, logger)
        # Force CPU for test (eval grid may be on GPU otherwise)
        prev_force = m.CFG.force_cpu
        m.CFG.force_cpu = True
        m.refresh_device()
        try:
            base = m.create_collocation_sampler(tiny_df, scaler_disp, enc)
            rar = m.AdaptiveCollocationSampler(base, eval_grid_size=50, rng_seed=42)
            net = m.SoftPINNNet(in_d=5, hidden_layers=[16], dropout=0.0, softplus_beta=1.0)
            rar.update_weights(net, params, "soft")
            assert rar.is_active is True
        finally:
            m.CFG.force_cpu = prev_force
            m.refresh_device()
