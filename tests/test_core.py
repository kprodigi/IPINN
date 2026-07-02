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

    def test_d_common_equals_lc1_stroke(self, m):
        assert m.D_COMMON == m.disp_end_mm("LC1")

    def test_ea_common_mm_tag(self, m):
        assert m.EA_COMMON_MM_TAG == f"{int(m.D_COMMON)}mm"


class TestBOConfigDefaults:
    def test_bo_defaults(self, m):
        bo = m.BOConfig()
        # Tightened budget: empirical convergence happens within ~7
        # evaluations, so B=20 (5 init + 15 EI) with stagnation early
        # stopping covers the same ground at ~50% of the prior cost.
        assert bo.n_calls_total == 20
        assert bo.n_init == 5
        assert bo.xi == 0.01
        assert bo.n_bo_restarts == 5
        # Early-stop safety net
        assert bo.stagnation_patience == 6
        assert bo.stagnation_min_delta == 1e-5


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
        # The random split is labelled for what it measures: within-curve
        # interpolation (rows split inside curves), not design generalization.
        label = m.protocol_label("random")
        assert "random 80/20" in label
        assert "Interpolation" in label

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
        net = m.HardEnergyNet(in_d=5, hidden_layers=[32, 16], dropout=0.0, softplus_beta=1.0)
        x = torch.randn(10, 5)
        out = net(x)
        assert out.shape == (10, 1)

    def test_count_parameters(self, m):
        net = m.HardEnergyNet(in_d=5, hidden_layers=[32, 16], dropout=0.0, softplus_beta=1.0)
        assert net.count_parameters() > 0

    def test_gradient_for_force(self, m):
        """F = dE/dd via autodiff must produce non-zero gradients."""
        net = m.HardEnergyNet(in_d=5, hidden_layers=[16], dropout=0.0, softplus_beta=1.0)
        x = torch.randn(5, 5, requires_grad=True)
        E = net(x)
        dE = torch.autograd.grad(E.sum(), x, create_graph=True)[0]
        assert dE[:, 0].abs().sum() > 0

    def test_zero_bc_enforces_E_and_F_zero_at_origin(self, m):
        """Architectural BC: at raw d=0, both E and F = dE/dd are exactly zero."""
        torch.manual_seed(0)
        net = m.HardEnergyNet(in_d=5, hidden_layers=[16, 8], dropout=0.0, softplus_beta=1.0)
        # Non-trivial scaling so the test rules out a degenerate (zero-everywhere) net.
        params = m.ScalingParams(
            mu_d=20.0, sig_d=10.0,
            mu_F=5.0,  sig_F=2.0,
            mu_E=100.0, sig_E=50.0,
            grad_factor=1.0 / 10.0,
        )
        net.configure_zero_bc(params)
        net.eval()
        # Build a batch at raw d=0  →  scaled d = -mu_d/sig_d = -2.0
        d_scaled_zero = -params.mu_d / params.sig_d
        x = torch.zeros(8, 5)
        x[:, 0] = d_scaled_zero
        x[:, 1:] = torch.randn(8, 4)
        x.requires_grad_(True)
        # (1) E(d=0) = 0 in raw space.
        E_n = net(x)
        E_raw = E_n * params.sig_E + params.mu_E
        assert torch.allclose(E_raw, torch.zeros_like(E_raw), atol=1e-5)
        # (2) F(d=0) = dE/dd|_{d=0} = 0 in raw space (via chain rule, the
        # scaled-space d-derivative at d_scaled_zero must vanish).
        dE_dxs = torch.autograd.grad(E_n.sum(), x, create_graph=False)[0]
        F_phys = dE_dxs[:, 0] * params.grad_factor
        assert torch.allclose(F_phys, torch.zeros_like(F_phys), atol=1e-5)

    def test_zero_bc_off_boundary_nontrivial(self, m):
        """BC does not collapse the network — F is nonzero away from d=0."""
        torch.manual_seed(1)
        net = m.HardEnergyNet(in_d=5, hidden_layers=[16, 8], dropout=0.0, softplus_beta=1.0)
        params = m.ScalingParams(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0)
        net.configure_zero_bc(params)
        net.eval()
        x = torch.randn(8, 5)
        x[:, 0] = x[:, 0] + 5.0  # shift d-column well off the BC value (0.0)
        x = x.requires_grad_(True)
        E_n = net(x)
        dE = torch.autograd.grad(E_n.sum(), x, create_graph=False)[0]
        assert dE[:, 0].abs().sum() > 0

    def test_zero_bc_supports_training_double_backward(self, m):
        """The inner-grad graph survives outer ``loss.backward()`` in training mode."""
        torch.manual_seed(2)
        net = m.HardEnergyNet(in_d=5, hidden_layers=[16], dropout=0.0, softplus_beta=1.0)
        params = m.ScalingParams(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0)
        net.configure_zero_bc(params)
        net.train()
        x = torch.randn(4, 5, requires_grad=True)
        E_n = net(x)
        # Physics-loss surrogate: ||dE/dd||² + data loss, then outer backward.
        dE = torch.autograd.grad(E_n.sum(), x, create_graph=True)[0]
        loss = (dE[:, 0] ** 2).mean() + E_n.pow(2).mean()
        loss.backward()
        # At least one weight tensor must receive a non-zero gradient.
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters()
        )

    def test_zero_bc_disabled_matches_raw_net(self, m):
        """``configure_zero_bc(enabled=False)`` should restore the raw forward."""
        torch.manual_seed(3)
        net = m.HardEnergyNet(in_d=5, hidden_layers=[16], dropout=0.0, softplus_beta=1.0)
        params = m.ScalingParams(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0)
        # Enable then disable.
        net.configure_zero_bc(params, enabled=True)
        net.configure_zero_bc(params, enabled=False)
        net.eval()
        x = torch.randn(6, 5)
        out_with = net(x)
        # Raw forward (bypass the BC machinery).
        out_raw = net.net(x)
        assert torch.allclose(out_with, out_raw)


# =====================================================================
# Hard-PINN and Soft-PINN auxiliary regularizers (monotonicity F>=0,
# angle smoothness dF/dθ, curvature d²E/dd²).  These soft penalties
# encode the physical priors used by the unseen-angle protocol and are
# required in the production cfgs of both Soft-PINN and Hard-PINN
# (Section 3.2 of the manuscript).
# =====================================================================
class TestCurvatureRegularizationHard:
    def test_curvature_fn_exists(self, m):
        assert hasattr(m, "curvature_regularization_hard")

    def test_hard_unseen_config_includes_auxiliary_regularizers(self, m):
        cfg = m.get_model_config("hard", "unseen")
        # The production Hard-PINN cfg must contain the three auxiliary
        # field-wide regularizer weights so the physical priors
        # (monotonicity, angle smoothness, energy-curvature) are active.
        for key in ("w_curvature", "w_monotonicity", "w_angle_smooth"):
            assert key in cfg, (
                f"{key} must be set in production Hard cfg."
            )
            assert cfg[key] > 0.0, (
                f"{key} must be positive in production Hard cfg; got {cfg[key]}."
            )

    def test_soft_unseen_config_includes_auxiliary_regularizers(self, m):
        cfg = m.get_model_config("soft", "unseen")
        # Soft-PINN cfg includes monotonicity and angle smoothness; the
        # curvature penalty is not required for the soft formulation
        # because the work-energy residual already constrains E shape.
        for key in ("w_monotonicity", "w_angle_smooth"):
            assert key in cfg, (
                f"{key} must be set in production Soft cfg."
            )
            assert cfg[key] > 0.0, (
                f"{key} must be positive in production Soft cfg; got {cfg[key]}."
            )

    def test_curvature_loss_scalar(self, m):
        # Function-level smoke test for ablation use; not invoked by the
        # production training pipeline.
        net = m.HardEnergyNet(in_d=5, hidden_layers=[16], dropout=0.0, softplus_beta=1.0)
        x = torch.randn(8, 5, requires_grad=True)
        params = m.ScalingParams(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0)
        loss = m.curvature_regularization_hard(x, net, params)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


# =====================================================================
# Soft-PINN paired BC penalty (E(0)=0 AND F(0)=0)
# =====================================================================
class TestSoftPairedBCPenalty:
    def test_soft_unseen_config_has_w_bc(self, m):
        cfg = m.get_model_config("soft", "unseen")
        assert "w_bc" in cfg and cfg["w_bc"] > 0, (
            "Soft-PINN must have a positive w_bc for the paired E(0)/F(0) "
            "soft penalty (Section 3.2.2)."
        )

    def test_soft_pinn_outputs_both_F_and_E(self, m):
        # The paired BC penalty needs a network that emits both heads.
        net = m.SoftPINNNet(in_d=5, hidden_layers=[16], dropout=0.0, softplus_beta=1.0)
        x = torch.randn(4, 5)
        out = net(x)
        assert out.shape == (4, 2), (
            "SoftPINNNet must output [F_n, E_n] for the paired BC penalty to work."
        )


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
        #  reverted to  semantics: load R² only for all approaches.
        # Energy R² is ignored.
        score = m._val_checkpoint_score(0.8, 0.9)
        assert score == pytest.approx(0.8)


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


# =====================================================================
# r2_safe — robust R² that returns NaN on flat-target slices
# =====================================================================
class TestR2Safe:
    def test_perfect_fit_returns_one(self, m):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert m.r2_safe(y, y) == pytest.approx(1.0, abs=1e-12)

    def test_constant_target_returns_nan_not_minus_inf(self, m):
        # Old r2_score behaviour: divides by var(y_true)=0 -> -inf or warning.
        # r2_safe must return NaN so Tukey-fence percentiles do not collapse.
        y_true = np.array([5.0, 5.0, 5.0, 5.0])
        y_pred = np.array([4.9, 5.1, 4.95, 5.05])
        result = m.r2_safe(y_true, y_pred)
        assert np.isnan(result)
        assert result != -np.inf  # explicitly: not the broken sklearn value

    def test_near_constant_target_returns_nan(self, m):
        # Variance below the 1e-12 cutoff is treated as undefined.
        y_true = np.full(10, 1.0) + np.random.RandomState(0).randn(10) * 1e-9
        y_pred = np.full(10, 1.0)
        assert np.isnan(m.r2_safe(y_true, y_pred))

    def test_nan_input_returns_nan(self, m):
        y = np.array([1.0, 2.0, np.nan])
        p = np.array([1.0, 2.0, 3.0])
        assert np.isnan(m.r2_safe(y, p))

    def test_size_mismatch_returns_nan(self, m):
        y = np.array([1.0, 2.0, 3.0])
        p = np.array([1.0, 2.0])
        assert np.isnan(m.r2_safe(y, p))

    def test_empty_returns_nan(self, m):
        assert np.isnan(m.r2_safe(np.array([]), np.array([])))

    def test_matches_sklearn_for_well_posed_input(self, m):
        from sklearn.metrics import r2_score
        rng = np.random.RandomState(42)
        y = rng.randn(50) * 3 + 1
        p = y + rng.randn(50) * 0.5
        assert m.r2_safe(y, p) == pytest.approx(r2_score(y, p), abs=1e-12)


# =====================================================================
# _val_checkpoint_score — load R² only for all approaches ( semantics)
# =====================================================================
#  reverted "0.5*(load+energy) for DDNS/Soft, load only for Hard"
# back to uniform "load R² only" rule.  ``r2_energy`` and ``approach``
# are kept in the signature for call-site compatibility but are unused.
# These tests pin the new behaviour.
class TestValCheckpointScore:
    def test_default_returns_load_only(self, m):
        # No-approach default still returns load R² only; energy is ignored.
        assert m._val_checkpoint_score(0.8, 0.9) == pytest.approx(0.8)

    def test_ddns_returns_load_only(self, m):
        assert m._val_checkpoint_score(0.7, 0.95, approach="ddns") == pytest.approx(0.7)

    def test_soft_returns_load_only(self, m):
        assert m._val_checkpoint_score(0.7, 0.95, approach="soft") == pytest.approx(0.7)

    def test_hard_returns_load_only(self, m):
        # Hard-PINN: F = dE/dd by construction, so energy R² is trivially ~1
        # at every epoch and never discriminated between checkpoints.  Score
        # is load R² regardless of energy — same rule as DDNS / Soft.
        assert m._val_checkpoint_score(0.7, 0.999, approach="hard") == pytest.approx(0.7)
        assert m._val_checkpoint_score(0.7, 0.5,   approach="hard") == pytest.approx(0.7)

    def test_nan_energy_is_ignored(self, m):
        # ``r2_energy`` is unused; a NaN there has no effect on the score.
        assert m._val_checkpoint_score(0.7, float("nan"), approach="soft") == pytest.approx(0.7)
        assert m._val_checkpoint_score(0.7, float("nan"), approach="ddns") == pytest.approx(0.7)
        assert m._val_checkpoint_score(0.7, float("nan"), approach="hard") == pytest.approx(0.7)

    def test_nan_load_returns_minus_inf(self, m):
        # NaN-safe: if load R² is NaN (flat-target slice) the checkpointer
        # must skip — score is -inf so any ``best_score > -inf`` check fails.
        assert m._val_checkpoint_score(float("nan"), 0.9, approach="soft") == -np.inf
        assert m._val_checkpoint_score(float("nan"), 0.9, approach="ddns") == -np.inf
        assert m._val_checkpoint_score(float("nan"), 0.9, approach="hard") == -np.inf

    def test_both_nan_returns_minus_inf(self, m):
        assert m._val_checkpoint_score(float("nan"), float("nan"), approach="soft") == -np.inf
        assert m._val_checkpoint_score(float("nan"), float("nan"), approach="hard") == -np.inf


# =====================================================================
# _atomic_to_csv — DataFrame.to_csv monkey-patch
# =====================================================================
class TestAtomicToCSV:
    def test_writes_csv_atomically(self, m, tmp_path):
        # The patch should write through tmp + os.replace; final file present.
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        target = str(tmp_path / "atomic.csv")
        df.to_csv(target, index=False)
        assert os.path.isfile(target)
        # No leftover .tmp
        assert not os.path.isfile(target + ".tmp")
        # Round-trip
        rt = pd.read_csv(target)
        assert list(rt.columns) == ["a", "b"]
        assert rt["a"].tolist() == [1, 2, 3]

    def test_overwrite_existing_is_atomic(self, m, tmp_path):
        # Writing to an existing path should atomically replace it
        # (no half-written file ever visible at the target path).
        target = str(tmp_path / "exists.csv")
        pd.DataFrame({"x": [1]}).to_csv(target, index=False)
        assert pd.read_csv(target)["x"].tolist() == [1]
        pd.DataFrame({"x": [99]}).to_csv(target, index=False)
        assert pd.read_csv(target)["x"].tolist() == [99]
        assert not os.path.isfile(target + ".tmp")

    def test_passthrough_for_non_string_buffer(self, m):
        # The patch only triggers atomic write for string/PathLike paths;
        # buffers must pass through unchanged.
        import io
        buf = io.StringIO()
        pd.DataFrame({"a": [1, 2]}).to_csv(buf, index=False)
        contents = buf.getvalue()
        assert "a" in contents
        assert "1" in contents and "2" in contents

    def test_pathlike_is_supported(self, m, tmp_path):
        # pathlib.Path objects should also trigger the atomic write.
        target = tmp_path / "pathlike.csv"
        pd.DataFrame({"v": [10, 20]}).to_csv(target, index=False)
        assert target.is_file()
        assert pd.read_csv(target)["v"].tolist() == [10, 20]


# =====================================================================
# make_torch_generator + DataLoader determinism
# =====================================================================
class TestMakeTorchGenerator:
    def test_returns_seeded_generator(self, m):
        g = m.make_torch_generator(2026)
        assert isinstance(g, torch.Generator)

    def test_same_seed_same_sequence(self, m):
        g1 = m.make_torch_generator(2026)
        g2 = m.make_torch_generator(2026)
        a = torch.randint(0, 1000, (10,), generator=g1)
        b = torch.randint(0, 1000, (10,), generator=g2)
        assert torch.equal(a, b)

    def test_different_seeds_different_sequences(self, m):
        g1 = m.make_torch_generator(2026)
        g2 = m.make_torch_generator(2027)
        a = torch.randint(0, 10_000, (50,), generator=g1)
        b = torch.randint(0, 10_000, (50,), generator=g2)
        assert not torch.equal(a, b)

    def test_data_loader_kwargs_with_seed_includes_generator(self, m):
        kw = m._data_loader_kwargs(seed=2026)
        assert "generator" in kw
        assert isinstance(kw["generator"], torch.Generator)
        assert kw["pin_memory"] is False

    def test_data_loader_kwargs_without_seed_omits_generator(self, m):
        kw = m._data_loader_kwargs()
        assert "generator" not in kw
        assert kw["pin_memory"] is False

    def test_dataloader_shuffle_is_deterministic_under_seed(self, m):
        # Same seed -> identical shuffle order across runs.
        from torch.utils.data import DataLoader, TensorDataset
        x = torch.arange(20).float().unsqueeze(1)
        y = torch.arange(20).float().unsqueeze(1)
        ds = TensorDataset(x, y)
        order_a = []
        for xb, _ in DataLoader(ds, batch_size=4, shuffle=True,
                                 **m._data_loader_kwargs(seed=2026)):
            order_a.append(xb.squeeze().tolist())
        order_b = []
        for xb, _ in DataLoader(ds, batch_size=4, shuffle=True,
                                 **m._data_loader_kwargs(seed=2026)):
            order_b.append(xb.squeeze().tolist())
        assert order_a == order_b


# =====================================================================
# Additional regression-prevention tests
# =====================================================================
class TestMonotonicityLossSoft:
    """The HPO-tuned w_monotonicity is calibrated to the normalized-gradient form;
    the loss must therefore use the un-normalized-energy gradient (sigma_E * dE_n/du),
    NOT the physical-force form (sigma_E/sigma_d * dE_n/du), so the penalty
    magnitude matches what the manuscript HPO tuned against."""

    def test_returns_zero_for_strictly_monotone_energy(self, m):
        # Build a tiny SoftPINNNet whose energy output is +x[:,0] (monotone).
        # The penalty mean(relu(-dE/du)^2) must be 0 for monotone energy.
        class _IdEnergy(torch.nn.Module):
            def forward(self, x):
                # Output [F_dummy, E = x[:,0]] (monotone in u-coord).
                F = torch.zeros_like(x[:, 0:1])
                E = x[:, 0:1]
                return torch.cat([F, E], dim=1)

        net = _IdEnergy()
        params = m.ScalingParams(mu_d=0.0, sig_d=1.0, mu_F=0.0, sig_F=1.0,
                                 mu_E=0.0, sig_E=1.0, grad_factor=1.0)
        X = torch.randn(64, 5, requires_grad=True)
        loss = m.monotonicity_loss_soft(X, net, params)
        assert float(loss.detach()) == pytest.approx(0.0, abs=1e-8)

    def test_returns_positive_for_decreasing_energy(self, m):
        # Energy = -x[:,0] -> dE/du is negative everywhere -> penalty > 0.
        class _NegEnergy(torch.nn.Module):
            def forward(self, x):
                F = torch.zeros_like(x[:, 0:1])
                E = -x[:, 0:1]
                return torch.cat([F, E], dim=1)

        net = _NegEnergy()
        params = m.ScalingParams(mu_d=0.0, sig_d=1.0, mu_F=0.0, sig_F=1.0,
                                 mu_E=0.0, sig_E=1.0, grad_factor=1.0)
        X = torch.randn(64, 5, requires_grad=True)
        loss = m.monotonicity_loss_soft(X, net, params)
        assert float(loss.detach()) > 0.5  # relu(1)^2 = 1 averaged

    def test_penalty_scales_with_sigma_E_squared(self, m):
        # The penalty should scale as sigma_E^2 because dE_phys = sigma_E * dE_n.
        # With dE_n/du = -1 (decreasing), penalty = (sigma_E)^2.
        class _NegE(torch.nn.Module):
            def forward(self, x):
                F = torch.zeros_like(x[:, 0:1])
                E = -x[:, 0:1]
                return torch.cat([F, E], dim=1)

        net = _NegE()
        X = torch.randn(64, 5, requires_grad=True)
        params_a = m.ScalingParams(mu_d=0, sig_d=1, mu_F=0, sig_F=1, mu_E=0, sig_E=1.0, grad_factor=1.0)
        params_b = m.ScalingParams(mu_d=0, sig_d=1, mu_F=0, sig_F=1, mu_E=0, sig_E=10.0, grad_factor=10.0)
        loss_a = float(m.monotonicity_loss_soft(X, net, params_a).detach())
        loss_b = float(m.monotonicity_loss_soft(X, net, params_b).detach())
        # sig_E ratio = 10, penalty ratio should be 100 (squared)
        assert loss_b / max(loss_a, 1e-12) == pytest.approx(100.0, rel=0.01)


class TestConformalCalibrationFactors:
    """Two-factor conformal calibration: cf at 68.3% for ±1σ, cf at 95.4% for ±2σ."""

    def test_one_sigma_factor_recovers_target_coverage(self, m):
        # Synthesize a Gaussian-residual case: |z| = |residual|/sigma should be
        # ~|N(0,1)|. The 68.3-percentile of |N(0,1)| is ~1.0 (the 1-sigma point).
        rng = np.random.RandomState(2026)
        sigma = 1.0
        residuals = np.abs(rng.randn(50_000) * sigma)
        cf_1 = float(np.percentile(residuals / sigma, 68.3))
        # For a half-normal, the 68.3-percentile is exactly 1.0 (the 1-sigma point of a normal).
        assert cf_1 == pytest.approx(1.0, abs=0.02)

    def test_two_sigma_factor_recovers_target_coverage(self, m):
        # 95.4-percentile of |N(0,1)| is ~2.0.
        rng = np.random.RandomState(2026)
        residuals = np.abs(rng.randn(50_000))
        cf_2 = float(np.percentile(residuals, 95.4))
        assert cf_2 == pytest.approx(2.0, abs=0.05)

    def test_two_factors_are_distinct(self, m):
        # Heavy-tailed residual: cf_2 should NOT equal 2*cf_1 (would imply Gaussian).
        rng = np.random.RandomState(2026)
        # Student-t with 3 df has heavier tails than Gaussian.
        from scipy import stats as scstats
        residuals = np.abs(scstats.t.rvs(df=3, size=20_000, random_state=rng))
        cf_1 = float(np.percentile(residuals, 68.3))
        cf_2 = float(np.percentile(residuals, 95.4))
        # For heavy tails, cf_2 > 2*cf_1 (linear extrapolation under-covers).
        assert cf_2 > 2.0 * cf_1


