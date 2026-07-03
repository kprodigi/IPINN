"""Property tests for the Hard-PINN architecture variants (Tier 1 + Tier 2).

The variants' whole value is that physics plausibility is a THEOREM, not a
soft penalty — so these tests check the guarantees exactly, at inputs far
outside the training range (the regime the soft penalties never see).
"""

import logging

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")


@pytest.fixture()
def params(m):
    return m.ScalingParams(mu_d=40.0, sig_d=25.0, mu_F=8.0, sig_F=4.0,
                           mu_E=300.0, sig_E=200.0, grad_factor=200.0 / 25.0)


def _random_inputs(params, n=2048, seed=0, d_max_mm=200.0, theta_max_rad=3.14159):
    """Inputs deliberately beyond the measured range (d to 200mm, θ to 180°)."""
    g = torch.Generator().manual_seed(seed)
    d_raw = torch.rand(n, 1, generator=g) * d_max_mm
    d_s = (d_raw - params.mu_d) / params.sig_d
    th = torch.rand(n, 1, generator=g) * theta_max_rad
    lc = torch.nn.functional.one_hot(torch.randint(0, 2, (n,), generator=g), 2).float()
    return torch.cat([d_s, torch.sin(th), torch.cos(th), lc], dim=1)


class TestMonotoneGuarantees:
    @pytest.mark.parametrize("factory", [
        lambda m: m.MonotoneHardEnergyNet(5, [32, 16], 0.0, 10.0),
        lambda m: m.SeparableHardEnergyNet(5, [32, 16], 0.0, 10.0, n_basis=4, monotone=True),
    ])
    def test_force_nonnegative_everywhere(self, m, params, factory):
        torch.manual_seed(1)
        net = factory(m)
        net.configure_zero_bc(params)
        net.eval()
        X = _random_inputs(params).requires_grad_(True)
        E = net(X)
        dE_dd = torch.autograd.grad(E.sum(), X)[0][:, m.U_COL]
        assert float(dE_dd.min()) >= -1e-8, "monotone variant produced dE/dd < 0"

    @pytest.mark.parametrize("factory", [
        lambda m: m.MonotoneHardEnergyNet(5, [16, 8], 0.0, 10.0),
        lambda m: m.SeparableHardEnergyNet(5, [16, 8], 0.0, 10.0, monotone=True),
        lambda m: m.SeparableHardEnergyNet(5, [16, 8], 0.0, 10.0, monotone=False),
    ])
    def test_raw_energy_zero_at_d0_exactly(self, m, params, factory):
        torch.manual_seed(2)
        net = factory(m)
        net.configure_zero_bc(params)
        net.eval()
        X = _random_inputs(params, n=256)
        X0 = X.clone()
        X0[:, m.U_COL] = -params.mu_d / params.sig_d  # raw d = 0
        E_raw = net(X0) * params.sig_E + params.mu_E
        assert float(E_raw.abs().max()) < 1e-3, "raw E(d=0) != 0"

    def test_monotone_energy_nonnegative_for_positive_d(self, m, params):
        torch.manual_seed(3)
        net = m.MonotoneHardEnergyNet(5, [16, 8], 0.0, 10.0)
        net.configure_zero_bc(params)
        net.eval()
        X = _random_inputs(params, n=1024)
        E_raw = net(X) * params.sig_E + params.mu_E
        assert float(E_raw.min()) >= -1e-3, "monotone+E(0)=0 must give E >= 0 for d >= 0"

    def test_double_backward_training_compatible(self, m, params):
        torch.manual_seed(4)
        for net in (m.MonotoneHardEnergyNet(5, [16, 8], 0.1, 10.0),
                    m.SeparableHardEnergyNet(5, [16, 8], 0.1, 10.0, monotone=True)):
            net.configure_zero_bc(params)
            net.train()
            X = _random_inputs(params, n=64).requires_grad_(True)
            E = net(X)
            dE = torch.autograd.grad(E.sum(), X, create_graph=True)[0][:, m.U_COL]
            loss = (dE ** 2).mean() + E.pow(2).mean()
            loss.backward()  # physics losses need grad-of-grad
            assert any(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in net.parameters())


class TestSeparableCapacity:
    def test_theta_dependence_is_single_linear_map(self, m):
        net = m.SeparableHardEnergyNet(5, [32, 16], 0.0, 10.0, n_basis=4)
        # The whole point of Tier 2: θ/LC capacity limited to one linear layer
        # (z_dim x n_basis weights + n_basis biases).
        assert isinstance(net.phi, torch.nn.Linear)
        assert net.phi.weight.shape == (4, 4)  # n_basis x z_dim

    def test_positive_linear_weights_positive(self, m):
        pl = m.PositiveLinear(8, 4)
        assert float(pl.effective_weight.min()) > 0.0


class TestFactory:
    @pytest.fixture()
    def cfg(self):
        return {"hidden_layers": [16, 8], "dropout": 0.0, "softplus_beta": 10.0}

    def test_default_is_published_mlp(self, m, cfg, params):
        m.CFG.hard_architecture = ""
        net = m.make_hard_energy_net(5, cfg, params)
        assert type(net).__name__ == "HardEnergyNet"

    def test_cfg_and_global_override(self, m, cfg, params):
        try:
            net = m.make_hard_energy_net(5, {**cfg, "architecture": "monotone"}, params)
            assert type(net).__name__ == "MonotoneHardEnergyNet"
            m.CFG.hard_architecture = "monotone_separable"
            net = m.make_hard_energy_net(5, cfg, params)
            assert type(net).__name__ == "SeparableHardEnergyNet" and net.monotone
        finally:
            m.CFG.hard_architecture = ""

    def test_unknown_architecture_rejected(self, m, cfg):
        with pytest.raises(ValueError):
            m.make_hard_energy_net(5, {**cfg, "architecture": "quantum"}, None)

    def test_state_dict_round_trip(self, m, params):
        torch.manual_seed(5)
        a = m.SeparableHardEnergyNet(5, [16, 8], 0.0, 10.0, monotone=True)
        a.configure_zero_bc(params)
        b = m.SeparableHardEnergyNet(5, [16, 8], 0.0, 10.0, monotone=True)
        b.load_state_dict(a.state_dict())
        x = _random_inputs(params, n=32)
        assert torch.allclose(a(x), b(x))


class TestInterpretabilityArtifacts:
    """The explainability layer must be exact, not approximate."""

    @pytest.fixture()
    def quiet(self):
        lg = logging.getLogger("t_interp")
        lg.setLevel(logging.CRITICAL)
        lg.addHandler(logging.NullHandler())
        return lg

    def test_variance_decomposition_exact_attribution(self, m, quiet):
        import itertools
        angles = np.linspace(45, 70, 51)
        # EA depends only on theta; IPF only on LC -> exact 100/0 split.
        rows = [{"angle": a, "lc": lc, "EA": 2.0 * a, "IPF": 1.0 + (lc == "LC2")}
                for a, lc in itertools.product(angles, ["LC1", "LC2"])]
        df = m.compute_design_variance_decomposition(pd.DataFrame(rows), quiet, None)
        ea = df[df["Quantity"].str.startswith("EA")].iloc[0]
        ipf = df[df["Quantity"] == "IPF"].iloc[0]
        assert ea["S_theta"] == pytest.approx(1.0, abs=1e-6)
        assert ipf["S_LC"] == pytest.approx(1.0, abs=1e-6)

    def test_variance_decomposition_detects_interaction(self, m, quiet):
        import itertools
        angles = np.linspace(45, 70, 51)
        # Multiplicative theta x LC surface -> nonzero interaction share.
        rows = [{"angle": a, "lc": lc, "EA": a * (1.0 if lc == "LC1" else 2.0),
                 "IPF": 1.0} for a, lc in itertools.product(angles, ["LC1", "LC2"])]
        df = m.compute_design_variance_decomposition(pd.DataFrame(rows), quiet, None)
        ea = df[df["Quantity"].str.startswith("EA")].iloc[0]
        assert ea["S_interaction"] > 0.005
        assert abs(ea["S_theta"] + ea["S_LC"] + ea["S_interaction"] - 1.0) < 1e-6

    def test_inverse_explanation_decomposes_objective(self, m, quiet, tmp_path):
        res = [{"target_ea": 30.0, "target_ipf": 0.6, "prob_weight": 0.0,
                "target_info": {"id": "T1"},
                "gpbo_best": {"x_best": 57.0, "lc": "LC1", "y_best": 0.001,
                              "pred_ea": 30.9, "pred_ipf": 0.594}}]
        df = m.table_inverse_design_explanation(res, None, None, str(tmp_path), quiet)
        row = df.iloc[0]
        # With 1/target^2 weighting: sqrt(term) == relative error, exactly.
        assert float(row["EA_rel_error_pct"]) == pytest.approx(3.0, abs=0.01)
        assert float(row["IPF_rel_error_pct"]) == pytest.approx(1.0, abs=0.01)
        assert row["dominant_term"] == "EA_fit"
        total = (float(row["EA_fit_term"]) + float(row["IPF_fit_term"])
                 + float(row["classifier_penalty_term"]))
        assert total == pytest.approx(0.001, rel=1e-3)
        assert (tmp_path / "Table_inverse_design_explanation.csv").exists()

    def test_separable_figure_is_faithful_readout(self, m, params, tmp_path, quiet):
        torch.manual_seed(6)

        class _Enc:
            categories_ = [np.array(["LC1", "LC2"])]

        net = m.SeparableHardEnergyNet(5, [16, 8], 0.0, 10.0, n_basis=3, monotone=True)
        net.configure_zero_bc(params)
        out = m.fig_separable_interpretability([net], _Enc(), params, str(tmp_path), quiet)
        assert out is not None and (tmp_path / "Fig_separable_interpretability.png").exists()
        # Non-separable models: no figure, no crash.
        plain = m.HardEnergyNet(5, [16, 8], 0.0, 10.0)
        assert m.fig_separable_interpretability([plain], _Enc(), params, str(tmp_path), quiet) is None
