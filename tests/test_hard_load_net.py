"""Unit tests for HardLoadNet (flipped Hard-PINN parameterization).

Verifies the four architectural properties that justify the load-primary
parameterization against the existing energy-primary one:

  1. F(d=0) = 0 by construction (architectural BC)
  2. E(d=0) = 0 by construction (∫₀⁰ F dδ = 0, no model-side correction needed)
  3. The trapezoidal Ê(d) is a self-consistent integral of the model's F̂(d)
     — independent recomputation by torch.trapezoid recovers the same value
     to within 5e-3 relative error on smooth random initializations.
  4. The integration grid is dense enough that the K=64 default agrees with
     K=256 to within 1e-3 — i.e., the discretization error is dwarfed by the
     fit error we care about.
  5. Backprop through the integration delivers nonzero, non-NaN gradients
     to every backbone parameter.
"""
import math

import numpy as np
import pytest
import torch

from composite_design import (
    HARD_LOAD_K_INTEGRATION,
    HardLoadNet,
    ScalingParams,
    U_COL,
    hard_load_compute_energy,
    hard_load_predict_load_energy,
)


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------
@pytest.fixture()
def params():
    """Plausible scaling params for a synthetic crashworthiness dataset.

    mu_d, sig_d in mm; mu_F in kN; mu_E in J.  Values picked to be different
    from each other so any unit-confusion bug would show up.
    """
    return ScalingParams(
        mu_d=50.0, sig_d=30.0,
        mu_F=4.5, sig_F=2.0,
        mu_E=180.0, sig_E=120.0,
        grad_factor=120.0 / 30.0,  # sig_E / sig_d (unused here but kept consistent)
    )


@pytest.fixture()
def model(params):
    """Small HardLoadNet on CPU with BC active.  3 input features:
    [d_scaled, sin(θ), cos(θ)] — no LC one-hot, simpler test geometry.
    """
    torch.manual_seed(0)
    m = HardLoadNet(in_d=3, hidden_layers=[16, 16], dropout=0.0, softplus_beta=10.0)
    m.configure_zero_bc(params)
    m.eval()
    return m


def _make_X(d_mm: torch.Tensor, params: ScalingParams, theta_deg: float = 60.0) -> torch.Tensor:
    """Build [B, 3] feature matrix at the given displacement(s) and angle."""
    d_scaled = (d_mm - params.mu_d) / params.sig_d
    theta_rad = math.radians(theta_deg)
    sin_t = torch.full_like(d_scaled, math.sin(theta_rad))
    cos_t = torch.full_like(d_scaled, math.cos(theta_rad))
    return torch.stack([d_scaled, sin_t, cos_t], dim=1)


# ----------------------------------------------------------------------------
# Test 1.  F(d=0) = 0 by construction
# ----------------------------------------------------------------------------
def test_F_at_zero_is_zero_for_every_theta(model, params):
    """F_raw(d=0; θ, LC) = 0 for every angle, by the architectural correction."""
    d_zero = torch.zeros(6)
    thetas = [45.0, 50.0, 55.0, 60.0, 65.0, 70.0]
    for theta in thetas:
        X = _make_X(d_zero, params, theta_deg=theta)
        F_n = model(X)
        F_raw = F_n * params.sig_F + params.mu_F
        # raw F at d=0 should be exactly 0 (within float32 roundoff)
        assert torch.allclose(
            F_raw, torch.zeros_like(F_raw), atol=1e-5
        ), f"F_raw(d=0, θ={theta}) = {F_raw.numpy()} ≠ 0"


# ----------------------------------------------------------------------------
# Test 2.  E(d=0) = 0 by construction (trapezoidal integral from 0 to 0)
# ----------------------------------------------------------------------------
def test_E_at_zero_is_zero(model, params):
    """E_raw(d=0; θ, LC) = ∫₀⁰ F dδ = 0, no architectural correction needed."""
    d_zero = torch.zeros(3)
    X = _make_X(d_zero, params, theta_deg=60.0)
    F_n, E_n = hard_load_compute_energy(model, X, params, K_integration=32)
    E_raw = E_n * params.sig_E + params.mu_E
    assert torch.allclose(
        E_raw, torch.zeros_like(E_raw), atol=1e-5
    ), f"E_raw(d=0) = {E_raw.numpy()} ≠ 0"


# ----------------------------------------------------------------------------
# Test 3.  Trapezoidal Ê is a self-consistent integral of F̂
# ----------------------------------------------------------------------------
def test_E_matches_independent_trapezoid_of_F(model, params):
    """Pick d_target=120mm, recompute ∫₀^120 F̂ dδ via torch.trapezoid on a
    fine grid, and check it matches hard_load_compute_energy's output.
    """
    d_target = torch.tensor([120.0])
    X_target = _make_X(d_target, params, theta_deg=55.0)

    # Reference: dense torch.trapezoid on K=257 points
    K_ref = 257
    alpha = torch.linspace(0.0, 1.0, K_ref)
    d_grid_mm = d_target * alpha                                   # [K_ref]
    X_grid = _make_X(d_grid_mm.view(-1), params, theta_deg=55.0)  # [K_ref, 3]
    with torch.no_grad():
        F_n_grid = model(X_grid).squeeze(-1)                       # [K_ref]
    F_raw_grid = F_n_grid * params.sig_F + params.mu_F
    E_raw_ref = torch.trapezoid(F_raw_grid, d_grid_mm)              # scalar

    # System under test: default K=64
    _, E_n = hard_load_compute_energy(model, X_target, params, K_integration=64)
    E_raw_sut = E_n.squeeze() * params.sig_E + params.mu_E

    rel_err = float(abs(E_raw_sut - E_raw_ref) / max(1e-6, abs(E_raw_ref)))
    assert rel_err < 5e-3, (
        f"E_raw mismatch: SUT={float(E_raw_sut):.4f} ref={float(E_raw_ref):.4f}"
        f" rel_err={rel_err:.4g}"
    )


# ----------------------------------------------------------------------------
# Test 4.  K=64 is dense enough — agrees with K=256 to within 1e-3
# ----------------------------------------------------------------------------
def test_default_K_is_dense_enough(model, params):
    """Doubling K from 64 to 256 should change E by less than 0.1% relative."""
    d_target = torch.tensor([30.0, 80.0, 120.0])
    X = _make_X(d_target, params, theta_deg=60.0)
    _, E_n_64 = hard_load_compute_energy(model, X, params, K_integration=64)
    _, E_n_256 = hard_load_compute_energy(model, X, params, K_integration=256)
    E_64 = (E_n_64 * params.sig_E + params.mu_E).squeeze()
    E_256 = (E_n_256 * params.sig_E + params.mu_E).squeeze()
    rel = (E_64 - E_256).abs() / E_256.abs().clamp_min(1e-3)
    assert (rel < 1e-3).all(), (
        f"K=64 vs K=256 disagrees by more than 0.1%: rel_err={rel.numpy()}"
    )


# ----------------------------------------------------------------------------
# Test 5.  Backprop through the integral delivers gradients
# ----------------------------------------------------------------------------
def test_backprop_through_integral_updates_every_param(model, params):
    """A simple MSE loss on E_n must produce non-zero, finite gradients on
    every backbone parameter — confirming autograd flows through all K
    integration nodes, not just the target d.

    The final-layer bias is allowed to have zero gradient.  It is
    structurally redundant under the BC subtraction
    ``F_n(x) = net(x) − net(x|d=0) + c_{0,F}``: the bias appears in both
    ``net(x)`` and ``net(x|d=0)`` and cancels, so ∂loss/∂b ≡ 0 by
    construction.  This is the same property as :class:`HardEnergyNet`
    and is not a bug.
    """
    model.train()
    d_target = torch.linspace(10.0, 120.0, 16).view(-1)
    X = _make_X(d_target, params, theta_deg=60.0)
    _, E_n = hard_load_compute_energy(model, X, params, K_integration=32)
    target = torch.zeros_like(E_n) + 0.5  # arbitrary nonzero target
    loss = ((E_n - target) ** 2).mean()
    loss.backward()
    # Identify the final-layer bias (last Linear's bias) — exempted as above.
    final_bias = list(model.net[-1].parameters())[-1]
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
        if p is final_bias:
            # Structurally zero — see docstring above.
            assert p.grad.abs().sum() == 0, (
                f"final-layer bias unexpectedly has nonzero grad on {name}"
            )
        else:
            assert p.grad.abs().sum() > 0, f"zero grad on {name}"


# ----------------------------------------------------------------------------
# Test 6.  predict wrapper round-trips through numpy correctly
# ----------------------------------------------------------------------------
def test_hard_load_predict_load_energy_returns_numpy(model, params):
    """The eval wrapper returns 1-D numpy arrays of raw kN and J."""
    d_target = torch.linspace(0.0, 130.0, 32).view(-1)
    X = _make_X(d_target, params, theta_deg=60.0)
    F_arr, E_arr = hard_load_predict_load_energy(model, X, params, K_integration=64)
    assert isinstance(F_arr, np.ndarray) and F_arr.ndim == 1 and F_arr.size == 32
    assert isinstance(E_arr, np.ndarray) and E_arr.ndim == 1 and E_arr.size == 32
    # F at the first row (d=0) should be 0
    assert abs(float(F_arr[0])) < 1e-4
    # E at the first row should be 0
    assert abs(float(E_arr[0])) < 1e-4
    # E should be non-decreasing on average (F is typically positive in crash data,
    # but for a randomly initialized net we can only assert E[d=0] = 0).
