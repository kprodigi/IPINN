# -*- coding: utf-8 -*-
"""
================================================================================
COMPARE SOFT-PINN vs HARD-PINN — physics-correctness diagnostics
================================================================================
Loads the M=20 ensemble bundles produced by stage2_v16.py for Soft and Hard
and computes the metrics that demonstrate the architectural-BC + autograd
contribution of Hard-PINN beyond R² alone.

Hard-PINN's value is NOT primarily about R² lift over Soft-PINN (the gap is
typically ~0.02-0.03 on this problem).  It is about CATEGORICAL physics
guarantees that Soft-PINN can only approximate:

   1. F(d=0) AND E(d=0):
        Soft  →  small but non-zero residual (~10⁻¹ units)
        Hard  →  exactly zero by construction (machine precision)

      Hard's ``HardEnergyNet.configure_zero_bc`` enforces BOTH BCs
      architecturally via slope-subtraction:
        E_corrected(x) = E_net(x) − E_net(x|d=0)
                         − (d_s − d_s0) · ∂E_net/∂d_s|_{x|d=0} + c_{0,E}
      so E(d=0) = c_{0,E} (raw 0) and ∂E/∂d_s|_{d=0} = 0 (raw F(0) = 0).

   2. Force-energy thermodynamic identity F = dE/dd:
        Soft  →  residual penalised in loss; small but nonzero
        Hard  →  F is computed AS dE/dd via autograd; residual = 0 always

   3. Hyperparameter count for BC enforcement:
        Soft  →  w_bc tuned (and a separate physics weight w_phys)
        Hard  →  no w_bc (architectural)

   4. Worst-seed behaviour and ensemble interval coverage — usually
      tighter for Hard since its physics is exact.

Usage:
    python hpo/compare_soft_vs_hard.py \\
        --soft_bundle  results_stage2_v16/stage2_soft_bundle.pt \\
        --hard_bundle  results_stage2_v16/stage2_hard_bundle.pt \\
        --output_dir   results_stage2_v16/

Writes:
    soft_vs_hard_comparison.json   per-metric numbers, JSON
    soft_vs_hard_comparison.log    human-readable comparison
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch

_HPO_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HPO_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import composite_design as cd  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning, module=r"matplotlib")


# =============================================================================
# Helpers
# =============================================================================
def _make_logger(log_path: str) -> logging.Logger:
    log = logging.getLogger("compare_soft_vs_hard")
    log.setLevel(logging.INFO)
    log.handlers = []
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(ch)
    return log


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    return obj


def _reconstruct_models(bundle: Dict, approach: str, in_d: int, device: torch.device) -> List:
    """Rebuild models from the bundle's saved state_dicts."""
    cfg = bundle["cfg"]
    models = []
    for entry in bundle["models_state"]:
        sd = entry["state_dict"]
        if approach in ("soft", "ddns"):
            m = cd.SoftPINNNet(in_d, cfg["hidden_layers"], cfg["dropout"],
                               cfg["softplus_beta"]).to(device)
            # Soft uses soft-penalty BC (not architectural).  DDNS BC is disabled.
            m.configure_zero_bc(bundle["params"], enabled=False)
        else:  # hard
            m = cd.HardEnergyNet(in_d, cfg["hidden_layers"], cfg["dropout"],
                                 cfg["softplus_beta"]).to(device)
            # Hard uses architectural BC.
            m.configure_zero_bc(bundle["params"], enabled=True)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    return models


# =============================================================================
# Physics-correctness metrics
# =============================================================================
def boundary_residuals(models: List, approach: str, train_df, scaler_disp, enc,
                       params, device) -> Tuple[np.ndarray, np.ndarray]:
    """Compute |F(d=0)| and |E(d=0)| in raw units for every model on every
    unique (angle, LC) combination in train_df.

    Hard-PINN returns 0 exactly (machine precision) due to architectural
    subtraction trick.  Soft-PINN returns small but nonzero residuals
    (the soft `w_bc` penalty pulls them toward zero but does not enforce
    it).  This is the categorical demonstration of architectural vs soft BC.
    """
    import pandas as pd

    combos = train_df[["Angle", "LC"]].drop_duplicates().reset_index(drop=True)
    bc_df = combos.copy()
    bc_df["disp_mm"] = 0.0
    bc_df["load_kN"] = 0.0    # placeholder; not used
    bc_df["energy_J"] = 0.0   # placeholder; not used
    X_bc = cd.to_tensor(cd.build_features(bc_df, scaler_disp, enc))

    abs_F_list, abs_E_list = [], []
    for model in models:
        if approach in ("soft", "ddns"):
            with torch.no_grad():
                pred = model(X_bc)
                F_n = pred[:, 0]
                E_n = pred[:, 1]
            F_raw = F_n.detach().cpu().numpy() * params.sig_F + params.mu_F
            E_raw = E_n.detach().cpu().numpy() * params.sig_E + params.mu_E
        else:
            # Hard: F via autograd; must keep grad context enabled.
            X_bc_g = X_bc.detach().clone().requires_grad_(True)
            E_g = model(X_bc_g).squeeze(-1)
            grads = torch.autograd.grad(E_g.sum(), X_bc_g, create_graph=False)[0]
            # F = dE/dd in raw kN once we multiply by grad_factor
            F_raw = (grads[:, cd.U_COL] * params.grad_factor).detach().cpu().numpy()
            E_raw = E_g.detach().cpu().numpy() * params.sig_E + params.mu_E
        abs_F_list.append(np.abs(F_raw))
        abs_E_list.append(np.abs(E_raw))

    # Stack: (M_models, N_combos) — every combo evaluated by every model
    return np.stack(abs_F_list, axis=0), np.stack(abs_E_list, axis=0)


def force_energy_consistency(models: List, approach: str, val_df, scaler_disp, enc,
                              params, device) -> np.ndarray:
    """Compute |F_predicted − dE/dd_predicted| (raw units, kN) per val point,
    averaged across models.  Hard returns zero by construction; Soft returns
    a small but nonzero residual.  This is the second categorical guarantee.
    """
    Xv = cd.to_tensor(cd.build_features(val_df, scaler_disp, enc))

    residuals = []  # one array per model
    for model in models:
        if approach in ("soft", "ddns"):
            # Both F and E come from the same network as separate outputs.
            # F is NOT automatically dE/dd — that's enforced only by w_phys.
            Xv_g = Xv.detach().clone().requires_grad_(True)
            pred = model(Xv_g)
            E_n = pred[:, 1]
            F_n_direct = pred[:, 0]
            # dE/dd in normalised space → raw F
            dE_dd_grads = torch.autograd.grad(E_n.sum(), Xv_g, create_graph=False)[0]
            F_via_grad_norm = dE_dd_grads[:, cd.U_COL] * params.grad_factor
            F_via_grad_norm = (F_via_grad_norm - params.mu_F) / max(1e-12, params.sig_F)
            # Compare raw F units
            F_direct_raw = F_n_direct.detach().cpu().numpy() * params.sig_F + params.mu_F
            F_grad_raw   = F_via_grad_norm.detach().cpu().numpy() * params.sig_F + params.mu_F
            residuals.append(np.abs(F_direct_raw - F_grad_raw))
        else:  # hard — F IS dE/dd by construction, residual is 0
            residuals.append(np.zeros(len(val_df)))

    return np.stack(residuals, axis=0)  # (M, N_val)


def ensemble_predict(models: List, approach: str, val_df, scaler_disp, scaler_out,
                     enc, params, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (per_member_F: M×N, per_member_E: M×N, val_true_F, val_true_E)."""
    Xv = cd.to_tensor(cd.build_features(val_df, scaler_disp, enc))
    y_true = val_df[["load_kN", "energy_J"]].values

    all_F, all_E = [], []
    for model in models:
        if approach in ("soft", "ddns"):
            with torch.no_grad():
                pred = model(Xv)
                F_raw = (pred[:, 0] * params.sig_F + params.mu_F).cpu().numpy()
                E_raw = (pred[:, 1] * params.sig_E + params.mu_E).cpu().numpy()
        else:  # hard
            F_raw, E_raw = cd.hard_pinn_predict_load_energy(model, Xv, params)
        all_F.append(F_raw)
        all_E.append(E_raw)
    return np.stack(all_F, axis=0), np.stack(all_E, axis=0), y_true[:, 0], y_true[:, 1]


def coverage_2sigma(per_member_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of val points falling within mean ± 2·std across ensemble members.
    Uncalibrated (no conformal scaling); useful for relative comparison."""
    mean = per_member_pred.mean(axis=0)
    std  = per_member_pred.std(axis=0, ddof=0)
    low, high = mean - 2 * std, mean + 2 * std
    inside = (y_true >= low) & (y_true <= high)
    return float(inside.mean())


def r2_per_member(per_member_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Per-member R² (load).  Returns array of length M."""
    out = []
    yt_mean = y_true.mean()
    ss_tot = float(np.sum((y_true - yt_mean) ** 2))
    for i in range(per_member_pred.shape[0]):
        ss_res = float(np.sum((y_true - per_member_pred[i]) ** 2))
        out.append(1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))
    return np.array(out, dtype=float)


# =============================================================================
# Main
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Soft-PINN vs Hard-PINN physics-correctness comparison.")
    p.add_argument("--soft_bundle", required=True, help="Path to stage2_soft_bundle.pt")
    p.add_argument("--hard_bundle", required=True, help="Path to stage2_hard_bundle.pt")
    p.add_argument("--output_dir",  default="./results_stage2_v16")
    p.add_argument("--force_cpu",   action="store_true")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "soft_vs_hard_comparison.log")
    logger = _make_logger(log_path)

    device = torch.device("cpu" if args.force_cpu or not torch.cuda.is_available() else "cuda")
    cd.CFG.force_cpu = bool(args.force_cpu)
    cd.refresh_device()

    logger.info("=" * 80)
    logger.info("SOFT-PINN vs HARD-PINN — physics-correctness comparison")
    logger.info("=" * 80)
    logger.info(f"  soft_bundle = {args.soft_bundle}")
    logger.info(f"  hard_bundle = {args.hard_bundle}")
    logger.info(f"  device      = {device}")

    soft_bundle = torch.load(args.soft_bundle, weights_only=False, map_location="cpu")
    hard_bundle = torch.load(args.hard_bundle, weights_only=False, map_location="cpu")

    # Sanity: same data split + scalers (both bundles share base_seed=2026)
    if not soft_bundle["val_df"].equals(hard_bundle["val_df"]):
        logger.warning("  Soft and Hard val_df differ — comparison may be invalid")
    val_df = soft_bundle["val_df"]

    soft_scaler_disp = soft_bundle["scaler_disp"]
    soft_scaler_out  = soft_bundle["scaler_out"]
    soft_enc         = soft_bundle["enc"]
    soft_params      = soft_bundle["params"]

    hard_scaler_disp = hard_bundle["scaler_disp"]
    hard_scaler_out  = hard_bundle["scaler_out"]
    hard_enc         = hard_bundle["enc"]
    hard_params      = hard_bundle["params"]

    train_df_for_combos = soft_bundle["train_df"]  # (angle, LC) combos used for d=0 collocation

    # Reconstruct ensembles
    Xv = cd.to_tensor(cd.build_features(val_df, soft_scaler_disp, soft_enc))
    in_d = Xv.shape[1]
    logger.info(f"  feature dim in_d = {in_d}  |  M_soft = {len(soft_bundle['models_state'])}  M_hard = {len(hard_bundle['models_state'])}")

    soft_models = _reconstruct_models(soft_bundle, "soft", in_d, device)
    hard_models = _reconstruct_models(hard_bundle, "hard", in_d, device)

    # =====================================================================
    # METRIC 1: Boundary-condition residuals  |F(d=0)|, |E(d=0)|
    # =====================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("METRIC 1 — Boundary-condition residuals at d=0 (raw units)")
    logger.info("=" * 80)
    soft_F0, soft_E0 = boundary_residuals(soft_models, "soft", train_df_for_combos,
                                          soft_scaler_disp, soft_enc, soft_params, device)
    hard_F0, hard_E0 = boundary_residuals(hard_models, "hard", train_df_for_combos,
                                          hard_scaler_disp, hard_enc, hard_params, device)

    def _summ(name: str, arr: np.ndarray, units: str):
        return (f"  {name:>20s}:  mean={arr.mean():.6f}  median={np.median(arr):.6f}  "
                f"p95={np.percentile(arr,95):.6f}  max={arr.max():.6f}  [{units}]")

    logger.info(_summ("Soft |F(0)|", soft_F0, "kN"))
    logger.info(_summ("Hard |F(0)|", hard_F0, "kN  (machine precision = 0)"))
    logger.info(_summ("Soft |E(0)|", soft_E0, "J"))
    logger.info(_summ("Hard |E(0)|", hard_E0, "J  (machine precision = 0)"))
    bc_F_ratio = soft_F0.mean() / max(hard_F0.mean(), 1e-12)
    bc_E_ratio = soft_E0.mean() / max(hard_E0.mean(), 1e-12)
    logger.info(f"  Soft/Hard ratio for |F(0)|: {bc_F_ratio:.2e}")
    logger.info(f"  Soft/Hard ratio for |E(0)|: {bc_E_ratio:.2e}")

    # =====================================================================
    # METRIC 2: Force-energy consistency |F − dE/dd|
    # =====================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("METRIC 2 — Force–energy consistency residual |F − dE/dd| (raw kN)")
    logger.info("=" * 80)
    soft_fe_res = force_energy_consistency(soft_models, "soft", val_df,
                                           soft_scaler_disp, soft_enc, soft_params, device)
    hard_fe_res = force_energy_consistency(hard_models, "hard", val_df,
                                           hard_scaler_disp, hard_enc, hard_params, device)

    logger.info(_summ("Soft |F − dE/dd|", soft_fe_res, "kN"))
    logger.info(_summ("Hard |F − dE/dd|", hard_fe_res, "kN  (= 0 by construction)"))
    fe_ratio = soft_fe_res.mean() / max(hard_fe_res.mean(), 1e-12)
    logger.info(f"  Soft/Hard residual ratio: {fe_ratio:.2e}")

    # =====================================================================
    # METRIC 3: R² (mean of per-member + ensemble-aggregated)
    # =====================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("METRIC 3 — R² comparison (the standard headline metric)")
    logger.info("=" * 80)
    soft_F_pred, soft_E_pred, y_true_F, y_true_E = ensemble_predict(
        soft_models, "soft", val_df, soft_scaler_disp, soft_scaler_out, soft_enc, soft_params, device)
    hard_F_pred, hard_E_pred, _, _ = ensemble_predict(
        hard_models, "hard", val_df, hard_scaler_disp, hard_scaler_out, hard_enc, hard_params, device)

    soft_r2_load = r2_per_member(soft_F_pred, y_true_F)
    hard_r2_load = r2_per_member(hard_F_pred, y_true_F)

    soft_ens_F = soft_F_pred.mean(axis=0)
    hard_ens_F = hard_F_pred.mean(axis=0)
    soft_ens_E = soft_E_pred.mean(axis=0)
    hard_ens_E = hard_E_pred.mean(axis=0)

    def _r2(yt, yp):
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    soft_ens_load_r2  = _r2(y_true_F, soft_ens_F)
    hard_ens_load_r2  = _r2(y_true_F, hard_ens_F)
    soft_ens_eng_r2   = _r2(y_true_E, soft_ens_E)
    hard_ens_eng_r2   = _r2(y_true_E, hard_ens_E)

    logger.info(f"  Soft per-member load R²:  mean={soft_r2_load.mean():.4f}  std={soft_r2_load.std():.4f}  "
                f"min={soft_r2_load.min():.4f}  max={soft_r2_load.max():.4f}  p5={np.percentile(soft_r2_load,5):.4f}")
    logger.info(f"  Hard per-member load R²:  mean={hard_r2_load.mean():.4f}  std={hard_r2_load.std():.4f}  "
                f"min={hard_r2_load.min():.4f}  max={hard_r2_load.max():.4f}  p5={np.percentile(hard_r2_load,5):.4f}")
    logger.info(f"  Soft ensemble load R²:    {soft_ens_load_r2:.4f}")
    logger.info(f"  Hard ensemble load R²:    {hard_ens_load_r2:.4f}")
    logger.info(f"  Δ (Hard − Soft) ensemble: {hard_ens_load_r2 - soft_ens_load_r2:+.4f}")
    logger.info(f"  Soft ensemble energy R²:  {soft_ens_eng_r2:.4f}")
    logger.info(f"  Hard ensemble energy R²:  {hard_ens_eng_r2:.4f}")

    # =====================================================================
    # METRIC 4: 2σ ensemble interval coverage (uncalibrated)
    # =====================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("METRIC 4 — Ensemble ±2σ interval coverage (uncalibrated)")
    logger.info("=" * 80)
    soft_cov_F = coverage_2sigma(soft_F_pred, y_true_F)
    hard_cov_F = coverage_2sigma(hard_F_pred, y_true_F)
    soft_cov_E = coverage_2sigma(soft_E_pred, y_true_E)
    hard_cov_E = coverage_2sigma(hard_E_pred, y_true_E)
    logger.info(f"  Soft load coverage (target ≥95%):   {soft_cov_F*100:.1f}%")
    logger.info(f"  Hard load coverage (target ≥95%):   {hard_cov_F*100:.1f}%")
    logger.info(f"  Soft energy coverage:               {soft_cov_E*100:.1f}%")
    logger.info(f"  Hard energy coverage:               {hard_cov_E*100:.1f}%")

    # =====================================================================
    # METRIC 5: Hyperparameter count
    # =====================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("METRIC 5 — Tuned-hyperparameter count")
    logger.info("=" * 80)
    soft_cfg = soft_bundle["cfg"]
    hard_cfg = hard_bundle["cfg"]
    soft_hp_names = sorted(soft_cfg.keys())
    hard_hp_names = sorted(hard_cfg.keys())
    soft_phys_hp = [k for k in soft_hp_names if k in {"w_phys", "w_bc", "w_data_load", "w_data_energy",
                                                       "w_monotonicity", "w_angle_smooth", "smooth_delta_deg",
                                                       "colloc_ratio"}]
    hard_phys_hp = [k for k in hard_hp_names if k in {"w_load", "w_energy", "w_monotonicity", "w_angle_smooth",
                                                       "w_curvature", "smooth_delta_deg", "colloc_ratio",
                                                       "grad_clip", "warmup_epochs", "swa_pct"}]
    logger.info(f"  Soft physics/loss hyperparams ({len(soft_phys_hp)}): {soft_phys_hp}")
    logger.info(f"  Hard physics/loss hyperparams ({len(hard_phys_hp)}): {hard_phys_hp}")
    logger.info(f"  Note: Hard's `w_load` and `w_energy` re-weight DATA terms; the FORCE-ENERGY")
    logger.info(f"        coupling (Soft's `w_phys`) is removed entirely in Hard — that's the")
    logger.info(f"        architectural saving.  Hard's `w_bc` is also removed.")

    # =====================================================================
    # WRITE JSON SUMMARY
    # =====================================================================
    out_json = os.path.join(args.output_dir, "soft_vs_hard_comparison.json")
    payload = _json_safe({
        "metric_1_boundary": {
            "soft_abs_F0":  {"mean": soft_F0.mean(), "median": np.median(soft_F0),
                              "p95": np.percentile(soft_F0, 95), "max": soft_F0.max()},
            "hard_abs_F0":  {"mean": hard_F0.mean(), "median": np.median(hard_F0),
                              "p95": np.percentile(hard_F0, 95), "max": hard_F0.max()},
            "soft_abs_E0":  {"mean": soft_E0.mean(), "median": np.median(soft_E0),
                              "p95": np.percentile(soft_E0, 95), "max": soft_E0.max()},
            "hard_abs_E0":  {"mean": hard_E0.mean(), "median": np.median(hard_E0),
                              "p95": np.percentile(hard_E0, 95), "max": hard_E0.max()},
            "soft_over_hard_ratio_F": bc_F_ratio,
            "soft_over_hard_ratio_E": bc_E_ratio,
        },
        "metric_2_force_energy_consistency": {
            "soft_residual": {"mean": soft_fe_res.mean(), "median": np.median(soft_fe_res),
                              "p95": np.percentile(soft_fe_res, 95), "max": soft_fe_res.max()},
            "hard_residual": {"mean": hard_fe_res.mean(), "median": np.median(hard_fe_res),
                              "p95": np.percentile(hard_fe_res, 95), "max": hard_fe_res.max()},
            "soft_over_hard_ratio":   fe_ratio,
        },
        "metric_3_r2": {
            "soft_per_member_mean":     soft_r2_load.mean(),
            "soft_per_member_std":      soft_r2_load.std(),
            "soft_per_member_min":      soft_r2_load.min(),
            "soft_per_member_max":      soft_r2_load.max(),
            "soft_per_member_p5":       np.percentile(soft_r2_load, 5),
            "hard_per_member_mean":     hard_r2_load.mean(),
            "hard_per_member_std":      hard_r2_load.std(),
            "hard_per_member_min":      hard_r2_load.min(),
            "hard_per_member_max":      hard_r2_load.max(),
            "hard_per_member_p5":       np.percentile(hard_r2_load, 5),
            "soft_ensemble_load_r2":    soft_ens_load_r2,
            "hard_ensemble_load_r2":    hard_ens_load_r2,
            "soft_ensemble_energy_r2":  soft_ens_eng_r2,
            "hard_ensemble_energy_r2":  hard_ens_eng_r2,
            "ensemble_load_r2_delta":   hard_ens_load_r2 - soft_ens_load_r2,
        },
        "metric_4_coverage": {
            "soft_load_coverage_2sigma":   soft_cov_F,
            "hard_load_coverage_2sigma":   hard_cov_F,
            "soft_energy_coverage_2sigma": soft_cov_E,
            "hard_energy_coverage_2sigma": hard_cov_E,
        },
        "metric_5_hp_count": {
            "soft_phys_hyperparams": soft_phys_hp,
            "hard_phys_hyperparams": hard_phys_hp,
        },
    })
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    logger.info("")
    logger.info(f"  Wrote: {out_json}")
    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
