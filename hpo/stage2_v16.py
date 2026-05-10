# -*- coding: utf-8 -*-
"""
================================================================================
STAGE 2 — production retrain at full budget using v_16's hardcoded HPO-best cfg
================================================================================
Trains an ``M=20`` ensemble for one approach (ddns / soft / hard) on the
unseen-θ=60° protocol, using v_16's hardcoded ``cfg_*`` from
``composite_design_v16.py`` lines 272-329 — with the v_20 architectural BC
applied to Soft and Hard (BC disabled for DDNS, per its data-driven framing).

Why: v_16's documented production R² is 0.849 (Hard).  Reproducing it in
v_20's pipeline gives us the paper-grade forward-prediction number, plus
a clean comparison of "v_16 hyperparameters under v_20's architectural BC"
vs the original soft-BC v_16.

Usage (one approach per SLURM job; run all three in parallel):
    python hpo/stage2_v16.py --approach hard --output_dir ./results_stage2 \\
        --data_dir ./data --n_ensemble 20

Outputs (per ``--output_dir``):
    stage2_<approach>_results.json         per-member R², mean ± std, ensemble metrics
    stage2_<approach>_log.txt              full training log
    stage2_<approach>_bundle.pt            trained models + scalers (torch.save)
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from typing import Dict, List

import numpy as np
import torch

# stage2_v16.py lives in ``hpo/``; composite_design_v20.py is at the repo root.
_HPO_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HPO_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import composite_design_v20 as cd  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning, module=r"matplotlib")


# =============================================================================
# v_16's HARDCODED HPO-BEST CFG — verbatim from composite_design_v16.py
# (lines 272-329 of v_16; only the protocol="unseen" branch).  Documented R²
# in v_16 source comments:
#   ddns: best val load R² = 0.7835  (arch [128, 64, 32])
#   soft: best val load R² = 0.8012  (arch [256, 128, 64])
#   hard: best val load R² = 0.8499  (arch [32, 32])  ← the 0.85 target
# =============================================================================
V16_CFGS = {
    "ddns": {
        "optimizer":          "adam",
        "lr":                 4.2123162503e-05,
        "weight_decay":       3.1582563297e-05,
        "batch_size":         64,
        "hidden_layers":      [128, 64, 32],
        "dropout":            0.016401,
        "softplus_beta":      18.9027,
        "smoothl1_beta":      1.0838,
        "w_data_load":        3.568932,
        "w_data_energy":      3.451798,
        "w_phys":             0.0,
        "w_bc":               0.0,
        "colloc_ratio":       0.0,
        "epochs":             600,
        "eval_every":         25,
        "earlystop_patience_evals": 15,
        "earlystop_min_delta":      1e-5,
        "sched_patience":     58,
        "sched_factor":       0.4589,
    },
    "soft": {
        "optimizer":          "adam",
        "lr":                 4.2358412564e-03,
        "weight_decay":       1.5707123457e-04,
        "batch_size":         32,
        "hidden_layers":      [256, 128, 64],
        "dropout":            0.000137,
        "softplus_beta":      11.0650,
        "smoothl1_beta":      1.1584,
        "w_data_load":        3.609280,
        "w_data_energy":      1.704464,
        "w_phys":             3.689334,
        # ``w_bc`` is in v_16 cfg but UNUSED in v_20 — the architectural
        # E(0)=0 correction replaces the soft penalty.  Kept here so the cfg
        # round-trips exactly with v_16 source.
        "w_bc":               0.851661,
        "colloc_ratio":       3.697626,
        "w_monotonicity":     4.957592,
        "w_angle_smooth":     0.028448,
        "smooth_delta_deg":   2.0000,
        "extrapolate_angles": True,
        "epochs":             800,
        "eval_every":         25,
        "earlystop_patience_evals": 15,
        "earlystop_min_delta":      1e-5,
        "sched_patience":     71,
        "sched_factor":       0.6547,
    },
    "hard": {
        "optimizer":          "adamw",
        "lr":                 4.0e-05,
        "weight_decay":       5.27e-04,
        "batch_size":         16,
        "hidden_layers":      [32, 32],
        "dropout":            0.0003,
        "softplus_beta":      13.82,
        "smoothl1_beta":      0.143,
        "w_load":             6.0,
        "w_energy":           7.0,
        "grad_clip":          1.63,
        "w_monotonicity":     5.0,
        "w_angle_smooth":     0.03,
        "w_curvature":        0.005,
        "smooth_delta_deg":   1.38,
        "colloc_ratio":       1.86,
        "extrapolate_angles": True,
        "epochs":             800,
        "eval_every":         20,
        "earlystop_patience_evals": 15,
        "earlystop_min_delta":      1e-5,
        "sched_patience":     73,
        "sched_factor":       0.37,
    },
}


def _make_logger(approach: str, log_path: str) -> logging.Logger:
    log = logging.getLogger(f"stage2.{approach}")
    log.setLevel(logging.INFO)
    log.handlers = []
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(ch)
    return log


def _patch_get_model_config(approach: str, v16_cfg: Dict):
    """Monkey-patch ``cd.get_model_config`` to return v_16's cfg for this
    approach on the unseen protocol.  Other (approach, protocol) pairs fall
    back to v_20's defaults."""
    original = cd.get_model_config

    def patched(a: str, p: str = "random", w_phys_override=None):
        if a == approach and p == "unseen":
            cfg = dict(v16_cfg)
            # Honour w_phys_override hook (used by Soft physics-weight ablation)
            if w_phys_override is not None and "w_phys" in cfg:
                cfg["w_phys"] = float(w_phys_override)
            return cfg
        return original(a, p, w_phys_override)

    cd.get_model_config = patched
    return original


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    p = argparse.ArgumentParser(
        description="Stage 2 production retrain with v_16 hardcoded cfg in v_20 pipeline.")
    p.add_argument("--approach", choices=["ddns", "soft", "hard"], required=True)
    p.add_argument("--data_dir",   default="./data")
    p.add_argument("--output_dir", default="./results_stage2")
    p.add_argument("--n_ensemble", type=int, default=20,
                   help="Number of ensemble members (default 20, matching v_16 production).")
    p.add_argument("--seed",       type=int, default=2026)
    p.add_argument("--force_cpu",  action="store_true")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"stage2_{args.approach}_log.txt")
    logger = _make_logger(args.approach, log_path)

    # Apply CFG overrides (mirrors what main_v20 does)
    cd.CFG.n_ensemble = int(args.n_ensemble)
    cd.CFG.seed       = int(args.seed)
    cd.CFG.seed_base  = int(args.seed)
    cd.CFG.split_seed = int(args.seed)
    cd.CFG.force_cpu  = bool(args.force_cpu)
    cd.refresh_device()
    cd.set_publication_style()

    v16_cfg = V16_CFGS[args.approach]
    _patch_get_model_config(args.approach, v16_cfg)

    logger.info("=" * 80)
    logger.info(f"STAGE 2 v_16-cfg production retrain — approach={args.approach}")
    logger.info(f"  M={cd.CFG.n_ensemble}  epochs={v16_cfg['epochs']}  "
                f"arch={v16_cfg['hidden_layers']}  batch={v16_cfg['batch_size']}")
    logger.info(f"  unseen-θ={cd.CFG.theta_star}° protocol")
    logger.info(f"  output_dir={args.output_dir}")
    logger.info("=" * 80)

    # Data + preprocessors
    df_all = cd.load_data(args.data_dir, logger)
    train_df, val_df = cd.split_unseen_angle(df_all, cd.CFG.theta_star, logger)
    scaler_disp, scaler_out, enc, params = cd.create_preprocessors(train_df, logger)

    # Train ensemble (v_20's train_ensemble already does Tukey-fence convergence
    # filtering + per-member metrics + ensemble aggregate + diagnostics).
    t0 = time.time()
    result = cd.train_ensemble(
        args.approach, train_df, val_df,
        scaler_disp, scaler_out, enc, params,
        "unseen", logger,
    )
    elapsed = time.time() - t0

    # Per-member R² stats
    member_load_r2  = [m["load_r2"]   for m in result["member_metrics"]]
    member_energy_r2 = [m.get("energy_r2", float("nan"))
                       for m in result["member_metrics"]]
    mean_load   = float(np.mean(member_load_r2))
    std_load    = float(np.std(member_load_r2, ddof=0))
    min_load    = float(np.min(member_load_r2))
    max_load    = float(np.max(member_load_r2))

    # Ensemble-aggregate metrics (predictions averaged across members, then R²)
    ens_load_r2   = float(result["metrics"].get("load_r2", float("nan")))
    ens_energy_r2 = float(result["metrics"].get("energy_r2", float("nan")))

    logger.info("\n" + "=" * 80)
    logger.info(f"STAGE 2 RESULTS — {args.approach.upper()}")
    logger.info("=" * 80)
    logger.info(f"  Members trained: M_total={result['M_total']}  M_eff={result['M_eff']}")
    logger.info(f"  Per-member load R²:")
    logger.info(f"    mean = {mean_load:.4f}  std = {std_load:.4f}")
    logger.info(f"    min  = {min_load:.4f}  max = {max_load:.4f}")
    logger.info(f"  Per-member load R² (sorted):")
    for i, r2 in enumerate(sorted(member_load_r2, reverse=True), 1):
        logger.info(f"    rank {i:2d}: {r2:.4f}")
    logger.info(f"  Ensemble-aggregated R²:")
    logger.info(f"    load   = {ens_load_r2:.4f}")
    logger.info(f"    energy = {ens_energy_r2:.4f}")
    logger.info(f"  Total wall: {elapsed/3600:.2f} h  ({elapsed:.0f} s)")
    logger.info(f"  Avg per-model: {result['avg_training_time']:.0f} s")

    # Persist results JSON
    out_json = os.path.join(args.output_dir, f"stage2_{args.approach}_results.json")
    results_payload = _json_safe({
        "approach":             args.approach,
        "protocol":             "unseen",
        "theta_star_deg":       cd.CFG.theta_star,
        "v16_cfg":              v16_cfg,
        "n_ensemble_requested": int(args.n_ensemble),
        "M_total":              int(result["M_total"]),
        "M_eff":                int(result["M_eff"]),
        "convergence_fence":    float(result.get("convergence_fence", float("-inf"))),
        "per_member_load_r2":   member_load_r2,
        "per_member_energy_r2": member_energy_r2,
        "mean_load_r2":         mean_load,
        "std_load_r2":          std_load,
        "min_load_r2":          min_load,
        "max_load_r2":          max_load,
        "ensemble_load_r2":     ens_load_r2,
        "ensemble_energy_r2":   ens_energy_r2,
        "avg_training_time_s":  float(result["avg_training_time"]),
        "total_wall_s":         float(elapsed),
    })
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(results_payload, fh, indent=2)
        fh.write("\n")
    logger.info(f"\n  Wrote: {out_json}")

    # Persist trained models + scalers as torch.save bundle (for downstream
    # use: figure regeneration, inverse-design retrain, etc.).
    bundle = {
        "approach":     args.approach,
        "v16_cfg":      v16_cfg,
        "models_state": [{"state_dict": {k: v.detach().cpu()
                                         for k, v in m.state_dict().items()},
                          "init_args":  None}  # rebuild via cfg if needed
                         for m in result["models"]],
        "scaler_disp":  scaler_disp,
        "scaler_out":   scaler_out,
        "enc":          enc,
        "params":       params,
        "train_df":     train_df,
        "val_df":       val_df,
        "member_metrics": result["member_metrics"],
        "ensemble_metrics": result["metrics"],
    }
    out_pt = os.path.join(args.output_dir, f"stage2_{args.approach}_bundle.pt")
    torch.save(bundle, out_pt)
    logger.info(f"  Wrote: {out_pt}")

    logger.info("\n" + "=" * 80)
    logger.info("STAGE 2 COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
