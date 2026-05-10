# -*- coding: utf-8 -*-
"""
================================================================================
STAGE 2 — production retrain at full budget (single approach × M=20)
================================================================================
Trains an ``M=20`` ensemble for ONE approach (ddns / soft / hard) on the
unseen-θ=60° protocol using whatever ``composite_design_v20.get_model_config``
currently returns for that approach.

As of commit baking-v_16-cfg, ``get_model_config(_, 'unseen')`` returns the
v_16 hardcoded HPO-best cfg whose production retrain produced:
    DDNS:  R²_load = 0.7835  (arch [128, 64, 32])
    Soft:  R²_load = 0.8012  (arch [256, 128, 64])
    Hard:  R²_load = 0.8499  (arch [32, 32])     ← the 0.85+ target

Why a per-approach launcher rather than ``--mode forward``:
``main_v20 --mode forward`` trains both the random AND unseen protocols and
saves a forward bundle for downstream inverse + figure pipeline.  This
launcher is tighter — one approach, unseen protocol only, M=20 ensemble,
writes a focused results JSON + bundle.  Use this for the paper R² number;
use ``--mode forward`` when you need the full forward bundle for the
figure-rendering pipeline.

Usage (one approach per SLURM job; run all three in parallel via
slurm/submit_stage2_v16.sh):
    python hpo/stage2_v16.py --approach hard --output_dir ./results_stage2 \\
        --data_dir ./data --n_ensemble 20

Outputs (per ``--output_dir``):
    stage2_<approach>_results.json   per-member R², mean ± std, ensemble metrics
    stage2_<approach>_log.txt        full training log
    stage2_<approach>_bundle.pt      trained models + scalers (torch.save)
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
        description="Stage 2 production retrain (one approach × M=20 ensemble, unseen-θ=60°).")
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

    # Pull cfg from the main module (now v_16's hardcoded best for unseen).
    cfg = cd.get_model_config(args.approach, "unseen")

    logger.info("=" * 80)
    logger.info(f"STAGE 2 production retrain — approach={args.approach}")
    logger.info(f"  M={cd.CFG.n_ensemble}  epochs={cfg['epochs']}  "
                f"arch={cfg['hidden_layers']}  batch={cfg['batch_size']}")
    logger.info(f"  unseen-θ={cd.CFG.theta_star}° protocol")
    logger.info(f"  output_dir={args.output_dir}")
    logger.info("=" * 80)

    # Data + preprocessors
    df_all = cd.load_data(args.data_dir, logger)
    train_df, val_df = cd.split_unseen_angle(df_all, cd.CFG.theta_star, logger)
    scaler_disp, scaler_out, enc, params = cd.create_preprocessors(train_df, logger)

    # Train ensemble (v_20's train_ensemble does Tukey-fence convergence
    # filtering + per-member metrics + ensemble aggregate + diagnostics).
    t0 = time.time()
    result = cd.train_ensemble(
        args.approach, train_df, val_df,
        scaler_disp, scaler_out, enc, params,
        "unseen", logger,
    )
    elapsed = time.time() - t0

    # Per-member R² stats
    member_load_r2   = [m["load_r2"]                          for m in result["member_metrics"]]
    member_energy_r2 = [m.get("energy_r2", float("nan"))      for m in result["member_metrics"]]
    mean_load = float(np.mean(member_load_r2))
    std_load  = float(np.std(member_load_r2, ddof=0))
    min_load  = float(np.min(member_load_r2))
    max_load  = float(np.max(member_load_r2))

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
    payload = _json_safe({
        "approach":             args.approach,
        "protocol":             "unseen",
        "theta_star_deg":       cd.CFG.theta_star,
        "cfg":                  cfg,
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
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    logger.info(f"\n  Wrote: {out_json}")

    # Persist trained models + scalers as torch.save bundle
    bundle = {
        "approach":         args.approach,
        "cfg":              cfg,
        "models_state":     [{"state_dict": {k: v.detach().cpu()
                                             for k, v in m.state_dict().items()}}
                             for m in result["models"]],
        "scaler_disp":      scaler_disp,
        "scaler_out":       scaler_out,
        "enc":              enc,
        "params":           params,
        "train_df":         train_df,
        "val_df":           val_df,
        "member_metrics":   result["member_metrics"],
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
