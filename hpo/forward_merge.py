# -*- coding: utf-8 -*-
"""
================================================================================
FORWARD — merge per-member partial bundles into the final ensemble
================================================================================
Companion to :mod:`hpo.forward_member` (the per-member training launcher).
Collects all ``parts_<approach>/member_*.pt`` files produced by parallel
SLURM array tasks and emits the SAME outputs as :mod:`hpo.forward_member`:

    forward_<approach>_results.json   per-member R², mean ± std, ensemble metrics
    forward_<approach>_log.txt        merge log
    forward_<approach>_bundle.pt      trained models + scalers (torch.save)

The merge applies the same Tukey-fence convergence filter on training-set
R² as :func:`composite_design.train_ensemble`, so the merged outputs
match the sequential ``forward_member.py`` outputs (modulo non-determinism from
CUDA kernels across different GPUs — bit-for-bit equality is not
guaranteed when members are trained on different hardware).

Failure handling: members whose part file is missing (e.g. a SLURM task
that died before writing) or that wrote a ``.FAILED.json`` marker are
reported and skipped; the merge proceeds with the surviving members
provided at least one survived.

Usage:
    python hpo/forward_merge.py --approach hard \\
        --data_dir ./data --output_dir ./results_forward --n_ensemble 20
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from typing import Dict, List, Optional

import numpy as np
import torch

_HPO_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HPO_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import composite_design as cd  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning, module=r"matplotlib")


def _make_logger(approach: str, log_path: str) -> logging.Logger:
    log = logging.getLogger(f"forward.merge.{approach}")
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


def _build_model(approach: str, cfg: Dict, in_d: int) -> torch.nn.Module:
    """Re-instantiate the per-approach network with the same shape used at
    training time.  Mirrors what train_ddns / train_soft / train_hard do.
    """
    if approach in ("ddns", "soft"):
        return cd.SoftPINNNet(
            in_d, cfg["hidden_layers"], cfg["dropout"], cfg["softplus_beta"],
        )
    if approach == "hard":
        return cd.HardEnergyNet(
            in_d, cfg["hidden_layers"], cfg["dropout"], cfg["softplus_beta"],
        )
    raise ValueError(f"Unknown approach: {approach}")


def main():
    p = argparse.ArgumentParser(
        description="Forward merge step (gather per-member partial bundles).")
    p.add_argument("--approach", choices=["ddns", "soft", "hard"], required=True)
    p.add_argument("--data_dir",   default="./data")
    p.add_argument("--output_dir", default="./results_forward")
    p.add_argument("--n_ensemble", type=int, default=20,
                   help="Expected ensemble size M.  Members with missing or "
                        "failed part files are tolerated; this just controls "
                        "the warning threshold and the metadata recorded.")
    p.add_argument("--seed",       type=int, default=2026)
    p.add_argument("--force_cpu",  action="store_true")
    p.add_argument("--dry_run",    action="store_true",
                   help="Smoke mode (matches the per-member launcher's flag).")
    p.add_argument("--theta_star", type=float, default=None,
                   help="Held-out angle (degrees) for the unseen-angle "
                        "protocol.  When set, the merge reads from "
                        "``parts_<approach>_t<theta>/`` (the per-fold "
                        "directory written by forward_member.py with the "
                        "same flag) and writes the resulting bundle as "
                        "``forward_<approach>_t<theta>_bundle.pt`` so LOAO "
                        "folds do not overwrite each other.")
    args = p.parse_args()

    # Resolve theta_star and decide the output filename suffix BEFORE
    # opening the log file, so per-fold logs go to distinct paths.
    theta_for_paths: Optional[int] = None
    if args.theta_star is not None:
        cd.CFG.theta_star = float(args.theta_star)
        theta_for_paths = int(round(float(args.theta_star)))
        approach_tag = f"{args.approach}_t{theta_for_paths}"
    else:
        approach_tag = args.approach

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"forward_{approach_tag}_log.txt")
    logger = _make_logger(args.approach, log_path)

    cd.CFG.n_ensemble = int(args.n_ensemble)
    cd.CFG.seed       = int(args.seed)
    cd.CFG.seed_base  = int(args.seed)
    cd.CFG.split_seed = int(args.seed)
    cd.CFG.force_cpu  = bool(args.force_cpu)
    cd.CFG.dry_run    = bool(args.dry_run)
    cd.refresh_device()
    cd.set_publication_style()

    if theta_for_paths is not None:
        parts_dir = os.path.join(
            args.output_dir, f"parts_{args.approach}_t{theta_for_paths}",
        )
    else:
        parts_dir = os.path.join(args.output_dir, f"parts_{args.approach}")
    if not os.path.isdir(parts_dir):
        raise FileNotFoundError(
            f"Per-member parts directory not found: {parts_dir}.  "
            f"Did forward_member.py run?",
        )

    cfg = cd.get_model_config(args.approach, "unseen")

    logger.info("=" * 80)
    logger.info(f"FORWARD MERGE — approach={args.approach}")
    logger.info(f"  Expected M={args.n_ensemble}  arch={cfg['hidden_layers']}")
    logger.info(f"  parts_dir={parts_dir}")
    logger.info(f"  output_dir={args.output_dir}")
    logger.info("=" * 80)

    # Reload data & preprocessors EXACTLY as the per-member jobs did, so the
    # val-set features fed to the merged ensemble match what each member saw.
    df_all = cd.load_data(args.data_dir, logger)
    train_df, val_df = cd.split_unseen_angle(df_all, cd.CFG.theta_star, logger)
    scaler_disp, scaler_out, enc, params = cd.create_preprocessors(train_df, logger)

    # Collect partials.
    parts: List[Dict] = []
    failed: List[Dict] = []
    missing: List[int] = []
    for idx in range(args.n_ensemble):
        part_path = os.path.join(parts_dir, f"member_{idx:02d}.pt")
        fail_path = os.path.join(parts_dir, f"member_{idx:02d}.FAILED.json")
        if os.path.exists(part_path):
            try:
                # weights_only=False because the part file contains the cfg dict
                # and history dict alongside the state_dict.  All written by us
                # in the same process; safe.
                parts.append(torch.load(part_path, map_location="cpu", weights_only=False))
            except TypeError:
                # torch < 2.4: weights_only kw not supported
                parts.append(torch.load(part_path, map_location="cpu"))
        elif os.path.exists(fail_path):
            with open(fail_path, encoding="utf-8") as fh:
                failed.append(json.load(fh))
        else:
            missing.append(idx)

    if failed:
        logger.warning(f"  {len(failed)} members FAILED during training:")
        for f in failed:
            logger.warning(
                f"    M{f.get('member_idx', '?')}: "
                f"{f.get('exception_type', '?')}: {f.get('exception_repr', '?')}"
            )
    if missing:
        logger.warning(f"  {len(missing)} member part files MISSING: {missing}")
    if not parts:
        raise RuntimeError(
            "No surviving per-member partial bundles.  "
            "Cannot construct the ensemble.  Inspect logs in "
            f"{parts_dir} for training failures.",
        )

    # Sort by member_idx so the bundle's models[] list is deterministic.
    parts.sort(key=lambda d: int(d.get("member_idx", -1)))
    logger.info(f"  Loaded {len(parts)} partial bundles "
                f"(member indices: {[int(d['member_idx']) for d in parts]})")

    # Reconstruct models on the active DEVICE, ready for evaluate_ensemble.
    models: List[torch.nn.Module] = []
    for part in parts:
        in_d = int(part["in_d"])
        model = _build_model(args.approach, part["cfg"], in_d).to(cd.DEVICE)
        if args.approach == "hard":
            model.configure_zero_bc(params)
        elif args.approach == "soft":
            # train_soft uses enabled=False — Soft uses the soft w_bc penalty.
            model.configure_zero_bc(params, enabled=False)
        else:  # ddns
            model.configure_zero_bc(params, enabled=False)
        model.load_state_dict({k: v.to(cd.DEVICE) for k, v in part["state_dict"].items()})
        model.eval()
        models.append(model)

    # --- Convergence filter (Tukey fence on TRAINING-set R²) ------------------
    # Mirrors composite_design.train_ensemble at lines ~2295-2342.
    train_r2_scores = [float(p["train_metrics"]["load_r2"]) for p in parts]
    k_iqr = cd.CFG.convergence_filter_iqr
    M_total = len(models)
    fence = float("-inf")
    if k_iqr > 0 and M_total >= 5:
        q1 = float(np.percentile(train_r2_scores, 25))
        q3 = float(np.percentile(train_r2_scores, 75))
        iqr = q3 - q1
        if iqr > 0.01:
            fence = q1 - k_iqr * iqr
            keep_mask = [r2 >= fence for r2 in train_r2_scores]
        else:
            keep_mask = [True] * M_total
        M_eff = sum(keep_mask)
        logger.info(f"  Convergence filter (Tukey fence, k={k_iqr}):")
        logger.info(f"    Train R² stats: Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f}")
        if iqr > 0.01:
            logger.info(f"    Fence = Q1 - {k_iqr}*IQR = {fence:.4f}")
        else:
            logger.info("    IQR < 0.01: all members converged similarly, skipping filter")
        n_discarded = M_total - M_eff
        if n_discarded > 0:
            discarded = [parts[i]["member_idx"] for i, keep in enumerate(keep_mask) if not keep]
            logger.info(f"    M_total={M_total}, M_eff={M_eff}, discarded={n_discarded}")
            for i, keep in enumerate(keep_mask):
                if not keep:
                    logger.info(
                        f"    Discarded M{parts[i]['member_idx']}: "
                        f"train R²={train_r2_scores[i]:.4f} < fence {fence:.4f}"
                    )
            models = [m for m, k in zip(models, keep_mask) if k]
            parts  = [p for p, k in zip(parts, keep_mask) if k]
        else:
            logger.info(f"    All {M_total} members above fence (no outliers)")
    else:
        M_eff = M_total
    # Abort hard when fewer than MIN_SURVIVORS members survived.  A
    # 1- or 2-member "ensemble" would still write a bundle but is not a
    # meaningful representation of the search and will produce
    # misleading R^2 confidence intervals downstream.  3 is the
    # conventional minimum for any std/IQR-based reporting.
    MIN_SURVIVORS = 3
    if len(models) < MIN_SURVIVORS:
        raise RuntimeError(
            f"Only {len(models)} members survived the Tukey-fence convergence "
            f"filter (minimum required: {MIN_SURVIVORS}).  M_total={M_total}, "
            f"failed={len(failed)}, missing={len(missing)}.  Rerun the failed/"
            f"missing members or relax the filter via "
            f"composite_design.CFG.convergence_filter_iqr before merging.",
        )

    # Ensemble metrics (predictions averaged across surviving members).
    ens_metrics = cd.evaluate_ensemble(
        models, args.approach, val_df, scaler_disp, scaler_out, enc, params,
    )

    # Per-member R² stats (over surviving members, same convention as forward_member.py).
    member_load_r2   = [float(p["val_metrics"]["load_r2"])   for p in parts]
    member_energy_r2 = [float(p["val_metrics"].get("energy_r2", float("nan"))) for p in parts]
    mean_load = float(np.mean(member_load_r2))
    std_load  = float(np.std(member_load_r2, ddof=0))
    min_load  = float(np.min(member_load_r2))
    max_load  = float(np.max(member_load_r2))
    ens_load_r2   = float(ens_metrics.get("load_r2", float("nan")))
    ens_energy_r2 = float(ens_metrics.get("energy_r2", float("nan")))
    avg_training_time = float(np.mean([p["training_time"] for p in parts]))
    total_wall = float(np.sum([p["training_time"] for p in parts]))  # sum across parallel tasks

    logger.info("\n" + "=" * 80)
    logger.info(f"FORWARD MERGE RESULTS — {args.approach.upper()}")
    logger.info("=" * 80)
    logger.info(f"  Members surviving filter: M_total={M_total}  M_eff={len(models)}")
    logger.info(f"  Per-member load R²:")
    logger.info(f"    mean = {mean_load:.4f}  std = {std_load:.4f}")
    logger.info(f"    min  = {min_load:.4f}  max = {max_load:.4f}")
    logger.info(f"  Per-member load R² (sorted):")
    for i, r2 in enumerate(sorted(member_load_r2, reverse=True), 1):
        logger.info(f"    rank {i:2d}: {r2:.4f}")
    logger.info(f"  Ensemble-aggregated R²:")
    logger.info(f"    load   = {ens_load_r2:.4f}")
    logger.info(f"    energy = {ens_energy_r2:.4f}")
    logger.info(f"  Cumulative member training time: {total_wall/3600:.2f} h  ({total_wall:.0f} s)")
    logger.info(f"  Avg per-member training:        {avg_training_time:.0f} s")

    # Persist results JSON.  When ``--theta_star`` was given, the
    # ``approach_tag`` already encodes the held-out angle so per-fold LOAO
    # results never collide.
    out_json = os.path.join(args.output_dir, f"forward_{approach_tag}_results.json")
    payload = _json_safe({
        "approach":             args.approach,
        "protocol":             "unseen",
        "theta_star_deg":       cd.CFG.theta_star,
        "cfg":                  cfg,
        "n_ensemble_requested": int(args.n_ensemble),
        "M_total":              int(M_total),
        "M_eff":                int(len(models)),
        # When no fence was applied (M_total < 5 OR IQR ≤ 0.01) ``fence``
        # stays at ``-inf``, which json.dump serializes as the
        # non-standard literal ``-Infinity``.  Emit ``null`` instead so
        # the JSON parses everywhere.
        "convergence_fence":    (float(fence) if np.isfinite(fence) else None),
        "member_indices":       [int(p["member_idx"]) for p in parts],
        "per_member_load_r2":   member_load_r2,
        "per_member_energy_r2": member_energy_r2,
        "mean_load_r2":         mean_load,
        "std_load_r2":          std_load,
        "min_load_r2":          min_load,
        "max_load_r2":          max_load,
        "ensemble_load_r2":     ens_load_r2,
        "ensemble_energy_r2":   ens_energy_r2,
        "avg_training_time_s":  avg_training_time,
        "cumulative_train_s":   total_wall,
        "failed_members":       failed,
        "missing_member_idx":   missing,
    })
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    logger.info(f"\n  Wrote: {out_json}")

    # Persist final bundle (same shape as forward_member.py writes).
    bundle = {
        "approach":         args.approach,
        "cfg":              cfg,
        "models_state":     [{"state_dict": {k: v.detach().cpu()
                                             for k, v in m.state_dict().items()}}
                             for m in models],
        "scaler_disp":      scaler_disp,
        "scaler_out":       scaler_out,
        "enc":              enc,
        "params":           params,
        "train_df":         train_df,
        "val_df":           val_df,
        "member_metrics":   [p["val_metrics"] for p in parts],
        "ensemble_metrics": ens_metrics,
    }
    out_pt = os.path.join(args.output_dir, f"forward_{approach_tag}_bundle.pt")
    torch.save(bundle, out_pt)
    logger.info(f"  Wrote: {out_pt}")
    logger.info("\n" + "=" * 80)
    logger.info("FORWARD MERGE COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
