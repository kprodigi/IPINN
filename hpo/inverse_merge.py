# -*- coding: utf-8 -*-
"""
================================================================================
INVERSE-DESIGN — merge per-member partial bundles into a pretrained surrogate
================================================================================
Companion to :mod:`hpo.inverse_member`.  Collects
``parts_inverse_hard/member_*.pt`` files produced by parallel SLURM array
tasks, applies the same Tukey-fence convergence filter that
``composite_design.train_full_data_hard_pinn`` would have applied,
and writes ONE bundle the downstream GP-BO + inverse-design analysis
pipeline can consume:

    <output_dir>/inverse_pretrained_hard.pt   contains state_dicts + scalers +
                                              enc + params + member metadata

To feed this bundle into the existing inverse-design pipeline, run::

    python composite_design.py --mode inverse \\
        --output_dir <output_dir> \\
        --use_pretrained_inverse <output_dir>/inverse_pretrained_hard.pt

The ``--mode inverse`` path detects the flag, skips
``train_full_data_hard_pinn`` (which would otherwise retrain from scratch
sequentially), and proceeds directly to classifier training + GP-BO + the
rest of the inverse-design analyses.

Failure handling: members whose part file is missing or that wrote a
``.FAILED.json`` marker are reported and skipped; the merge proceeds with
the surviving members provided at least one survived.

Usage:
    python hpo/inverse_merge.py \\
        --data_dir ./data --output_dir ./results_paper --n_ensemble 20
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from typing import Dict, List

import numpy as np
import torch

_HPO_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HPO_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import composite_design as cd  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning, module=r"matplotlib")


def _make_logger(log_path: str) -> logging.Logger:
    log = logging.getLogger("inverse.merge")
    log.setLevel(logging.INFO)
    log.handlers = []
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(ch)
    return log


def main():
    p = argparse.ArgumentParser(
        description="Merge per-member inverse-design partials into one "
                    "pretrained surrogate bundle.")
    p.add_argument("--data_dir",   default="./data")
    p.add_argument("--output_dir", default="./results_paper")
    p.add_argument("--n_ensemble", type=int, default=20)
    p.add_argument("--seed",       type=int, default=2026)
    p.add_argument("--force_cpu",  action="store_true")
    p.add_argument("--dry_run",    action="store_true")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "inverse_merge_log.txt")
    logger = _make_logger(log_path)

    cd.CFG.force_cpu = bool(args.force_cpu)
    cd.CFG.dry_run   = bool(args.dry_run)
    cd.CFG.seed       = int(args.seed)
    cd.CFG.seed_base  = int(args.seed)
    cd.CFG.split_seed = int(args.seed)
    cd.refresh_device()
    cd.set_publication_style()

    parts_dir = os.path.join(args.output_dir, "parts_inverse_hard")
    if not os.path.isdir(parts_dir):
        raise FileNotFoundError(
            f"Per-member parts directory not found: {parts_dir}.  "
            f"Did inverse_member.py run?",
        )

    logger.info("=" * 80)
    logger.info("INVERSE-DESIGN MERGE")
    logger.info(f"  parts_dir={parts_dir}")
    logger.info(f"  output_dir={args.output_dir}")
    logger.info("=" * 80)

    # Collect partials.
    parts: List[Dict] = []
    failed: List[Dict] = []
    missing: List[int] = []
    for idx in range(args.n_ensemble):
        part_path = os.path.join(parts_dir, f"member_{idx:02d}.pt")
        fail_path = os.path.join(parts_dir, f"member_{idx:02d}.FAILED.json")
        if os.path.exists(part_path):
            try:
                parts.append(torch.load(part_path, map_location="cpu", weights_only=False))
            except TypeError:
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
            "No surviving per-member partial bundles.  Cannot construct the "
            f"inverse-design surrogate.  Inspect logs in {parts_dir}.",
        )

    # Sort by member_idx so the bundle's models[] list is deterministic.
    parts.sort(key=lambda d: int(d.get("member_idx", -1)))
    logger.info(f"  Loaded {len(parts)} partial bundles "
                f"(member indices: {[int(d['member_idx']) for d in parts]})")

    # Preprocessors should be identical across partials (deterministic from
    # df_all).  Take from the first partial; verify the rest agree on the
    # scaler statistics — this would catch the rare case where someone
    # re-ran individual members against a modified data directory.
    scaler_disp = parts[0]["scaler_disp"]
    scaler_out  = parts[0]["scaler_out"]
    enc         = parts[0]["enc"]
    params      = parts[0]["params"]
    cfg         = parts[0]["cfg"]
    in_d        = int(parts[0]["in_d"])
    for p in parts[1:]:
        if not np.allclose(
            scaler_disp.mean_, p["scaler_disp"].mean_, rtol=0, atol=1e-12,
        ) or not np.allclose(
            scaler_disp.scale_, p["scaler_disp"].scale_, rtol=0, atol=1e-12,
        ):
            raise RuntimeError(
                f"Preprocessor mismatch between member {parts[0]['member_idx']} "
                f"and member {p['member_idx']}.  Some partial bundles were "
                f"trained against a different data directory; cannot merge.",
            )
        if int(p["in_d"]) != in_d:
            raise RuntimeError(
                f"Feature-dimension mismatch between member "
                f"{parts[0]['member_idx']} (in_d={in_d}) and member "
                f"{p['member_idx']} (in_d={p['in_d']}).  Cannot merge.",
            )

    # Convergence filter (Tukey fence on TRAINING-set R²).  Mirrors the
    # filter inside train_full_data_hard_pinn at lines ~1236-1254.
    train_r2_scores = [float(p["train_r2"]) for p in parts]
    k_iqr = cd.CFG.convergence_filter_iqr
    M_total = len(parts)
    fence = float("-inf")
    keep_mask = [True] * M_total
    if k_iqr > 0 and M_total >= 5:
        q1 = float(np.percentile(train_r2_scores, 25))
        q3 = float(np.percentile(train_r2_scores, 75))
        iqr = q3 - q1
        if iqr > 0.01:
            fence = q1 - k_iqr * iqr
            keep_mask = [r2 >= fence for r2 in train_r2_scores]
        logger.info(f"  Convergence filter (Tukey fence, k={k_iqr}):")
        logger.info(f"    Train R² stats: Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f}")
        if iqr > 0.01:
            logger.info(f"    Fence = Q1 - {k_iqr}*IQR = {fence:.4f}")
        else:
            logger.info("    IQR < 0.01: skipping filter")
        M_eff = sum(keep_mask)
        n_discarded = M_total - M_eff
        if n_discarded > 0:
            logger.info(f"    M_total={M_total}, M_eff={M_eff}, discarded={n_discarded}")
            for i, keep in enumerate(keep_mask):
                if not keep:
                    logger.info(
                        f"    Discarded M{parts[i]['member_idx']}: "
                        f"train R²={train_r2_scores[i]:.4f} < fence {fence:.4f}"
                    )
        else:
            logger.info(f"    All {M_total} members above fence")
    # Hard abort when fewer than MIN_SURVIVORS members survived (mirrors
    # the forward-merge floor).  Writing a 1- or 2-member pretrained
    # inverse surrogate would silently produce misleading GP-BO posterior
    # widths downstream.
    MIN_SURVIVORS = 3
    if sum(keep_mask) < MIN_SURVIVORS:
        raise RuntimeError(
            f"Only {sum(keep_mask)} members survived the Tukey-fence "
            f"convergence filter (minimum required: {MIN_SURVIVORS}).  "
            f"M_total={M_total}, failed={len(failed)}, missing={len(missing)}.  "
            f"Rerun the failed/missing members or relax the filter via "
            f"composite_design.CFG.convergence_filter_iqr before merging.",
        )

    surviving = [p for p, k in zip(parts, keep_mask) if k]
    inv_models_state = [
        {"state_dict": p["state_dict"], "in_d": p["in_d"],
         "member_idx": int(p["member_idx"]), "train_r2": float(p["train_r2"])}
        for p in surviving
    ]

    # Validate that we can actually reconstruct one model (catches state_dict /
    # arch mismatches before downstream loaders blow up).
    test_model = cd.HardEnergyNet(
        in_d, cfg["hidden_layers"], cfg["dropout"], cfg["softplus_beta"]
    )
    test_model.configure_zero_bc(params)
    test_model.load_state_dict(surviving[0]["state_dict"])
    logger.info(f"  Reconstruction smoke check OK ({sum(p.numel() for p in test_model.parameters())} params).")

    # Write the pretrained bundle.  The downstream consumer is
    # composite_design._train_inverse_and_analyze invoked with
    # ``--use_pretrained_inverse <this path>`` (we add that flag in a
    # companion patch to composite_design).
    out_path = os.path.join(args.output_dir, "inverse_pretrained_hard.pt")
    bundle = {
        "approach":         "hard",
        "kind":             "inverse_full_data_pretrained",
        "cfg":              cfg,
        "in_d":             in_d,
        "scaler_disp":      scaler_disp,
        "scaler_out":       scaler_out,
        "enc":              enc,
        "params":           params,
        "inv_models_state": inv_models_state,
        "member_train_r2":  train_r2_scores,
        "M_total":          int(M_total),
        "M_eff":            int(sum(keep_mask)),
        # ``fence`` stays at -inf when no fence was applied (M < 5 or
        # IQR ≤ 0.01).  Emit None → JSON null so downstream loaders do
        # not have to special-case non-standard ``-Infinity`` literals.
        "convergence_fence": (float(fence) if np.isfinite(fence) else None),
        "failed_members":   failed,
        "missing_member_idx": missing,
        "seed_base":        int(args.seed),
    }
    torch.save(bundle, out_path)
    logger.info(f"\n  Wrote: {out_path}")

    # Also write a small JSON summary for human-readable inspection.
    summary_path = os.path.join(args.output_dir, "inverse_pretrained_hard_summary.json")
    summary = {
        "M_total":            int(M_total),
        "M_eff":              int(sum(keep_mask)),
        "member_indices":     [int(p["member_idx"]) for p in surviving],
        "train_r2_per_member": train_r2_scores,
        "mean_train_r2":      float(np.mean(train_r2_scores)),
        "min_train_r2":       float(np.min(train_r2_scores)),
        "max_train_r2":       float(np.max(train_r2_scores)),
        "convergence_fence":  (float(fence) if np.isfinite(fence) else None),
        "failed_members":     failed,
        "missing_member_idx": missing,
        "bundle_path":        os.path.abspath(out_path),
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    logger.info(f"  Wrote: {summary_path}")

    logger.info("\n" + "=" * 80)
    logger.info("INVERSE-DESIGN MERGE COMPLETE")
    logger.info("=" * 80)
    logger.info("Next step: feed the pretrained surrogate into the inverse-")
    logger.info("design pipeline (skips training, runs classifier + GP-BO + ")
    logger.info("ablations):")
    logger.info("")
    logger.info(f"  python composite_design.py --mode inverse \\")
    logger.info(f"      --output_dir {args.output_dir} \\")
    logger.info(f"      --use_pretrained_inverse {out_path}")
    logger.info("")


if __name__ == "__main__":
    main()
