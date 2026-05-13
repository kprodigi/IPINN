# -*- coding: utf-8 -*-
"""
================================================================================
STAGE 2 — per-member retrain (one ensemble member per SLURM task)
================================================================================
Train ONE (or a small list) of the M ensemble members for the Stage 2
production retrain, so the M=20 ensemble can be parallelised across GPU
nodes via a SLURM array job.

Companion to :mod:`hpo.stage2_v16` (which runs the full M=20 sequentially
on a single GPU).  Combined with :mod:`hpo.merge_stage2_members` (final
aggregation + Tukey-fence filter + ensemble metrics), this lets us cut the
wall-clock from ~M × per-member to (M / N_gpus) × per-member.

Reproducibility: each member m uses seed = ``seed + m * 1000`` — identical to
:func:`composite_design_v20.train_ensemble`'s loop.  So member m trained by
this launcher matches member m trained by ``stage2_v16.py`` bit-for-bit.

Outputs (per ``--output_dir``):
    parts_<approach>/member_<idx>.pt   per-member partial bundle (state_dict,
                                       train_r2, val_metrics, history, meta)
    parts_<approach>/member_<idx>.log  per-member training log

Usage (one member per SLURM task):
    python hpo/stage2_member.py --approach hard --member_idx 0 \\
        --data_dir ./data --output_dir ./results_stage2_v16

Or train a contiguous range of members in one task (e.g. if you have fewer
GPUs than M):
    python hpo/stage2_member.py --approach hard --members 0,1 ...
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

# stage2_member.py lives in ``hpo/``; composite_design_v20.py is at the repo root.
_HPO_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HPO_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import composite_design_v20 as cd  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning, module=r"matplotlib")


def _make_logger(approach: str, member_idx: int, log_path: str) -> logging.Logger:
    log = logging.getLogger(f"stage2.{approach}.m{member_idx}")
    log.setLevel(logging.INFO)
    log.handlers = []
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(ch)
    return log


def _parse_members(spec: str) -> List[int]:
    """Parse ``--members 0,1,2`` into ``[0, 1, 2]``."""
    return [int(x) for x in spec.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser(
        description="Stage 2 per-member retrain (one approach, one or more "
                    "members, unseen-theta=60deg).  Use with "
                    "submit_stage2_hard_array.sh.")
    p.add_argument("--approach", choices=["ddns", "soft", "hard"], required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--member_idx", type=int,
                       help="Single member index (0-based).  Convenient when "
                            "SLURM_ARRAY_TASK_ID directly maps to the member.")
    group.add_argument("--members", type=str,
                       help="Comma-separated list of member indices to train "
                            "in this task (e.g. '0,10' for two members).")
    p.add_argument("--data_dir",   default="./data")
    p.add_argument("--output_dir", default="./results_stage2_v16")
    p.add_argument("--n_ensemble", type=int, default=20,
                   help="Total ensemble size M (used only for seed allocation; "
                        "must match what merge_stage2_members.py expects).")
    p.add_argument("--seed",       type=int, default=2026)
    p.add_argument("--force_cpu",  action="store_true")
    p.add_argument("--dry_run",    action="store_true",
                   help="Smoke mode: shrinks training budgets via "
                        "composite_design_v20._dry_run_shrink_training_cfg.  "
                        "Useful for CI; NOT for production retrain.")
    args = p.parse_args()

    members = [args.member_idx] if args.member_idx is not None else _parse_members(args.members)
    if not members:
        raise ValueError("No member indices specified.")
    for m in members:
        if not (0 <= m < args.n_ensemble):
            raise ValueError(
                f"member index {m} is out of range [0, {args.n_ensemble - 1}]."
            )

    parts_dir = os.path.join(args.output_dir, f"parts_{args.approach}")
    os.makedirs(parts_dir, exist_ok=True)

    # Apply CFG overrides (mirrors stage2_v16.py).  n_ensemble is set to the
    # FULL ensemble size so seed_base * 1000 indexing matches the sequential
    # launcher's bit-for-bit.
    cd.CFG.n_ensemble = int(args.n_ensemble)
    cd.CFG.seed       = int(args.seed)
    cd.CFG.seed_base  = int(args.seed)
    cd.CFG.split_seed = int(args.seed)
    cd.CFG.force_cpu  = bool(args.force_cpu)
    cd.CFG.dry_run    = bool(args.dry_run)
    cd.refresh_device()
    cd.set_publication_style()

    cfg = cd.get_model_config(args.approach, "unseen")

    # Bootstrap logger for the setup phase (per-member logger reset inside loop).
    setup_log_path = os.path.join(
        parts_dir, f"member_{members[0]:02d}_setup.log"
    )
    setup_logger = _make_logger(args.approach, members[0], setup_log_path)
    setup_logger.info("=" * 80)
    setup_logger.info(f"STAGE 2 per-member retrain — approach={args.approach}")
    setup_logger.info(f"  Training members: {members}")
    setup_logger.info(f"  M_total={cd.CFG.n_ensemble}  epochs={cfg['epochs']}  "
                      f"arch={cfg['hidden_layers']}  batch={cfg['batch_size']}")
    setup_logger.info(f"  unseen-θ={cd.CFG.theta_star}° protocol")
    setup_logger.info(f"  parts_dir={parts_dir}")
    setup_logger.info("=" * 80)

    # Data + preprocessors (deterministic given args.seed).  These MUST be
    # identical across all jobs so the partial bundles compose into a coherent
    # ensemble.  Bit-for-bit reproducibility: load_data, split_unseen_angle,
    # and create_preprocessors are all deterministic in CFG.seed/split_seed.
    df_all = cd.load_data(args.data_dir, setup_logger)
    train_df, val_df = cd.split_unseen_angle(df_all, cd.CFG.theta_star, setup_logger)
    scaler_disp, scaler_out, enc, params = cd.create_preprocessors(train_df, setup_logger)

    train_fn = {
        "ddns": cd.train_ddns,
        "soft": cd.train_soft,
        "hard": cd.train_hard,
    }[args.approach]

    for member_idx in members:
        seed_m = cd.CFG.seed_base + member_idx * 1000
        log_path = os.path.join(parts_dir, f"member_{member_idx:02d}.log")
        logger = _make_logger(args.approach, member_idx, log_path)
        logger.info("=" * 80)
        logger.info(f"  Training member {member_idx}  (seed={seed_m})")
        logger.info("=" * 80)
        t0 = time.time()
        try:
            model, hist, _r2, meta = train_fn(
                train_df, val_df, scaler_disp, scaler_out, enc, params,
                seed_m, "unseen", logger,
            )
        except (RuntimeError, ValueError) as ex:
            logger.error(
                f"  TRAINING FAILED ({type(ex).__name__}): {ex}.  "
                f"No partial bundle written for member {member_idx}."
            )
            # Don't crash the SLURM task — write a failure marker so merge can
            # report cleanly.
            fail_path = os.path.join(parts_dir, f"member_{member_idx:02d}.FAILED.json")
            with open(fail_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {"member_idx": member_idx, "exception_type": type(ex).__name__,
                     "exception_repr": repr(ex)}, fh, indent=2,
                )
            continue
        elapsed = time.time() - t0

        # Per-member metrics: train R² (for Tukey filter) and val metrics.
        train_metrics = cd.evaluate_model(
            model, args.approach, train_df, scaler_disp, scaler_out, enc, params,
        )
        val_metrics = cd.evaluate_model(
            model, args.approach, val_df, scaler_disp, scaler_out, enc, params,
        )
        logger.info(f"  M{member_idx}: train R²_load={train_metrics['load_r2']:.4f}  "
                    f"val R²_load={val_metrics['load_r2']:.4f}  time={elapsed:.0f}s")

        # Persist partial bundle.  ``state_dict`` is the bare network's; the
        # merge script reconstructs the model + applies configure_zero_bc.
        # ``in_d`` is preserved so the merge can re-instantiate without
        # re-touching the training data.
        in_d = cd.build_features(train_df.head(1), scaler_disp, enc).shape[1]
        part = {
            "approach":       args.approach,
            "member_idx":     int(member_idx),
            "seed":           int(seed_m),
            "in_d":           int(in_d),
            "cfg":            cfg,
            "state_dict":     {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "train_metrics":  train_metrics,
            "val_metrics":    val_metrics,
            "history":        hist,
            "meta":           meta,
            "training_time":  float(elapsed),
        }
        part_path = os.path.join(parts_dir, f"member_{member_idx:02d}.pt")
        torch.save(part, part_path)
        logger.info(f"  Wrote: {part_path}")

    setup_logger.info("=" * 80)
    setup_logger.info("STAGE 2 PER-MEMBER COMPLETE")
    setup_logger.info("=" * 80)


if __name__ == "__main__":
    main()
