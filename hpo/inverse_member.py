# -*- coding: utf-8 -*-
"""
================================================================================
INVERSE-DESIGN — per-member retrain of the full-data Hard-PINN surrogate
================================================================================
Trains ONE (or a small list) of the M full-data Hard-PINN ensemble members
that ``composite_design_v20.train_full_data_hard_pinn`` would otherwise
produce sequentially.  Parallelise the inverse-design surrogate training
across GPU nodes via a SLURM array job.

Companion to :mod:`hpo.merge_inverse_members` (the aggregator).  Together
they replace the sequential M=20 loop inside ``train_full_data_hard_pinn``
with a SLURM array of M independent tasks + one merge job.

Reproducibility: each member m uses ``seed = seed + m * 100`` — identical
to the stride used by ``train_full_data_hard_pinn`` (note: this is a
DIFFERENT stride from the unseen-protocol stage 2 launcher, which uses
stride 1000; the two strides ensure full-data and validation ensemble
members are initialised independently for m >= 1).

Implementation: the per-member launcher invokes
``train_full_data_hard_pinn`` directly with ``CFG.n_ensemble = 1`` and
``CFG.seed_base`` shifted by ``member_idx * 100``, so the function trains
exactly one model whose effective seed matches what the sequential M-loop
would have used for that member index.  No code duplication.

Outputs (per ``--output_dir``):
    parts_inverse_hard/member_<idx>.pt   per-member partial bundle
    parts_inverse_hard/member_<idx>.log  per-member training log

Usage (one member per SLURM task):
    python hpo/inverse_member.py --member_idx 0 \\
        --data_dir ./data --output_dir ./results_paper

Or train a contiguous range of members in one task:
    python hpo/inverse_member.py --members 0,1 ...
================================================================================
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
import warnings
from typing import Dict, List

import numpy as np
import torch

_HPO_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HPO_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import composite_design_v20 as cd  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning, module=r"matplotlib")


def _make_logger(member_idx: int, log_path: str) -> logging.Logger:
    log = logging.getLogger(f"inverse.m{member_idx}")
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
    return [int(x) for x in spec.split(",") if x.strip()]


def _build_cfg_for_save(dry_run: bool) -> Dict:
    """Re-build the cfg that ``train_full_data_hard_pinn`` will use, for
    saving alongside each partial bundle.  Mirrors the in-place overrides
    inside that function (epochs=1500, batch_size=128, warmup_epochs=150).
    The merge step re-derives this; we save it for traceability.
    """
    cfg = cd.get_model_config("hard", "unseen")
    cfg = dict(cfg)  # don't mutate the cached config
    cfg["epochs"] = 1500
    cfg["batch_size"] = 128
    cfg["warmup_epochs"] = 150
    if dry_run:
        cfg["epochs"] = min(int(cfg["epochs"]), 12)
        cfg["warmup_epochs"] = min(int(cfg.get("warmup_epochs", 6)), 4)
        cd._dry_run_shrink_training_cfg(cfg)
    return cfg


def main():
    p = argparse.ArgumentParser(
        description="Inverse-design per-member retrain of the full-data "
                    "Hard-PINN surrogate.  Use with "
                    "submit_inverse_hard_array.sh.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--member_idx", type=int,
                       help="Single member index (0-based).")
    group.add_argument("--members", type=str,
                       help="Comma-separated list of member indices.")
    p.add_argument("--data_dir",   default="./data")
    p.add_argument("--output_dir", default="./results_paper")
    p.add_argument("--n_ensemble", type=int, default=20,
                   help="Total ensemble size M (used only for seed allocation; "
                        "must match what merge_inverse_members.py expects).")
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
                f"member index {m} is out of range [0, {args.n_ensemble - 1}].",
            )

    parts_dir = os.path.join(args.output_dir, "parts_inverse_hard")
    os.makedirs(parts_dir, exist_ok=True)

    base_seed = int(args.seed)
    # Apply CFG flags that don't depend on the member.
    cd.CFG.force_cpu = bool(args.force_cpu)
    cd.CFG.dry_run   = bool(args.dry_run)
    cd.refresh_device()
    cd.set_publication_style()

    # Bootstrap logger for the setup phase.
    setup_log_path = os.path.join(parts_dir, f"member_{members[0]:02d}_setup.log")
    setup_logger = _make_logger(members[0], setup_log_path)
    setup_logger.info("=" * 80)
    setup_logger.info("INVERSE-DESIGN per-member retrain (full-data Hard-PINN)")
    setup_logger.info(f"  Training members: {members}")
    setup_logger.info(f"  M_total={args.n_ensemble}  base_seed={base_seed}")
    setup_logger.info(f"  parts_dir={parts_dir}")
    setup_logger.info("=" * 80)

    # Load data once.  This is identical across all tasks (deterministic).
    df_all = cd.load_data(args.data_dir, setup_logger)

    cfg_template = _build_cfg_for_save(bool(args.dry_run))

    for member_idx in members:
        per_member_log_path = os.path.join(parts_dir, f"member_{member_idx:02d}.log")
        logger = _make_logger(member_idx, per_member_log_path)
        member_seed = base_seed + member_idx * 100
        logger.info("=" * 80)
        logger.info(f"  Training inverse-Hard member {member_idx}  (seed={member_seed})")
        logger.info("=" * 80)

        # CFG hack: trick train_full_data_hard_pinn into training one model
        # whose effective seed matches member_idx.  Inside the function:
        #     seed = CFG.seed_base + m_idx * 100   for m_idx in range(CFG.n_ensemble)
        # With n_ensemble=1 we have m_idx=0, so we set seed_base to be the
        # desired effective seed.
        prev_n_ens = cd.CFG.n_ensemble
        prev_seed_base = cd.CFG.seed_base
        cd.CFG.n_ensemble = 1
        cd.CFG.seed       = member_seed
        cd.CFG.seed_base  = member_seed
        cd.CFG.split_seed = member_seed

        t0 = time.time()
        try:
            inv_models, scaler_disp, scaler_out, enc, params = \
                cd.train_full_data_hard_pinn(df_all, logger)
        except (RuntimeError, ValueError) as ex:
            logger.error(
                f"  TRAINING FAILED ({type(ex).__name__}): {ex}.  "
                f"No partial bundle written for member {member_idx}."
            )
            fail_path = os.path.join(parts_dir, f"member_{member_idx:02d}.FAILED.json")
            with open(fail_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {"member_idx": member_idx, "exception_type": type(ex).__name__,
                     "exception_repr": repr(ex)}, fh, indent=2,
                )
            # Restore CFG and continue to next member (or exit).
            cd.CFG.n_ensemble = prev_n_ens
            cd.CFG.seed_base  = prev_seed_base
            continue
        elapsed = time.time() - t0

        # Restore CFG (so subsequent members in the same task get clean state).
        cd.CFG.n_ensemble = prev_n_ens
        cd.CFG.seed_base  = prev_seed_base

        if len(inv_models) != 1:
            logger.error(
                f"  train_full_data_hard_pinn returned {len(inv_models)} models, "
                f"expected 1.  Aborting member {member_idx}."
            )
            continue
        model = inv_models[0]

        # Re-compute train-set R²_load for the Tukey filter at merge time.
        # train_full_data_hard_pinn already computes this internally but doesn't
        # return it; recomputing on the (eval) model is cheap and deterministic.
        X_full = cd.build_features(df_all, scaler_disp, enc)
        X_tensor = cd.to_tensor(X_full)
        y_full_np = df_all[["load_kN", "energy_J"]].values
        Fv, _ = cd.hard_pinn_predict_load_energy(model, X_tensor, params)
        train_r2 = float(cd.r2_safe(y_full_np[:, 0], Fv))
        logger.info(
            f"  M{member_idx}: train R²_load={train_r2:.4f}  time={elapsed:.0f}s"
        )

        # Persist partial bundle.  Each task re-fits preprocessors
        # deterministically from df_all; we still save scalers/encoder/params
        # in the FIRST member's bundle for the merge script's convenience.
        in_d = int(X_full.shape[1])
        part = {
            "approach":       "hard",
            "kind":           "inverse_full_data",
            "member_idx":     int(member_idx),
            "seed":           int(member_seed),
            "in_d":           in_d,
            "cfg":            cfg_template,
            "state_dict":     {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "scaler_disp":    copy.deepcopy(scaler_disp),
            "scaler_out":     copy.deepcopy(scaler_out),
            "enc":            copy.deepcopy(enc),
            "params":         params,
            "train_r2":       train_r2,
            "training_time":  float(elapsed),
        }
        part_path = os.path.join(parts_dir, f"member_{member_idx:02d}.pt")
        torch.save(part, part_path)
        logger.info(f"  Wrote: {part_path}")

    setup_logger.info("=" * 80)
    setup_logger.info("INVERSE-DESIGN PER-MEMBER COMPLETE")
    setup_logger.info("=" * 80)


if __name__ == "__main__":
    main()
