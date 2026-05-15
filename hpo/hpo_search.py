# -*- coding: utf-8 -*-
"""
================================================================================
HPO SEARCH — Optuna TPE for the three forward-design surrogates
================================================================================
Tunes the hyperparameters of one of the three forward surrogates
(DDNS / Soft-PINN / Hard-PINN) under the unseen-angle protocol
(theta_star = 60 deg).  The objective is the mean validation load R^2
across ``--n_seeds`` independently trained ensemble members for each
trial.  Designed to run distributed across SLURM workers via a single
shared SQLite-backed study, with full resume-from-preemption support.

Sampling policy
---------------
* ``n_startup_trials`` (default 15) are drawn uniformly at random from
  the per-approach search space.  These give the TPE model a diverse
  initial set of observations so it does not exploit prematurely.
* After startup, Optuna's multivariate TPE sampler with grouped
  categorical parameters proposes subsequent trials.
* An optional small set of **warm-start** parameter dicts can be
  enqueued before any TPE proposals so that the search starts with
  one or more informed prior configurations.  Warm starts use the
  same parameter names as the search space; missing parameters are
  freely sampled.  Pass ``--no_warm_starts`` to disable.

Per-trial evaluation
--------------------
For each trial:
  1. Patch the parameters into a working configuration dict
     (deep-copied from ``get_model_config(approach, "unseen")``).
  2. Train ``--n_seeds`` ensemble members (each with a different seed
     drawn from ``base_seed + 1000 * k``) on the unseen-angle training
     split.  Each member uses the same epoch budget controlled by
     ``--hpo_epochs``.
  3. Evaluate every member on the validation set (theta_star = 60 deg).
  4. The trial's reported value is the mean validation load R^2 across
     the surviving members.

Distributed execution
---------------------
The study is persisted in a SQLite file via Optuna's RDB storage.  Many
workers can attach simultaneously; Optuna serialises trial scheduling
through SQLite locks so each trial is run by exactly one worker.  Pass
the same ``--output_dir`` and ``--study_name`` from each worker to
share the study.

Usage
-----
    # One approach per invocation (SLURM-friendly):
    python hpo/hpo_search.py --approach soft --n_trials 100 --output_dir ./hpo_out
    python hpo/hpo_search.py --approach hard --n_trials 120 --output_dir ./hpo_out
    python hpo/hpo_search.py --approach ddns --n_trials 80  --output_dir ./hpo_out

    # Quick local smoke (5 trials, tiny budget, ~3 min on CPU):
    python hpo/hpo_search.py --approach hard --n_trials 5 --hpo_epochs 20 \
        --n_seeds 1 --output_dir ./hpo_smoke --dry_run

    # Resume after preemption — same --output_dir + --study_name picks
    # up the existing trials:
    python hpo/hpo_search.py --approach soft --n_trials 100 --output_dir ./hpo_out

    # Multiple workers sharing one study (different GPUs on one node):
    CUDA_VISIBLE_DEVICES=0 python hpo/hpo_search.py --approach hard --output_dir ./hpo_out &
    CUDA_VISIBLE_DEVICES=1 python hpo/hpo_search.py --approach hard --output_dir ./hpo_out &

Outputs (per ``--output_dir``)
------------------------------
    hpo_study_<approach>.db                 SQLite study (resumable; source of truth)
    hpo_best_params_<approach>.json         Best params so far — rewritten
                                            atomically after every completed
                                            trial; copy-pasteable into the
                                            ``get_model_config`` cfg.  Also
                                            includes a ``best_metrics`` block
                                            with mean/std train and val
                                            R^2_load + R^2_energy across
                                            seeds, and a ``best_per_seed_metrics``
                                            list for reproducibility.
    hpo_history_<approach>.csv              Every trial.  Columns include the
                                            objective value plus
                                            ``mean_train_load_r2``,
                                            ``mean_val_load_r2``,
                                            ``mean_train_energy_r2``,
                                            ``mean_val_energy_r2``,
                                            std versions, the train-val gap,
                                            and all searched hyperparameters.
                                            Rewritten after every trial.
    hpo_run_state_<approach>.json           Compact progress snapshot
                                            (completed/failed/running counts,
                                            current best, last trial)
    hpo_best_snapshots_<approach>/          Audit trail: a JSON copy of the
        best_at_trial_NNNN.json             best-params dict (incl. metrics)
                                            at the moment each new best was set
    hpo_log_<approach>.txt                  Full Optuna log

Crash / preemption safety
-------------------------
Three layers of resilience:

1. **Per-trial atomic checkpoints.** After every trial completes, the
   best-params JSON, full history CSV, and run-state JSON are rewritten
   atomically (``write to tmp + os.replace``).  A worker killed mid-write
   either leaves the OLD file or the NEW file — never a half-written one.

2. **Heartbeat-based trial revival.** Workers ping the SQLite study every
   ``heartbeat_interval`` seconds (default 300 s).  If a worker dies and
   its trial's heartbeat is older than ``grace_period`` (default 900 s),
   Optuna marks the trial FAILED and ``RetryFailedTrialCallback`` re-enqueues
   it (up to 2 retries) so the search resumes seamlessly.

3. **Resume-from-DB.** On startup the script counts completed trials in
   the SQLite DB and runs only ``n_trials - n_done`` more.  Resubmitting
   the same SLURM command after a job aborts is sufficient to continue
   the search.  The first checkpoint write inside the resumed run also
   refreshes the on-disk JSONs to reflect the current best.
================================================================================
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.storages import RDBStorage, RetryFailedTrialCallback
except ImportError:  # pragma: no cover
    sys.stderr.write(
        "Optuna is required for HPO.  Install: pip install 'optuna>=3.4,<5'\n"
    )
    raise

_HPO_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HPO_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import composite_design as cd  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module=r"matplotlib")
warnings.filterwarnings(
    "ignore",
    category=optuna.exceptions.ExperimentalWarning,
    message=r"Argument ``(multivariate|group)`` is an experimental feature.*",
)


# =============================================================================
# SEARCH SPACES
#
# Architecture is encoded as a STRING key (e.g. ``"128-64-32"``) rather than
# a tuple/list of ints.  Optuna's SQLite persistence layer round-trips
# str/int/float/bool/None losslessly; tuples are silently converted to lists
# on reload, which then trips
# ``CategoricalDistribution does not support dynamic value space.``
# once TPE's ``multivariate=True, group=True`` engages after the startup
# trials.  Strings sidestep this entirely.
#
# The Soft-PINN search space includes ``w_phys`` (work-energy residual
# penalty weight) and ``w_bc`` (paired E(0)/F(0) soft-penalty weight) but
# does NOT include weights for monotonicity / angle-smoothness / curvature —
# those are not part of the simplified Soft loss formulation.  Similarly,
# the Hard-PINN search space contains only data-fit and stabilisation
# hyperparameters since the three core physics constraints are enforced
# architecturally.
# =============================================================================
HL_DDNS_SOFT = {
    "64-32":      [64, 32],
    "128-64":     [128, 64],
    "128-64-32":  [128, 64, 32],
    "256-128":    [256, 128],
    "256-128-64": [256, 128, 64],
}
HL_HARD = {
    "32-32":      [32, 32],
    "64-32":      [64, 32],
    "64-64":      [64, 64],
    "64-64-64":   [64, 64, 64],
    "128-64":     [128, 64],
    "128-64-32":  [128, 64, 32],
}


def suggest_ddns(trial: "optuna.Trial") -> Dict:
    """Search space for the data-driven baseline (no physics losses)."""
    hl_key = trial.suggest_categorical("hidden_layers", list(HL_DDNS_SOFT.keys()))
    return {
        "hidden_layers":   list(HL_DDNS_SOFT[hl_key]),
        "batch_size":      trial.suggest_categorical("batch_size", [32, 64, 128]),
        "lr":              trial.suggest_float("lr",             1e-6, 1e-3, log=True),
        "weight_decay":    trial.suggest_float("weight_decay",   1e-6, 1e-3, log=True),
        "dropout":         trial.suggest_float("dropout",        0.0,  0.10),
        "softplus_beta":   trial.suggest_float("softplus_beta",  5.0,  30.0),
        "smoothl1_beta":   trial.suggest_float("smoothl1_beta",  0.1,  3.0),
        "w_data_load":     trial.suggest_float("w_data_load",    0.5,  10.0, log=True),
        "w_data_energy":   trial.suggest_float("w_data_energy",  0.5,  10.0, log=True),
        "sched_patience":  trial.suggest_int  ("sched_patience", 20,   100),
        "sched_factor":    trial.suggest_float("sched_factor",   0.2,  0.8),
    }


def suggest_soft(trial: "optuna.Trial") -> Dict:
    """Search space for the Soft-PINN.

    The Soft-PINN loss contains the data terms (w_data_load, w_data_energy),
    the work-energy residual loss (w_phys), and the paired boundary-condition
    soft penalty (w_bc) that enforces BOTH E(0)=0 AND F(0)=0.  No additional
    field-wide regularisers (monotonicity, angle smoothness, curvature) are
    applied: the three core physics constraints common to both PINN
    variants are exactly the ones searched here.
    """
    hl_key = trial.suggest_categorical("hidden_layers", list(HL_DDNS_SOFT.keys()))
    return {
        "hidden_layers":   list(HL_DDNS_SOFT[hl_key]),
        "batch_size":      trial.suggest_categorical("batch_size", [32, 64, 128]),
        "lr":              trial.suggest_float("lr",             1e-5, 5e-2, log=True),
        "weight_decay":    trial.suggest_float("weight_decay",   1e-6, 5e-3, log=True),
        "dropout":         trial.suggest_float("dropout",        0.0,  0.05),
        "softplus_beta":   trial.suggest_float("softplus_beta",  5.0,  25.0),
        "smoothl1_beta":   trial.suggest_float("smoothl1_beta",  0.1,  3.0),
        "w_data_load":     trial.suggest_float("w_data_load",    0.5,  10.0, log=True),
        "w_data_energy":   trial.suggest_float("w_data_energy",  0.1,  10.0, log=True),
        "w_phys":          trial.suggest_float("w_phys",         0.01, 10.0, log=True),
        "w_bc":            trial.suggest_float("w_bc",           0.01, 5.0,  log=True),
        "colloc_ratio":    trial.suggest_float("colloc_ratio",   1.0,  6.0),
        "sched_patience":  trial.suggest_int  ("sched_patience", 20,   100),
        "sched_factor":    trial.suggest_float("sched_factor",   0.2,  0.8),
    }


def suggest_hard(trial: "optuna.Trial") -> Dict:
    """Search space for the Hard-PINN.

    The Hard-PINN loss reduces to the data terms alone (w_load, w_energy)
    because the three core physics constraints (work-energy identity F = dE/dd
    via autograd, and the two boundary conditions E(0) = 0 and F(0) = 0 via
    slope-subtraction) are enforced architecturally and do not appear in the
    loss.  The remaining hyperparameters are the network architecture,
    optimisation knobs, and the stabilisation parameters (warmup_epochs,
    swa_pct) required by the warmup + cosine + SWA training schedule.

    The batch_size choices exclude 8 because empirically a Hard-PINN trial
    at batch_size=8 takes ~5.5 h per ensemble member, which makes the search
    impractically slow.  16 and 32 remain in the modern-ML regime and cut
    per-trial wall-clock by 2-4x without giving up generalisation.
    """
    hl_key = trial.suggest_categorical("hidden_layers", list(HL_HARD.keys()))
    return {
        "hidden_layers":   list(HL_HARD[hl_key]),
        "batch_size":      trial.suggest_categorical("batch_size", [16, 32]),
        "lr":              trial.suggest_float("lr",             1e-6, 5e-3, log=True),
        "weight_decay":    trial.suggest_float("weight_decay",   1e-5, 5e-2, log=True),
        "dropout":         trial.suggest_float("dropout",        0.0,  0.05),
        "softplus_beta":   trial.suggest_float("softplus_beta",  5.0,  25.0),
        "smoothl1_beta":   trial.suggest_float("smoothl1_beta",  0.05, 1.0),
        "w_load":          trial.suggest_float("w_load",         1.0,  20.0, log=True),
        "w_energy":        trial.suggest_float("w_energy",       1.0,  20.0, log=True),
        "grad_clip":       trial.suggest_float("grad_clip",      0.5,  3.0),
        "warmup_epochs":   trial.suggest_int  ("warmup_epochs",  40,   150),
        "swa_pct":         trial.suggest_float("swa_pct",        0.10, 0.35),
    }


SUGGESTERS = {"ddns": suggest_ddns, "soft": suggest_soft, "hard": suggest_hard}
HL_BY_APPROACH = {
    "ddns": HL_DDNS_SOFT,
    "soft": HL_DDNS_SOFT,
    "hard": HL_HARD,
}


# =============================================================================
# WARM-START SEEDS
#
# One informed prior dict per approach is enqueued before any TPE proposals.
# Parameter names must match the suggester keys exactly; categorical values
# must match the suggester's choice strings (e.g. ``"32-32"``, not
# ``[32, 32]``).  Any parameter missing from a warm-start dict is freely
# sampled by Optuna for that trial.  Pass ``--no_warm_starts`` to skip.
# =============================================================================
WARM_START = {
    "ddns": [
        {
            "hidden_layers":   "128-64-32",
            "batch_size":      64,
            "lr":              4.21e-05,
            "weight_decay":    3.16e-05,
            "dropout":         0.016,
            "softplus_beta":   18.90,
            "smoothl1_beta":   1.08,
            "w_data_load":     3.57,
            "w_data_energy":   3.45,
            "sched_patience":  58,
            "sched_factor":    0.46,
        },
    ],
    "soft": [
        {
            "hidden_layers":   "256-128-64",
            "batch_size":      32,
            "lr":              4.24e-03,
            "weight_decay":    1.57e-04,
            "dropout":         1.4e-04,
            "softplus_beta":   11.06,
            "smoothl1_beta":   1.16,
            "w_data_load":     3.61,
            "w_data_energy":   1.70,
            "w_phys":          3.69,
            "w_bc":            0.85,
            "colloc_ratio":    3.70,
            "sched_patience":  71,
            "sched_factor":    0.65,
        },
    ],
    "hard": [
        {
            "hidden_layers":   "32-32",
            "batch_size":      16,
            "lr":              4.0e-05,
            "weight_decay":    5.27e-04,
            "dropout":         0.0003,
            "softplus_beta":   13.82,
            "smoothl1_beta":   0.143,
            "w_load":          6.0,
            "w_energy":        7.0,
            "grad_clip":       1.63,
            "warmup_epochs":   80,
            "swa_pct":         0.20,
        },
    ],
}


# =============================================================================
# TRIAL EVALUATION
# =============================================================================
def _build_trial_cfg(approach: str, params: Dict) -> Dict:
    """Patch the trial parameters into the production base config."""
    cfg = copy.deepcopy(cd.get_model_config(approach, "unseen"))
    cfg.update(params)
    return cfg


def _train_and_score_member(
    approach: str,
    cfg: Dict,
    seed: int,
    df_tr,
    df_val,
    scaler_disp,
    scaler_out,
    enc,
    params,
    logger: logging.Logger,
) -> Dict[str, float]:
    """Train one ensemble member with the trial cfg and return all four
    summary metrics: train and val R^2 for both load and energy.

    Returns a dict with keys
        ``train_load_r2``, ``val_load_r2``,
        ``train_energy_r2``, ``val_energy_r2``.
    The optimiser objective uses ``val_load_r2``; the other three are
    persisted via ``trial.set_user_attr`` so every HPO trial's CSV row
    carries the full train-vs-val picture, including the train < val
    pattern characteristic of the unseen-angle protocol.
    """
    train_fn = {"ddns": cd.train_ddns, "soft": cd.train_soft, "hard": cd.train_hard}[approach]
    _orig_get_cfg = cd.get_model_config

    def _patched_get_cfg(app, protocol, w_phys_override=None):
        if app == approach:
            patched = copy.deepcopy(cfg)
            if w_phys_override is not None and "w_phys" in patched:
                patched["w_phys"] = float(w_phys_override)
            return patched
        return _orig_get_cfg(app, protocol, w_phys_override)

    cd.get_model_config = _patched_get_cfg
    try:
        model, history, best_r2, meta = train_fn(
            df_tr, df_val, scaler_disp, scaler_out, enc, params,
            seed, "unseen", logger,
        )
        # Evaluate on BOTH splits so trials record train vs val side-by-side.
        # ``evaluate_model`` returns load_r2 + energy_r2 on the supplied df.
        val_m = cd.evaluate_model(
            model, approach, df_val, scaler_disp, scaler_out, enc, params,
        )
        train_m = cd.evaluate_model(
            model, approach, df_tr, scaler_disp, scaler_out, enc, params,
        )
        return {
            "train_load_r2":   float(train_m["load_r2"]),
            "val_load_r2":     float(val_m["load_r2"]),
            "train_energy_r2": float(train_m.get("energy_r2", float("nan"))),
            "val_energy_r2":   float(val_m.get("energy_r2", float("nan"))),
        }
    finally:
        cd.get_model_config = _orig_get_cfg


def make_objective(
    approach: str,
    df_tr,
    df_val,
    scaler_disp,
    scaler_out,
    enc,
    params,
    n_seeds: int,
    hpo_epochs: int,
    base_seed: int,
    logger: logging.Logger,
):
    """Build the Optuna objective callable for ``approach``."""
    suggester = SUGGESTERS[approach]

    def objective(trial: "optuna.Trial") -> float:
        trial_params = suggester(trial)
        trial_params["epochs"] = int(hpo_epochs)
        trial_params.setdefault("eval_every", max(1, hpo_epochs // 8))
        cfg = _build_trial_cfg(approach, trial_params)

        # Per-seed metrics; we average across seeds for the trial summary.
        per_seed: List[Dict[str, float]] = []
        t_trial = time.time()
        for k in range(n_seeds):
            # Use a prime stride (17) shifted by one stride from the base
            # so HPO trial seeds never coincide with any production-ensemble
            # member seed.  Forward production uses ``base + 1000 * m``;
            # inverse production uses ``base + 100 * m``; the prime 17 is
            # coprime with both 1000 and 100, and the ``+ 17`` shift ensures
            # ``k = 0`` is also offset from ``base``.  Without this offset,
            # HPO trials at default ``--seed 2026`` would train on the
            # identical RNG state as forward member 0 / 1 / 2 ..., biasing
            # the HPO objective toward those specific trajectories.
            seed_k = base_seed + 17 * (k + 1)
            seed_metrics = _train_and_score_member(
                approach, cfg, seed_k,
                df_tr, df_val, scaler_disp, scaler_out, enc, params,
                logger,
            )
            per_seed.append(seed_metrics)
            logger.info(
                f"  trial {trial.number:03d}  seed {k+1}/{n_seeds}  "
                f"train_R2_load={seed_metrics['train_load_r2']:.4f}  "
                f"val_R2_load={seed_metrics['val_load_r2']:.4f}  "
                f"train_R2_energy={seed_metrics['train_energy_r2']:.4f}  "
                f"val_R2_energy={seed_metrics['val_energy_r2']:.4f}"
            )
            # Report the running mean of the OBJECTIVE (val load R^2) to
            # Optuna so a pruner could act on the partial trial.
            running_mean_val_load = float(np.mean(
                [s["val_load_r2"] for s in per_seed]
            ))
            trial.report(running_mean_val_load, step=k)

        # Trial-level summaries: mean and std across seeds for every metric.
        def _agg(key: str) -> Tuple[float, float]:
            vs = np.array([s[key] for s in per_seed], dtype=float)
            return float(vs.mean()), float(vs.std(ddof=0))

        mean_val_load,   std_val_load   = _agg("val_load_r2")
        mean_train_load, std_train_load = _agg("train_load_r2")
        mean_val_eng,    std_val_eng    = _agg("val_energy_r2")
        mean_train_eng,  std_train_eng  = _agg("train_energy_r2")

        # Persist all four metrics + the train-vs-val gap on the trial so
        # the on-disk history CSV and best-params JSON include the full
        # picture, not just the objective.  ``trial.user_attrs`` round-trips
        # through Optuna's SQLite storage and is exposed in the history
        # callback.
        trial.set_user_attr("mean_val_load_r2",    mean_val_load)
        trial.set_user_attr("mean_train_load_r2",  mean_train_load)
        trial.set_user_attr("mean_val_energy_r2",  mean_val_eng)
        trial.set_user_attr("mean_train_energy_r2", mean_train_eng)
        trial.set_user_attr("std_val_load_r2",     std_val_load)
        trial.set_user_attr("std_train_load_r2",   std_train_load)
        trial.set_user_attr("train_val_load_gap",  mean_train_load - mean_val_load)
        trial.set_user_attr("per_seed_metrics",    per_seed)
        trial.set_user_attr("n_seeds_completed",   len(per_seed))

        elapsed = time.time() - t_trial
        gap = mean_train_load - mean_val_load
        gap_sign = "+" if gap >= 0 else ""
        logger.info(
            f"trial {trial.number:03d} DONE  "
            f"mean_train_R2_load={mean_train_load:.4f}  "
            f"mean_val_R2_load={mean_val_load:.4f}  "
            f"gap={gap_sign}{gap:.4f}  "
            f"(n_seeds={n_seeds}, wall={elapsed:.0f}s)"
        )
        return mean_val_load

    return objective


# =============================================================================
# STUDY ORCHESTRATION
# =============================================================================
def _make_logger(approach: str, log_path: str) -> logging.Logger:
    logger = logging.getLogger(f"hpo.{approach}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    ch = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


def _suggester_keys(approach: str) -> List[str]:
    """Parameter names emitted by the approach's suggester.  Used to filter
    warm starts so they only carry parameters that are still in the search
    space."""
    if approach == "ddns":
        return [
            "hidden_layers", "batch_size", "lr", "weight_decay", "dropout",
            "softplus_beta", "smoothl1_beta", "w_data_load", "w_data_energy",
            "sched_patience", "sched_factor",
        ]
    if approach == "soft":
        return [
            "hidden_layers", "batch_size", "lr", "weight_decay", "dropout",
            "softplus_beta", "smoothl1_beta", "w_data_load", "w_data_energy",
            "w_phys", "w_bc", "colloc_ratio", "sched_patience", "sched_factor",
        ]
    if approach == "hard":
        return [
            "hidden_layers", "batch_size", "lr", "weight_decay", "dropout",
            "softplus_beta", "smoothl1_beta", "w_load", "w_energy",
            "grad_clip", "warmup_epochs", "swa_pct",
        ]
    raise ValueError(f"Unknown approach: {approach}")


def _enqueue_warm_starts(
    study: "optuna.Study", approach: str, logger: logging.Logger,
) -> None:
    """Enqueue the WARM_START dicts as the first Optuna trials.

    Skipped on resume if the same params already appear in the study.
    Warm-start dicts are filtered to keys in the current search space so
    that an obsolete warm start (e.g. one that referenced a removed
    regulariser) does not crash the enqueue step.
    """
    allowed = set(_suggester_keys(approach))
    for ws in WARM_START.get(approach, []):
        ws_filtered = {k: v for k, v in ws.items() if k in allowed}
        if not ws_filtered:
            continue
        try:
            study.enqueue_trial(ws_filtered, skip_if_exists=True)
        except Exception as e:  # pragma: no cover
            logger.warning(f"  warm start skipped: {e}")
        else:
            logger.info(
                f"  enqueued warm start: hidden_layers="
                f"{ws_filtered.get('hidden_layers', '?')}"
            )


# =============================================================================
# CHECKPOINTING — atomic writes after every completed trial
#
# Optuna's SQLite DB is the source of truth (every trial is committed on
# completion).  The plain-text checkpoints below mirror that DB on disk so
# the best parameters and full trial history are inspectable / pasteable
# without opening the DB, AND so they survive any preemption.
#
# Each writer writes to ``<path>.tmp`` then ``os.replace``s the temp file
# over the final path.  ``os.replace`` is atomic on POSIX and Windows, so
# a worker killed mid-write either sees the OLD file or the NEW file —
# never a half-written one.
# =============================================================================
def _atomic_write_text(path: str, content: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(content)
    os.replace(tmp, path)


def _safe_best_trial(study: "optuna.Study"):
    """Return ``study.best_trial`` or ``None`` if no trial has completed.

    Optuna's ``best_trial`` property raises ``ValueError("Record does not
    exist.")`` (not returns ``None``) when the study has zero COMPLETE
    trials.  Wrap the access so callers can do a simple ``if best:``.
    """
    try:
        return study.best_trial
    except ValueError:
        return None


def _best_payload(study: "optuna.Study") -> Optional[Dict]:
    """Build the best-trial JSON payload.  Returns None if no completed trial."""
    best = _safe_best_trial(study)
    if best is None:
        return None
    approach = study.user_attrs.get("approach")
    n_completed = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    n_failed = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL
    )
    n_running = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.RUNNING
    )
    payload: Dict = {
        "approach":      approach,
        "best_value":    float(best.value) if best.value is not None else None,
        "best_trial":    int(best.number),
        "best_params":   {k: v for k, v in best.params.items()},
        "n_trials":      len(study.trials),
        "n_completed":   n_completed,
        "n_failed":      n_failed,
        "n_running":     n_running,
        "last_updated":  time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # Surface the train/val metric breakdown stored as user_attrs by the
    # objective.  These let the headline JSON answer "what's the train R^2
    # at the best trial?" without parsing the history CSV.
    metric_keys = [
        "mean_train_load_r2",   "mean_val_load_r2",
        "mean_train_energy_r2", "mean_val_energy_r2",
        "std_train_load_r2",    "std_val_load_r2",
        "train_val_load_gap",   "n_seeds_completed",
    ]
    best_metrics = {
        k: best.user_attrs.get(k) for k in metric_keys if k in best.user_attrs
    }
    if best_metrics:
        payload["best_metrics"] = best_metrics
    # Full per-seed breakdown for paper-quality reproducibility (small).
    if "per_seed_metrics" in best.user_attrs:
        payload["best_per_seed_metrics"] = best.user_attrs["per_seed_metrics"]
    if "hidden_layers" in best.params and approach in HL_BY_APPROACH:
        hl_key = best.params["hidden_layers"]
        hl_map = HL_BY_APPROACH[approach]
        if hl_key in hl_map:
            payload["best_params_resolved"] = dict(best.params)
            payload["best_params_resolved"]["hidden_layers"] = list(hl_map[hl_key])
    return payload


def _history_csv_content(study: "optuna.Study") -> str:
    """Serialise every trial to CSV.  Columns:

      trial, value, state, datetime_start, datetime_complete, duration_sec,
      mean_train_load_r2, mean_val_load_r2,
      mean_train_energy_r2, mean_val_energy_r2,
      std_train_load_r2, std_val_load_r2,
      train_val_load_gap, n_seeds_completed,
      <every searched hyperparameter>

    The train/val R^2 columns are pulled from the per-trial ``user_attrs``
    set by ``make_objective``.  This lets a single CSV answer both
    "what's the best val R^2_load" (= ``value``) and "how does train R^2
    compare to val R^2 for the recovered config" (= the metric columns).
    """
    import io
    all_param_keys: set = set()
    for t in study.trials:
        all_param_keys.update(t.params.keys())
    param_keys = sorted(all_param_keys)
    # Fixed-order metric columns so the CSV header is stable across runs.
    metric_cols = [
        "mean_train_load_r2",   "mean_val_load_r2",
        "mean_train_energy_r2", "mean_val_energy_r2",
        "std_train_load_r2",    "std_val_load_r2",
        "train_val_load_gap",   "n_seeds_completed",
    ]
    headers = (
        ["trial", "value", "state", "datetime_start", "datetime_complete",
         "duration_sec"]
        + metric_cols
        + param_keys
    )
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers)
    writer.writeheader()
    for t in study.trials:
        dur = ""
        if t.datetime_start and t.datetime_complete:
            dur = f"{(t.datetime_complete - t.datetime_start).total_seconds():.1f}"
        row = {
            "trial": t.number,
            "value": "" if t.value is None else f"{t.value:.6f}",
            "state": str(t.state).split(".")[-1],
            "datetime_start":
                t.datetime_start.isoformat(timespec="seconds") if t.datetime_start else "",
            "datetime_complete":
                t.datetime_complete.isoformat(timespec="seconds") if t.datetime_complete else "",
            "duration_sec": dur,
        }
        for k in metric_cols:
            v = t.user_attrs.get(k)
            if v is None:
                row[k] = ""
            elif isinstance(v, float):
                row[k] = f"{v:.6f}"
            else:
                row[k] = v
        for k in param_keys:
            row[k] = t.params.get(k, "")
        writer.writerow(row)
    return buf.getvalue()


def _run_state_payload(study: "optuna.Study", last_trial: "optuna.trial.FrozenTrial") -> Dict:
    """Compact JSON describing the current state of the search.  Useful for
    quick at-a-glance progress checks without parsing the SQLite DB."""
    n_completed = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    n_failed = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL
    )
    n_running = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.RUNNING
    )
    n_pruned = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    )
    last_value = (
        float(last_trial.value) if last_trial.value is not None else None
    )
    best = _safe_best_trial(study)
    return {
        "approach":             study.user_attrs.get("approach"),
        "n_trials_total":       len(study.trials),
        "n_completed":          n_completed,
        "n_failed":             n_failed,
        "n_running":            n_running,
        "n_pruned":             n_pruned,
        "best_value":           float(best.value) if best is not None and best.value is not None else None,
        "best_trial_number":    int(best.number) if best is not None else None,
        "last_trial_number":    int(last_trial.number),
        "last_trial_state":     str(last_trial.state).split(".")[-1],
        "last_trial_value":     last_value,
        "last_updated":         time.strftime("%Y-%m-%d %H:%M:%S %Z").strip(),
    }


def make_checkpoint_callback(
    approach: str, output_dir: str, logger: logging.Logger,
):
    """Return an ``study.optimize(callbacks=[...])`` callback that runs after
    every trial and atomically rewrites:

      hpo_best_params_<approach>.json   — best params so far (resumable)
      hpo_history_<approach>.csv        — every trial's params + R^2 + state
      hpo_run_state_<approach>.json     — compact progress snapshot

    A best-params snapshot is also written to
    ``hpo_best_params_<approach>.json.<trial>.snap`` whenever a NEW best is
    found, so a complete audit trail of how the best evolved is preserved.
    """
    best_path     = os.path.join(output_dir, f"hpo_best_params_{approach}.json")
    history_path  = os.path.join(output_dir, f"hpo_history_{approach}.csv")
    state_path    = os.path.join(output_dir, f"hpo_run_state_{approach}.json")
    snap_dir      = os.path.join(output_dir, f"hpo_best_snapshots_{approach}")
    os.makedirs(snap_dir, exist_ok=True)

    # Track the best-value-so-far across callback invocations so we can
    # detect "new best" without re-reading the file.
    state = {"best_value_seen": float("-inf")}

    def _callback(study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
        try:
            # 1) Best params JSON
            payload = _best_payload(study)
            if payload is not None:
                _atomic_write_text(best_path, json.dumps(payload, indent=2) + "\n")
                # If this trial set a new best, also drop a numbered snapshot
                # so we have an audit trail.
                best_value = payload.get("best_value")
                if best_value is not None and best_value > state["best_value_seen"]:
                    state["best_value_seen"] = best_value
                    snap_path = os.path.join(
                        snap_dir,
                        f"best_at_trial_{trial.number:04d}.json",
                    )
                    _atomic_write_text(snap_path, json.dumps(payload, indent=2) + "\n")
            # 2) Full trial history CSV
            _atomic_write_text(history_path, _history_csv_content(study))
            # 3) Compact run-state JSON
            _atomic_write_text(
                state_path,
                json.dumps(_run_state_payload(study, trial), indent=2) + "\n",
            )
            # 4) Concise log line so the user can monitor progress.
            #    Shows train R^2 alongside the val R^2 objective so the
            #    train-vs-val gap is visible at every trial completion.
            best_for_log = _safe_best_trial(study)
            best_str = (
                f"{best_for_log.value:.4f}"
                if best_for_log is not None and best_for_log.value is not None
                else "n/a"
            )
            t_train = trial.user_attrs.get("mean_train_load_r2")
            t_val   = trial.user_attrs.get("mean_val_load_r2")
            train_str = "n/a" if t_train is None else f"{t_train:.4f}"
            val_str   = (
                "n/a" if t_val is None else f"{t_val:.4f}"
            ) if trial.value is None else f"{trial.value:.4f}"
            logger.info(
                f"  [checkpoint] trial {trial.number} "
                f"state={str(trial.state).split('.')[-1]} "
                f"train_R2_load={train_str}  val_R2_load={val_str} "
                f"| best_val_R2_load_so_far={best_str}"
            )
        except Exception as exc:  # pragma: no cover
            # Never let a checkpointing failure abort the study; the SQLite
            # DB is the source of truth and will still hold the trial.
            logger.warning(
                f"  [checkpoint] write failed for trial {trial.number}: "
                f"{type(exc).__name__}: {exc}  (continuing; SQLite is intact)",
            )

    return _callback


# Backward-compatible export wrappers, used at the very end of ``main()``.
def _export_best(study, out_path, logger):
    payload = _best_payload(study)
    if payload is None:
        logger.warning("  No completed trials; skipping best-params export.")
        return
    _atomic_write_text(out_path, json.dumps(payload, indent=2) + "\n")
    logger.info(f"  Wrote: {out_path}")


def _export_history(study, out_path, logger):
    _atomic_write_text(out_path, _history_csv_content(study))
    logger.info(f"  Wrote: {out_path}")


# =============================================================================
# CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser(
        description="Optuna TPE HPO for the unseen-angle forward-design "
                    "surrogates (DDNS / Soft-PINN / Hard-PINN).",
    )
    p.add_argument("--approach", choices=["ddns", "soft", "hard"], required=True,
                   help="Which surrogate's hyperparameters to tune.")
    p.add_argument("--n_trials", type=int, default=100,
                   help="Total Optuna trials (including the startup random "
                        "trials).")
    p.add_argument("--n_startup_trials", type=int, default=15,
                   help="Random-search trials before TPE engages.")
    p.add_argument("--n_seeds", type=int, default=2,
                   help="Ensemble members trained per trial; the trial's "
                        "objective is the mean validation load R^2.")
    p.add_argument("--hpo_epochs", type=int, default=200,
                   help="Per-trial training epochs (smaller than production "
                        "to keep the search tractable).")
    p.add_argument("--data_dir",   default="./data")
    p.add_argument("--output_dir", default="./hpo_out")
    p.add_argument("--study_name", default=None,
                   help="Optuna study name.  Defaults to ``hpo_<approach>`` "
                        "so multiple approaches do not collide in one "
                        "--output_dir.")
    p.add_argument("--seed", type=int, default=2026,
                   help="Base seed; member k of a trial uses seed + 1000*k.")
    p.add_argument("--force_cpu", action="store_true")
    p.add_argument("--dry_run", action="store_true",
                   help="CI/smoke: shrink training budgets globally via "
                        "composite_design._dry_run_shrink_training_cfg.")
    p.add_argument("--no_warm_starts", action="store_true",
                   help="Disable the WARM_START priors and rely entirely on "
                        "the random-startup + TPE sampler.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, f"hpo_log_{args.approach}.txt")
    logger = _make_logger(args.approach, log_path)

    study_name = args.study_name or f"hpo_{args.approach}"
    db_path = os.path.join(args.output_dir, f"hpo_study_{args.approach}.db")
    storage_url = f"sqlite:///{os.path.abspath(db_path)}"

    cd.CFG.force_cpu = bool(args.force_cpu)
    cd.CFG.dry_run = bool(args.dry_run)
    cd.refresh_device()
    cd.set_publication_style()

    logger.info("=" * 80)
    logger.info(f"HPO SEARCH — approach={args.approach}")
    logger.info(f"  study_name = {study_name}")
    logger.info(f"  storage    = {storage_url}")
    logger.info(f"  n_trials   = {args.n_trials}")
    logger.info(f"  n_startup  = {args.n_startup_trials}")
    logger.info(f"  n_seeds    = {args.n_seeds}")
    logger.info(f"  hpo_epochs = {args.hpo_epochs}")
    logger.info(f"  seed       = {args.seed}")
    logger.info("=" * 80)

    df_all = cd.load_data(args.data_dir, logger)
    train_df, val_df = cd.split_unseen_angle(df_all, cd.CFG.theta_star, logger)
    scaler_disp, scaler_out, enc, params = cd.create_preprocessors(train_df, logger)

    sampler = TPESampler(
        n_startup_trials=int(args.n_startup_trials),
        multivariate=True,
        group=True,
        seed=int(args.seed),
    )
    # RDBStorage with a heartbeat: every running worker pings the SQLite
    # study every ``heartbeat_interval`` seconds.  If a worker dies (SLURM
    # preemption, OOM, hardware fault) and its trial's heartbeat is older
    # than ``grace_period``, Optuna marks that trial FAILED and the
    # ``RetryFailedTrialCallback`` re-enqueues it (up to ``max_retry``
    # times) so the search resumes seamlessly across worker restarts.
    #
    # heartbeat_interval=300s and grace_period=900s give Hard-PINN trials
    # (~3 h each at hpo_epochs=250) plenty of margin: a healthy worker
    # writes 36 heartbeats per trial; a stalled worker is detected in
    # under 15 minutes.
    storage = RDBStorage(
        url=storage_url,
        heartbeat_interval=300,
        grace_period=900,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=2),
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )
    study.set_user_attr("approach", args.approach)
    logger.info(f"  Resumed/created study with {len(study.trials)} prior trial(s).")

    if not args.no_warm_starts:
        _enqueue_warm_starts(study, args.approach, logger)

    objective = make_objective(
        args.approach,
        train_df, val_df, scaler_disp, scaler_out, enc, params,
        n_seeds=int(args.n_seeds),
        hpo_epochs=int(args.hpo_epochs),
        base_seed=int(args.seed),
        logger=logger,
    )

    n_done = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    )
    n_remaining = max(0, int(args.n_trials) - n_done)
    logger.info(f"  Completed so far: {n_done}; running {n_remaining} more.")

    # If trials already exist in the DB (e.g. resuming after preemption),
    # write an immediate checkpoint snapshot so the on-disk JSONs reflect
    # the current state even before any new trial completes in this run.
    if _safe_best_trial(study) is not None:
        try:
            payload = _best_payload(study)
            _atomic_write_text(
                os.path.join(args.output_dir, f"hpo_best_params_{args.approach}.json"),
                json.dumps(payload, indent=2) + "\n",
            )
            _atomic_write_text(
                os.path.join(args.output_dir, f"hpo_history_{args.approach}.csv"),
                _history_csv_content(study),
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(f"  initial checkpoint write failed: {exc}")

    checkpoint_cb = make_checkpoint_callback(args.approach, args.output_dir, logger)

    if n_remaining > 0:
        # ``catch`` keeps the worker alive if a single trial errors out —
        # the offending trial is marked FAILED in the DB and TPE continues
        # with the next.  Without this, a single OOM or NaN-grad explosion
        # would kill the whole worker, leaving 9 / 10 GPUs running and 1
        # idle.
        study.optimize(
            objective,
            n_trials=n_remaining,
            callbacks=[checkpoint_cb],
            catch=(RuntimeError, ValueError, KeyError, IndexError, MemoryError),
            show_progress_bar=False,
        )

    # Final exports (same content as the per-trial checkpoints; safe to
    # rewrite at the end as a sanity guarantee).
    best_path = os.path.join(args.output_dir, f"hpo_best_params_{args.approach}.json")
    hist_path = os.path.join(args.output_dir, f"hpo_history_{args.approach}.csv")
    _export_best(study, best_path, logger)
    _export_history(study, hist_path, logger)

    logger.info("=" * 80)
    logger.info("HPO SEARCH COMPLETE")
    best_final = _safe_best_trial(study)
    if best_final is not None:
        logger.info(f"  best_value = {best_final.value:.6f}")
        logger.info(f"  best_trial = {best_final.number}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
