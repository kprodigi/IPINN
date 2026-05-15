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
    hpo_study_<approach>.db          SQLite study (resumable, per-approach)
    hpo_best_params_<approach>.json  Best params, copy-pasteable into the
                                     ``get_model_config`` hard-coded cfg
    hpo_history_<approach>.csv       Every trial's params + R^2 + status
    hpo_log_<approach>.txt           Full Optuna log
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
from typing import Dict, List, Optional

import numpy as np

try:
    import optuna
    from optuna.samplers import TPESampler
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
) -> float:
    """Train one ensemble member with the trial cfg and return val load R^2."""
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
        metrics = cd.evaluate_model(
            model, approach, df_val, scaler_disp, scaler_out, enc, params,
        )
        return float(metrics["load_r2"])
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

        member_r2s: List[float] = []
        t_trial = time.time()
        for k in range(n_seeds):
            seed_k = base_seed + 1000 * k
            r2 = _train_and_score_member(
                approach, cfg, seed_k,
                df_tr, df_val, scaler_disp, scaler_out, enc, params,
                logger,
            )
            member_r2s.append(r2)
            logger.info(
                f"  trial {trial.number:03d}  seed {k+1}/{n_seeds}  "
                f"val_R2_load={r2:.4f}"
            )
            trial.report(float(np.mean(member_r2s)), step=k)

        mean_r2 = float(np.mean(member_r2s))
        elapsed = time.time() - t_trial
        logger.info(
            f"trial {trial.number:03d} DONE  mean_val_R2_load={mean_r2:.4f}  "
            f"(n_seeds={n_seeds}, wall={elapsed:.0f}s)"
        )
        return mean_r2

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


def _export_best(
    study: "optuna.Study", out_path: str, logger: logging.Logger,
) -> None:
    """Write the best trial's params to a JSON file in a copy-pasteable form."""
    if not study.best_trial:
        logger.warning("  No completed trials yet; skipping best-params export.")
        return
    best = study.best_trial
    out: Dict = {
        "approach":      study.user_attrs.get("approach"),
        "best_value":    float(best.value) if best.value is not None else None,
        "best_trial":    int(best.number),
        "best_params":   {k: v for k, v in best.params.items()},
        "n_trials":      len(study.trials),
        "n_completed":   sum(
            1 for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ),
    }
    if "hidden_layers" in best.params:
        hl_key = best.params["hidden_layers"]
        hl_map = HL_BY_APPROACH[study.user_attrs.get("approach", "ddns")]
        out["best_params_resolved"] = dict(best.params)
        out["best_params_resolved"]["hidden_layers"] = list(hl_map[hl_key])
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    logger.info(f"  Wrote: {out_path}")


def _export_history(
    study: "optuna.Study", out_path: str, logger: logging.Logger,
) -> None:
    """Dump every trial's (number, value, state, params) to CSV."""
    all_param_keys: set = set()
    for t in study.trials:
        all_param_keys.update(t.params.keys())
    param_keys = sorted(all_param_keys)
    headers = ["trial", "value", "state"] + param_keys
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for t in study.trials:
            row = {
                "trial": t.number,
                "value": "" if t.value is None else f"{t.value:.6f}",
                "state": str(t.state).split(".")[-1],
            }
            for k in param_keys:
                row[k] = t.params.get(k, "")
            writer.writerow(row)
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
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
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

    if n_remaining > 0:
        study.optimize(objective, n_trials=n_remaining, show_progress_bar=False)

    best_path = os.path.join(args.output_dir, f"hpo_best_params_{args.approach}.json")
    hist_path = os.path.join(args.output_dir, f"hpo_history_{args.approach}.csv")
    _export_best(study, best_path, logger)
    _export_history(study, hist_path, logger)

    logger.info("=" * 80)
    logger.info("HPO SEARCH COMPLETE")
    if study.best_trial:
        logger.info(f"  best_value = {study.best_value:.6f}")
        logger.info(f"  best_trial = {study.best_trial.number}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
