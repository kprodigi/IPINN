# -*- coding: utf-8 -*-
"""
================================================================================
TUNE_V20  —  Hyperparameter optimization for v_20's forward training
================================================================================
Optuna TPE search over physics-loss weights and key optimizer hyperparameters
for each of DDNS, Soft-PINN, and Hard-PINN on the unseen-θ=60° protocol.
Written for HPC submission: SQLite-backed study survives SLURM preemption,
single-GPU-per-worker, resumable.

Per-trial budget: M = ``--n_seeds`` (default 2) members trained for at most
``--hpo_epochs`` epochs each.  This matches v_19's HPO convention.

Objective: mean validation **load R²** on the unseen protocol — maximised.
Mirrors v_20's checkpoint metric (load R² only for every approach).

Search space rationale
----------------------
We tune the *active* loss weights (``w_phys`` for Soft, ``w_load``/``w_energy``
/``w_curvature`` for Hard, etc.) plus the optimization knobs that interact
with them (``lr``, ``weight_decay``, ``dropout``, ``softplus_beta``,
``smoothl1_beta``).  Architecture (``hidden_layers``) is held fixed at the
v_19 HPO-found values so this run is a re-tune around the architectural
``E(0)=0`` BC and the load-only checkpoint, not a rediscovery of arch.
``w_bc`` is excluded because v_20's architectural correction makes it dead.

Usage
-----
    # one approach per invocation (SLURM-friendly):
    python tune_v20.py --approach soft  --n_trials 80 --output_dir ./hpo_v20
    python tune_v20.py --approach hard  --n_trials 100 --output_dir ./hpo_v20
    python tune_v20.py --approach ddns  --n_trials 60 --output_dir ./hpo_v20

    # quick local sanity (5 trials, tiny budget, ~5 min):
    python tune_v20.py --approach soft --n_trials 5 --hpo_epochs 30 --n_seeds 1 \\
        --output_dir ./hpo_dry --dry_hpo

    # resume after preemption — same --output_dir and --study_name picks up
    # any existing trials from the SQLite DB:
    python tune_v20.py --approach soft --n_trials 80 --output_dir ./hpo_v20

    # multiple workers on different GPUs hitting the same study DB:
    CUDA_VISIBLE_DEVICES=0 python tune_v20.py --approach soft ... &
    CUDA_VISIBLE_DEVICES=1 python tune_v20.py --approach soft ... &

Outputs
-------
    <output_dir>/
        ├── tune_v20_study_<approach>.db          ← SQLite study (resumable, per-approach)
        ├── best_params_<approach>.json           ← copy-pasteable into v_20
        ├── trial_history_<approach>.csv          ← every trial's params + R²
        └── hpo_log_<approach>.txt                ← full Optuna log
================================================================================
"""

import argparse
import copy
import json
import logging
import os
import sys
import time
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Optuna is a required dependency — install with: pip install "optuna>=3.4,<5"
try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError as e:  # pragma: no cover
    sys.stderr.write("Optuna is required for HPO.  Install: pip install 'optuna>=3.4,<5'\n")
    raise

# Reuse v_20's training infrastructure verbatim.  Importing this module also
# applies its atomic-CSV monkey-patch and warning filters.
#
# tune_v20.py lives in ``hpo/``; composite_design.py lives at the repo
# root.  Insert the parent directory on ``sys.path`` before the import so
# this script works whether it's invoked as ``python hpo/tune_v20.py`` from
# the repo root or as ``python tune_v20.py`` from inside ``hpo/``.
_HPO_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HPO_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import composite_design as cd  # noqa: E402

# Silence Optuna's per-trial INFO chatter on stdout — we already log to file.
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module=r"matplotlib")

# Suppress the one-time ExperimentalWarning for TPESampler's
# ``multivariate=True`` / ``group=True`` arguments — both widely used and
# stable, but Optuna marks them as experimental.
#
# DO NOT suppress the UserWarning about non-primitive categorical choices.
# Optuna's SQLite layer round-trips str/int/float/bool/None losslessly,
# but tuples become lists on reload, causing TPE's distribution-compatibility
# check to raise ``CategoricalDistribution does not support dynamic value
# space.`` after ``n_startup_trials``.  Earlier we silenced this warning
# thinking it was cosmetic and the resulting crash burned ~10 GPU-hours.
# Leave it loud so future tuple regressions surface immediately.
warnings.filterwarnings(
    "ignore",
    category=optuna.exceptions.ExperimentalWarning,
    message=r"Argument ``(multivariate|group)`` is an experimental feature.*",
)


# =============================================================================
# SEARCH SPACES — per approach, ranges set 1–2 orders of magnitude around the
# v_19 HPO-found values so the TPE has room to move but doesn't wander far.
#
# Architecture is encoded as a STRING key (e.g. ``"128-64-32"``) rather than a
# tuple/list of ints.  Optuna's SQLite persistence layer round-trips str/int/
# float/bool/None losslessly; tuples are silently converted to lists on
# reload, which then trips ``CategoricalDistribution does not support dynamic
# value space.`` once TPE's ``multivariate=True, group=True`` engages after
# ``n_startup_trials`` (the persisted distribution's value type no longer
# matches the new trial's choice type).  Strings sidestep this entirely.
# =============================================================================
HL_DDNS_SOFT = {
    "64-32":      [64, 32],
    "128-64":     [128, 64],
    "128-64-32":  [128, 64, 32],
    "256-128":    [256, 128],
    "256-128-64": [256, 128, 64],
}
HL_HARD = {
    "32-32":     [32, 32],
    "64-32":     [64, 32],
    "64-64":     [64, 64],
    "64-64-64":  [64, 64, 64],   # added — best old Hard run (R2=0.840) used this
    "128-64":    [128, 64],
    "128-64-32": [128, 64, 32],
}


def suggest_ddns(trial: "optuna.Trial") -> Dict:
    # DDNS uses SoftPINNNet (same backbone as Soft-PINN) — same architecture
    # choices.  No physics weights to tune; just data weights, optimization
    # knobs, and architecture.
    hl_key = trial.suggest_categorical("hidden_layers", list(HL_DDNS_SOFT.keys()))
    return {
        "hidden_layers":   list(HL_DDNS_SOFT[hl_key]),
        "batch_size":      trial.suggest_categorical("batch_size", [32, 64, 128]),
        "lr":              trial.suggest_float("lr",            1e-6, 1e-3, log=True),
        "weight_decay":    trial.suggest_float("weight_decay",  1e-6, 1e-3, log=True),
        "dropout":         trial.suggest_float("dropout",       0.0,  0.10),
        "softplus_beta":   trial.suggest_float("softplus_beta", 5.0,  30.0),
        "smoothl1_beta":   trial.suggest_float("smoothl1_beta", 0.1,  3.0),
        "w_data_load":     trial.suggest_float("w_data_load",   0.5,  10.0, log=True),
        "w_data_energy":   trial.suggest_float("w_data_energy", 0.5,  10.0, log=True),
        "sched_patience":  trial.suggest_int  ("sched_patience", 20,  100),
        "sched_factor":    trial.suggest_float("sched_factor",  0.2,  0.8),
    }


def suggest_soft(trial: "optuna.Trial") -> Dict:
    # v_19's HPO winner [256, 128] is included so TPE can rediscover it if
    # it remains optimal under v_20's architectural BC.
    hl_key = trial.suggest_categorical("hidden_layers", list(HL_DDNS_SOFT.keys()))
    return {
        "hidden_layers":   list(HL_DDNS_SOFT[hl_key]),
        "batch_size":      trial.suggest_categorical("batch_size", [32, 64, 128]),
        "lr":              trial.suggest_float("lr",            1e-5, 5e-2, log=True),
        "weight_decay":    trial.suggest_float("weight_decay",  1e-6, 5e-3, log=True),
        "dropout":         trial.suggest_float("dropout",       0.0,  0.05),
        "softplus_beta":   trial.suggest_float("softplus_beta", 5.0,  25.0),
        "smoothl1_beta":   trial.suggest_float("smoothl1_beta", 0.1,  3.0),
        "w_data_load":     trial.suggest_float("w_data_load",   0.5,  10.0, log=True),
        "w_data_energy":   trial.suggest_float("w_data_energy", 0.1,  10.0, log=True),
        "w_phys":          trial.suggest_float("w_phys",        0.01, 10.0, log=True),
        "w_monotonicity":  trial.suggest_float("w_monotonicity", 0.1, 50.0, log=True),
        "w_angle_smooth":  trial.suggest_float("w_angle_smooth", 1e-3, 1.0, log=True),
        "smooth_delta_deg": trial.suggest_float("smooth_delta_deg", 1.0, 5.0),
        "colloc_ratio":    trial.suggest_float("colloc_ratio",  1.0,  6.0),
        "sched_patience":  trial.suggest_int  ("sched_patience", 20,  100),
        "sched_factor":    trial.suggest_float("sched_factor",  0.2,  0.8),
    }


def suggest_hard(trial: "optuna.Trial") -> Dict:
    # Hard-PINN architecture choices kept smaller — autograd through dE/dd is
    # VRAM-bounded so big nets are risky.  ``warmup_epochs`` and ``swa_pct``
    # are searched in narrow ranges around v_19's stability-tuned values
    # (80 and 0.20) so we don't leave the regime where Hard-PINN converges.
    #
    # ``batch_size`` excludes 8: an empirical Hard trial 0 with batch_size=8
    # took 5h 35m on a V100 (1102 minibatches/epoch × autograd-through-dE/dd
    # × 3 seeds × 200 epochs).  16 and 32 stay in the modern-ML regime and
    # cut per-trial wall by 2–4× without giving up meaningful generalization.
    hl_key = trial.suggest_categorical("hidden_layers", list(HL_HARD.keys()))
    return {
        "hidden_layers":   list(HL_HARD[hl_key]),
        "batch_size":      trial.suggest_categorical("batch_size", [16, 32]),
        "lr":              trial.suggest_float("lr",            1e-6, 5e-3, log=True),
        "weight_decay":    trial.suggest_float("weight_decay",  1e-5, 5e-2, log=True),
        "dropout":         trial.suggest_float("dropout",       0.0,  0.05),
        "softplus_beta":   trial.suggest_float("softplus_beta", 5.0,  25.0),
        "smoothl1_beta":   trial.suggest_float("smoothl1_beta", 0.05, 1.0),
        "w_load":          trial.suggest_float("w_load",        1.0,  20.0, log=True),
        "w_energy":        trial.suggest_float("w_energy",      1.0,  20.0, log=True),
        # Widened ranges below: an old Hard HPO best had w_monotonicity=0.094
        # (below the previous 0.5 lower bound) and w_curvature=0.010 (at the
        # previous 1e-2 upper bound).  Excluding those regions risks repeating
        # the search away from the known optimum.
        "w_monotonicity":  trial.suggest_float("w_monotonicity", 0.05, 30.0, log=True),
        "w_angle_smooth":  trial.suggest_float("w_angle_smooth", 1e-3, 0.1, log=True),
        "w_curvature":     trial.suggest_float("w_curvature",   1e-5, 0.1, log=True),
        "smooth_delta_deg": trial.suggest_float("smooth_delta_deg", 1.0, 5.0),
        "colloc_ratio":    trial.suggest_float("colloc_ratio",  1.0,  6.0),
        "grad_clip":       trial.suggest_float("grad_clip",     0.5,  3.0),
        "warmup_epochs":   trial.suggest_int  ("warmup_epochs", 40,   150),
        "swa_pct":         trial.suggest_float("swa_pct",       0.10, 0.35),
    }


# =============================================================================
# WARM-START SEEDS — best params from the prior (v_6 / v_16 / hpo_v3) HPO runs.
# Each approach gets one or more dicts that are enqueued as the FIRST trials
# of a fresh study so TPE has strong informed priors instead of rediscovering
# the good region from scratch.  Skipped on resume.
#
# Sources (from ``Old HPO Files/``):
#   - DDNS: tune_ddns.db best                       R2=0.7835
#   - Soft: tune_soft.db best                       R2=0.8012
#   - Hard: TWO seeds —
#       (a) v_16 production cfg ([32, 32], 800ep, M=20)            R2=0.849861
#           This is the highest documented Hard val-load R^2 — full-budget
#           production retrain of the v_16 hardcoded cfg.
#       (b) hpo_hardpinn_v3 trial 186 ([32, 32], M=2 seeds)
#           mean R2=0.8304 with std=0.0106 (lowest seed-to-seed std in the
#           v_3 study; useful as a stability prior for ensemble members).
#
# Param NAMES must match suggester keys exactly; categorical values must
# match choice strings (e.g. "32-32" not [32, 32]).  Any param missing from
# a warm-start dict is freely sampled by Optuna for that trial.
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
            "w_monotonicity":  4.96,
            "w_angle_smooth":  0.028,
            "smooth_delta_deg": 2.0,
            "colloc_ratio":    3.70,
            "sched_patience":  71,
            "sched_factor":    0.65,
        },
    ],
    "hard": [
        # (a) v_16 production cfg — highest documented Hard R^2 (0.849861).
        {
            "hidden_layers":    "32-32",
            "batch_size":       16,
            "lr":               4.0e-05,
            "weight_decay":     5.27e-04,
            "dropout":          0.0003,
            "softplus_beta":    13.82,
            "smoothl1_beta":    0.143,
            "w_load":           6.0,
            "w_energy":         7.0,
            "grad_clip":        1.63,
            "w_monotonicity":   5.0,
            "w_angle_smooth":   0.03,
            "w_curvature":      0.005,
            "smooth_delta_deg": 1.38,
            "colloc_ratio":     1.86,
            "warmup_epochs":    80,    # v_19 default — not specified in v_16 source
            "swa_pct":          0.20,  # v_19 default
            "sched_patience":   73,
            "sched_factor":     0.37,
        },
        # (b) hpo_hardpinn_v3 trial 186 — lowest seed-to-seed std (0.011) at
        # mean R2=0.8304 across 2 seeds.  Acts as a stability prior.
        {
            "hidden_layers":    "32-32",
            "batch_size":       32,
            "lr":               1.34e-05,
            "weight_decay":     0.00495,
            "dropout":          0.00172,
            "softplus_beta":    8.87,
            "smoothl1_beta":    0.229,
            "w_load":           4.58,
            "w_energy":         6.56,
            "grad_clip":        1.95,
            "w_monotonicity":   2.74,
            "w_angle_smooth":   0.0192,
            "w_curvature":      0.00155,
            "smooth_delta_deg": 2.50,
            "colloc_ratio":     3.90,
            "warmup_epochs":    80,
            "swa_pct":          0.20,
            "sched_patience":   60,
            "sched_factor":     0.50,
        },
    ],
}


SUGGESTERS = {"ddns": suggest_ddns, "soft": suggest_soft, "hard": suggest_hard}

TRAIN_FNS = {
    "ddns": cd.train_ddns,
    "soft": cd.train_soft,
    "hard": cd.train_hard,
}


# =============================================================================
# CONTEXT — load data once, share across trials
# =============================================================================
class HPOContext:
    """Caches the loaded dataset, train/val splits, and preprocessors so we
    don't pay the load cost on every trial."""

    def __init__(self, data_dir: str, logger: logging.Logger):
        logger.info("Loading data + building unseen-θ=60° split (one-time)...")
        self.df_all = cd.load_data(data_dir, logger)
        self.train_df_u, self.val_df_u = cd.split_unseen_angle(
            self.df_all, cd.CFG.theta_star, logger)
        (self.scaler_disp_u, self.scaler_out_u,
         self.enc_u, self.params_u) = cd.create_preprocessors(self.train_df_u, logger)
        logger.info(f"  N_train = {len(self.train_df_u)}  N_val = {len(self.val_df_u)}")


# =============================================================================
# OBJECTIVE — train M=n_seeds members with sampled cfg, return per-seed R²s
# =============================================================================
def evaluate_trial(approach: str, cfg: Dict, ctx: HPOContext,
                   n_seeds: int, base_seed: int,
                   logger: logging.Logger) -> Tuple[float, list]:
    """Train n_seeds members with the given cfg and return ``(mean R², list
    of per-seed R²s)``.  v_20 uses load R² as the checkpoint metric.

    Each member is trained independently; we don't apply the Tukey
    convergence filter here because the HPO budget is small and we want
    every trial's mean to reflect the cfg's *typical* behavior, not its
    best-of-N.

    Returning the per-seed list (not just the mean) lets the objective
    record std / min as trial user_attrs.  Low seed-to-seed std is a
    desirable property for the production retrain (M=20 ensembles), and
    a robust-mean objective (mean − α·std) is supported via CLI.
    """
    train_fn = TRAIN_FNS[approach]
    r2_list = []
    for s in range(n_seeds):
        seed = base_seed + s * 1000
        cd.set_seed(seed)
        try:
            _, _, best_r2, _ = train_fn(
                ctx.train_df_u, ctx.val_df_u,
                ctx.scaler_disp_u, ctx.scaler_out_u, ctx.enc_u, ctx.params_u,
                seed, "unseen", logger,
            )
        except Exception as e:
            logger.warning(f"  Trial seed={seed} failed: {type(e).__name__}: {e}")
            continue
        if not np.isfinite(best_r2):
            continue
        r2_list.append(float(best_r2))

    if not r2_list:
        # Optuna treats this as a failed trial; TPE skips it for guidance.
        raise optuna.exceptions.TrialPruned("No finite R² across all seeds")
    return float(np.mean(r2_list)), r2_list


def make_objective(approach: str, ctx: HPOContext, n_seeds: int,
                   base_seed: int, hpo_epochs: int,
                   logger: logging.Logger,
                   objective_mode: str = "mean",
                   robust_alpha: float = 1.0):
    """Closure-based objective; monkey-patches ``cd.get_model_config`` for
    the duration of each trial so v_20's training functions pick up the
    sampled cfg without modifying their signatures.

    Parameters
    ----------
    objective_mode : {"mean", "robust_mean"}
        ``"mean"`` (default): score = mean of per-seed R² — the same
        metric v_20's checkpointer uses.
        ``"robust_mean"``: score = mean − α·std.  Penalises seed-to-seed
        instability so trials whose ensemble members agree on R² are
        preferred over volatile ones.  Useful for picking a config that
        will retrain stably at production M=20.
    robust_alpha : float
        Weight on std in robust-mean mode.  α=1.0 means a config with
        mean=0.85, std=0.05 scores the same (0.80) as mean=0.80, std=0.0.
    """

    suggester = SUGGESTERS[approach]
    original_get_model_config = cd.get_model_config

    def patched_factory(sampled: Dict):
        """Return a get_model_config replacement that injects ``sampled`` for
        (approach, 'unseen') trials and falls back to the real cfg otherwise.

        Cap epochs at ``hpo_epochs`` for trial-speed.  Scale ``warmup_epochs``
        proportionally so the warmup-to-total ratio matches production (where
        ``epochs=800`` and ``warmup_epochs≈80``).  Without this, a sampled
        warmup of 150 inside a 200-epoch HPO trial would consume 75% of
        training — an artifact that would unfairly penalize that hyperparameter
        choice during HPO even though it works in production.
        """
        def patched(a, p, w_phys_override=None):
            base = original_get_model_config(a, p, w_phys_override)
            if a != approach or p != "unseen":
                return base
            cfg = {**base, **sampled}
            orig_epochs = int(cfg.get("epochs", hpo_epochs))
            cfg["epochs"] = min(orig_epochs, int(hpo_epochs))
            cfg["eval_every"] = max(1, cfg["epochs"] // 8)
            ratio = cfg["epochs"] / max(1, orig_epochs)
            if "warmup_epochs" in cfg:
                # Scale warmup proportionally + clamp to [2, epochs/2] for sanity.
                scaled = int(round(int(cfg["warmup_epochs"]) * ratio))
                cfg["warmup_epochs"] = max(2, min(scaled, cfg["epochs"] // 2))
            # respect dry_run if active
            if hasattr(cd, "_dry_run_shrink_training_cfg"):
                cd._dry_run_shrink_training_cfg(cfg)
            return cfg
        return patched

    def objective(trial: optuna.Trial) -> float:
        sampled = suggester(trial)
        cd.get_model_config = patched_factory(sampled)
        try:
            t0 = time.time()
            r2_mean, r2_list = evaluate_trial(
                approach, sampled, ctx, n_seeds, base_seed, logger)
            elapsed = time.time() - t0
            r2_arr = np.asarray(r2_list, dtype=float)
            r2_std = float(r2_arr.std(ddof=0)) if r2_arr.size > 1 else 0.0
            r2_min = float(r2_arr.min())
            # Always record ensemble-stability metrics for post-hoc analysis.
            trial.set_user_attr("seconds", elapsed)
            trial.set_user_attr("r2_per_seed", r2_list)
            trial.set_user_attr("r2_mean", r2_mean)
            trial.set_user_attr("r2_std", r2_std)
            trial.set_user_attr("r2_min", r2_min)
            # Score depends on the configured objective mode.
            if objective_mode == "robust_mean":
                score = r2_mean - robust_alpha * r2_std
                trial.set_user_attr("robust_score", score)
            else:
                score = r2_mean
            r2 = score
            seed_str = ", ".join(f"{x:.4f}" for x in r2_list)
            logger.info(
                f"  trial {trial.number:3d}  score={r2:.4f}  "
                f"mean={r2_mean:.4f} std={r2_std:.4f} min={r2_min:.4f}  "
                f"seeds=[{seed_str}]  ({elapsed:.1f}s)  {sampled}"
            )
            return r2
        finally:
            cd.get_model_config = original_get_model_config

    return objective


# =============================================================================
# RUN
# =============================================================================
def run_hpo(approach: str, output_dir: str, *,
            n_trials: int = 80, n_seeds: int = 2,
            hpo_epochs: int = 200, base_seed: int = 2026,
            study_name: Optional[str] = None,
            data_dir: str = ".",
            objective_mode: str = "mean",
            robust_alpha: float = 1.0) -> Dict:
    """Execute one Optuna study for ``approach``.  Returns the best params
    dict (already merged onto the v_19 base config so it's drop-in ready).

    ``objective_mode='robust_mean'`` switches the objective from mean R²
    to ``mean − α·std`` so trials with unstable seed-to-seed performance
    are penalised.  Useful when the production retrain uses a large
    ensemble (e.g. M=20) and ensemble stability matters as much as the
    point estimate.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"hpo_log_{approach}.txt")
    logger = _make_logger(approach, log_path)
    logger.info("=" * 80)
    logger.info(
        f"HPO V_20  approach={approach}  n_trials={n_trials}  "
        f"n_seeds={n_seeds}  epochs_cap={hpo_epochs}  "
        f"objective={objective_mode}"
        + (f"  alpha={robust_alpha}" if objective_mode == "robust_mean" else "")
    )
    logger.info("=" * 80)

    cd.refresh_device()
    cd.set_publication_style()
    ctx = HPOContext(data_dir, logger)

    if study_name is None:
        study_name = f"v20_{approach}_unseen"
    # Per-approach DB file so concurrent SLURM jobs don't race on
    # alembic schema init in a single shared SQLite file.  Each approach
    # owns its own DB; resume-after-preemption is unchanged.
    storage_path = os.path.join(output_dir, f"tune_v20_study_{approach}.db")
    storage = f"sqlite:///{storage_path}"

    # TPE configuration tuned for our 10–17 dim mixed search:
    #   - n_startup_trials=15 (default 10): more random trials before TPE
    #     surrogate kicks in, important when M=3 seed averaging adds noise.
    #   - n_ei_candidates=50 (default 24): more EI candidates per acquisition
    #     so TPE doesn't get stuck in a narrow region in higher-dim spaces.
    #   - multivariate=True + group=True: joint sampling over the categorical
    #     hidden_layers + batch_size pair (avoids independent-axis pathology).
    sampler = TPESampler(
        seed=base_seed,
        multivariate=True,
        group=True,
        n_startup_trials=15,
        n_ei_candidates=50,
    )
    study = optuna.create_study(
        study_name=study_name, storage=storage, sampler=sampler,
        direction="maximize", load_if_exists=True,
    )

    n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    logger.info(f"Existing completed trials: {n_done}")

    # Warm-start: enqueue prior-HPO best params as the first N trials of a
    # fresh study (one or more dicts per approach).  Skipped on resume
    # (study.trials non-empty) so we don't burn duplicate trials after
    # preemption.
    if not study.trials and approach in WARM_START:
        seeds = WARM_START[approach]
        for i, seed_params in enumerate(seeds):
            try:
                study.enqueue_trial(seed_params, skip_if_exists=True)
                logger.info(
                    f"Warm-start #{i} enqueued ({len(seed_params)} params, "
                    f"approach={approach}); will run as trial {i}."
                )
            except Exception as ex:  # pragma: no cover
                # If a warm-start key/value falls outside the current search
                # space, Optuna raises at trial-run time, not at enqueue.
                # Log but don't abort — TPE will still produce trial i from
                # scratch.
                logger.warning(f"Warm-start #{i} enqueue failed: {ex}")

    todo = max(0, n_trials - n_done)
    if todo == 0:
        logger.info("All requested trials already completed; nothing to do.")
    else:
        logger.info(f"Running {todo} new trials (target {n_trials} total)...")
        objective = make_objective(approach, ctx, n_seeds, base_seed,
                                   hpo_epochs, logger,
                                   objective_mode=objective_mode,
                                   robust_alpha=robust_alpha)
        study.optimize(objective, n_trials=todo, gc_after_trial=True,
                       show_progress_bar=False)

    # Best params + drop-in cfg for v_20 ----------------------------------
    if not study.trials:
        logger.warning("Study has no trials.")
        return {}
    best = study.best_trial
    logger.info("=" * 80)
    logger.info(f"Best trial: #{best.number}  load_R2 = {best.value:.4f}")
    for k, v in best.params.items():
        logger.info(f"  {k} = {v}")

    # Build a drop-in cfg by merging onto v_20's base config for that approach
    base_cfg = cd.get_model_config(approach, "unseen")
    drop_in = {**base_cfg, **best.params}
    drop_in.pop("w_phys_override", None)

    out_json = os.path.join(output_dir, f"best_params_{approach}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "approach": approach,
            "protocol": "unseen",
            "best_load_r2_mean": float(best.value),
            "n_seeds_per_trial": n_seeds,
            "hpo_epochs_cap": hpo_epochs,
            "n_trials_completed": len([t for t in study.trials
                                        if t.state == optuna.trial.TrialState.COMPLETE]),
            "best_params": best.params,
            "drop_in_cfg": _json_safe(drop_in),
        }, f, indent=2, default=str)
    logger.info(f"Wrote: {out_json}")

    # Also a CSV trial history for plotting / sanity checks.  Includes
    # ensemble-stability metrics (r2_mean, r2_std, r2_min) so post-hoc
    # analysis can pick a low-std config near the top, not just the
    # absolute best mean.  ``score`` is what TPE optimised (mean by
    # default; mean − α·std under --objective robust_mean); ``r2_mean``
    # is always the unpenalised mean for cross-mode comparison.
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        ua = t.user_attrs
        seeds = ua.get("r2_per_seed", [])
        rows.append({
            "trial":      t.number,
            "score":      t.value,
            "r2_mean":    ua.get("r2_mean", t.value),
            "r2_std":     ua.get("r2_std", float("nan")),
            "r2_min":     ua.get("r2_min", float("nan")),
            "r2_per_seed": ";".join(f"{x:.6f}" for x in seeds) if seeds else "",
            "seconds":    ua.get("seconds", float("nan")),
            **t.params,
        })
    if rows:
        csv_path = os.path.join(output_dir, f"trial_history_{approach}.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logger.info(f"Wrote: {csv_path}")

    logger.info("=" * 80)
    logger.info("HPO COMPLETE")
    logger.info("=" * 80)
    return drop_in


# =============================================================================
# Helpers
# =============================================================================
def _make_logger(approach: str, log_path: str) -> logging.Logger:
    log = logging.getLogger(f"hpo_v20.{approach}")
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
    """Recursively coerce numpy / pandas scalars to plain Python for json.dump."""
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


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="HPO for v_20 forward training (Optuna TPE on unseen-θ=60°).",
    )
    parser.add_argument("--approach", choices=["ddns", "soft", "hard", "all"],
                        required=True,
                        help="Which approach to tune.  'all' runs sequential "
                             "studies for ddns + soft + hard (long).")
    parser.add_argument("--data_dir",   type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="./hpo_v20")
    parser.add_argument("--n_trials",   type=int, default=80,
                        help="Target number of completed trials (resumed studies "
                             "subtract already-completed trials).")
    parser.add_argument("--n_seeds",    type=int, default=3,
                        help="Members trained per trial; mean load R² is the "
                             "objective.  Default 3 (v_19 HPO used 2; raised to 3 "
                             "in v_20 to reduce seed-noise that misled TPE on "
                             "marginal hyperparameter differences).")
    parser.add_argument("--hpo_epochs", type=int, default=200,
                        help="Cap on per-trial training epochs (vs ~600–800 for "
                             "the production run).  200 is a good balance.")
    parser.add_argument("--base_seed",  type=int, default=2026)
    parser.add_argument("--study_name", type=str, default=None,
                        help="Override the default 'v20_<approach>_unseen' name.")
    parser.add_argument("--dry_hpo",    action="store_true",
                        help="Tiny smoke (5 trials × M=1 × 30 epochs).  Use to "
                             "validate the script locally before SLURM.")
    parser.add_argument("--objective", choices=["mean", "robust_mean"],
                        default="mean",
                        help="'mean' (default) optimises mean load R² across "
                             "seeds — the v_20 checkpoint metric.  "
                             "'robust_mean' optimises mean − alpha*std so trials "
                             "with unstable seed-to-seed performance are "
                             "penalised.  Useful when the production retrain "
                             "uses a large ensemble (e.g. M=20) and stability "
                             "matters as much as the point estimate.")
    parser.add_argument("--robust_alpha", type=float, default=1.0,
                        help="Weight on std in --objective robust_mean.  "
                             "Default 1.0 means a (mean=0.85, std=0.05) trial "
                             "scores the same (0.80) as (mean=0.80, std=0.0).")
    args = parser.parse_args()

    if args.dry_hpo:
        args.n_trials   = min(args.n_trials, 5)
        args.n_seeds    = 1
        args.hpo_epochs = 30

    approaches = ["ddns", "soft", "hard"] if args.approach == "all" else [args.approach]
    for approach in approaches:
        run_hpo(approach=approach,
                output_dir=args.output_dir,
                n_trials=args.n_trials,
                n_seeds=args.n_seeds,
                hpo_epochs=args.hpo_epochs,
                base_seed=args.base_seed,
                study_name=args.study_name,
                data_dir=args.data_dir,
                objective_mode=args.objective,
                robust_alpha=args.robust_alpha)


if __name__ == "__main__":
    main()
