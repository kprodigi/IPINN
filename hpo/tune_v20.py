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
# tune_v20.py lives in ``hpo/``; composite_design_v20.py lives at the repo
# root.  Insert the parent directory on ``sys.path`` before the import so
# this script works whether it's invoked as ``python hpo/tune_v20.py`` from
# the repo root or as ``python tune_v20.py`` from inside ``hpo/``.
_HPO_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HPO_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import composite_design_v20 as cd  # noqa: E402

# Silence Optuna's per-trial INFO chatter on stdout — we already log to file.
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module=r"matplotlib")

# Optuna 4.x emits two classes of cosmetic warnings during HPO:
#   1. ExperimentalWarning for the TPESampler's ``multivariate=True`` and
#      ``group=True`` arguments — both widely used and stable; the warning
#      fires once at study creation.
#   2. UserWarning per-trial about categorical choices being a tuple or
#      list of ints (rather than primitive types).  Affects display in
#      Optuna's web UI only; the SQLite study DB persistence and
#      resume-after-preemption are unaffected.
# Both are noise in the SLURM run log; silence them narrowly so genuine
# warnings (training divergence, NaN losses, etc.) still surface.
warnings.filterwarnings(
    "ignore",
    category=optuna.exceptions.ExperimentalWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"Choices for a categorical distribution should be a tuple.*",
)


# =============================================================================
# SEARCH SPACES — per approach, ranges set 1–2 orders of magnitude around the
# v_19 HPO-found values so the TPE has room to move but doesn't wander far.
# =============================================================================
def suggest_ddns(trial: "optuna.Trial") -> Dict:
    # DDNS uses SoftPINNNet (same backbone as Soft-PINN) — same architecture
    # choices.  No physics weights to tune; just data weights, optimization
    # knobs, and architecture.
    hl_choices = [(64, 32), (128, 64), (128, 64, 32), (256, 128), (256, 128, 64)]
    return {
        "hidden_layers":   list(trial.suggest_categorical("hidden_layers", hl_choices)),
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
    # ``hidden_layers`` and ``batch_size`` are categorical (hashable tuples for
    # the SQLite study DB).  v_19's HPO winner [256, 128] is included so the
    # TPE can rediscover it if it remains optimal under v_20's architectural BC.
    hl_choices = [(64, 32), (128, 64), (128, 64, 32), (256, 128), (256, 128, 64)]
    return {
        "hidden_layers":   list(trial.suggest_categorical("hidden_layers", hl_choices)),
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
    hl_choices = [(32, 32), (64, 32), (64, 64), (128, 64), (128, 64, 32)]
    return {
        "hidden_layers":   list(trial.suggest_categorical("hidden_layers", hl_choices)),
        "batch_size":      trial.suggest_categorical("batch_size", [8, 16]),
        "lr":              trial.suggest_float("lr",            1e-6, 5e-3, log=True),
        "weight_decay":    trial.suggest_float("weight_decay",  1e-5, 5e-2, log=True),
        "dropout":         trial.suggest_float("dropout",       0.0,  0.05),
        "softplus_beta":   trial.suggest_float("softplus_beta", 5.0,  25.0),
        "smoothl1_beta":   trial.suggest_float("smoothl1_beta", 0.05, 1.0),
        "w_load":          trial.suggest_float("w_load",        1.0,  20.0, log=True),
        "w_energy":        trial.suggest_float("w_energy",      1.0,  20.0, log=True),
        "w_monotonicity":  trial.suggest_float("w_monotonicity", 0.5, 30.0, log=True),
        "w_angle_smooth":  trial.suggest_float("w_angle_smooth", 1e-3, 0.1, log=True),
        "w_curvature":     trial.suggest_float("w_curvature",   1e-5, 1e-2, log=True),
        "smooth_delta_deg": trial.suggest_float("smooth_delta_deg", 1.0, 5.0),
        "colloc_ratio":    trial.suggest_float("colloc_ratio",  1.0,  6.0),
        "grad_clip":       trial.suggest_float("grad_clip",     0.5,  2.0),
        "warmup_epochs":   trial.suggest_int  ("warmup_epochs", 40,   150),
        "swa_pct":         trial.suggest_float("swa_pct",       0.10, 0.35),
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
# OBJECTIVE — train M=n_seeds members with sampled cfg, return mean load R²
# =============================================================================
def evaluate_trial(approach: str, cfg: Dict, ctx: HPOContext,
                   n_seeds: int, base_seed: int, logger: logging.Logger) -> float:
    """Train n_seeds members with the given cfg and return mean validation
    load R² (the metric v_20's checkpoint score uses).

    Each member is trained independently; we don't apply the Tukey
    convergence filter here because the HPO budget is small and we want
    every trial's mean to reflect the cfg's *typical* behavior, not its
    best-of-N.
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
    return float(np.mean(r2_list))


def make_objective(approach: str, ctx: HPOContext, n_seeds: int,
                   base_seed: int, hpo_epochs: int,
                   logger: logging.Logger):
    """Closure-based objective; monkey-patches ``cd.get_model_config`` for
    the duration of each trial so v_20's training functions pick up the
    sampled cfg without modifying their signatures."""

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
            r2 = evaluate_trial(approach, sampled, ctx, n_seeds, base_seed, logger)
            elapsed = time.time() - t0
            logger.info(f"  trial {trial.number:3d}  load_R2={r2:.4f}  "
                        f"({elapsed:.1f}s)  {sampled}")
            # Optuna stores user attrs alongside the trial — useful for later analysis
            trial.set_user_attr("seconds", elapsed)
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
            data_dir: str = ".") -> Dict:
    """Execute one Optuna study for ``approach``.  Returns the best params
    dict (already merged onto the v_19 base config so it's drop-in ready)."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"hpo_log_{approach}.txt")
    logger = _make_logger(approach, log_path)
    logger.info("=" * 80)
    logger.info(f"HPO V_20  approach={approach}  n_trials={n_trials}  "
                f"n_seeds={n_seeds}  epochs_cap={hpo_epochs}")
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
    todo = max(0, n_trials - n_done)
    if todo == 0:
        logger.info("All requested trials already completed; nothing to do.")
    else:
        logger.info(f"Running {todo} new trials (target {n_trials} total)...")
        objective = make_objective(approach, ctx, n_seeds, base_seed,
                                   hpo_epochs, logger)
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

    # Also a CSV trial history for plotting / sanity checks
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        rows.append({
            "trial":      t.number,
            "load_r2":    t.value,
            "seconds":    t.user_attrs.get("seconds", float("nan")),
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
                data_dir=args.data_dir)


if __name__ == "__main__":
    main()
