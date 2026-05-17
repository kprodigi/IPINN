"""Smoke test: train HardEnergyNet vs HardLoadNet on θ=60 single-angle held-out.

A short head-to-head comparison to validate the flipped Hard-PINN parameterization
before committing to a full HPO sweep.  Both models train with identical HPO-best
hyperparameters from get_model_config('hard', 'unseen'); only ``hard_param``
differs ("energy" vs "load").

Pass criterion: HardLoadNet's val R²_load ≥ HardEnergyNet's val R²_load − 0.05
on θ=60 within a small epoch budget (default 150).  Stronger pass: load-primary
beats energy-primary outright.

Usage
-----
    python scripts/smoke_hard_load_vs_energy.py \\
        --data_dir ./data --epochs 150 --seed 2026

The script:
  1. Loads LC1/LC2 data, builds the θ=60 LOAO fold (train = all other angles).
  2. Trains HardEnergyNet (legacy) for ``epochs`` epochs.
  3. Trains HardLoadNet (flipped) for ``epochs`` epochs with the same HPs.
  4. Prints per-fold val R²_load, val R²_energy, train R²_load for both,
     plus a one-line verdict.

Output is plain stdout, kept under 200 lines so the verdict is grep-friendly:
    VERDICT: HardLoadNet val_R2_load=+0.7XX vs HardEnergyNet +0.8XX  -> better/worse/comparable
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import composite_design as cd  # noqa: E402


def setup_logger() -> logging.Logger:
    # Force UTF-8 on stdout so Greek letters (theta, mu) and arrows render
    # on Windows consoles (cp1252 default) without breaking the run.
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    lg = logging.getLogger("smoke")
    lg.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    lg.addHandler(h)
    return lg


def load_data(data_dir: Path, logger: logging.Logger) -> pd.DataFrame:
    lc1 = pd.read_excel(data_dir / "LC1.xlsx")
    lc2 = pd.read_excel(data_dir / "LC2.xlsx")
    lc1["LC"] = "LC1"
    lc2["LC"] = "LC2"
    df = pd.concat([lc1, lc2], ignore_index=True)
    logger.info(f"  Loaded {len(df)} rows. Angles: {sorted(df['Angle'].unique())}")
    return df


def make_loao_split(df: pd.DataFrame, theta_star: float):
    train = df[df["Angle"] != theta_star].copy().reset_index(drop=True)
    val = df[df["Angle"] == theta_star].copy().reset_index(drop=True)
    return train, val


def run_one(approach_param: str, df: pd.DataFrame, theta_star: float,
            epochs: int, seed: int, logger: logging.Logger) -> dict:
    train_df, val_df = make_loao_split(df, theta_star)
    logger.info(f"  [{approach_param}] LOAO θ*={theta_star}: train={len(train_df)} val={len(val_df)}")

    scaler_disp, scaler_out, enc, params = cd.create_preprocessors(train_df, logger)

    # Override the unseen-protocol Hard cfg with smoke-test budget + the
    # parameterisation under test.  train_hard merges this into its own
    # protocol cfg via the ``cfg_override`` kwarg.
    cfg_override = {
        "epochs": epochs,
        "eval_every": max(5, epochs // 15),
        # warmup_epochs scales with epochs (10% rule, matching unseen default)
        "warmup_epochs": max(5, epochs // 10),
        "hard_param": approach_param,
    }

    # Drive train_hard.  protocol="unseen" so the stabilized schedule path is taken.
    cd.set_seed(seed)
    t0 = time.time()
    model, history, best_r2_mean, meta = cd.train_hard(
        train_df, val_df, scaler_disp, scaler_out, enc, params,
        seed=seed, protocol="unseen", logger=logger, cfg_override=cfg_override,
    )
    elapsed = time.time() - t0

    # Re-evaluate final state on train + val with the appropriate predict fn.
    Xtr = cd.to_tensor(cd.build_features(train_df, scaler_disp, enc))
    Xv = cd.to_tensor(cd.build_features(val_df, scaler_disp, enc))
    y_tr = train_df[["load_kN", "energy_J"]].values
    y_v = val_df[["load_kN", "energy_J"]].values

    if approach_param == "load":
        Ftr, Etr = cd.hard_load_predict_load_energy(model, Xtr, params)
        Fv, Ev = cd.hard_load_predict_load_energy(model, Xv, params)
    else:
        Ftr, Etr = cd.hard_pinn_predict_load_energy(model, Xtr, params)
        Fv, Ev = cd.hard_pinn_predict_load_energy(model, Xv, params)

    return {
        "approach_param": approach_param,
        "train_R2_load": float(cd.r2_safe(y_tr[:, 0], Ftr)),
        "train_R2_energy": float(cd.r2_safe(y_tr[:, 1], Etr)),
        "val_R2_load": float(cd.r2_safe(y_v[:, 0], Fv)),
        "val_R2_energy": float(cd.r2_safe(y_v[:, 1], Ev)),
        "best_r2_mean": float(best_r2_mean),
        "elapsed_sec": elapsed,
        "n_params": meta["n_params"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--theta_star", type=float, default=60.0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--skip_energy", action="store_true",
                    help="Skip the HardEnergyNet baseline (use prior published numbers instead)")
    args = ap.parse_args()

    logger = setup_logger()
    logger.info("=" * 78)
    logger.info(f"SMOKE: HardLoadNet vs HardEnergyNet  θ*={args.theta_star}  epochs={args.epochs}")
    logger.info("=" * 78)

    df = load_data(Path(args.data_dir), logger)

    results = {}
    if not args.skip_energy:
        logger.info("")
        logger.info("[1/2] Training HardEnergyNet (legacy, energy-primary)...")
        results["energy"] = run_one("energy", df, args.theta_star, args.epochs, args.seed, logger)

    logger.info("")
    logger.info("[2/2] Training HardLoadNet (flipped, load-primary)...")
    results["load"] = run_one("load", df, args.theta_star, args.epochs, args.seed, logger)

    # Summary
    logger.info("")
    logger.info("=" * 78)
    logger.info("RESULTS")
    logger.info("=" * 78)
    logger.info(f"  {'param':>8s}  {'train_R2_F':>10s}  {'val_R2_F':>10s}  "
                f"{'train_R2_E':>10s}  {'val_R2_E':>10s}  {'time_s':>7s}  {'n_par':>6s}")
    for ap_name, r in results.items():
        logger.info(
            f"  {ap_name:>8s}  {r['train_R2_load']:+10.4f}  {r['val_R2_load']:+10.4f}  "
            f"{r['train_R2_energy']:+10.4f}  {r['val_R2_energy']:+10.4f}  "
            f"{r['elapsed_sec']:7.1f}  {r['n_params']:6d}"
        )

    # One-line verdict for log-grep
    load_v = results["load"]["val_R2_load"]
    if "energy" in results:
        energy_v = results["energy"]["val_R2_load"]
        delta = load_v - energy_v
        if delta > 0.02:
            tag = "BETTER"
        elif delta > -0.05:
            tag = "COMPARABLE"
        else:
            tag = "WORSE"
        logger.info("")
        logger.info(
            f"VERDICT: HardLoadNet val_R2_load={load_v:+.4f} vs "
            f"HardEnergyNet {energy_v:+.4f}  delta={delta:+.4f}  -> {tag}"
        )
        return 0 if tag != "WORSE" else 1
    else:
        logger.info("")
        logger.info(f"VERDICT: HardLoadNet val_R2_load={load_v:+.4f}  (no energy baseline this run)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
