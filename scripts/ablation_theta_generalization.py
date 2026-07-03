"""θ-generalization ablation: Hard-PINN architecture variants vs the data floor.

Motivation: F = ∂E/∂d constrains only the displacement direction — it cannot
improve generalization across the design variable θ.  This harness measures
what actually does.  For each architecture variant and each held-out angle it
trains a small Hard-PINN ensemble on the remaining angles and reports the
DESIGN-LEVEL error (EA@D_COMMON, IPF vs experiment at the held-out angle),
alongside the model-free floor: linear interpolation of the experimental
design curve over θ.  A variant is only useful where it approaches (or, via
better inductive bias, beats) that floor while staying physically plausible.

Variants (see composite_design.make_hard_energy_net):
  mlp                — published production architecture (baseline)
  monotone           — Tier 1: F ≥ 0 and E(0)=0 by construction
  separable          — Tier 2: low-order θ-dependence  E = Σ φ_k(θ,LC)·B_k(d)
  monotone_separable — Tier 1+2 combined

Usage (full ablation, HPC):
    python scripts/ablation_theta_generalization.py --data_dir ./data \
        --output_dir ./results_ablation --members 4

Smoke (CPU, minutes):
    python scripts/ablation_theta_generalization.py --data_dir ./data \
        --output_dir ./results_ablation_smoke --members 2 --epochs 40 \
        --batch_size 32 --angles 60 --force_cpu

Outputs:
    Table_ablation_theta_generalization.csv   per (variant, θ*, LC) detail
    Table_ablation_summary.csv                per-variant means vs the floor
    Fig_ablation_theta_generalization.png     design-level MAPE bar chart
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import composite_design as cd  # noqa: E402


def _make_logger(path: str) -> logging.Logger:
    log = logging.getLogger("ablation_theta")
    log.setLevel(logging.INFO)
    log.handlers = []
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


def _interp_floor(dm: pd.DataFrame, theta: float, lc: str) -> Dict[str, float]:
    """Model-free floor: linear interpolation of the experimental design curve."""
    sub = dm[dm["LC"].astype(str) == lc].sort_values("Angle")
    a = sub["Angle"].astype(float).values
    out = {}
    m = a != theta
    for col, key in [("EA_common", "EA"), ("IPF", "IPF")]:
        y = sub[col].astype(float).values
        y_true = float(y[~m][0])
        y_pred = float(np.interp(theta, a[m], y[m]))
        out[f"{key}_exp"] = y_true
        out[f"{key}_floor_err_pct"] = abs(y_pred - y_true) / max(abs(y_true), 1e-12) * 100
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--output_dir", default="./results_ablation")
    p.add_argument("--variants", default="mlp,monotone,separable,monotone_separable",
                   help="Comma-separated subset of: " + ",".join(cd.HARD_ARCHITECTURES))
    p.add_argument("--angles", default="45,50,55,60,65,70",
                   help="Comma-separated held-out angles (each trained on the rest).")
    p.add_argument("--members", type=int, default=4,
                   help="Ensemble members per (variant, angle) cell (default 4).")
    p.add_argument("--epochs", type=int, default=0,
                   help="Cap training epochs (0 = full tuned budget; warmup is "
                        "rescaled proportionally when capped).")
    p.add_argument("--batch_size", type=int, default=0,
                   help="Override batch size (0 = tuned config; larger batches "
                        "speed up CPU smoke runs).")
    p.add_argument("--n_basis", type=int, default=4,
                   help="Number of displacement basis functions for the separable variants.")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--force_cpu", action="store_true")
    args = p.parse_args()

    variants = [v.strip().lower() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in cd.HARD_ARCHITECTURES:
            raise SystemExit(f"Unknown variant {v!r}; choose from {cd.HARD_ARCHITECTURES}")
    angles = [float(a) for a in args.angles.split(",") if a.strip()]

    os.makedirs(args.output_dir, exist_ok=True)
    logger = _make_logger(os.path.join(args.output_dir, "ablation_log.txt"))

    cd.CFG.force_cpu = bool(args.force_cpu)
    cd.CFG.seed = cd.CFG.seed_base = cd.CFG.split_seed = int(args.seed)
    cd.refresh_device()
    cd.set_publication_style()

    # Config shim: apply epoch/batch overrides coherently (warmup must shrink
    # with the epoch cap or the cosine schedule never leaves warmup).
    _orig_get_cfg = cd.get_model_config

    def _patched_get_cfg(approach: str, protocol: str = "random", w_phys_override=None):
        cfg = _orig_get_cfg(approach, protocol, w_phys_override)
        if approach == "hard":
            cfg = dict(cfg)
            if args.epochs > 0:
                cfg["epochs"] = int(args.epochs)
                if "warmup_epochs" in cfg:
                    cfg["warmup_epochs"] = max(1, int(args.epochs) // 10)
            if args.batch_size > 0:
                cfg["batch_size"] = int(args.batch_size)
            cfg["n_basis"] = int(args.n_basis)
        return cfg

    cd.get_model_config = _patched_get_cfg

    df_all = cd.load_data(args.data_dir, logger)
    dm = cd.compute_design_space_metrics(df_all, logger)
    cd.enrich_df_metrics_ea_common(dm, df_all, logger=logger)

    logger.info("=" * 78)
    logger.info(f"θ-GENERALIZATION ABLATION — variants={variants}  angles={angles}")
    logger.info(f"  members={args.members}  epochs={'full' if args.epochs == 0 else args.epochs}"
                f"  batch={'cfg' if args.batch_size == 0 else args.batch_size}"
                f"  device={'cpu' if args.force_cpu else 'auto'}")
    logger.info("=" * 78)

    rows: List[Dict] = []
    t_start = time.time()
    for variant in variants:
        cd.CFG.hard_architecture = "" if variant == "mlp" else variant
        for theta in angles:
            t0 = time.time()
            cd.CFG.theta_star = float(theta)
            train_df, val_df = cd.split_unseen_angle(df_all, float(theta), logger)
            scaler_disp, scaler_out, enc, params = cd.create_preprocessors(train_df, logger)

            models = []
            for m_i in range(args.members):
                seed_m = int(args.seed) + m_i * 1000
                try:
                    model, _hist, _r2, _meta = cd.train_hard(
                        train_df, val_df, scaler_disp, scaler_out, enc, params,
                        seed_m, "unseen", logger)
                    models.append(model)
                except (RuntimeError, ValueError) as exc:
                    logger.warning(f"  member {m_i} failed ({variant}, θ*={theta:g}): {exc}")
            if not models:
                logger.warning(f"  SKIP {variant} θ*={theta:g}: no members trained")
                continue

            ens = cd.evaluate_ensemble(models, "hard", val_df, scaler_disp,
                                       scaler_out, enc, params)

            # Faithful decomposition readout for the separable variants:
            # E = Σ φ_k(θ,LC)·[B_k(d)−B_k(0)] plotted directly from the model.
            if variant in ("separable", "monotone_separable"):
                try:
                    cd.fig_separable_interpretability(
                        models, enc, params, args.output_dir, logger,
                        tag=f"{variant}_t{theta:g}")
                except Exception as exc:
                    logger.warning(f"  interpretability figure skipped: {exc}")

            for lc in sorted(dm["LC"].astype(str).unique()):
                m = cd.compute_ea_ipf_ensemble(models, "hard", float(theta), lc,
                                               scaler_disp, enc, params,
                                               d_eval=cd.D_COMMON)
                floor = _interp_floor(dm, float(theta), lc)
                ea_exp, ipf_exp = floor["EA_exp"], floor["IPF_exp"]
                pl = m.get("plausibility", {})
                rows.append({
                    "variant": variant,
                    "theta_star": theta,
                    "LC": lc,
                    "M": len(models),
                    "load_R2_pointwise": round(float(ens.get("load_r2", float("nan"))), 4),
                    "energy_R2_pointwise": round(float(ens.get("energy_r2", float("nan"))), 4),
                    "EA_exp_J": round(ea_exp, 3),
                    "EA_pred_J": round(float(m["EA"]), 3),
                    "EA_err_pct": round(abs(m["EA"] - ea_exp) / max(abs(ea_exp), 1e-12) * 100, 2),
                    "EA_floor_err_pct": round(floor["EA_floor_err_pct"], 2),
                    "IPF_exp_kN": round(ipf_exp, 4),
                    "IPF_pred_kN": round(float(m["IPF"]), 4),
                    "IPF_err_pct": round(abs(m["IPF"] - ipf_exp) / max(abs(ipf_exp), 1e-12) * 100, 2),
                    "IPF_floor_err_pct": round(floor["IPF_floor_err_pct"], 2),
                    "neg_force_frac": pl.get("neg_force_frac", float("nan")),
                    "nonmono_energy_frac": pl.get("nonmono_energy_frac", float("nan")),
                    "neg_ea_frac": pl.get("neg_ea_frac", float("nan")),
                })
            logger.info(f"  [{variant} θ*={theta:g}] done in {time.time() - t0:.0f}s "
                        f"(M={len(models)}, load R²={ens.get('load_r2', float('nan')):.3f})")

    cd.CFG.hard_architecture = ""
    cd.get_model_config = _orig_get_cfg

    if not rows:
        raise SystemExit("No results produced.")
    df = pd.DataFrame(rows)
    detail_path = os.path.join(args.output_dir, "Table_ablation_theta_generalization.csv")
    df.to_csv(detail_path, index=False)
    logger.info(f"Saved: {os.path.basename(detail_path)} ({len(df)} rows)")

    # Per-variant summary vs the floor.
    summ = df.groupby("variant").agg(
        EA_MAPE=("EA_err_pct", "mean"),
        IPF_MAPE=("IPF_err_pct", "mean"),
        EA_floor_MAPE=("EA_floor_err_pct", "mean"),
        IPF_floor_MAPE=("IPF_floor_err_pct", "mean"),
        load_R2_mean=("load_R2_pointwise", "mean"),
        neg_force_frac_mean=("neg_force_frac", "mean"),
        nonmono_energy_frac_mean=("nonmono_energy_frac", "mean"),
    ).round(3).reindex([v for v in variants if v in set(df["variant"])])
    summ_path = os.path.join(args.output_dir, "Table_ablation_summary.csv")
    summ.to_csv(summ_path)
    logger.info(f"Saved: {os.path.basename(summ_path)}")
    logger.info("\n" + summ.to_string())

    # Figure: design-level MAPE per variant, floor as a reference line.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    x = np.arange(len(summ))
    for ax, col, floor_col, title, unit in [
        (axes[0], "EA_MAPE", "EA_floor_MAPE", f"EA@{cd.EA_COMMON_MM_TAG} design-level error", "%"),
        (axes[1], "IPF_MAPE", "IPF_floor_MAPE", "IPF design-level error", "%"),
    ]:
        ax.bar(x, summ[col].values, 0.6, color="#0072B2", edgecolor="black", linewidth=0.8)
        floor_val = float(summ[floor_col].mean())
        ax.axhline(floor_val, color="#D55E00", linestyle="--", linewidth=1.6,
                   label=f"interpolation floor ({floor_val:.1f}%)")
        ax.set_xticks(x)
        ax.set_xticklabels([v.replace("_", "\n") for v in summ.index], fontsize=9)
        ax.set_ylabel(f"Mean abs. error at held-out θ ({unit})")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    fig.suptitle("Hard-PINN θ-generalization ablation (design level, held-out angles)")
    fig.savefig(os.path.join(args.output_dir, "Fig_ablation_theta_generalization.png"),
                dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved: Fig_ablation_theta_generalization.png")
    logger.info(f"TOTAL wall time: {(time.time() - t_start) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
