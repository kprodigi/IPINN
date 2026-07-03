"""Held-out-design generalization harness: LOAO + LOCO, skill-scored, calibrated.

Motivation: F = dE/dd constrains only the displacement direction — it cannot
improve generalization across the design variable θ.  This harness measures
what actually does, and reports it the way reviewers can trust:

PROTOCOLS
  loao  hold out BOTH loading cases of one angle (design-generalization test;
        boundary folds 45/70 are one-sided extrapolation and are stratified)
  loco  hold out ONE (θ, LC) curve, keep the other LC at that angle
        (cross-loading transfer at fixed geometry; 12 folds)

METRICS (per held-out curve / design)
  design-level : EA@D_COMMON and IPF percent error vs experiment — the
                 quantities forward/inverse design actually operates on
  curve-level  : NRMSE (range-normalized), raw R², and a LEVEL/SHAPE
                 decomposition (bias, R²_shape after removing the scalar
                 offset) — raw R² against a single curve's own mean is driven
                 negative by pure level shifts, exactly what single-specimen
                 scatter produces
  skill        : S = 1 − err_model/err_floor vs TWO floors — linear
                 interpolation of the experimental design metrics over θ, and
                 a 2-parameter mechanics-trend fit (best candidate H(θ) form
                 selected on training folds only)
  plausibility : fraction of ensemble members violating F ≥ 0 / monotone E /
                 EA > 0 at the held-out design
  calibration  : jackknife+ prediction intervals over folds with EMPIRICAL
                 coverage (each fold's interval never uses its own residual)

Variants (see composite_design.make_hard_energy_net):
  mlp                — published production architecture (baseline)
  monotone           — Tier 1: F ≥ 0 and E(0)=0 by construction
  separable          — Tier 2: low-order θ-dependence  E = Σ φ_k(θ,LC)·B_k(d)
  monotone_separable — Tier 1+2 combined

Full run (HPC):
    python scripts/ablation_theta_generalization.py --data_dir ./data \
        --output_dir ./results_ablation --members 4
    python scripts/ablation_theta_generalization.py --data_dir ./data \
        --output_dir ./results_ablation_loco --protocol loco --members 4

Smoke (CPU, minutes):
    ... --members 2 --epochs 60 --batch_size 32 --angles 60 --force_cpu

Outputs:
    Table_ablation_theta_generalization.csv   per-fold detail (all metrics)
    Table_ablation_summary.csv                per-variant aggregate + strata
    Table_ablation_jackknife_coverage.csv     calibrated-interval coverage
    Fig_ablation_theta_generalization.png     design-level error vs floors
    Fig_ablation_skill.png                    per-fold skill scores
    Fig_separable_interpretability_*.png      decomposition (separable runs)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

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

BOUNDARY_ANGLES = (45.0, 70.0)


def _make_logger(path: str) -> logging.Logger:
    log = logging.getLogger("ablation_theta")
    log.setLevel(logging.INFO)
    log.handlers = []
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def _interp_floor(dm: pd.DataFrame, theta: float, lc: str) -> Dict[str, float]:
    """Model-free floor: linear interpolation of the experimental design curve."""
    sub = dm[dm["LC"].astype(str) == lc].sort_values("Angle")
    a = sub["Angle"].astype(float).values
    out: Dict[str, float] = {}
    m = a != theta
    for col, key in [("EA_common", "EA"), ("IPF", "IPF")]:
        y = sub[col].astype(float).values
        y_true = float(y[~m][0])
        y_pred = float(np.interp(theta, a[m], y[m]))
        out[f"{key}_exp"] = y_true
        out[f"{key}_floor_err_pct"] = abs(y_pred - y_true) / max(abs(y_true), 1e-12) * 100
    return out


def _mechanics_floor(dm: pd.DataFrame, theta: float, lc: str) -> Dict[str, float]:
    """2-parameter mechanics-trend floor (candidate H(θ) fit on training folds)."""
    sub = dm[dm["LC"].astype(str) == lc].sort_values("Angle")
    a = sub["Angle"].astype(float).values
    m = a != theta
    out: Dict[str, float] = {}
    for col, key in [("EA_common", "EA"), ("IPF", "IPF")]:
        y = sub[col].astype(float).values
        y_true = float(y[~m][0])
        pred, fname = cd.mechanics_trend_baseline(a[m], y[m], theta)
        out[f"{key}_mech_err_pct"] = abs(pred - y_true) / max(abs(y_true), 1e-12) * 100
        out[f"{key}_mech_form"] = fname
    return out


def _evaluate_cell(models, theta: float, lc: str, df_all: pd.DataFrame,
                   dm: pd.DataFrame, scaler_disp, enc, params) -> Dict:
    """All metrics for one held-out (θ, LC) design cell."""
    row: Dict = {}
    # Design level
    m = cd.compute_ea_ipf_ensemble(models, "hard", float(theta), lc,
                                   scaler_disp, enc, params, d_eval=cd.D_COMMON)
    floor = _interp_floor(dm, float(theta), lc)
    mech = _mechanics_floor(dm, float(theta), lc)
    ea_exp, ipf_exp = floor["EA_exp"], floor["IPF_exp"]
    ea_err = abs(m["EA"] - ea_exp) / max(abs(ea_exp), 1e-12) * 100
    ipf_err = abs(m["IPF"] - ipf_exp) / max(abs(ipf_exp), 1e-12) * 100
    row.update({
        "EA_exp_J": round(ea_exp, 3), "EA_pred_J": round(float(m["EA"]), 3),
        "EA_err_pct": round(ea_err, 2),
        "EA_floor_err_pct": round(floor["EA_floor_err_pct"], 2),
        "EA_mech_err_pct": round(mech["EA_mech_err_pct"], 2),
        "EA_mech_form": mech["EA_mech_form"],
        "EA_skill_vs_floor": round(cd.skill_score(ea_err, floor["EA_floor_err_pct"]), 3),
        "IPF_exp_kN": round(ipf_exp, 4), "IPF_pred_kN": round(float(m["IPF"]), 4),
        "IPF_err_pct": round(ipf_err, 2),
        "IPF_floor_err_pct": round(floor["IPF_floor_err_pct"], 2),
        "IPF_mech_err_pct": round(mech["IPF_mech_err_pct"], 2),
        "IPF_mech_form": mech["IPF_mech_form"],
        "IPF_skill_vs_floor": round(cd.skill_score(ipf_err, floor["IPF_floor_err_pct"]), 3),
    })
    pl = m.get("plausibility", {})
    row.update({
        "neg_force_frac": pl.get("neg_force_frac", float("nan")),
        "nonmono_energy_frac": pl.get("nonmono_energy_frac", float("nan")),
        "neg_ea_frac": pl.get("neg_ea_frac", float("nan")),
    })
    # Curve level (level/shape decomposition) against the held-out curve
    g = df_all[(df_all["LC"].astype(str) == lc)
               & (df_all["Angle"].astype(float) == float(theta))].sort_values("disp_mm")
    if len(g) >= 10:
        disps = g["disp_mm"].values.astype(float)
        Fm, _Fs, Em, _Es = cd.predict_curve_ensemble(models, "hard", float(theta), lc,
                                                     disps, scaler_disp, enc, params)
        cm = cd.curve_error_metrics(g["load_kN"].values.astype(float), Fm)
        row.update({
            "load_R2_raw": round(cm["R2_raw"], 4),
            "load_R2_shape": round(cm["R2_shape"], 4),
            "load_NRMSE": round(cm["NRMSE_range"], 4),
            "load_bias_kN": round(cm["bias"], 4),
            "load_pearson_r": round(cm["pearson_r"], 4),
        })
        cme = cd.curve_error_metrics(g["energy_J"].values.astype(float), Em)
        row.update({"energy_R2_raw": round(cme["R2_raw"], 4),
                    "energy_NRMSE": round(cme["NRMSE_range"], 4)})
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--output_dir", default="./results_ablation")
    p.add_argument("--protocol", choices=["loao", "loco"], default="loao",
                   help="loao: hold out both LCs of an angle; loco: hold out one (θ,LC) curve.")
    p.add_argument("--variants", default="mlp,monotone,separable,monotone_separable",
                   help="Comma-separated subset of: " + ",".join(cd.HARD_ARCHITECTURES))
    p.add_argument("--angles", default="45,50,55,60,65,70",
                   help="Comma-separated held-out angles.")
    p.add_argument("--members", type=int, default=4,
                   help="Ensemble members per fold cell (default 4).")
    p.add_argument("--epochs", type=int, default=0,
                   help="Cap training epochs (0 = full tuned budget; warmup rescaled).")
    p.add_argument("--batch_size", type=int, default=0,
                   help="Override batch size (0 = tuned config).")
    p.add_argument("--n_basis", type=int, default=4,
                   help="Displacement basis functions for the separable variants.")
    p.add_argument("--theta_features", choices=["default", "mechanics"], default="default",
                   help="θ embedding: raw Fourier (default) or candidate kinematic coordinates.")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Miscoverage level for jackknife+ intervals (default 0.1).")
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
    cd.CFG.theta_feature_map = "mechanics" if args.theta_features == "mechanics" else ""
    cd.refresh_device()
    cd.set_publication_style()

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
    lcs = sorted(df_all["LC"].astype(str).unique())

    # Fold definitions
    if args.protocol == "loao":
        folds: List[Tuple[float, Optional[str]]] = [(th, None) for th in angles]
    else:
        folds = [(th, lc) for th in angles for lc in lcs]

    logger.info("=" * 78)
    logger.info(f"HELD-OUT-DESIGN ABLATION — protocol={args.protocol}  variants={variants}")
    logger.info(f"  folds={[(t, l or 'both') for t, l in folds]}")
    logger.info(f"  members={args.members}  epochs={'full' if args.epochs == 0 else args.epochs}"
                f"  batch={'cfg' if args.batch_size == 0 else args.batch_size}"
                f"  θ-features={args.theta_features}  device={'cpu' if args.force_cpu else 'auto'}")
    logger.info("=" * 78)

    rows: List[Dict] = []
    t_start = time.time()
    for variant in variants:
        cd.CFG.hard_architecture = "" if variant == "mlp" else variant
        for theta, held_lc in folds:
            t0 = time.time()
            cd.CFG.theta_star = float(theta)
            if held_lc is None:
                train_df, val_df = cd.split_unseen_angle(df_all, float(theta), logger)
            else:
                mask = ((df_all["Angle"].astype(float) == float(theta))
                        & (df_all["LC"].astype(str) == held_lc))
                train_df = df_all.loc[~mask].reset_index(drop=True)
                val_df = df_all.loc[mask].reset_index(drop=True)
                logger.info(f"LOCO split ({held_lc}@{theta:g}°): train={len(train_df)}, "
                            f"val={len(val_df)}")
            scaler_disp, scaler_out, enc, params = cd.create_preprocessors(train_df, logger)

            models = []
            for m_i in range(args.members):
                seed_m = int(args.seed) + m_i * 1000
                try:
                    model, _h, _r, _me = cd.train_hard(train_df, val_df, scaler_disp,
                                                       scaler_out, enc, params,
                                                       seed_m, "unseen", logger)
                    models.append(model)
                except (RuntimeError, ValueError) as exc:
                    logger.warning(f"  member {m_i} failed ({variant}, θ*={theta:g}, "
                                   f"lc={held_lc or 'both'}): {exc}")
            if not models:
                logger.warning(f"  SKIP {variant} θ*={theta:g} lc={held_lc or 'both'}: "
                               f"no members trained")
                continue

            if variant in ("separable", "monotone_separable"):
                try:
                    cd.fig_separable_interpretability(
                        models, enc, params, args.output_dir, logger,
                        tag=f"{variant}_t{theta:g}" + (f"_{held_lc}" if held_lc else ""))
                except Exception as exc:
                    logger.warning(f"  interpretability figure skipped: {exc}")

            held_cells = [(theta, lc) for lc in lcs] if held_lc is None else [(theta, held_lc)]
            for th_c, lc_c in held_cells:
                base = {
                    "variant": variant,
                    "protocol": args.protocol,
                    "theta_star": th_c,
                    "LC": lc_c,
                    "fold_type": ("boundary" if float(th_c) in BOUNDARY_ANGLES else "interior"),
                    "fold": (f"{lc_c}@{th_c:g}" if args.protocol == "loco" else f"θ*={th_c:g}"),
                    "M": len(models),
                }
                base.update(_evaluate_cell(models, th_c, lc_c, df_all, dm,
                                           scaler_disp, enc, params))
                rows.append(base)
            logger.info(f"  [{variant} θ*={theta:g} lc={held_lc or 'both'}] done in "
                        f"{time.time() - t0:.0f}s (M={len(models)})")

    cd.CFG.hard_architecture = ""
    cd.CFG.theta_feature_map = ""
    cd.get_model_config = _orig_get_cfg

    if not rows:
        raise SystemExit("No results produced.")
    df = pd.DataFrame(rows)
    detail_path = os.path.join(args.output_dir, "Table_ablation_theta_generalization.csv")
    df.to_csv(detail_path, index=False)
    logger.info(f"Saved: {os.path.basename(detail_path)} ({len(df)} rows)")

    # ---- Aggregates: overall + stratified by fold type ----------------------
    def _agg(sub: pd.DataFrame) -> Dict:
        return {
            "EA_MAPE": sub["EA_err_pct"].mean(),
            "IPF_MAPE": sub["IPF_err_pct"].mean(),
            "EA_floor_MAPE": sub["EA_floor_err_pct"].mean(),
            "IPF_floor_MAPE": sub["IPF_floor_err_pct"].mean(),
            "EA_mech_MAPE": sub["EA_mech_err_pct"].mean(),
            "IPF_mech_MAPE": sub["IPF_mech_err_pct"].mean(),
            "EA_skill_mean": sub["EA_skill_vs_floor"].mean(),
            "IPF_skill_mean": sub["IPF_skill_vs_floor"].mean(),
            "load_R2_shape_mean": sub.get("load_R2_shape", pd.Series(dtype=float)).mean(),
            "load_NRMSE_mean": sub.get("load_NRMSE", pd.Series(dtype=float)).mean(),
            "n_cells": len(sub),
        }
    summ_rows = []
    for variant in variants:
        sv = df[df["variant"] == variant]
        if not len(sv):
            continue
        for stratum, sub in [("all", sv),
                             ("interior", sv[sv["fold_type"] == "interior"]),
                             ("boundary", sv[sv["fold_type"] == "boundary"])]:
            if len(sub):
                summ_rows.append({"variant": variant, "stratum": stratum, **_agg(sub)})
    summ = pd.DataFrame(summ_rows).round(3)
    summ_path = os.path.join(args.output_dir, "Table_ablation_summary.csv")
    summ.to_csv(summ_path, index=False)
    logger.info(f"Saved: {os.path.basename(summ_path)}")
    logger.info("\n" + summ.to_string(index=False))

    # ---- Jackknife+ calibrated intervals + empirical coverage ---------------
    jk_rows = []
    for variant in variants:
        for lc in lcs:
            sub = df[(df["variant"] == variant) & (df["LC"] == lc)].sort_values("theta_star")
            if len(sub) < 3:
                continue
            for metric, pred_c, exp_c in [("EA", "EA_pred_J", "EA_exp_J"),
                                          ("IPF", "IPF_pred_kN", "IPF_exp_kN")]:
                tbl, cov = cd.jackknife_plus_intervals(
                    sub[pred_c].values.astype(float),
                    sub[exp_c].values.astype(float), alpha=args.alpha)
                jk_rows.append({"variant": variant, "LC": lc, "metric": metric,
                                "n_folds": len(tbl),
                                "nominal_coverage": 1.0 - args.alpha,
                                "empirical_coverage": round(cov, 3),
                                "median_half_width": round(float(tbl["half_width"].median()), 4)})
    if jk_rows:
        jk = pd.DataFrame(jk_rows)
        jk.to_csv(os.path.join(args.output_dir, "Table_ablation_jackknife_coverage.csv"),
                  index=False)
        logger.info("Saved: Table_ablation_jackknife_coverage.csv")
        logger.info("\n" + jk.to_string(index=False))

    # ---- Figures -------------------------------------------------------------
    plotted = [v for v in variants if v in set(df["variant"])]
    x = np.arange(len(plotted))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, col, floor_col, mech_col, title in [
        (axes[0], "EA_err_pct", "EA_floor_err_pct", "EA_mech_err_pct",
         f"EA@{cd.EA_COMMON_MM_TAG} design-level error"),
        (axes[1], "IPF_err_pct", "IPF_floor_err_pct", "IPF_mech_err_pct",
         "IPF design-level error"),
    ]:
        means = [df[df["variant"] == v][col].mean() for v in plotted]
        ax.bar(x, means, 0.55, color="#0072B2", edgecolor="black", linewidth=0.8,
               label="model")
        ax.axhline(df[floor_col].mean(), color="#D55E00", ls="--", lw=1.6,
                   label=f"interp. floor ({df[floor_col].mean():.1f}%)")
        ax.axhline(df[mech_col].mean(), color="#009E73", ls=":", lw=1.6,
                   label=f"mechanics trend ({df[mech_col].mean():.1f}%)")
        ax.set_xticks(x)
        ax.set_xticklabels([v.replace("_", "\n") for v in plotted], fontsize=9)
        ax.set_ylabel("Mean abs. error at held-out design (%)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    cd.add_subplot_label(axes[0], "a")
    cd.add_subplot_label(axes[1], "b")
    fig.suptitle(f"Held-out-design generalization ({args.protocol.upper()}, design level)")
    fig.savefig(os.path.join(args.output_dir, "Fig_ablation_theta_generalization.png"),
                dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Per-fold skill figure
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    fold_labels = sorted(df["fold"].unique(), key=lambda s: (str(s)))
    width = 0.8 / max(len(plotted), 1)
    cmapv = plt.get_cmap("tab10")
    for ax, sk_col, title in [(axes[0], "EA_skill_vs_floor", f"EA@{cd.EA_COMMON_MM_TAG} skill"),
                              (axes[1], "IPF_skill_vs_floor", "IPF skill")]:
        for vi, v in enumerate(plotted):
            sv = df[df["variant"] == v].groupby("fold")[sk_col].mean()
            vals = [sv.get(f, np.nan) for f in fold_labels]
            ax.bar(np.arange(len(fold_labels)) + vi * width, vals, width,
                   color=cmapv(vi), edgecolor="black", linewidth=0.5, label=v)
        ax.axhline(0.0, color="black", lw=1.2)
        ax.set_xticks(np.arange(len(fold_labels)) + width * (len(plotted) - 1) / 2)
        ax.set_xticklabels(fold_labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    axes[0].set_ylabel("Skill  $S = 1 - err_{model}/err_{floor}$")
    axes[0].legend(fontsize=8)
    cd.add_subplot_label(axes[0], "a")
    cd.add_subplot_label(axes[1], "b")
    fig.suptitle("Per-fold skill vs the model-free interpolation floor (S>0 beats it)")
    fig.savefig(os.path.join(args.output_dir, "Fig_ablation_skill.png"),
                dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved: Fig_ablation_theta_generalization.png + Fig_ablation_skill.png")
    logger.info(f"TOTAL wall time: {(time.time() - t_start) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
