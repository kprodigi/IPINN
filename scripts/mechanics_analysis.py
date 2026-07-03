"""Data-only mechanics analysis: mode signatures, densification kinematics,
master-curve collapse.  No training required — runs in seconds on the raw
experiments and produces the Phase-2 evidence artifacts:

    Table_crush_mode_signatures.csv     per-curve mechanics descriptors
    Fig_mode_signatures.png             regime-separation evidence
    Table_densification_kinematics.csv  d_dens / plateau / stiffness / IPF
                                        regressed on candidate H(θ) forms
    Fig_densification_kinematics.png    onset vs θ with the best kinematic fit
    Table_master_curve_collapse.csv     self-similarity quantification
    Fig_master_curve_collapse.png       raw vs mechanics-scaled overlays

Usage:
    python scripts/mechanics_analysis.py --data_dir ./data --output_dir ./results_mechanics
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import composite_design as cd  # noqa: E402


def _make_logger(path: str) -> logging.Logger:
    log = logging.getLogger("mechanics_analysis")
    log.setLevel(logging.INFO)
    log.handlers = []
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    # UTF-8-safe console on Windows (θ/σ/² in log lines otherwise crash cp1252).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def fig_densification_kinematics(sig_df, kin_df, output_dir, logger):
    """Onset (and plateau force) vs θ with the best-fitting kinematic candidate."""
    cd.set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    colors = {"LC1": cd.COLORS.get("LC1", "#0072B2"), "LC2": cd.COLORS.get("LC2", "#D55E00")}
    markers = {"LC1": "o", "LC2": "s"}
    th_grid = np.linspace(44, 71, 200)
    for panel, qty, ylabel in ((0, "d_dens_mm", "Densification onset $d_{dens}$ (mm)"),
                               (1, "F_plateau_kN", "Plateau force $F_{plateau}$ (kN)")):
        ax = axes[panel]
        for lc in sorted(sig_df["LC"].unique()):
            sub = sig_df[sig_df["LC"] == lc]
            mask = np.isfinite(sub[qty].values.astype(float))
            ax.scatter(sub["Angle_deg"][mask], sub[qty][mask], s=70,
                       marker=markers.get(lc), color=colors.get(lc),
                       edgecolor="black", linewidth=0.7, label=f"{lc} measured")
            fits = kin_df[(kin_df["LC"] == lc) & (kin_df["Quantity"] == qty)]
            if len(fits):
                best = fits.loc[fits["R2_fit"].idxmax()]
                f = cd.THETA_KINEMATIC_CANDIDATES[best["candidate"]](th_grid)
                ax.plot(th_grid, best["a"] + best["b"] * f, "--", color=colors.get(lc),
                        lw=1.5, label=f"{lc} fit: {best['candidate']} (R²={best['R2_fit']:.2f})")
        ax.set_xlabel("Interior angle $\\theta$ (°)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")
    cd.add_subplot_label(axes[0], "a")
    cd.add_subplot_label(axes[1], "b")
    fig.suptitle("Kinematic trends: candidate $H(\\theta)$ fits to measured mechanics quantities")
    out = os.path.join(output_dir, "Fig_densification_kinematics.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {os.path.basename(out)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--output_dir", default="./results_mechanics")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = _make_logger(os.path.join(args.output_dir, "mechanics_log.txt"))

    df_all = cd.load_data(args.data_dir, logger)
    logger.info("=" * 78)
    logger.info("MECHANICS ANALYSIS (data-only): signatures, kinematics, master-curve")
    logger.info("=" * 78)

    sig_df = cd.compute_mode_signature_table(df_all, logger, args.output_dir)
    logger.info("\n" + sig_df.round(3).to_string(index=False))

    kin_df = cd.fit_densification_kinematics(sig_df, logger, args.output_dir)

    cd.fig_mode_signatures(sig_df, args.output_dir, logger)
    fig_densification_kinematics(sig_df, kin_df, args.output_dir, logger)
    collapse_df = cd.fig_master_curve_collapse(df_all, sig_df, args.output_dir, logger)
    if collapse_df is not None:
        logger.info("\nMaster-curve collapse quality:\n" + collapse_df.to_string(index=False))


if __name__ == "__main__":
    main()
