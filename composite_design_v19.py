# -*- coding: utf-8 -*-
"""
================================================================================
IPINN CRASHWORTHINESS FRAMEWORK
================================================================================
Physics-Informed Neural Networks for Forward Prediction and Inverse Design
of Hexagonal Composite Ring Structures under Quasi-Static Crushing
================================================================================
"""

import os
import sys
import glob
import copy
import json
import pickle
import random
import warnings
import argparse
import time
import logging
from dataclasses import dataclass, field, asdict, replace
from typing import Dict, Tuple, List, Optional, Callable, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.swa_utils import AveragedModel

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    matthews_corrcoef,
)
from sklearn.calibration import calibration_curve

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.ticker import AutoMinorLocator

warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\.nn\.modules\.loss")
warnings.filterwarnings("ignore", category=UserWarning, module=r"matplotlib")
warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn")

# Optional imports with graceful fallback
try:
    from scipy.signal import find_peaks
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    def find_peaks(x, prominence=0.0, width=None):
        """Fallback peak detection."""
        x = np.asarray(x).flatten()
        peaks = []
        for i in range(1, len(x) - 1):
            if x[i] > x[i - 1] and x[i] > x[i + 1]:
                peaks.append(i)
        return np.asarray(peaks, dtype=int), {}

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
    HAS_SKLEARN_GP = True
except ImportError:
    HAS_SKLEARN_GP = False

try:
    from skopt import gp_minimize as skopt_gp_minimize
    from skopt.space import Real, Categorical
    HAS_SKOPT = True
except (ImportError, TypeError, AttributeError) as _skopt_err:
    HAS_SKOPT = False
    # scikit-optimize may fail with numpy >= 2.0 (removed np.int, np.float)
    # Install compatible version: pip install "scikit-optimize>=0.10.1"

try:
    from botorch.fit import fit_gpytorch_mll
    from botorch.models.gp_regression import SingleTaskGP
    from botorch.models.model_list_gp_regression import ModelListGP
    from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
    from botorch.utils.transforms import normalize, unnormalize
    from botorch.utils.sampling import draw_sobol_samples
    from botorch.optim.optimize import optimize_acqf
    from botorch.acquisition.multi_objective.logei import (
        qLogNoisyExpectedHypervolumeImprovement,
    )
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
    from botorch.utils.multi_objective.pareto import is_non_dominated
    from botorch.exceptions import BadInitialCandidatesWarning
    HAS_BOTORCH = True
except ImportError:
    HAS_BOTORCH = False
    BadInitialCandidatesWarning = UserWarning  # type: ignore


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
class _SafeUnicodeStreamHandler(logging.StreamHandler):
    """Console StreamHandler that never crashes on Unicode (e.g. cp1252 Windows consoles)."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        except UnicodeEncodeError:
            try:
                msg = self.format(record).encode("ascii", errors="replace").decode("ascii")
                self.stream.write(msg + self.terminator)
                self.flush()
            except Exception:
                pass


def setup_logging(output_dir: str) -> logging.Logger:
    """Configure logging to both file and console."""
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger("PINN_Crashworthiness")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fh = logging.FileHandler(os.path.join(output_dir, "run_log.txt"), mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    ch = _SafeUnicodeStreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# =============================================================================
# PUBLICATION STYLE CONFIGURATION
# =============================================================================
# =============================================================================
# PUBLICATION FIGURE SCALING
# =============================================================================
# Composite Structures full-page width: 190mm = 7.48 inches.
# Elsevier minimum finished text: 7pt (6pt subscript).
# All figures are saved at 600 DPI (≥500 DPI combination art requirement).
#
# Problem: figures of different widths get scaled differently when printed
# at full-page width. Reference width is PRINT_WIDTH_IN (single column).
#
# Wide figures used to multiply font sizes by width/7.48, which made on-screen
# and PNG exports look huge and caused label/legend/title overlap. We cap the
# scale at 1.0 so fonts never exceed the single-column reference size.
#
# Target sizes (reference at fig_width >= PRINT_WIDTH_IN):
#   axis labels:    9 pt
#   subplot titles: 9 pt (bold)
#   tick labels:    8 pt
#   legend:         8 pt
#   panel labels:   9.5 pt (bold)
#   line width:     0.8 pt
# =============================================================================

PRINT_WIDTH_IN = 7.48  # Composite Structures full-page width in inches


def scaled_fonts(fig_width: float) -> dict:
    """Return font sizes for a figure of width ``fig_width`` (inches).

    Scale is ``min(fig_width / PRINT_WIDTH_IN, 1.0)`` so wide multi-panel figures
    do not get oversized type. Narrow figures scale down to a floor for readability.
    """
    s_raw = float(fig_width) / PRINT_WIDTH_IN
    s = min(s_raw, 1.0)
    s = max(s, 0.58)  # floor for very small axes (e.g. dense grids)
    return {
        "label":   round(9 * s, 1),
        "title":   round(8.5 * s, 1),
        "tick":    round(8 * s, 1),
        "legend":  round(7.5 * s, 1),
        "panel":   round(9.5 * s, 1),
        "annot":   round(7 * s, 1),
        "suptitle": round(10 * s, 1),
        "linewidth": round(0.8 * s, 2),
        "markersize": round(4 * s, 1),
        "axes_lw": round(0.7 * s, 2),
        "tick_major": round(4 * s, 1),
        "tick_minor": round(2.5 * s, 1),
    }


def apply_fig_style(fig, axes=None, fig_width: float = None, logger: Optional[logging.Logger] = None):
    """Apply scaled font sizes to all axes in a figure for print uniformity.
    
    Call AFTER creating subplots and setting labels/titles, but BEFORE savefig.
    If axes is None, uses fig.get_axes(). If fig_width is None, reads from fig.
    """
    _log = logger if logger is not None else logging.getLogger(__name__)
    if fig_width is None:
        fig_width = fig.get_size_inches()[0]
    if axes is None:
        all_axes = fig.get_axes()
    else:
        all_axes = list(np.array(axes).flat) if hasattr(axes, '__iter__') else [axes]
    
    sf = scaled_fonts(fig_width)
    for ax in all_axes:
        try:
            ax.tick_params(axis="both", labelsize=sf["tick"],
                           length=sf["tick_major"], width=sf["axes_lw"],
                           which='major')
            ax.tick_params(length=sf["tick_minor"], width=sf["axes_lw"] * 0.7,
                           which='minor')
            for spine in ax.spines.values():
                spine.set_linewidth(sf["axes_lw"])
            if ax.get_xlabel():
                ax.set_xlabel(ax.get_xlabel(), fontsize=sf["label"])
            if ax.get_ylabel():
                ax.set_ylabel(ax.get_ylabel(), fontsize=sf["label"])
            if ax.get_title():
                ax.set_title(ax.get_title(), fontsize=sf["title"])
            legend = ax.get_legend()
            if legend:
                for text in legend.get_texts():
                    text.set_fontsize(sf["legend"])
                try:
                    legend.prop.set_size(sf["legend"])
                except Exception as ex:
                    _log.debug("apply_fig_style: legend.prop sizing skipped: %s", ex)
        except Exception as ex:
            _log.debug("apply_fig_style: axis styling skipped: %s", ex)
    # Figure-level legends (e.g. fig.legend) are not tied to a single Axes
    for leg in getattr(fig, "legends", []) or []:
        try:
            for text in leg.get_texts():
                text.set_fontsize(sf["legend"])
            try:
                leg.prop.set_size(sf["legend"])
            except Exception as ex:
                _log.debug("apply_fig_style: fig legend prop sizing skipped: %s", ex)
        except Exception as ex:
            _log.debug("apply_fig_style: fig legend styling skipped: %s", ex)
    # Colorbar labels / tick labels on inset colorbar axes
    for ax in all_axes:
        try:
            ylab = ax.get_ylabel()
            if ylab and getattr(ax, "yaxis", None) is not None:
                ax.yaxis.label.set_fontsize(sf["label"])
            xlab = ax.get_xlabel()
            if xlab and getattr(ax, "xaxis", None) is not None:
                ax.xaxis.label.set_fontsize(sf["label"])
        except Exception as ex:
            _log.debug("apply_fig_style: colorbar label styling skipped: %s", ex)
    # Scale suptitle if present
    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(sf["suptitle"])


def set_publication_style():
    """Set matplotlib parameters for publication-quality figures.
    
    These are DEFAULT rcParams. Individual figure functions should call
    apply_fig_style() before saving to ensure print-uniform font scaling.
    The rcParams here target a ~10-inch figure width (the median).
    """
    plt.rcParams.update({
        "figure.figsize": (8, 6), "figure.dpi": 150, "figure.facecolor": "white",
        "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.12,
        "font.size": 10, "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 10, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.linewidth": 1.0, "axes.grid": True, "axes.axisbelow": True,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.size": 5, "ytick.major.size": 5,
        "xtick.minor.size": 3.0, "ytick.minor.size": 3.0,
        "xtick.major.width": 0.9, "ytick.major.width": 0.9,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "legend.fontsize": 8.5, "legend.frameon": True, "legend.framealpha": 0.95,
        "lines.linewidth": 1.5, "lines.markersize": 5,
        "grid.alpha": 0.25, "grid.linewidth": 0.5, "grid.linestyle": "--",
        "errorbar.capsize": 3,
    })


# =============================================================================
# COLOR AND STYLE DEFINITIONS
# =============================================================================
COLORS = {
    "LC1": "#4DBBD5", "LC2": "#E64B35",
    "ddns": "#E64B35", "soft": "#4DBBD5", "hard": "#7B2D8E",
    "data": "#000000", "experiment": "#000000",
    "gpbo": "#F39B7F",
}

MARKERS = {"LC1": "o", "LC2": "s", "ddns": "o", "soft": "D", "hard": "^",
           "gpbo": "^"}

LC_MARKERS = {
    "LC1": MARKERS.get("LC1", "o"),
    "LC2": MARKERS.get("LC2", "s"),
}

LINESTYLES = {"ddns": "--", "soft": "-.", "hard": "-", "data": "-",
              "gpbo": "-",
              "LC1": "-", "LC2": "--"}

HATCHES = {"ddns": "//", "soft": "\\\\", "hard": "", "random": "", "unseen": "//"}
FILLSTYLES = {"ddns": "none", "soft": "full", "hard": "full"}

MODEL_LABELS = {"ddns": "DDNS", "soft": "Soft-PINN", "hard": "Hard-PINN"}


# =============================================================================
# REPRODUCIBILITY
# =============================================================================
def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# [CHANGE A] LC-SPECIFIC DISPLACEMENT RANGE HELPER
# =============================================================================
def disp_end_mm(lc: str) -> float:
    """
    Return the maximum displacement (stroke) for a given loading configuration.
    LC1 -> 80 mm, LC2 -> 130 mm.
    """
    lc_str = str(lc).upper().strip()
    if lc_str in ["LC1", "0"]:
        return 80.0
    elif lc_str in ["LC2", "1"]:
        return 130.0
    else:
        if "1" in lc_str and "2" not in lc_str:
            return 80.0
        return 130.0


def get_n_steps_curve(lc: str) -> int:
    """Return number of steps for curve evaluation based on LC."""
    d_end = disp_end_mm(lc)
    return max(161, int(d_end * 2) + 1)


# Common displacement for cross-LC comparison in inverse design.
# LC1 has d_end=80mm, LC2 has d_end=130mm.  Evaluating both at d_common=80mm
# ensures the EA comparison is not confounded by LC2's longer crush path.
D_COMMON = 80.0  # mm
# CSV / log fragment for EA at D_COMMON (inverse objective); keep in sync if D_COMMON changes.
EA_COMMON_MM_TAG = f"{int(D_COMMON)}mm"


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class Config:
    """Global configuration with all parameters documented."""
    seed: int = 2026
    split_seed: int = 2026
    test_size: float = 0.20
    n_ensemble: int = 20
    bootstrap: bool = True
    seed_base: int = 2026
    force_cpu: bool = False
    angle_opt_min: float = 45.0
    angle_opt_max: float = 70.0
    angle_grid_step: float = 0.05
    output_dir: str = "./results"
    theta_star: float = 60.0
    unseen_fixed_epochs: int = 300
    run_gpbo: bool = True
    run_ablation: bool = True
    run_reviewer_proof: bool = True  # V5: Reviewer-proof analyses
    run_mobo_qnehvi: bool = True  # qNEHVI MOBO on (theta, LC) vs EA@D_COMMON & IPF (requires botorch)
    # When True, missing optional deps (skopt for GP-BO, botorch for MOBO) abort at startup.
    strict_paper_deps: bool = False
    # strict_paper_deps also requires MOBO to run (no --no_mobo_qnehvi) for a complete submission bundle.
    # Expensive inverse ablations (extra GP-BO runs per target); enable with --inverse_ablation.
    run_inverse_ablation: bool = False
    inverse_ablation_max_targets: int = 2
    # Validation-row inverse stress test (uses random-protocol val rows not in train).
    run_inverse_stress_validation: bool = True
    inverse_stress_max_targets: int = 5
    # Per-member forward spread at reported optimum (cheap).
    run_inverse_member_spread: bool = True
    # CI / smoke: tiny budgets, no MOBO, no reviewer extras, GP-BO replaced by coarse grid inverse.
    dry_run: bool = False
    show_plots: bool = False
    save_plots: bool = True
    # Convergence filter: discard members that are statistical outliers in
    # training-set R² using the Tukey fence (Q1 - 1.5*IQR).  This is the
    # standard boxplot outlier rule and adapts to the difficulty of each
    # protocol/approach combination.  A member below the fence learned
    # substantially less than its peers from similar data, indicating a
    # convergence failure rather than genuine epistemic disagreement.
    # Set to 0.0 to disable (fence_multiplier controls the strictness).
    convergence_filter_iqr: float = 1.5  # Tukey fence multiplier (1.5=standard, 3.0=extreme only)
    # Include design-space rows with stroke within this many mm of LC-specific d_end.
    design_space_stroke_margin_mm: float = 2.0
    # Leave-one-angle-out cross-validation (forward model robustness)
    run_loao_cv: bool = True
    # Residual-based adaptive refinement (RAR) for collocation sampling
    run_rar: bool = True
    rar_interval: int = 50   # epochs between RAR weight updates
    rar_warmup: int = 100    # start RAR after this many epochs (uniform before)


CFG = Config()


def protocol_label(protocol: str) -> str:
    """Human-readable protocol caption (``CFG.theta_star`` for unseen holdout)."""
    if protocol == "random":
        return "Random 80/20 Split"
    if protocol == "unseen":
        return rf"Unseen Angle $\theta^*={CFG.theta_star:g}^\circ$"
    return str(protocol)


def _ensemble_std_along_members(arr: np.ndarray) -> np.ndarray:
    """Std along ensemble axis 0; ddof=1 only if M>1 else zeros (avoids NaNs for M=1)."""
    a = np.asarray(arr, dtype=np.float64)
    if a.shape[0] <= 1:
        if a.ndim == 1:
            return np.zeros((), dtype=np.float64)
        return np.zeros(a.shape[1:], dtype=np.float64)
    return np.std(a, axis=0, ddof=1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() and (not CFG.force_cpu) else "cpu")
# Hard-PINN eval uses autograd over displacement; batching avoids VRAM spikes on large tensors.
HARD_PINN_EVAL_BATCH = 4096


def refresh_device() -> None:
    """Recompute global DEVICE (call after CLI updates CFG.force_cpu)."""
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() and (not CFG.force_cpu) else "cpu")


def _data_loader_kwargs() -> Dict[str, Any]:
    """DataLoader options: pin_memory disabled because to_tensor() places data on DEVICE directly."""
    return {"pin_memory": False}


@dataclass
class BOConfig:
    """GP-BO configuration.
    
    Uses skopt.gp_minimize with joint (theta, LC) search space.
    A single GP with Matern kernel over theta and categorical LC dimension
    shares information across loading conditions, producing well-resolved
    posteriors for both LCs.

    Default budget is ``n_calls_total`` and ``n_init`` below (see runtime logs).
    """
    n_calls_total: int = 100
    n_init: int = 20
    xi: float = 0.01
    n_candidates: int = 500
    gp_restarts: int = 3
    seed: int = 2026
    prob_weight: float = 0.02          # ensemble classifier penalty weight (auto-tuned if 'auto')
    beta_robust: float = 0.0           # ensemble uncertainty penalty weight (0=off, auto-tuned in pipeline)
    run_classifier_ablation: bool = True  # run with vs without penalty comparison
    lambda_sweep: bool = True           # run lambda sensitivity analysis
    # Tikhonov regularization for ill-posedness: gamma * (theta - theta_center)^2
    gamma_tikhonov: float = 0.0        # auto-tuned if 0; explicitly set >0 to override
    theta_center: float = 57.5         # midpoint of [45, 70] angle range
    # Multi-start BO: run GP-BO n_bo_restarts times with different seeds, keep best
    n_bo_restarts: int = 5


BO_CFG = BOConfig()


@dataclass
class MOBOQNEHVICfg:
    """Multi-objective BO (qNEHVI) on (angle, LC) after dense landscape (BoTorch).

    qNEHVI matches qEHVI in the noiseless limit and scales better in batch q
    (Daulton et al., NeurIPS 2021). Reference point follows BoTorch convention
    for maximization: strictly worse than the objective vectors of interest.
    """
    n_init: int = 14
    n_batches: int = 22
    q: int = 2
    mc_samples: int = 64
    num_restarts: int = 8
    raw_samples: int = 256
    ref_margin_frac: float = 0.03


MOBO_QNEHVI_CFG = MOBOQNEHVICfg()


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================
def _dry_run_shrink_training_cfg(cfg: Dict) -> None:
    """In-place shrink of training budgets when ``CFG.dry_run`` (CI / smoke)."""
    if not getattr(CFG, "dry_run", False):
        return
    cap_ep = 10
    cfg["epochs"] = min(int(cfg.get("epochs", 100)), cap_ep)
    cfg["eval_every"] = max(1, min(4, cfg["epochs"] // 2))
    if "earlystop_patience_evals" in cfg:
        cfg["earlystop_patience_evals"] = min(int(cfg["earlystop_patience_evals"]), 2)
    we = cfg.get("warmup_epochs")
    if we is not None:
        cfg["warmup_epochs"] = min(int(we), max(2, cfg["epochs"] // 2))


def get_model_config(approach: str, protocol: str = "random", w_phys_override: float = None) -> Dict:
    """Get model configuration with protocol-specific hyperparameters."""
    
    if protocol == "unseen":
        # --- Tuned hyperparameters for unseen θ=60° ---
        # Best val load R2 = 0.783475, arch = [128, 64, 32]
        cfg_ddns = {
            "optimizer": "adam", "lr": 4.2123162503e-05,
            "weight_decay": 3.1582563297e-05, "batch_size": 64,
            "hidden_layers": [128, 64, 32], "dropout": 0.016401,
            "softplus_beta": 18.9027, "smoothl1_beta": 1.0838,
            "w_data_load": 3.568932, "w_data_energy": 3.451798,
            "w_phys": 0.0, "w_bc": 0.0, "colloc_ratio": 0.0,
            "epochs": 600, "eval_every": 25,
            "earlystop_patience_evals": 15, "earlystop_min_delta": 1e-5,
            "sched_patience": 58, "sched_factor": 0.4589,
        }
        
        # Best val load R2 = 0.804971, arch = [256, 128]
        # HPO v3: 95 trials, Optuna TPE, 2-seed evaluation
        # w_phys=0.5195, w_mono=4.098, w_smooth=0.0194, delta=2.66 deg
        cfg_soft = {
            "optimizer": "adam", "lr": 8.0733040807e-03,
            "weight_decay": 6.8420377257e-04, "batch_size": 64,
            "hidden_layers": [256, 128], "dropout": 0.007671,
            "softplus_beta": 12.0831, "smoothl1_beta": 1.0266,
            "w_data_load": 2.969519, "w_data_energy": 0.953040,
            "w_phys": 0.519484 if w_phys_override is None else w_phys_override,
            "w_bc": 0.599104, "colloc_ratio": 3.670053,
            "w_monotonicity": 4.097050,
            "w_angle_smooth": 0.019446,
            "smooth_delta_deg": 2.6603,
            "extrapolate_angles": True,
            "epochs": 800, "eval_every": 25,
            "earlystop_patience_evals": 15, "earlystop_min_delta": 1e-5,
            "sched_patience": 55, "sched_factor": 0.4574,
        }
        
        # Best val load R2 = 0.850764, arch = [128, 64]
        # HPO v3: 154 trials, Optuna TPE, 2-seed evaluation
        # Stabilized training: warmup=80 epochs, cosine annealing, SWA last 20%
        # w_mono=7.720, w_smooth=0.0161, w_curv=0.0013
        cfg_hard = {
            "optimizer": "adam", "lr": 9.9507487403e-05,
            "weight_decay": 3.7459350574e-03, "batch_size": 8,
            "hidden_layers": [128, 64], "dropout": 0.005504,
            "softplus_beta": 11.6712, "smoothl1_beta": 0.1176,
            "w_load": 6.8031, "w_energy": 8.6549,
            "grad_clip": 0.9834,
            "w_monotonicity": 7.719974,
            "w_angle_smooth": 0.016094,
            "smooth_delta_deg": 1.9329,
            "colloc_ratio": 3.5795,
            "extrapolate_angles": True,
            "epochs": 800, "eval_every": 20,
            "earlystop_patience_evals": 20, "earlystop_min_delta": 1e-5,
            # Stabilization params (fixed, not searched)
            "warmup_epochs": 80,
            "swa_pct": 0.20,
            "eta_min": 1e-6,
        }
    else:
        cfg_soft = {
            "optimizer": "adamw", "lr": 5.26e-4, "weight_decay": 1e-6,
            "batch_size": 32, "hidden_layers": [64, 64, 64], "dropout": 0.016,
            "softplus_beta": 8.0, "smoothl1_beta": 0.77,
            "w_data_load": 1.0, "w_data_energy": 1.2,
            "w_phys": 1.0 if w_phys_override is None else w_phys_override,
            "w_bc": 0.0, "colloc_ratio": 1.8, "epochs": 2000, "eval_every": 20,
            "earlystop_patience_evals": 15, "earlystop_min_delta": 1e-5,
            "sched_patience": 50, "sched_factor": 0.8,
        }
        cfg_ddns = copy.deepcopy(cfg_soft)
        cfg_ddns["w_phys"] = 0.0
        cfg_ddns["w_bc"] = 0.0
        cfg_ddns["colloc_ratio"] = 0.0
        
        cfg_hard = {
            "optimizer": "adamw", "lr": 1.5e-4, "weight_decay": 1e-5,
            "batch_size": 64, "hidden_layers": [32, 32], "dropout": 0.0,
            "softplus_beta": 8.0, "smoothl1_beta": 0.05,
            "w_load": 3.0, "w_energy": 3.0, "epochs": 2000, "eval_every": 20,
            "earlystop_patience_evals": 15, "earlystop_min_delta": 1e-5,
            "sched_patience": 40, "sched_factor": 0.7,
        }
    
    cfg = {"ddns": cfg_ddns, "soft": cfg_soft, "hard": cfg_hard}[approach]
    _dry_run_shrink_training_cfg(cfg)
    return cfg


# =============================================================================
# [CHANGE E] INVERSE DESIGN TARGETS - Generated from data
# =============================================================================
INVERSE_TARGETS = []  # Populated dynamically from empirical data


def ea_at_common_from_row(
    row: pd.Series,
    df_all: Optional[pd.DataFrame] = None,
    logger: Optional[logging.Logger] = None,
) -> float:
    """Energy absorbed from 0 to D_COMMON (mm) for one design-space row.

    Matches ``compute_ea_ipf_ensemble(..., d_eval=D_COMMON)`` when curve data exist.
    If ``df_all`` lacks a dense curve, falls back to ``F_mean * min(D_COMMON, d_end)``
    (crude mean-force proxy—not exact absorbed energy on a nonlinear curve).
    """
    d_end = float(row["disp_end"])
    if d_end <= D_COMMON:
        return float(row["EA"])
    if df_all is not None:
        sub = df_all[(df_all["LC"] == row["LC"]) & (df_all["Angle"] == row["Angle"])].sort_values("disp_mm")
        if len(sub) > 1:
            ea_interp = float(np.interp(D_COMMON, sub["disp_mm"].values, sub["energy_J"].values))
            ea0 = float(np.interp(0.0, sub["disp_mm"].values, sub["energy_J"].values))
            return ea_interp - ea0
    approx = float(row["F_mean"]) * min(D_COMMON, d_end)
    if logger is not None:
        logger.debug(
            "ea_at_common_from_row: F_mean stroke approximation (LC=%s Angle=%s)",
            row.get("LC"), row.get("Angle"),
        )
    return approx


def enrich_df_metrics_ea_common(df_metrics: pd.DataFrame, df_all: pd.DataFrame,
                                logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Add ``EA_common`` column: EA absorbed to ``D_COMMON`` for every configuration."""
    df_metrics["EA_common"] = [
        ea_at_common_from_row(row, df_all, logger=logger) for _, row in df_metrics.iterrows()]
    if logger is not None:
        logger.info(f"  df_metrics: added EA_common (EA to d={D_COMMON:.0f} mm) for classifier / inverse / β tuning")
    return df_metrics


def generate_feasible_targets(df_metrics: pd.DataFrame, logger: logging.Logger,
                               df_all: pd.DataFrame = None) -> List[Dict]:
    """Generate inverse targets from **paired** observed (EA@D_COMMON, IPF) samples.

    Targets are real rows chosen at spread positions along sorted EA_common, so
    each (EA, IPF) pair is jointly observed—not independent marginal quantiles,
    which can lie outside the empirical attainable set.
    """
    n = len(df_metrics)
    if n == 0:
        return []
    if "EA_common" in df_metrics.columns:
        ea_c = df_metrics["EA_common"].values.astype(float)
    else:
        ea_c = np.array([
            ea_at_common_from_row(row, df_all, logger=logger) for _, row in df_metrics.iterrows()])
    ipf = df_metrics["IPF"].values.astype(float)
    order = np.argsort(ea_c)
    denom = max(n - 1, 1)
    quantiles = [0.20, 0.35, 0.50, 0.65, 0.80]
    ids = ["T1", "T2", "T3", "T4", "T5"]
    rationales = [
        "Paired sample at ~20th pct EA@D_COMMON (low crush energy)",
        "Paired sample at ~35th pct EA@D_COMMON",
        "Paired sample at ~50th pct EA@D_COMMON (median spread)",
        "Paired sample at ~65th pct EA@D_COMMON",
        "Paired sample at ~80th pct EA@D_COMMON (high crush energy)",
    ]
    targets = []
    for tid, q, rat in zip(ids, quantiles, rationales):
        pos = int(round(q * denom))
        pos = min(max(pos, 0), n - 1)
        i = int(order[pos])
        targets.append({
            "id": tid,
            "EA": float(ea_c[i]),
            "IPF": float(ipf[i]),
            "d_eval": D_COMMON,
            "rationale": rat,
        })
    logger.info(f"  Generated paired feasible targets (EA to d={D_COMMON:.0f} mm + matching IPF):")
    for t in targets:
        logger.info(f"    {t['id']}: EA@{D_COMMON:.0f}mm={t['EA']:.2f}J, IPF={t['IPF']:.3f}kN - {t['rationale']}")
    return targets


def generate_pareto_targets(
    pareto_front_df: pd.DataFrame,
    logger: logging.Logger,
    n_targets: int = 5
) -> List[Dict]:
    """Generate inverse design targets from evenly-spaced Pareto front points.

    Selects n_targets points along the Pareto front by arc-length
    parameterization in normalized (EA, IPF) space, ensuring even coverage
    of the trade-off. These targets are guaranteed feasible by construction.
    """
    if len(pareto_front_df) < n_targets:
        logger.warning(f"  Pareto front has only {len(pareto_front_df)} points, "
                      f"cannot select {n_targets} targets")
        return []

    df = pareto_front_df.sort_values("EA").reset_index(drop=True)
    ea_min, ea_max = df["EA"].min(), df["EA"].max()
    ipf_min, ipf_max = df["IPF"].min(), df["IPF"].max()
    ea_n = (df["EA"].values - ea_min) / (ea_max - ea_min + 1e-12)
    ipf_n = (df["IPF"].values - ipf_min) / (ipf_max - ipf_min + 1e-12)

    # Cumulative arc-length in normalized space
    arc = np.zeros(len(df))
    for i in range(1, len(df)):
        arc[i] = arc[i-1] + np.sqrt((ea_n[i] - ea_n[i-1])**2 + (ipf_n[i] - ipf_n[i-1])**2)

    # Select evenly-spaced points along arc
    target_arcs = np.linspace(0, arc[-1], n_targets)
    selected_indices = list(dict.fromkeys(int(np.argmin(np.abs(arc - ta))) for ta in target_arcs))

    targets = []
    for k, idx in enumerate(selected_indices):
        row = df.iloc[idx]
        targets.append({
            "id": f"P{k+1}",
            "EA": float(row["EA"]),
            "IPF": float(row["IPF"]),
            "d_eval": D_COMMON,
            "angle_hint": float(row["angle"]),
            "lc_hint": str(row["lc"]),
            "rationale": f"Pareto front point {k+1}/{len(selected_indices)} (arc-length spaced)",
        })

    logger.info(f"  Generated {len(targets)} Pareto-optimal inverse targets:")
    for t in targets:
        logger.info(f"    {t['id']}: EA@{EA_COMMON_MM_TAG}={t['EA']:.2f}J, IPF={t['IPF']:.3f}kN "
                    f"(hint: theta={t['angle_hint']:.1f} deg, {t['lc_hint']})")
    return targets


# =============================================================================
# PREPROCESSING
# =============================================================================
@dataclass
class ScalingParams:
    """Scaling parameters for gradient computation."""
    mu_d: float
    sig_d: float
    mu_F: float
    sig_F: float
    mu_E: float
    sig_E: float
    grad_factor: float


def create_preprocessors(train_df: pd.DataFrame, logger: logging.Logger) -> Tuple:
    """Create and fit preprocessors on training data only."""
    scaler_disp = StandardScaler()
    scaler_out = StandardScaler()
    try:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
    
    scaler_disp.fit(train_df[["disp_mm"]].values.astype(float))
    scaler_out.fit(train_df[["load_kN", "energy_J"]].values.astype(float))
    enc.fit(train_df[["LC"]])
    
    params = ScalingParams(
        mu_d=float(scaler_disp.mean_[0]), sig_d=float(scaler_disp.scale_[0]),
        mu_F=float(scaler_out.mean_[0]), sig_F=float(scaler_out.scale_[0]),
        mu_E=float(scaler_out.mean_[1]), sig_E=float(scaler_out.scale_[1]),
        grad_factor=float(scaler_out.scale_[1] / max(1e-12, scaler_disp.scale_[0])),
    )
    logger.info(f"Scaler params: μ_F={params.mu_F:.4f}, σ_F={params.sig_F:.4f}")
    return scaler_disp, scaler_out, enc, params


U_COL = 0


def build_features(df: pd.DataFrame, scaler_disp: StandardScaler, enc: OneHotEncoder) -> np.ndarray:
    """Build input feature matrix."""
    disp_scaled = scaler_disp.transform(df[["disp_mm"]].values.astype(float)).astype(np.float32)
    theta_rad = np.deg2rad(df[["Angle"]].values.astype(float)).astype(np.float32)
    sin_t = np.sin(theta_rad).astype(np.float32)
    cos_t = np.cos(theta_rad).astype(np.float32)
    lc_oh = enc.transform(df[["LC"]]).astype(np.float32)
    return np.hstack([disp_scaled, sin_t, cos_t, lc_oh]).astype(np.float32)


def build_targets(df: pd.DataFrame, scaler_out: StandardScaler) -> np.ndarray:
    """Build target matrix."""
    return scaler_out.transform(df[["load_kN", "energy_J"]].values.astype(float)).astype(np.float32)


def to_tensor(x: np.ndarray) -> torch.Tensor:
    """Convert numpy array to torch tensor on the global DEVICE."""
    if x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]:
        return torch.from_numpy(x).to(device=DEVICE, dtype=torch.float32)
    return torch.tensor(x, dtype=torch.float32, device=DEVICE)


def hard_pinn_predict_load_energy(
    model: nn.Module,
    X: torch.Tensor,
    params: ScalingParams,
    batch_size: int = HARD_PINN_EVAL_BATCH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batched Hard-PINN forward: physical load (kN) and energy (J).

    Avoids holding the full validation/training graph for dE/dd at once, which
    can exhaust GPU memory on ~10k+ rows or wide ensembles.
    """
    model.eval()
    n = int(X.shape[0])
    f_parts: List[np.ndarray] = []
    e_parts: List[np.ndarray] = []
    for start in range(0, n, batch_size):
        Xb = X[start : start + batch_size].detach().clone().requires_grad_(True)
        E_n = model(Xb)
        dE = torch.autograd.grad(
            E_n, Xb, torch.ones_like(E_n), create_graph=False, retain_graph=False
        )[0]
        Fb = (dE[:, U_COL : U_COL + 1] * params.grad_factor).detach().cpu().numpy().reshape(-1)
        Eb = (E_n.detach() * params.sig_E + params.mu_E).cpu().numpy().reshape(-1)
        f_parts.append(Fb)
        e_parts.append(Eb)
    return np.concatenate(f_parts), np.concatenate(e_parts)



def train_full_data_hard_pinn(df_all: pd.DataFrame, logger: logging.Logger) -> Tuple[List[nn.Module], StandardScaler, StandardScaler, OneHotEncoder, ScalingParams]:
    """
    Train Hard-PINN ensemble on 100% of data for maximum accuracy in inverse design.
    
    No holdout is used because:
    1. The forward model has already been validated using dual protocols
    2. For inverse design, we want the most accurate surrogate possible
    3. The inverse design validation comes from comparing predicted vs actual (EA, IPF)
    
    Returns:
        models: List of trained Hard-PINN models (ensemble)
        scaler_disp: Displacement scaler (fitted on all data)
        scaler_out: Output scaler (fitted on all data)
        enc: One-hot encoder for LC
        params: Scaling parameters
    """
    logger.info("  Training Hard-PINN ensemble on 100% of data for inverse design...")
    logger.info(f"    Total observations: {len(df_all)}")
    logger.info(f"    Angles: {sorted(df_all['Angle'].unique())}")
    logger.info(f"    LCs: {sorted(df_all['LC'].unique())}")
    
    # Create preprocessors on full data
    scaler_disp = StandardScaler()
    scaler_out = StandardScaler()
    try:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
    
    scaler_disp.fit(df_all[["disp_mm"]].values.astype(float))
    scaler_out.fit(df_all[["load_kN", "energy_J"]].values.astype(float))
    enc.fit(df_all[["LC"]])
    
    params = ScalingParams(
        mu_d=float(scaler_disp.mean_[0]), sig_d=float(scaler_disp.scale_[0]),
        mu_F=float(scaler_out.mean_[0]), sig_F=float(scaler_out.scale_[0]),
        mu_E=float(scaler_out.mean_[1]), sig_E=float(scaler_out.scale_[1]),
        grad_factor=float(scaler_out.scale_[1] / max(1e-12, scaler_disp.scale_[0])),
    )
    
    # [V8] Use unseen protocol config for full-data training.
    # The unseen config has the optimized architecture [128, 64] and
    # stabilized training (warmup + cosine + SWA). The regularizers
    # (monotonicity, smoothness, curvature) remain beneficial even when
    # all angles are present because they encode universal physics priors.
    cfg = get_model_config("hard", protocol="unseen")
    cfg["epochs"] = 1500  # Full data (incl. 60°) converges faster than unseen protocol
    cfg["batch_size"] = 128  # Larger batch for 10500 samples (unseen used 8 for 8816)
    cfg["warmup_epochs"] = 150  # Scale warmup proportionally: 80/800 ≈ 150/1500
    if CFG.dry_run:
        cfg["epochs"] = min(int(cfg["epochs"]), 12)
        cfg["warmup_epochs"] = min(int(cfg.get("warmup_epochs", 6)), 4)
        _dry_run_shrink_training_cfg(cfg)

    # Build features and targets for full data
    X_full = build_features(df_all, scaler_disp, enc)
    Y_full = build_targets(df_all, scaler_out)
    X_tensor = to_tensor(X_full)
    Y_tensor = to_tensor(Y_full)
    y_full_np = df_all[["load_kN", "energy_J"]].values
    
    # Collocation-based regularizer setup (mirrors train_hard)
    w_mono = cfg.get("w_monotonicity", 0.0)
    w_smooth = cfg.get("w_angle_smooth", 0.0)
    smooth_delta = cfg.get("smooth_delta_deg", 1.5)
    colloc_ratio = cfg.get("colloc_ratio", 0.0)
    extrapolate = cfg.get("extrapolate_angles", True)

    colloc_sampler = None
    if w_mono > 0 or w_smooth > 0:
        colloc_sampler = create_collocation_sampler(df_all, scaler_disp, enc, extrapolate_angles=extrapolate)
        logger.info(f"    Collocation regularizers active: w_mono={w_mono}, w_smooth={w_smooth}, "
                    f"colloc_ratio={colloc_ratio}, extrapolate={extrapolate}")
    
    models = []
    train_r2_scores = []
    for m_idx in range(CFG.n_ensemble):
        # Seed stride = 100 (distinct from train_ensemble's stride of 1000).
        # Different strides ensure full-data and validation ensemble members
        # are initialized independently for m_idx >= 1.
        seed = CFG.seed_base + m_idx * 100
        set_seed(seed)
        rng = np.random.default_rng(seed)
        
        # Bootstrap resample if enabled
        if CFG.bootstrap:
            indices = rng.integers(0, len(X_full), len(X_full))
            X_train = X_tensor[indices]
            Y_train = Y_tensor[indices]
        else:
            X_train = X_tensor
            Y_train = Y_tensor
        
        # Create and train model
        d_zero_scaled = float(-scaler_disp.mean_[0] / scaler_disp.scale_[0])
        model = HardEnergyNet(X_full.shape[1], cfg["hidden_layers"], cfg["dropout"], cfg["softplus_beta"],
                              d_zero_scaled=d_zero_scaled).to(DEVICE)

        if cfg["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        
        # [V8] Stabilized training for inverse design Hard-PINN
        warmup_ep = cfg.get("warmup_epochs", 80)
        total_ep = cfg["epochs"]
        scheduler = WarmupCosineScheduler(optimizer, warmup_ep, total_ep, eta_min=cfg.get("eta_min", 1e-6))
        swa_start = int(total_ep * (1.0 - cfg.get("swa_pct", 0.20)))
        swa_model = AveragedModel(model, device=DEVICE)
        swa_active = False
        
        # Training loop
        dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=False, **_data_loader_kwargs()
        )
        
        for epoch in range(cfg["epochs"]):
            model.train()
            epoch_loss = 0.0
            for Xb, Yb in loader:
                Xb = Xb.requires_grad_(True)
                optimizer.zero_grad()
                E_pred = model(Xb)
                dE_dX = torch.autograd.grad(E_pred.sum(), Xb, create_graph=True)[0]
                F_phys = dE_dX[:, U_COL:U_COL+1] * params.grad_factor
                # Normalize force to match train_hard: loss weights were HPO-tuned
                # for normalized-scale losses. Computing in physical kN would scale
                # the force term by sig_F^2 relative to what HPO intended.
                F_n = (F_phys - params.mu_F) / params.sig_F
                E_target = Yb[:, 1:2]
                loss_F = F.smooth_l1_loss(F_n, Yb[:, 0:1], beta=cfg["smoothl1_beta"])
                loss_E = F.mse_loss(E_pred, E_target)
                loss = cfg["w_load"] * loss_F + cfg["w_energy"] * loss_E
                
                # Collocation-based regularizers (same as train_hard)
                if colloc_sampler is not None:
                    n_colloc = max(1, int(colloc_ratio * Xb.shape[0])) if colloc_ratio > 0 else max(1, Xb.shape[0] // 2)
                    Xc = colloc_sampler(n_colloc, rng)
                    
                    if w_mono > 0:
                        loss_mono = monotonicity_loss_hard(Xc, model, params)
                        loss = loss + w_mono * loss_mono
                    
                    if w_smooth > 0:
                        n_smooth = max(1, n_colloc // 2)
                        loss_smooth = angle_smoothness_loss_hard(model, colloc_sampler, n_smooth, rng, params, smooth_delta)
                        loss = loss + w_smooth * loss_smooth

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.get("grad_clip", 1.0))
                optimizer.step()
                epoch_loss += loss.item()
            
            # [V8] Stabilized schedule + SWA
            scheduler.step()
            if epoch >= swa_start:
                swa_active = True
                swa_model.update_parameters(model)
        
        # [V8] Use SWA model for final evaluation if active
        eval_model = swa_model.module if swa_active else model
        
        # Compute training-set R² for convergence check (batched: full tensor grad is VRAM-heavy)
        eval_model.eval()
        Fv, _ = hard_pinn_predict_load_energy(eval_model, X_tensor, params)
        tr_r2 = float(r2_score(y_full_np[:, 0], Fv))
        train_r2_scores.append(tr_r2)
        
        # Store the SWA-averaged model if active, otherwise the base model
        if swa_active:
            model.load_state_dict(swa_model.module.state_dict())
        models.append(model)
        logger.info(f"    Ensemble member {m_idx + 1}/{CFG.n_ensemble}: "
                    f"trained on {len(X_train)} samples, train R²={tr_r2:.4f}")
    
    # Convergence filter (Tukey fence)
    k_iqr = CFG.convergence_filter_iqr
    M_total = len(models)
    if k_iqr > 0 and M_total >= 5:
        q1 = float(np.percentile(train_r2_scores, 25))
        q3 = float(np.percentile(train_r2_scores, 75))
        iqr = q3 - q1
        fence = q1 - k_iqr * iqr if iqr > 0.01 else float('-inf')
        keep_mask = [r2 >= fence for r2 in train_r2_scores]
        M_eff = sum(keep_mask)
        if M_eff < M_total:
            discarded = [(i, train_r2_scores[i]) for i, k in enumerate(keep_mask) if not k]
            logger.info(f"    Convergence filter (Tukey fence={fence:.4f}): "
                        f"{M_total} -> {M_eff} members")
            for idx, r2 in discarded:
                logger.info(f"      Discarded member {idx+1}: train R²={r2:.4f}")
            models = [m for m, k in zip(models, keep_mask) if k]
        else:
            logger.info(f"    Convergence filter: all {M_total} members above fence ({fence:.4f})")
    
    logger.info(f"  Full-data Hard-PINN ensemble training complete "
                f"({len(models)}/{M_total} models)")
    
    return models, scaler_disp, scaler_out, enc, params


# =============================================================================
# DATA LOADING AND VALIDATION
# =============================================================================
def validate_input_data(df: pd.DataFrame, logger: logging.Logger) -> Tuple[bool, List[str]]:
    """Validate input data structure rigorously."""
    issues = []
    required_cols = ["disp_mm", "load_kN", "energy_J", "Angle", "LC"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")
        return False, issues
    
    for col in required_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            issues.append(f"Column '{col}' has {n_missing} missing values")
    
    return len(issues) == 0, issues


def load_data(data_dir: str, logger: logging.Logger) -> pd.DataFrame:
    """Load and validate experimental data from Excel/CSV files."""
    logger.info("LOADING DATA")
    
    files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    if not files:
        files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not files:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    
    df_list = []
    for f in files:
        try:
            df = pd.read_excel(f) if f.endswith(".xlsx") else pd.read_csv(f)
            col_map = {}
            for c in df.columns:
                cl = c.lower().strip()
                if "disp" in cl or cl == "d" or cl == "u": col_map[c] = "disp_mm"
                elif "load" in cl or "force" in cl or cl == "f": col_map[c] = "load_kN"
                elif "energy" in cl or cl == "e": col_map[c] = "energy_J"
                elif "angle" in cl or "theta" in cl: col_map[c] = "Angle"
                elif cl == "lc" or "loading" in cl: col_map[c] = "LC"
            df.rename(columns=col_map, inplace=True)
            
            if "LC" not in df.columns:
                fname_upper = os.path.basename(f).upper()
                if "LC1" in fname_upper: df["LC"] = "LC1"
                elif "LC2" in fname_upper: df["LC"] = "LC2"
                else: continue
            
            req = ["disp_mm", "load_kN", "energy_J", "Angle", "LC"]
            if all(r in df.columns for r in req):
                df = df[req].dropna()
                df_list.append(df)
                logger.info(f"  Loaded {len(df)} rows from {os.path.basename(f)}")
        except Exception as e:
            logger.error(f"  Error loading {f}: {e}")
    
    if not df_list:
        raise ValueError("No valid data files loaded")
    
    df_all = pd.concat(df_list, ignore_index=True)
    df_all["disp_mm"] = df_all["disp_mm"].astype(float)
    df_all["load_kN"] = df_all["load_kN"].astype(float)
    df_all["energy_J"] = df_all["energy_J"].astype(float)
    df_all["Angle"] = df_all["Angle"].astype(float)
    df_all["LC"] = df_all["LC"].astype(str)
    
    logger.info(f"Total: {len(df_all)} rows, Angles: {sorted(df_all['Angle'].unique())}")
    for lc in df_all["LC"].unique():
        lc_data = df_all[df_all["LC"] == lc]
        logger.info(f"  {lc}: disp range = [{lc_data['disp_mm'].min():.1f}, {lc_data['disp_mm'].max():.1f}] mm")
    
    return df_all


def split_random_80_20(df: pd.DataFrame, seed: int, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified random 80-20 split when every stratum has >=2 rows; else unstratified."""
    df = df.copy()
    df["_strat"] = df["Angle"].astype(str) + "_" + df["LC"].astype(str)
    vc = df["_strat"].value_counts()
    if vc.min() >= 2:
        train_df, val_df = train_test_split(
            df, test_size=CFG.test_size, random_state=seed, stratify=df["_strat"])
    else:
        logger.warning(
            "  Random split: stratify disabled (some Angle_LC strata have <2 rows); "
            "using unstratified train_test_split.",
        )
        train_df, val_df = train_test_split(df, test_size=CFG.test_size, random_state=seed)
    train_df = train_df.drop(columns=["_strat"]).reset_index(drop=True)
    val_df = val_df.drop(columns=["_strat"]).reset_index(drop=True)
    logger.info(f"Random split: train={len(train_df)}, val={len(val_df)}")
    return train_df, val_df


def split_unseen_angle(df: pd.DataFrame, theta_star: float, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out all data for angle theta_star."""
    val_df = df[df["Angle"] == theta_star].copy().reset_index(drop=True)
    train_df = df[df["Angle"] != theta_star].copy().reset_index(drop=True)
    logger.info(f"Unseen-angle split (θ*={theta_star}°): train={len(train_df)}, val={len(val_df)}")
    return train_df, val_df


#

# =============================================================================
# NEURAL NETWORK MODELS
# =============================================================================
class SoftPINNNet(nn.Module):
    """MLP for DDNS and Soft-PINN (outputs both F and E)."""
    
    def __init__(self, in_d: int, hidden_layers: List[int], dropout: float, softplus_beta: float):
        super().__init__()
        layers = []
        d = in_d
        for h in hidden_layers:
            layers.append(nn.Linear(d, h))
            layers.append(nn.Softplus(beta=softplus_beta))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HardEnergyNet(nn.Module):
    """MLP for Hard-PINN (outputs only E; F derived by differentiation).

    Boundary enforcement: E = g(d) * NN(x) where g(d) = d_scaled - d_zero_scaled.
    Since d_zero_scaled is the scaled displacement value at d_phys=0, we have
    g(d_phys=0) = 0 and thus E(d_phys=0) = 0 exactly by construction.
    This matches the structural enforcement of F = dE/dd.
    """

    def __init__(self, in_d: int, hidden_layers: List[int], dropout: float,
                 softplus_beta: float, d_zero_scaled: float = 0.0):
        super().__init__()
        layers = []
        d = in_d
        for h in hidden_layers:
            layers.append(nn.Linear(d, h))
            layers.append(nn.Softplus(beta=softplus_beta))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
        self.register_buffer("d_zero_scaled", torch.tensor(d_zero_scaled, dtype=torch.float32))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = x[:, 0:1] - self.d_zero_scaled  # zero when d_phys = 0
        return g * self.net(x)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# PHYSICS LOSS FUNCTIONS
# =============================================================================
def compute_physics_residual(Xin: torch.Tensor, F_n: torch.Tensor, E_n: torch.Tensor, 
                             params: ScalingParams) -> torch.Tensor:
    """Compute physics residual: dE/dd - F."""
    dE_dX = torch.autograd.grad(E_n, Xin, torch.ones_like(E_n), create_graph=True)[0]
    dE_dd = dE_dX[:, U_COL:U_COL+1] * params.grad_factor
    F_phys = F_n * params.sig_F + params.mu_F
    return dE_dd - F_phys


def physics_loss_soft(Xin: torch.Tensor, F_n: torch.Tensor, E_n: torch.Tensor,
                      params: ScalingParams) -> torch.Tensor:
    """Soft physics loss for Soft-PINN."""
    res = compute_physics_residual(Xin, F_n, E_n, params)
    return torch.mean((res / params.sig_F) ** 2)


def create_collocation_sampler(train_df: pd.DataFrame, scaler_disp: StandardScaler, 
                                enc: OneHotEncoder, extrapolate_angles: bool = False) -> Callable:
    """Create collocation point sampler with LC-specific displacement bounds.
    
    When extrapolate_angles=True (for unseen-angle protocol), sample angles
    uniformly across the full range [45°, 70°] — including the held-out angle.
    This lets the physics constraint dE/dd = F regularize the model's predictions
    at angles where no data exists, providing genuine extrapolation benefit.
    """
    lc_categories = [str(x) for x in enc.categories_[0].tolist()]
    
    disp_bounds = {}
    for lc in lc_categories:
        sub = train_df[train_df["LC"] == lc]
        d_max = min(float(sub["disp_mm"].max()), disp_end_mm(lc))
        disp_bounds[lc] = (float(sub["disp_mm"].min()), d_max)
    
    if extrapolate_angles:
        # Cover ALL angles including unseen ones
        theta_min = 45.0
        theta_max = 70.0
    else:
        theta_min = float(train_df["Angle"].min())
        theta_max = float(train_df["Angle"].max())
    
    def sample(n_colloc: int, rng: np.random.Generator) -> torch.Tensor:
        lc_idx = rng.integers(0, len(lc_categories), n_colloc)
        lc_onehot = np.eye(len(lc_categories), dtype=np.float32)[lc_idx]
        disp = np.array([rng.uniform(*disp_bounds[lc_categories[i]]) for i in lc_idx]).reshape(-1, 1)
        disp_scaled = scaler_disp.transform(disp).astype(np.float32)
        theta_deg = rng.uniform(theta_min, theta_max, (n_colloc, 1))
        theta_rad = np.deg2rad(theta_deg).astype(np.float32)
        X = np.hstack([disp_scaled, np.sin(theta_rad), np.cos(theta_rad), lc_onehot])
        return to_tensor(X)
    
    return sample


class AdaptiveCollocationSampler:
    """Residual-based Adaptive Refinement (RAR) collocation sampler (Lu et al., 2021).

    Maintains a fixed evaluation grid; periodically re-weights sampling probabilities
    proportional to physics residual magnitude so that collocation points concentrate
    where the constraint is most violated (typically near unseen angles).
    """

    def __init__(self, base_sampler: Callable, eval_grid_size: int = 500, rng_seed: int = 0):
        self.base_sampler = base_sampler
        rng = np.random.default_rng(rng_seed)
        self.eval_grid = base_sampler(eval_grid_size, rng)  # fixed tensor
        self.weights = np.ones(eval_grid_size, dtype=np.float64) / eval_grid_size
        self.is_active = False

    @torch.no_grad()
    def update_weights(self, model: nn.Module, params, approach: str) -> None:
        """Re-compute residual-based sampling weights on the evaluation grid."""
        model.eval()
        Xg = self.eval_grid.clone().requires_grad_(True)
        with torch.enable_grad():
            if approach == "hard":
                E_n = model(Xg)
                dE = torch.autograd.grad(E_n.sum(), Xg, create_graph=False)[0]
                F_derived = dE[:, U_COL:U_COL + 1] * params.grad_factor
                # For Hard-PINN F=dE/dd is exact; use |F| as proxy for where physics matters most
                residual = torch.abs(F_derived).squeeze().cpu().numpy()
            else:  # soft / ddns
                pred = model(Xg)
                F_n, E_n = pred[:, 0:1], pred[:, 1:2]
                dE_dX = torch.autograd.grad(E_n.sum(), Xg, create_graph=False)[0]
                dE_dd = dE_dX[:, U_COL:U_COL + 1] * params.grad_factor
                F_phys = F_n * params.sig_F + params.mu_F
                residual = torch.abs(dE_dd - F_phys).detach().squeeze().cpu().numpy()
        eps = 1e-8
        w = residual + eps
        self.weights = w / w.sum()
        self.is_active = True

    def sample(self, n_colloc: int, rng: np.random.Generator) -> torch.Tensor:
        """Draw collocation points proportional to residual weights with small perturbation."""
        if not self.is_active:
            return self.base_sampler(n_colloc, rng)
        idx = rng.choice(len(self.weights), size=n_colloc, replace=True, p=self.weights)
        pts = self.eval_grid[idx].clone()
        # Small displacement perturbation to avoid memorising exact grid points
        noise = torch.randn_like(pts[:, 0:1]) * 0.05
        pts[:, 0:1] = pts[:, 0:1] + noise
        return pts


def angle_smoothness_loss_soft(model: nn.Module, colloc_sampler: Callable,
                                n_pts: int, rng: np.random.Generator,
                                params: ScalingParams, delta_deg: float = 1.0) -> torch.Tensor:
    """Angle-smoothness regularization for Soft-PINN / DDNS.
    
    Penalizes rapid prediction changes w.r.t. angle:
        L_smooth = E[ ||f(θ+δ) − f(θ)||² ] / δ²
    
    This encodes the inductive bias that "60° should behave like its neighbors",
    which is exactly the physics-informed prior needed for unseen-angle generalization.
    """
    X_base = colloc_sampler(n_pts, rng)  # [n, features]
    
    # Perturb angle: features are [disp_scaled, sin(θ), cos(θ), lc_oh]
    # Extract sin/cos, compute θ, perturb, recompute sin/cos
    sin_t = X_base[:, 1:2].detach().clone()
    cos_t = X_base[:, 2:3].detach().clone()
    theta_rad = torch.atan2(sin_t, cos_t)
    delta_rad = np.deg2rad(delta_deg)
    theta_perturbed = theta_rad + delta_rad
    
    X_perturbed = X_base.detach().clone()
    X_perturbed[:, 1:2] = torch.sin(theta_perturbed)
    X_perturbed[:, 2:3] = torch.cos(theta_perturbed)
    
    pred_base = model(X_base)
    pred_pert = model(X_perturbed)
    
    # Penalize large changes in both F and E predictions
    diff = pred_base - pred_pert
    return torch.mean(diff ** 2) / (delta_rad ** 2)


def angle_smoothness_loss_hard(model: nn.Module, colloc_sampler: Callable,
                                n_pts: int, rng: np.random.Generator,
                                params: ScalingParams, delta_deg: float = 1.0) -> torch.Tensor:
    """Angle-smoothness regularization for Hard-PINN.
    
    Same concept as soft version but applied to energy output E and its
    gradient dE/dd (= force). Ensures both energy and derived force vary
    smoothly with angle.
    """
    X_base = colloc_sampler(n_pts, rng).requires_grad_(True)
    
    sin_t = X_base[:, 1:2].detach().clone()
    cos_t = X_base[:, 2:3].detach().clone()
    theta_rad = torch.atan2(sin_t, cos_t)
    delta_rad = np.deg2rad(delta_deg)
    theta_perturbed = theta_rad + delta_rad
    
    X_perturbed = X_base.detach().clone().requires_grad_(True)
    X_perturbed_data = X_perturbed.data.clone()
    X_perturbed_data[:, 1:2] = torch.sin(theta_perturbed)
    X_perturbed_data[:, 2:3] = torch.cos(theta_perturbed)
    X_perturbed = X_perturbed_data.requires_grad_(True)
    
    # Energy predictions at both angles
    E_base = model(X_base)
    E_pert = model(X_perturbed)
    
    # Also compare forces (dE/dd)
    dE_base = torch.autograd.grad(E_base.sum(), X_base, create_graph=True)[0]
    F_base = dE_base[:, U_COL:U_COL+1] * params.grad_factor
    
    dE_pert = torch.autograd.grad(E_pert.sum(), X_perturbed, create_graph=True)[0]
    F_pert = dE_pert[:, U_COL:U_COL+1] * params.grad_factor
    
    # Combined smoothness on both E and F
    E_diff = (E_base - E_pert) ** 2
    F_diff = (F_base - F_pert) ** 2
    return (torch.mean(E_diff) + torch.mean(F_diff)) / (delta_rad ** 2)


def monotonicity_loss_soft(Xin: torch.Tensor, model: nn.Module, 
                            params: ScalingParams) -> torch.Tensor:
    """Monotonicity constraint for Soft-PINN: energy must not decrease with displacement.
    
    Enforces dE/dd ≥ 0 (energy absorption is non-decreasing), which is a 
    universal thermodynamic constraint valid at ALL angles and loading configs.
    """
    Xin_g = Xin.requires_grad_(True) if not Xin.requires_grad else Xin
    pred = model(Xin_g)
    E_pred = pred[:, 1:2] * params.sig_E + params.mu_E
    dE_dX = torch.autograd.grad(E_pred.sum(), Xin_g, create_graph=True)[0]
    dE_dd = dE_dX[:, U_COL:U_COL+1]  # dE/d(disp_scaled)
    # Penalize negative energy gradients (energy should increase with displacement)
    return torch.mean(F.relu(-dE_dd) ** 2)


def monotonicity_loss_hard(Xin: torch.Tensor, model: nn.Module,
                            params: ScalingParams) -> torch.Tensor:
    """Monotonicity/non-negativity constraint for Hard-PINN: F = dE/dd ≥ 0.
    
    Since Hard-PINN derives F from dE/dd by construction, enforcing F ≥ 0
    is equivalent to enforcing energy monotonicity. This physically means
    the crushing force should be non-negative (the structure resists compression).
    """
    Xin_g = Xin.requires_grad_(True) if not Xin.requires_grad else Xin
    E_n = model(Xin_g)
    dE = torch.autograd.grad(E_n.sum(), Xin_g, create_graph=True)[0]
    F_phys = dE[:, U_COL:U_COL+1] * params.grad_factor
    # Penalize negative force (non-physical)
    return torch.mean(F.relu(-F_phys) ** 2)



def _val_checkpoint_score(r2_load: float, r2_energy: float) -> float:
    """Mean validation load/energy R² for checkpointing, LR schedule, and early stopping."""
    return 0.5 * (float(r2_load) + float(r2_energy))


class EarlyStopping:
    """Early stopping with patience."""
    def __init__(self, patience: int = 10, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.stop = False
    
    def __call__(self, val_score: float, epoch: int) -> bool:
        if self.best_score is None or val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay for Hard-PINN stabilization.
    
    Addresses the gradient amplification problem in Hard-PINN:
    F = dE/dd via autodiff means early-training weight noise is amplified
    into force-prediction noise. Warmup keeps the LR near zero during this
    critical phase, preventing catastrophic early weight updates.
    
    Cosine decay provides a deterministic, smooth schedule (unlike
    ReduceLROnPlateau which reacts to noisy validation signals).
    """
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, eta_min: float = 1e-6):
        self.optimizer = optimizer
        self.warmup = warmup_epochs
        self.total = total_epochs
        self.eta_min = eta_min
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.ep = 0
    
    def step(self):
        self.ep += 1
        if self.ep <= self.warmup:
            scale = self.ep / max(1, self.warmup)
        else:
            progress = (self.ep - self.warmup) / max(1, self.total - self.warmup)
            base_lr = max(self.base_lrs[0], 1e-12)
            scale = (self.eta_min / base_lr
                     + (1 - self.eta_min / base_lr)
                     * 0.5 * (1 + math.cos(math.pi * progress)))
        for pg, blr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = blr * scale
    
    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def train_ddns(train_df: pd.DataFrame, val_df: pd.DataFrame, scaler_disp: StandardScaler,
               scaler_out: StandardScaler, enc: OneHotEncoder, params: ScalingParams,
               seed: int, protocol: str, logger: logging.Logger) -> Tuple:
    """Train DDNS model.

    Returns
    -------
    model, history, best_val_score, meta
        ``best_val_score`` is the best epoch's mean validation R²,
        ``0.5 * (R²_load + R²_energy)``, used for checkpointing and early stopping.
    """
    set_seed(seed)
    cfg = get_model_config("ddns", protocol)
    t0 = time.time()
    
    Xtr = to_tensor(build_features(train_df, scaler_disp, enc))
    ytr = to_tensor(build_targets(train_df, scaler_out))
    Xv = to_tensor(build_features(val_df, scaler_disp, enc))
    y_val = val_df[["load_kN", "energy_J"]].values
    
    model = SoftPINNNet(Xtr.shape[1], cfg["hidden_layers"], cfg["dropout"], cfg["softplus_beta"]).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]) if cfg.get("optimizer", "adamw").lower() == "adam" else optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=cfg["sched_patience"], factor=cfg["sched_factor"], mode='max')
    mse = nn.MSELoss()
    sl1 = nn.SmoothL1Loss(beta=cfg["smoothl1_beta"])
    
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, Xtr.shape[0], Xtr.shape[0]) if CFG.bootstrap else np.arange(Xtr.shape[0])
    loader = DataLoader(
        TensorDataset(Xtr[idx], ytr[idx]), batch_size=cfg["batch_size"], shuffle=True, **_data_loader_kwargs()
    )
    
    history = {"epoch": [], "train_loss": [], "val_load_r2": [], "val_energy_r2": [],
               "phys_residual_rms": []}
    best_state, best_r2 = None, -1e9
    es = EarlyStopping(cfg["earlystop_patience_evals"], cfg["earlystop_min_delta"])

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        loss_sum, nb = 0.0, 0
        for Xb, yb in loader:
            pred = model(Xb)
            loss = cfg.get("w_data_load", 1.0) * sl1(pred[:, 0:1], yb[:, 0:1]) + cfg.get("w_data_energy", 1.2) * mse(pred[:, 1:2], yb[:, 1:2])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            nb += 1
        
        if ep % cfg["eval_every"] == 0:
            model.eval()
            with torch.no_grad():
                pv = model(Xv)
                Fv = (pv[:, 0] * params.sig_F + params.mu_F).cpu().numpy()
                Ev = (pv[:, 1] * params.sig_E + params.mu_E).cpu().numpy()
            r2_l = r2_score(y_val[:, 0], Fv)
            r2_e = r2_score(y_val[:, 1], Ev)
            val_score = _val_checkpoint_score(r2_l, r2_e)
            history["epoch"].append(ep)
            history["train_loss"].append(loss_sum / nb)
            history["val_load_r2"].append(r2_l)
            history["val_energy_r2"].append(r2_e)
            # Per-epoch physics residual on validation set (DDNS has no physics constraint)
            try:
                Xv_g = Xv.detach().clone().requires_grad_(True)
                with torch.enable_grad():
                    pv_g = model(Xv_g)
                    res = compute_physics_residual(Xv_g, pv_g[:, 0:1], pv_g[:, 1:2], params)
                    history["phys_residual_rms"].append(float(torch.sqrt(torch.mean(res ** 2)).item()))
            except Exception:
                history["phys_residual_rms"].append(float('nan'))
            sched.step(val_score)
            if val_score > best_r2:
                best_r2 = val_score
                best_state = copy.deepcopy(model.state_dict())
            if es(val_score, ep):
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, history, best_r2, {"training_time": time.time() - t0, "n_params": model.count_parameters()}


def train_soft(train_df: pd.DataFrame, val_df: pd.DataFrame, scaler_disp: StandardScaler,
               scaler_out: StandardScaler, enc: OneHotEncoder, params: ScalingParams,
               seed: int, protocol: str, logger: logging.Logger, w_phys_override: float = None) -> Tuple:
    """Train Soft-PINN model with enhanced physics constraints.
    
    For unseen-angle protocol, includes:
    - Extrapolating collocation (covers ALL angles including unseen)
    - Angle-smoothness regularization (smooth predictions across angles)
    - Monotonicity constraint (energy must increase with displacement)
    """
    set_seed(seed)
    cfg = get_model_config("soft", protocol, w_phys_override)
    t0 = time.time()
    rng = np.random.default_rng(seed + 200)
    
    Xtr = to_tensor(build_features(train_df, scaler_disp, enc))
    ytr = to_tensor(build_targets(train_df, scaler_out))
    Xv = to_tensor(build_features(val_df, scaler_disp, enc))
    y_val = val_df[["load_kN", "energy_J"]].values
    
    model = SoftPINNNet(Xtr.shape[1], cfg["hidden_layers"], cfg["dropout"], cfg["softplus_beta"]).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]) if cfg.get("optimizer", "adamw").lower() == "adam" else optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=cfg["sched_patience"], factor=cfg["sched_factor"], mode='max')
    mse = nn.MSELoss()
    sl1 = nn.SmoothL1Loss(beta=cfg["smoothl1_beta"])
    
    # Use extrapolating collocation for unseen protocol
    extrapolate = cfg.get("extrapolate_angles", False)
    colloc_sampler = create_collocation_sampler(train_df, scaler_disp, enc, extrapolate_angles=extrapolate)

    # RAR adaptive collocation (Lu et al., 2021)
    rar_sampler = None
    if getattr(CFG, 'run_rar', True) and not getattr(CFG, 'dry_run', False):
        rar_sampler = AdaptiveCollocationSampler(colloc_sampler, eval_grid_size=500, rng_seed=seed + 7)

    # New physics weights
    w_mono = cfg.get("w_monotonicity", 0.0)
    w_smooth = cfg.get("w_angle_smooth", 0.0)
    smooth_delta = cfg.get("smooth_delta_deg", 1.5)

    w_bc = cfg.get("w_bc", 0.0)
    X_bc_t = None
    if w_bc > 0:
        combos = train_df[["Angle", "LC"]].drop_duplicates().reset_index(drop=True)
        bc_df = combos.copy()
        bc_df["disp_mm"] = 0.0
        X_bc_t = to_tensor(build_features(bc_df, scaler_disp, enc))

    rng_b = np.random.default_rng(seed)
    idx = rng_b.integers(0, Xtr.shape[0], Xtr.shape[0]) if CFG.bootstrap else np.arange(Xtr.shape[0])
    loader = DataLoader(
        TensorDataset(Xtr[idx], ytr[idx]), batch_size=cfg["batch_size"], shuffle=True, **_data_loader_kwargs()
    )

    history = {"epoch": [], "train_loss": [], "phys_loss": [], "val_load_r2": [], "val_energy_r2": [],
               "phys_residual_rms": []}
    best_state, best_r2 = None, -1e9
    es = EarlyStopping(cfg["earlystop_patience_evals"], cfg["earlystop_min_delta"])
    
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        loss_sum, phys_sum, nb = 0.0, 0.0, 0
        for Xb, yb in loader:
            Xb.requires_grad_(True)
            pred = model(Xb)
            loss_data = cfg["w_data_load"] * sl1(pred[:, 0:1], yb[:, 0:1]) + cfg["w_data_energy"] * mse(pred[:, 1:2], yb[:, 1:2])
            lp_data = physics_loss_soft(Xb, pred[:, 0:1], pred[:, 1:2], params)
            n_colloc = max(1, int(cfg["colloc_ratio"] * Xb.shape[0]))
            _cs = rar_sampler if (rar_sampler is not None and rar_sampler.is_active) else colloc_sampler
            Xc = _cs.sample(n_colloc, rng).requires_grad_(True) if hasattr(_cs, 'sample') else _cs(n_colloc, rng).requires_grad_(True)
            pc = model(Xc)
            lp_colloc = physics_loss_soft(Xc, pc[:, 0:1], pc[:, 1:2], params)
            loss_phys = 0.5 * (lp_data + lp_colloc)
            loss = loss_data + cfg["w_phys"] * loss_phys
            
            # Monotonicity: energy must not decrease with displacement
            if w_mono > 0:
                loss_mono = monotonicity_loss_soft(Xc, model, params)
                loss = loss + w_mono * loss_mono
            
            # Angle smoothness: predictions should vary smoothly with angle
            if w_smooth > 0:
                n_smooth = max(1, n_colloc // 2)
                loss_smooth = angle_smoothness_loss_soft(model, colloc_sampler, n_smooth, rng, params, smooth_delta)
                loss = loss + w_smooth * loss_smooth
            
            if w_bc > 0 and X_bc_t is not None:
                pred_bc = model(X_bc_t)
                E_bc_phys = pred_bc[:, 1:2] * params.sig_E + params.mu_E
                loss = loss + w_bc * mse(E_bc_phys, torch.zeros_like(E_bc_phys))
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            phys_sum += loss_phys.item()
            nb += 1
        
        if ep % cfg["eval_every"] == 0:
            model.eval()
            with torch.no_grad():
                pv = model(Xv)
                Fv = (pv[:, 0] * params.sig_F + params.mu_F).cpu().numpy()
                Ev = (pv[:, 1] * params.sig_E + params.mu_E).cpu().numpy()
            r2_l = r2_score(y_val[:, 0], Fv)
            r2_e = r2_score(y_val[:, 1], Ev)
            val_score = _val_checkpoint_score(r2_l, r2_e)
            history["epoch"].append(ep)
            history["train_loss"].append(loss_sum / nb)
            history["phys_loss"].append(phys_sum / nb)
            history["val_load_r2"].append(r2_l)
            history["val_energy_r2"].append(r2_e)
            # Per-epoch physics residual on validation set
            try:
                Xv_g = Xv.detach().clone().requires_grad_(True)
                with torch.enable_grad():
                    pv_g = model(Xv_g)
                    res = compute_physics_residual(Xv_g, pv_g[:, 0:1], pv_g[:, 1:2], params)
                    history["phys_residual_rms"].append(float(torch.sqrt(torch.mean(res ** 2)).item()))
            except Exception:
                history["phys_residual_rms"].append(float('nan'))
            sched.step(val_score)
            if val_score > best_r2:
                best_r2 = val_score
                best_state = copy.deepcopy(model.state_dict())
            if es(val_score, ep):
                break
        # RAR weight update
        if rar_sampler is not None and ep % CFG.rar_interval == 0 and ep >= CFG.rar_warmup:
            rar_sampler.update_weights(model, params, "soft")

    if best_state:
        model.load_state_dict(best_state)
    return model, history, best_r2, {"training_time": time.time() - t0, "n_params": model.count_parameters(), "w_phys": cfg["w_phys"]}


def train_hard(train_df: pd.DataFrame, val_df: pd.DataFrame, scaler_disp: StandardScaler,
               scaler_out: StandardScaler, enc: OneHotEncoder, params: ScalingParams,
               seed: int, protocol: str, logger: logging.Logger) -> Tuple:
    """Train Hard-PINN model with enhanced regularization.

    The third return value is the best epoch's mean validation R²,
    ``0.5 * (R²_load + R²_energy)``, used for checkpointing (including SWA check).

    For unseen-angle protocol, includes:
    - Extrapolating collocation (covers ALL angles including unseen)
    - Angle-smoothness regularization on both E and F = dE/dd
    - Monotonicity constraint (F = dE/dd ≥ 0)
    - Curvature regularization (smooth force curves via d²E/dd²)
    - [V8] Stabilized training: LR warmup + cosine annealing + SWA
    """
    set_seed(seed)
    cfg = get_model_config("hard", protocol)
    t0 = time.time()
    rng = np.random.default_rng(seed + 300)
    
    Xtr = to_tensor(build_features(train_df, scaler_disp, enc))
    ytr = to_tensor(build_targets(train_df, scaler_out))
    Xv = to_tensor(build_features(val_df, scaler_disp, enc))
    y_val = val_df[["load_kN", "energy_J"]].values
    
    d_zero_scaled = float(-scaler_disp.mean_[0] / scaler_disp.scale_[0])
    model = HardEnergyNet(Xtr.shape[1], cfg["hidden_layers"], cfg["dropout"], cfg["softplus_beta"],
                          d_zero_scaled=d_zero_scaled).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]) if cfg.get("optimizer", "adamw").lower() == "adam" else optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    
    # [V8] Scheduler selection: stabilized (unseen) vs reactive (random)
    use_stabilized = "warmup_epochs" in cfg
    if use_stabilized:
        sched = WarmupCosineScheduler(opt, cfg["warmup_epochs"], cfg["epochs"],
                                      eta_min=cfg.get("eta_min", 1e-6))
        swa_start = int(cfg["epochs"] * (1.0 - cfg.get("swa_pct", 0.20)))
        swa_model = AveragedModel(model, device=DEVICE)
        swa_active = False
    else:
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=cfg["sched_patience"],
                                                      factor=cfg["sched_factor"], mode='max')
    
    mse = nn.MSELoss()
    sl1 = nn.SmoothL1Loss(beta=cfg["smoothl1_beta"])
    
    # Physics regularization setup
    w_mono = cfg.get("w_monotonicity", 0.0)
    w_smooth = cfg.get("w_angle_smooth", 0.0)
    smooth_delta = cfg.get("smooth_delta_deg", 1.5)
    colloc_ratio = cfg.get("colloc_ratio", 0.0)

    # Create collocation sampler if any physics regularization is active
    colloc_sampler = None
    if w_mono > 0 or w_smooth > 0:
        extrapolate = cfg.get("extrapolate_angles", False)
        colloc_sampler = create_collocation_sampler(train_df, scaler_disp, enc, extrapolate_angles=extrapolate)

    # RAR adaptive collocation
    rar_sampler = None
    if colloc_sampler is not None and getattr(CFG, 'run_rar', True) and not getattr(CFG, 'dry_run', False):
        rar_sampler = AdaptiveCollocationSampler(colloc_sampler, eval_grid_size=500, rng_seed=seed + 7)

    rng_b = np.random.default_rng(seed)
    idx = rng_b.integers(0, Xtr.shape[0], Xtr.shape[0]) if CFG.bootstrap else np.arange(Xtr.shape[0])
    loader = DataLoader(
        TensorDataset(Xtr[idx], ytr[idx]), batch_size=cfg["batch_size"], shuffle=True, **_data_loader_kwargs()
    )

    history = {"epoch": [], "train_loss": [], "val_load_r2": [], "val_energy_r2": [],
               "phys_residual_rms": []}
    best_state, best_r2 = None, -1e9
    # [V8] Disable early stopping for stabilized training: SWA needs the full
    # epoch budget to accumulate weight averages.
    es = None if use_stabilized else EarlyStopping(cfg["earlystop_patience_evals"], cfg["earlystop_min_delta"])
    
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        loss_sum, nb = 0.0, 0
        for Xb, yb in loader:
            Xb.requires_grad_(True)
            E_n = model(Xb)
            dE = torch.autograd.grad(E_n, Xb, torch.ones_like(E_n), create_graph=True)[0]
            F_phys = dE[:, U_COL:U_COL+1] * params.grad_factor
            F_n = (F_phys - params.mu_F) / params.sig_F
            loss = cfg["w_load"] * sl1(F_n, yb[:, 0:1]) + cfg["w_energy"] * mse(E_n, yb[:, 1:2])
            
            # Collocation-based physics regularization
            if colloc_sampler is not None:
                n_colloc = max(1, int(colloc_ratio * Xb.shape[0])) if colloc_ratio > 0 else max(1, Xb.shape[0] // 2)
                _cs = rar_sampler if (rar_sampler is not None and rar_sampler.is_active) else colloc_sampler
                Xc = _cs.sample(n_colloc, rng) if hasattr(_cs, 'sample') else _cs(n_colloc, rng)

                if w_mono > 0:
                    loss_mono = monotonicity_loss_hard(Xc, model, params)
                    loss = loss + w_mono * loss_mono
                
                if w_smooth > 0:
                    n_smooth = max(1, n_colloc // 2)
                    loss_smooth = angle_smoothness_loss_hard(model, colloc_sampler, n_smooth, rng, params, smooth_delta)
                    loss = loss + w_smooth * loss_smooth

            opt.zero_grad()
            loss.backward()
            if cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip"])
            opt.step()
            loss_sum += loss.item()
            nb += 1
        
        # [V8] Scheduler step: warmup+cosine every epoch, ReduceLROnPlateau on eval
        if use_stabilized:
            sched.step()
            if ep >= swa_start:
                swa_active = True
                swa_model.update_parameters(model)
        
        if ep % cfg["eval_every"] == 0:
            # [V8] Evaluate SWA model if active, otherwise base model
            eval_model = swa_model.module if (use_stabilized and swa_active) else model
            eval_model.eval()
            Fv, Ev = hard_pinn_predict_load_energy(eval_model, Xv, params)
            r2_l = r2_score(y_val[:, 0], Fv)
            r2_e = r2_score(y_val[:, 1], Ev)
            val_score = _val_checkpoint_score(r2_l, r2_e)
            history["epoch"].append(ep)
            history["train_loss"].append(loss_sum / nb)
            history["val_load_r2"].append(r2_l)
            history["val_energy_r2"].append(r2_e)
            # Hard-PINN: F=dE/dd by construction → residual ≈ 0; track data-fit residual instead
            history["phys_residual_rms"].append(float(np.sqrt(np.mean((Fv - y_val[:, 0]) ** 2))))
            if not use_stabilized:
                sched.step(val_score)
            if val_score > best_r2:
                best_r2 = val_score
                best_state = copy.deepcopy(eval_model.state_dict())
            if es is not None and es(val_score, ep):
                break
        # RAR weight update
        if rar_sampler is not None and ep % CFG.rar_interval == 0 and ep >= CFG.rar_warmup:
            rar_sampler.update_weights(model, params, "hard")

    # [V8] Final SWA evaluation
    if use_stabilized and swa_active:
        swa_model.module.eval()
        Fv, Ev = hard_pinn_predict_load_energy(swa_model.module, Xv, params)
        r2_swa = _val_checkpoint_score(
            float(r2_score(y_val[:, 0], Fv)),
            float(r2_score(y_val[:, 1], Ev)),
        )
        if r2_swa > best_r2:
            best_r2 = r2_swa
            best_state = copy.deepcopy(swa_model.module.state_dict())
    
    if best_state:
        model.load_state_dict(best_state)
    return model, history, best_r2, {"training_time": time.time() - t0, "n_params": model.count_parameters()}


def train_ensemble(approach: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
                   scaler_disp: StandardScaler, scaler_out: StandardScaler,
                   enc: OneHotEncoder, params: ScalingParams, protocol: str,
                   logger: logging.Logger) -> Dict:
    """Train ensemble of models with optional convergence filtering.
    
    After training all M members, each member's load R² is evaluated on the
    full training set (not its bootstrap sample).  Members whose training-set
    R² falls below the Tukey fence (Q1 - k*IQR, where k = CFG.convergence_filter_iqr)
    are discarded as convergence failures.  This is analogous to discarding
    diverged MCMC chains and is documented transparently via M_total, M_eff,
    and the discard log.
    
    The threshold is computed from the TRAINING set R² distribution to avoid
    any interaction with validation data in the filtering decision.
    """
    logger.info(f"  Training {approach.upper()} ensemble (M={CFG.n_ensemble})...")
    train_fn = {"ddns": train_ddns, "soft": train_soft, "hard": train_hard}[approach]
    models, histories, member_metrics, metas = [], [], [], []
    
    for m in range(CFG.n_ensemble):
        seed = CFG.seed_base + m * 1000
        model, hist, r2, meta = train_fn(train_df, val_df, scaler_disp, scaler_out, enc, params, seed, protocol, logger)
        models.append(model)
        histories.append(hist)
        metas.append(meta)
        metrics = evaluate_model(model, approach, val_df, scaler_disp, scaler_out, enc, params)
        member_metrics.append(metrics)
        logger.info(f"    M{m+1}: R²_load={metrics['load_r2']:.4f}, time={meta['training_time']:.1f}s")
    
    # --- Convergence filter (Tukey fence on training-set R²) ----------------
    # Evaluate every member on the FULL training set (not its bootstrap
    # sample) to get a comparable quality metric.  Then apply the standard
    # Tukey fence: discard members with train R² < Q1 - k*IQR where k is
    # CFG.convergence_filter_iqr (default 1.5).  This adapts to the task
    # difficulty and only removes genuine outliers.
    k_iqr = CFG.convergence_filter_iqr
    M_total = len(models)
    
    # Compute training-set R² for each member
    train_r2_scores = []
    for model in models:
        tr_metrics = evaluate_model(model, approach, train_df, scaler_disp, scaler_out, enc, params)
        train_r2_scores.append(tr_metrics["load_r2"])
    
    if k_iqr > 0 and M_total >= 5:
        q1 = float(np.percentile(train_r2_scores, 25))
        q3 = float(np.percentile(train_r2_scores, 75))
        iqr = q3 - q1
        # Guard: if IQR is tiny (all members converged similarly), don't filter.
        # Only apply the fence when members show meaningful variance (IQR > 0.01).
        if iqr > 0.01:
            fence = q1 - k_iqr * iqr
        else:
            fence = float('-inf')  # skip filter for tight distributions
        
        keep_mask = [r2 >= fence for r2 in train_r2_scores]
        M_eff = sum(keep_mask)
        n_discarded = M_total - M_eff
        
        logger.info(f"    Convergence filter (Tukey fence, k={k_iqr}):")
        logger.info(f"      Train R² stats: Q1={q1:.4f}, Q3={q3:.4f}, IQR={iqr:.4f}")
        if iqr > 0.01:
            logger.info(f"      Fence = Q1 - {k_iqr}*IQR = {fence:.4f}")
        else:
            logger.info(f"      IQR < 0.01: all members converged similarly, skipping filter")
        
        if n_discarded > 0:
            discarded_indices = [i for i, keep in enumerate(keep_mask) if not keep]
            logger.info(f"      M_total={M_total}, M_eff={M_eff}, discarded={n_discarded}")
            for idx in discarded_indices:
                logger.info(f"      Discarded M{idx+1}: train R²={train_r2_scores[idx]:.4f} < fence {fence:.4f}")
            
            # Filter all parallel lists
            models = [m for m, k in zip(models, keep_mask) if k]
            histories = [h for h, k in zip(histories, keep_mask) if k]
            member_metrics = [mm for mm, k in zip(member_metrics, keep_mask) if k]
            metas = [mt for mt, k in zip(metas, keep_mask) if k]
        else:
            logger.info(f"      All {M_total} members above fence (no outliers)")
    else:
        M_eff = M_total
        fence = float('-inf')
    
    # Safety: ensure at least 3 members survive
    if len(models) < 3:
        logger.warning(f"    WARNING: Only {len(models)} members survived filtering; "
                       f"consider increasing n_ensemble or decreasing k_iqr.")
    
    ens_metrics = evaluate_ensemble(models, approach, val_df, scaler_disp, scaler_out, enc, params)
    return {"models": models, "histories": histories, "member_metrics": member_metrics,
            "metrics": ens_metrics, "avg_training_time": np.mean([m["training_time"] for m in metas]),
            "n_params": metas[0]["n_params"],
            "M_total": M_total, "M_eff": len(models),
            "train_r2_scores": train_r2_scores,
            "convergence_fence": fence}


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================
def evaluate_model(model: nn.Module, approach: str, val_df: pd.DataFrame,
                   scaler_disp: StandardScaler, scaler_out: StandardScaler,
                   enc: OneHotEncoder, params: ScalingParams) -> Dict:
    """Evaluate single model."""
    model.eval()
    Xv = to_tensor(build_features(val_df, scaler_disp, enc))
    y_val = val_df[["load_kN", "energy_J"]].values
    
    if approach in ["ddns", "soft"]:
        with torch.no_grad():
            pv = model(Xv)
            Fv = (pv[:, 0] * params.sig_F + params.mu_F).cpu().numpy()
            Ev = (pv[:, 1] * params.sig_E + params.mu_E).cpu().numpy()
    else:
        Fv, Ev = hard_pinn_predict_load_energy(model, Xv, params)
    
    return {"load_r2": float(r2_score(y_val[:, 0], Fv)), "energy_r2": float(r2_score(y_val[:, 1], Ev)),
            "load_rmse": float(np.sqrt(mean_squared_error(y_val[:, 0], Fv))),
            "energy_rmse": float(np.sqrt(mean_squared_error(y_val[:, 1], Ev))),
            "load_mae": float(mean_absolute_error(y_val[:, 0], Fv)),
            "energy_mae": float(mean_absolute_error(y_val[:, 1], Ev)),
            "load_errors": Fv - y_val[:, 0], "energy_errors": Ev - y_val[:, 1],
            "predictions": {"load": Fv, "energy": Ev}, "true_values": {"load": y_val[:, 0], "energy": y_val[:, 1]}}


def evaluate_ensemble(models: List[nn.Module], approach: str, val_df: pd.DataFrame,
                      scaler_disp: StandardScaler, scaler_out: StandardScaler,
                      enc: OneHotEncoder, params: ScalingParams) -> Dict:
    """Evaluate ensemble of models."""
    Xv = to_tensor(build_features(val_df, scaler_disp, enc))
    y_val = val_df[["load_kN", "energy_J"]].values
    all_F, all_E = [], []
    
    for model in models:
        model.eval()
        if approach in ["ddns", "soft"]:
            with torch.no_grad():
                pv = model(Xv)
                all_F.append((pv[:, 0] * params.sig_F + params.mu_F).cpu().numpy())
                all_E.append((pv[:, 1] * params.sig_E + params.mu_E).cpu().numpy())
        else:
            Fb, Eb = hard_pinn_predict_load_energy(model, Xv, params)
            all_F.append(Fb)
            all_E.append(Eb)
    
    Fm = np.mean(all_F, axis=0)
    Fs = _ensemble_std_along_members(all_F)
    Em = np.mean(all_E, axis=0)
    Es = _ensemble_std_along_members(all_E)
    
    return {"load_r2": float(r2_score(y_val[:, 0], Fm)), "energy_r2": float(r2_score(y_val[:, 1], Em)),
            "load_rmse": float(np.sqrt(mean_squared_error(y_val[:, 0], Fm))),
            "energy_rmse": float(np.sqrt(mean_squared_error(y_val[:, 1], Em))),
            "load_mae": float(mean_absolute_error(y_val[:, 0], Fm)),
            "energy_mae": float(mean_absolute_error(y_val[:, 1], Em)),
            "load_errors": Fm - y_val[:, 0], "energy_errors": Em - y_val[:, 1],
            "predictions": {"load": Fm, "energy": Em, "load_std": Fs, "energy_std": Es},
            "true_values": {"load": y_val[:, 0], "energy": y_val[:, 1]}}


def compute_statistical_tests(dual_results: Dict, logger: logging.Logger) -> Dict:
    """Compute statistical significance tests between models.
    
    Uses Welch's independent-samples t-test (not paired) because the
    Tukey-fence convergence filter can discard different numbers of
    ensemble members per approach, making the arrays unequal length.
    Cohen's d is computed for independent samples:
        d = (mean1 - mean2) / s_pooled

    **Interpretation:** ensemble members are trained on the same data split and
    architecture; they are not strictly independent replicates. Welch's test is
    used here as a standardized effect-size summary; treat p-values as
    descriptive rather than exact confirmatory Type-I error rates.

    **Multiplicity:** Bonferroni adjustment is applied **within each protocol**
    (random vs unseen), over the family of pairwise approach comparisons reported
    for that protocol—not pooled across protocols.
    """
    if not HAS_SCIPY:
        logger.warning("SciPy not available, skipping statistical tests")
        return {}
    results = {}
    for protocol in ["random", "unseen"]:
        if protocol not in dual_results:
            continue
        results[protocol] = {}
        r2_vals = {a: [m["load_r2"] for m in dual_results[protocol][a]["member_metrics"]] 
                   for a in ["ddns", "soft", "hard"] if a in dual_results[protocol]}
        for a1, a2 in [("ddns", "soft"), ("ddns", "hard"), ("soft", "hard")]:
            if a1 in r2_vals and a2 in r2_vals:
                arr1 = np.array(r2_vals[a1])
                arr2 = np.array(r2_vals[a2])
                t_stat, t_pvalue = stats.ttest_ind(arr1, arr2, equal_var=False)
                # Cohen's d for independent samples (pooled std)
                n1, n2 = len(arr1), len(arr2)
                s_pooled = np.sqrt(((n1 - 1) * np.var(arr1, ddof=1) + (n2 - 1) * np.var(arr2, ddof=1)) / (n1 + n2 - 2))
                cohens_d = (np.mean(arr1) - np.mean(arr2)) / (s_pooled + 1e-12)
                results[protocol][f"{a1}_vs_{a2}"] = {"t_statistic": float(t_stat), "t_pvalue": float(t_pvalue), "cohens_d": float(cohens_d),
                                                       "n1": n1, "n2": n2}
                logger.info(f"  {protocol} {a1} vs {a2}: p={t_pvalue:.4f}, d={cohens_d:.3f} (n1={n1}, n2={n2})")

    # Bonferroni: correct within each protocol's comparison family only
    for protocol in results:
        n_family = max(len(results[protocol]), 1)
        for comp in results[protocol]:
            raw_p = results[protocol][comp]["t_pvalue"]
            m = max(n_family, 1)
            results[protocol][comp]["p_bonferroni"] = min(raw_p * m, 1.0)
            results[protocol][comp]["n_comparisons_family"] = m
            results[protocol][comp]["significant_bonferroni"] = results[protocol][comp]["p_bonferroni"] < 0.05
        if n_family > 1:
            logger.info(
                f"  Bonferroni ({protocol}): family of {n_family} comparisons, "
                f"α_adj per test ≈ 0.05/{n_family} = {0.05/n_family:.4f}"
            )
    return results


def compute_confidence_intervals(values: List[float], confidence: float = 0.95) -> Dict:
    """Compute confidence intervals using t-distribution."""
    n = len(values)
    mean = float(np.mean(values))
    if n <= 1:
        # Cannot compute CI with 0 or 1 sample
        return {"mean": mean, "std": float('nan'), "ci_lower": float('nan'), "ci_upper": float('nan')}
    std = float(np.std(values, ddof=1))
    if HAS_SCIPY:
        t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
    else:
        # Approximate t-critical for 95% CI using Cochran's formula
        # Exact for large n, conservative for small n
        t_crit = 1.96 + 2.4 / n if n > 2 else 12.706
    margin = t_crit * std / np.sqrt(n)
    return {"mean": mean, "std": std, "ci_lower": float(mean - margin), "ci_upper": float(mean + margin)}


# =============================================================================
# CRASHWORTHINESS METRICS - [CHANGE A] LC-specific displacement
# =============================================================================
def predict_curve_ensemble(models: List[nn.Module], approach: str, angle: float, lc: str,
                           disps: np.ndarray, scaler_disp: StandardScaler,
                           enc: OneHotEncoder, params: ScalingParams) -> Tuple:
    """Predict load-displacement curve using ensemble."""
    df = pd.DataFrame({"disp_mm": disps, "Angle": angle, "LC": lc})
    Xt = to_tensor(build_features(df, scaler_disp, enc))
    all_F, all_E = [], []
    
    for model in models:
        model.eval()
        if approach in ["ddns", "soft"]:
            with torch.no_grad():
                pv = model(Xt)
                all_F.append((pv[:, 0] * params.sig_F + params.mu_F).cpu().numpy())
                all_E.append((pv[:, 1] * params.sig_E + params.mu_E).cpu().numpy())
        else:
            Fb, Eb = hard_pinn_predict_load_energy(model, Xt, params)
            all_F.append(Fb)
            all_E.append(Eb)
    
    all_Fa = np.array(all_F)
    all_Ea = np.array(all_E)
    return (
        np.mean(all_Fa, axis=0),
        _ensemble_std_along_members(all_Fa),
        np.mean(all_Ea, axis=0),
        _ensemble_std_along_members(all_Ea),
    )


def predict_curve_best_member(models: List[nn.Module], approach: str, angle: float, lc: str,
                               disps: np.ndarray, scaler_disp: StandardScaler,
                               enc: OneHotEncoder, params: ScalingParams,
                               best_idx: int) -> Tuple:
    """Predict load-displacement curve using only the best ensemble member.
    
    Returns (F_best, E_best) - predictions from the single best model.
    Also returns ensemble std for uncertainty bands.
    """
    df = pd.DataFrame({"disp_mm": disps, "Angle": angle, "LC": lc})
    Xt = to_tensor(build_features(df, scaler_disp, enc))
    all_F, all_E = [], []
    
    for model in models:
        model.eval()
        if approach in ["ddns", "soft"]:
            with torch.no_grad():
                pv = model(Xt)
                all_F.append((pv[:, 0] * params.sig_F + params.mu_F).cpu().numpy())
                all_E.append((pv[:, 1] * params.sig_E + params.mu_E).cpu().numpy())
        else:
            Fb, Eb = hard_pinn_predict_load_energy(model, Xt, params)
            all_F.append(Fb)
            all_E.append(Eb)
    
    F_best = all_F[best_idx]
    E_best = all_E[best_idx]
    F_std = _ensemble_std_along_members(all_F)
    E_std = _ensemble_std_along_members(all_E)
    return F_best, F_std, E_best, E_std


def compute_ipf_robust(disps: np.ndarray, loads: np.ndarray, min_disp: float = 0.5, prom_frac: float = 0.05) -> Tuple[float, float]:
    """Compute Initial Peak Force robustly.

    Fallback: if no prominent peaks found after min_disp, use the maximum
    force in the first 25% of the displacement range (not the global max,
    which for progressive crushing could occur deep in the stroke).
    """
    peaks, _ = find_peaks(loads, prominence=prom_frac * (loads.max() - loads.min()))
    valid_peaks = [p for p in peaks if disps[p] > min_disp]
    if valid_peaks:
        idx = valid_peaks[0]
    else:
        # Restrict fallback to first 25% of displacement range
        d_max = disps[-1] if len(disps) > 0 else 1.0
        early_mask = disps <= max(d_max * 0.25, min_disp * 2)
        if np.any(early_mask):
            idx = int(np.argmax(loads[:np.sum(early_mask)]))
        else:
            idx = np.argmax(loads)
    return float(loads[idx]), float(disps[idx])


def compute_ea_ipf_ensemble(models: List[nn.Module], approach: str, angle: float, lc: str,
                             scaler_disp: StandardScaler, enc: OneHotEncoder, params: ScalingParams,
                             d_eval: float = None) -> Dict:
    """Compute EA and IPF using ensemble with per-member uncertainty propagation.

    FIX: The original derived EA and IPF from the ensemble-mean curve only,
    so EA_std/IPF_std were never populated.  This version computes EA_i and
    IPF_i independently for each ensemble member, then reports the mean and
    std across members.  This correctly propagates nonlinear uncertainty
    through peak-detection (IPF) and endpoint-extraction (EA).
    
    Parameters
    ----------
    d_eval : float, optional
        If specified, EA is computed at this displacement instead of ``d_end(lc)``.
        Used in inverse design for displacement-fair comparison across LCs by
        evaluating both at the same ``d_eval`` (typically ``D_COMMON``). The
        full curve is still predicted (for IPF and plotting); only the EA
        integration endpoint changes.
    """
    d_end = disp_end_mm(lc)
    n_steps = get_n_steps_curve(lc)
    disps = np.linspace(0, d_end, n_steps)

    # Build input tensor once
    df_tmp = pd.DataFrame({"disp_mm": disps, "Angle": angle, "LC": lc})
    Xt = to_tensor(build_features(df_tmp, scaler_disp, enc))

    all_F, all_E = [], []
    for model in models:
        model.eval()
        if approach in ["ddns", "soft"]:
            with torch.no_grad():
                pv = model(Xt)
                all_F.append((pv[:, 0] * params.sig_F + params.mu_F).cpu().numpy())
                all_E.append((pv[:, 1] * params.sig_E + params.mu_E).cpu().numpy())
        else:
            Fb, Eb = hard_pinn_predict_load_energy(model, Xt, params)
            all_F.append(Fb)
            all_E.append(Eb)

    all_F = np.array(all_F)  # (M, n_steps)
    all_E = np.array(all_E)

    # Ensemble-mean curves (backward-compatible, used for plotting)
    Fm = np.mean(all_F, axis=0)
    Fs = _ensemble_std_along_members(all_F)
    Em = np.mean(all_E, axis=0)
    Es = _ensemble_std_along_members(all_E)
    Em_corrected = Em - Em[0]

    # Per-member scalar metrics
    # When d_eval is specified (inverse design), compute EA at d_eval instead of d_end.
    # This ensures displacement-fair comparison: both LCs evaluated at same crush distance.
    # IPF is unaffected (occurs in first few mm, well before d_eval for any LC).
    d_metric = d_eval if (d_eval is not None and d_eval < d_end) else d_end
    if d_metric < d_end:
        # Interpolate energy at d_eval for each member
        ea_per_member = np.array([
            float(np.interp(d_metric, disps, all_E[i]) - all_E[i, 0])
            for i in range(len(models))
        ])
    else:
        ea_per_member = np.array([float(all_E[i, -1] - all_E[i, 0]) for i in range(len(models))])
    ipf_per_member = np.array([compute_ipf_robust(disps, all_F[i])[0] for i in range(len(models))])

    ea_mean = float(np.mean(ea_per_member))
    ipf_mean = float(np.mean(ipf_per_member))
    f_mean = ea_mean / d_metric if d_metric > 0 else 0.0
    cfe = f_mean / ipf_mean if ipf_mean > 1e-12 else 0.0
    _dd_m = 1 if len(models) > 1 else 0

    return {"EA": ea_mean, "IPF": ipf_mean,
            "EA_std": float(np.std(ea_per_member, ddof=_dd_m)),
            "IPF_std": float(np.std(ipf_per_member, ddof=_dd_m)),
            "F_mean": f_mean, "CFE": cfe,
            "disps": disps, "loads": Fm, "loads_std": Fs,
            "energies": Em_corrected, "energies_std": Es,
            "disp_end": d_end, "d_eval": d_metric,
            "ea_per_member": ea_per_member, "ipf_per_member": ipf_per_member}


def compute_design_space_metrics(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Compute EA and IPF from experimental data with LC-specific displacement ranges."""
    results = []
    n_skip = 0
    margin = float(getattr(CFG, "design_space_stroke_margin_mm", 2.0))
    for (lc, ang), group in df.groupby(["LC", "Angle"]):
        group = group.sort_values("disp_mm")
        d_end = disp_end_mm(lc)
        n_steps = get_n_steps_curve(lc)
        if group["disp_mm"].max() < d_end - margin:
            n_skip += 1
            continue
        disp_grid = np.linspace(0, d_end, n_steps)
        loads = np.interp(disp_grid, group["disp_mm"].values, group["load_kN"].values)
        energy = np.interp(disp_grid, group["disp_mm"].values, group["energy_J"].values)
        IPF, _ = compute_ipf_robust(disp_grid, loads)
        ea = float(energy[-1] - energy[0])
        f_mean = ea / d_end if d_end > 0 else 0.0
        cfe = f_mean / IPF if IPF > 1e-12 else 0.0
        results.append({"LC": lc, "Angle": ang, "EA": ea, "IPF": IPF,
                         "F_mean": f_mean, "CFE": cfe, "disp_end": d_end})
    if n_skip > 0 and logger is not None:
        logger.warning(
            "  compute_design_space_metrics: skipped %d (LC,Angle) groups with "
            "incomplete stroke (max disp < d_end - %.1f mm).",
            n_skip, margin,
        )
    return pd.DataFrame(results)


# =============================================================================
# ENSEMBLE CLASSIFIER FOR LOADING-CONDITION PLAUSIBILITY
# =============================================================================
# EA_common = energy absorbed to D_COMMON (see enrich_df_metrics_ea_common); aligns with inverse BO objective.
# Angle (deg) aligns the penalty with the candidate θ during BO: P(LC | θ, EA_common, IPF), not only marginal
# (EA, IPF), which better matches how the surrogate is queried along θ.
CLASSIFIER_FEATURES = ["EA_common", "IPF", "Angle"]


def _lc_label_to_binary(lc) -> int:
    """Map LC label to 0=LC1, 1=LC2 (explicit; avoids fragile ``'1' in str(lc)``)."""
    u = str(lc).upper().strip()
    if "LC2" in u:
        return 1
    return 0


def _make_lc_voting_classifier() -> VotingClassifier:
    """Shared soft-voting template for LC plausibility (keep in sync across train / lambda tuning)."""
    return VotingClassifier(
        estimators=[
            ("nb", GaussianNB(var_smoothing=1e-9)),
            ("svm", SVC(
                probability=True, kernel="rbf", gamma="scale",
                class_weight="balanced", random_state=CFG.seed,
            )),
            ("rf", RandomForestClassifier(
                n_estimators=100, random_state=CFG.seed, class_weight="balanced_subsample",
            )),
            ("mlp", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=5000,
                                  random_state=CFG.seed)),
        ],
        voting="soft",
    )


def train_lc_plausibility_classifier(
    df_metrics: pd.DataFrame, logger: logging.Logger
) -> Tuple[Any, StandardScaler, Dict]:
    """
    Train a calibrated soft-voting ensemble classifier that estimates
    P_ens(LC | EA, IPF).

    The classifier is used during inverse design to penalise candidate
    (theta, LC) pairs whose predicted (EA, IPF) are implausible for
    the proposed loading configuration.

    IMPORTANT -- Feature selection rationale:
    Features are ``EA_common`` (energy absorbed to ``D_COMMON`` for fair
    cross-LC comparison), ``IPF``, and ``Angle`` (degrees). The angle matches
    how inverse BO queries the surrogate along θ. F_mean / CFE were removed
    because ``d_end`` is LC-specific and would leak the label.

    Returns
    -------
    cal_ens : CalibratedClassifierCV
        Calibrated soft-voting ensemble.
    feat_scaler : StandardScaler
        Fitted on training [EA, IPF].
    diag : dict
        Diagnostic metrics (resubstitution, CV, per-learner).
    """
    logger.info("  Training LC plausibility ensemble classifier...")
    if "EA_common" not in df_metrics.columns:
        raise ValueError("train_lc_plausibility_classifier requires df_metrics['EA_common']; "
                         "call enrich_df_metrics_ea_common(df_metrics, df_all) first.")

    # --- Feature matrix and labels -------------------------------------------
    for c in CLASSIFIER_FEATURES:
        if c not in df_metrics.columns:
            raise ValueError(f"train_lc_plausibility_classifier: df_metrics missing column {c!r}")
    X_raw = df_metrics[CLASSIFIER_FEATURES].values.astype(np.float64)
    y = np.array([_lc_label_to_binary(lc) for lc in df_metrics["LC"].values], dtype=int)

    feat_scaler = StandardScaler()
    X_scaled = feat_scaler.fit_transform(X_raw)

    n_samples = len(y)
    n_lc0 = int((y == 0).sum())
    n_lc1 = int((y == 1).sum())
    logger.info(f"    Samples: {n_samples} (LC1={n_lc0}, LC2={n_lc1})")
    logger.info(f"    Features: {CLASSIFIER_FEATURES}")

    # --- Helper: build a fresh VotingClassifier instance ---------------------
    def _make_base():
        return _make_lc_voting_classifier()

    def _safe_inner_cv(y_train):
        """Compute inner CV folds ensuring >=2 and each fold has both classes."""
        min_class = int(min(np.sum(y_train == 0), np.sum(y_train == 1)))
        return max(2, min(5, min_class))

    # --- Base learners (for the final production model) ----------------------
    base_ens = _make_base()
    n_cv = max(2, min(5, min(n_lc0, n_lc1)))
    cal_ens = CalibratedClassifierCV(base_ens, method="sigmoid", cv=n_cv)

    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        cal_ens.fit(X_scaled, y)

    # --- Diagnostics: resubstitution (training) metrics ----------------------
    y_pred = cal_ens.predict(X_scaled)
    y_prob = cal_ens.predict_proba(X_scaled)
    acc = accuracy_score(y, y_pred)
    ll = log_loss(y, y_prob)
    brier = brier_score_loss(y, y_prob[:, 1])

    logger.info(f"    Resubstitution accuracy: {acc:.4f}")
    logger.info(f"    Log-loss: {ll:.4f},  Brier score: {brier:.4f}")

    # --- Cross-validation (LOO for small datasets, 5-fold otherwise) ---------
    if n_samples <= 30:
        from sklearn.model_selection import LeaveOneOut
        loo = LeaveOneOut()
        loo_preds = np.zeros(n_samples, dtype=int)
        loo_probs = np.zeros(n_samples)
        for train_idx, test_idx in loo.split(X_raw):
            # Refit scaler on train fold only to avoid data leakage
            _scaler = StandardScaler().fit(X_raw[train_idx])
            _X_tr = _scaler.transform(X_raw[train_idx])
            _X_te = _scaler.transform(X_raw[test_idx])
            _ens = CalibratedClassifierCV(
                _make_base(), method="sigmoid",
                cv=_safe_inner_cv(y[train_idx]))
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                _ens.fit(_X_tr, y[train_idx])
            loo_preds[test_idx[0]] = _ens.predict(_X_te)[0]
            loo_probs[test_idx[0]] = _ens.predict_proba(_X_te)[0, 1]
        cv_acc = accuracy_score(y, loo_preds)
        cv_brier = brier_score_loss(y, loo_probs)
        cv_method = "LOO"
        cv_y_store, cv_pred_store, cv_prob_store = y.copy(), loo_preds.copy(), loo_probs.copy()
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.seed)
        cv_preds = np.zeros(n_samples, dtype=int)
        cv_probs = np.zeros(n_samples)
        for train_idx, test_idx in skf.split(X_raw, y):
            # Refit scaler on train fold only to avoid data leakage
            _scaler = StandardScaler().fit(X_raw[train_idx])
            _X_tr = _scaler.transform(X_raw[train_idx])
            _X_te = _scaler.transform(X_raw[test_idx])
            _ens = CalibratedClassifierCV(
                _make_base(), method="sigmoid",
                cv=_safe_inner_cv(y[train_idx]))
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                _ens.fit(_X_tr, y[train_idx])
            cv_preds[test_idx] = _ens.predict(_X_te)
            cv_probs[test_idx] = _ens.predict_proba(_X_te)[:, 1]
        cv_acc = accuracy_score(y, cv_preds)
        cv_brier = brier_score_loss(y, cv_probs)
        cv_method = "5-fold stratified CV"
        cv_y_store, cv_pred_store, cv_prob_store = y.copy(), cv_preds.copy(), cv_probs.copy()

    mcc = float(matthews_corrcoef(y, cv_pred_store)) if len(np.unique(y)) > 1 else float("nan")
    logger.info(f"    {cv_method} accuracy: {cv_acc:.4f}, Brier: {cv_brier:.4f}, MCC: {mcc:.4f}")

    # --- Per-classifier accuracy via CV (NOT resubstitution) -----------------
    per_clf_acc = {}
    for name, clf_template in base_ens.estimators:
        if n_samples <= 30:
            from sklearn.model_selection import LeaveOneOut as _LOO
            _preds = np.zeros(n_samples, dtype=int)
            for train_idx, test_idx in _LOO().split(X_raw):
                _scaler = StandardScaler().fit(X_raw[train_idx])
                _X_tr = _scaler.transform(X_raw[train_idx])
                _X_te = _scaler.transform(X_raw[test_idx])
                _clf = type(clf_template)(**clf_template.get_params())
                _cal = CalibratedClassifierCV(
                    _clf, method="sigmoid",
                    cv=_safe_inner_cv(y[train_idx]))
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    _cal.fit(_X_tr, y[train_idx])
                _preds[test_idx[0]] = _cal.predict(_X_te)[0]
            per_clf_acc[name] = float(accuracy_score(y, _preds))
        else:
            _preds = np.zeros(n_samples, dtype=int)
            for train_idx, test_idx in StratifiedKFold(
                    n_splits=5, shuffle=True,
                    random_state=CFG.seed).split(X_raw, y):
                _scaler = StandardScaler().fit(X_raw[train_idx])
                _X_tr = _scaler.transform(X_raw[train_idx])
                _X_te = _scaler.transform(X_raw[test_idx])
                _clf = type(clf_template)(**clf_template.get_params())
                _cal = CalibratedClassifierCV(
                    _clf, method="sigmoid",
                    cv=_safe_inner_cv(y[train_idx]))
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    _cal.fit(_X_tr, y[train_idx])
                _preds[test_idx] = _cal.predict(_X_te)
            per_clf_acc[name] = float(accuracy_score(y, _preds))
    per_clf_acc["ensemble"] = float(cv_acc)
    logger.info(f"    Per-classifier CV accuracy: {per_clf_acc}")

    diag = {
        "resub_accuracy": acc,
        "resub_log_loss": ll,
        "resub_brier": brier,
        "cv_method": cv_method,
        "cv_accuracy": cv_acc,
        "cv_brier": cv_brier,
        "cv_mcc": mcc,
        "per_classifier_accuracy": per_clf_acc,
        "n_samples": n_samples,
        "n_features": len(CLASSIFIER_FEATURES),
        "feature_names": CLASSIFIER_FEATURES,
        "cv_y": cv_y_store,
        "cv_pred": cv_pred_store,
        "cv_prob_lc2": cv_prob_store,
    }

    return cal_ens, feat_scaler, diag


def compute_lc_penalty(
    cal_ens, feat_scaler: StandardScaler,
    ea: float, ipf: float,
    lc: str, prob_weight: float = 0.5,
    angle_deg: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Compute loading-consistency penalty: prob_weight * (-log p_LC).

    Uses the same feature vector as training (``CLASSIFIER_FEATURES``): by default
    ``EA_common``, ``IPF``, and ``Angle`` (degrees). Pass ``angle_deg`` for the BO
    candidate θ so the penalty matches P(LC | θ, EA, IPF).

    Returns:
        penalty: scaled negative log-likelihood
        p_lc: classifier probability for the proposed LC
    """
    n_in = int(getattr(feat_scaler, "n_features_in_", 2))
    if n_in >= 3:
        if angle_deg is None or not np.isfinite(angle_deg):
            raise ValueError("compute_lc_penalty: angle_deg required when classifier uses Angle feature")
        x = np.array([[ea, ipf, float(angle_deg)]], dtype=np.float64)
    else:
        x = np.array([[ea, ipf]], dtype=np.float64)
    x_scaled = feat_scaler.transform(x)
    proba = cal_ens.predict_proba(x_scaled)[0]  # [P(LC1), P(LC2)]
    lc_idx = _lc_label_to_binary(lc)
    p_lc = float(proba[lc_idx])
    phi = -np.log(max(p_lc, 1e-6))
    return float(prob_weight * phi), p_lc


def generate_classifier_diagnostics_table(
    cal_ens, feat_scaler: StandardScaler, diag: Dict,
    output_dir: str, logger: logging.Logger
) -> None:
    """Save classifier diagnostic table as CSV."""
    rows = []
    for name, acc in diag["per_classifier_accuracy"].items():
        rows.append({"Classifier": name.upper(), "CV_Accuracy": f"{acc:.4f}"})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "Table_classifier_per_learner.csv"), index=False)
    logger.info("  Saved: Table_classifier_per_learner.csv")

    summary = pd.DataFrame([{
        "Metric": "Resubstitution accuracy", "Value": f"{diag['resub_accuracy']:.4f}"},
        {"Metric": "Resubstitution log-loss", "Value": f"{diag['resub_log_loss']:.4f}"},
        {"Metric": "Resubstitution Brier score", "Value": f"{diag['resub_brier']:.4f}"},
        {"Metric": f"{diag['cv_method']} accuracy", "Value": f"{diag['cv_accuracy']:.4f}"},
        {"Metric": f"{diag['cv_method']} Brier score", "Value": f"{diag['cv_brier']:.4f}"},
        {"Metric": f"{diag['cv_method']} MCC", "Value": f"{diag.get('cv_mcc', float('nan')):.4f}"},
        {"Metric": "N samples", "Value": str(diag['n_samples'])},
        {"Metric": "N features", "Value": str(diag['n_features'])},
    ])
    summary.to_csv(os.path.join(output_dir, "Table_classifier_summary.csv"), index=False)
    logger.info("  Saved: Table_classifier_summary.csv")


def fig_classifier_decision_boundary(
    cal_ens, feat_scaler: StandardScaler, df_metrics: pd.DataFrame,
    output_dir: str, logger: logging.Logger
) -> None:
    """Plot classifier probability landscape in EA-IPF space."""
    ea_col = "EA_common" if "EA_common" in df_metrics.columns else "EA"
    ea_range = np.linspace(df_metrics[ea_col].min() * 0.8, df_metrics[ea_col].max() * 1.2, 100)
    ipf_range = np.linspace(df_metrics["IPF"].min() * 0.8, df_metrics["IPF"].max() * 1.2, 100)
    EA_grid, IPF_grid = np.meshgrid(ea_range, ipf_range)
    angle_ref = float(np.median(df_metrics["Angle"].astype(float).values))
    n_in = int(getattr(feat_scaler, "n_features_in_", 2))
    if n_in >= 3:
        ang_col = np.full(EA_grid.size, angle_ref, dtype=np.float64)
        X_grid = np.column_stack([EA_grid.ravel(), IPF_grid.ravel(), ang_col])
    else:
        X_grid = np.column_stack([EA_grid.ravel(), IPF_grid.ravel()])
    X_grid_scaled = feat_scaler.transform(X_grid)
    P_lc2 = cal_ens.predict_proba(X_grid_scaled)[:, 1].reshape(EA_grid.shape)

    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.5))

    # Panel (a): P(LC2 | indicators) heatmap
    ax = axes[0]
    im = ax.contourf(EA_grid, IPF_grid, P_lc2, levels=20, cmap="RdYlBu_r")
    ax.contour(EA_grid, IPF_grid, P_lc2, levels=[0.5], colors="black",
               linewidths=2, linestyles="--")
    for lc_val, marker, color, label in [("LC1", "o", "#1f77b4", "LC1"),
                                          ("LC2", "s", "#ff7f0e", "LC2")]:
        sub = df_metrics[df_metrics["LC"] == lc_val]
        ax.scatter(sub[ea_col], sub["IPF"], c=color, marker=marker,
                   edgecolors="black", s=80, zorder=5, label=label)
    ax.set_xlabel("Energy absorption to {:.0f} mm (J)".format(D_COMMON))
    ax.set_ylabel("Initial Peak Force IPF (kN)")
    ang_note = f" (slice at θ={angle_ref:.1f}°)" if n_in >= 3 else ""
    ax.set_title(f"(a) P(LC2 | features){ang_note}")
    ax.legend(loc="best")
    plt.colorbar(im, ax=ax, label="P(LC2)")

    # Panel (b): Penalty landscape
    ax2 = axes[1]
    Phi_lc1 = -np.log(np.maximum(1.0 - P_lc2, 1e-6))  # penalty if LC=LC1
    Phi_lc2 = -np.log(np.maximum(P_lc2, 1e-6))          # penalty if LC=LC2
    Phi_min = np.minimum(Phi_lc1, Phi_lc2)  # best-case penalty
    im2 = ax2.contourf(EA_grid, IPF_grid, Phi_min, levels=20, cmap="YlOrRd")
    ax2.contour(EA_grid, IPF_grid, P_lc2, levels=[0.5], colors="black",
                linewidths=2, linestyles="--")
    for lc_val, marker, color, label in [("LC1", "o", "#1f77b4", "LC1"),
                                          ("LC2", "s", "#ff7f0e", "LC2")]:
        sub = df_metrics[df_metrics["LC"] == lc_val]
        ax2.scatter(sub[ea_col], sub["IPF"], c=color, marker=marker,
                    edgecolors="black", s=80, zorder=5, label=label)
    ax2.set_xlabel("Energy absorption to {:.0f} mm (J)".format(D_COMMON))
    ax2.set_ylabel("Initial Peak Force IPF (kN)")
    ax2.set_title(f"(b) min-LC Penalty $\\Phi${ang_note}")
    ax2.legend(loc="best")
    plt.colorbar(im2, ax=ax2, label="$\\Phi$ (lower = more plausible)")

    fig.suptitle("Ensemble Classifier: Loading-Condition Plausibility",
                 fontweight="bold", y=0.98)
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.88])
    fig.savefig(os.path.join(output_dir, "Fig_classifier_decision_boundary.png"),
                dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: Fig_classifier_decision_boundary.png")


def auto_tune_lambda(
    cal_ens, feat_scaler: StandardScaler,
    df_metrics: pd.DataFrame, logger: logging.Logger,
    alpha: float = 0.20,
    convergence_pct: float = 10.0
) -> Tuple[float, Dict]:
    """
    Automatically tune the classifier penalty weight (lambda) using
    LOO cross-validated penalties on the training data.

    Principle
    ---------
    The penalty for the CORRECT LC at optimizer convergence should be
    a fixed fraction (alpha) of the expected converged fit error.
    If the fit error at convergence_pct % relative error on both EA
    and IPF is:

        fit_conv = 2 * (convergence_pct / 100)^2

    then lambda is chosen so that:

        lambda * mean_phi_correct = alpha * fit_conv

    This ensures the penalty is a mild regularizer (not dominant)
    for correct-LC solutions, while being ~5x larger for wrong-LC
    solutions (empirically measured discrimination ratio).

    Parameters
    ----------
    alpha : float
        Fraction of converged fit error allocated to penalty (default 0.20).
    convergence_pct : float
        Expected relative error (%) at optimizer convergence (default 10%).

    Returns
    -------
    lambda_opt : float
        Optimally tuned penalty weight.
    tuning_diag : dict
        Diagnostics: LOO penalties, discrimination ratio, etc.
    """
    logger.info("  Auto-tuning classifier penalty weight (lambda)...")

    import warnings as _w
    X_raw = df_metrics[CLASSIFIER_FEATURES].values.astype(np.float64)
    y = np.array([_lc_label_to_binary(lc) for lc in df_metrics["LC"].values], dtype=int)
    n = len(y)

    # LOO cross-validated probabilities (scaler refit on train fold only; matches training CV)
    from sklearn.model_selection import LeaveOneOut
    loo_probs = np.zeros(n)
    for train_idx, test_idx in LeaveOneOut().split(X_raw):
        _scaler = StandardScaler().fit(X_raw[train_idx])
        X_tr = _scaler.transform(X_raw[train_idx])
        X_te = _scaler.transform(X_raw[test_idx])
        _base = _make_lc_voting_classifier()
        min_class = int(min(np.sum(y[train_idx] == 0), np.sum(y[train_idx] == 1)))
        _cv = max(2, min(5, min_class))
        _ens = CalibratedClassifierCV(_base, method="sigmoid", cv=_cv)
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            _ens.fit(X_tr, y[train_idx])
        loo_probs[test_idx[0]] = _ens.predict_proba(X_te)[0, 1]

    # Penalties for correct and wrong LC
    p_correct = np.where(y == 1, loo_probs, 1.0 - loo_probs)
    p_wrong = 1.0 - p_correct
    phi_correct = -np.log(np.maximum(p_correct, 1e-6))
    phi_wrong = -np.log(np.maximum(p_wrong, 1e-6))

    mean_phi_correct = float(phi_correct.mean())
    mean_phi_wrong = float(phi_wrong.mean())
    discrim_ratio = mean_phi_wrong / mean_phi_correct if mean_phi_correct > 1e-12 else float("inf")

    # Derive optimal lambda
    fit_conv = 2.0 * (convergence_pct / 100.0) ** 2
    lambda_opt = alpha * fit_conv / mean_phi_correct if mean_phi_correct > 1e-12 else 0.01

    logger.info(f"    LOO mean penalty (correct LC): {mean_phi_correct:.4f}")
    logger.info(f"    LOO mean penalty (wrong LC):   {mean_phi_wrong:.4f}")
    logger.info(f"    Discrimination ratio:          {discrim_ratio:.2f}x")
    logger.info(f"    Assumed convergence:            {convergence_pct:.0f}% -> fit_conv={fit_conv:.6f}")
    logger.info(f"    Alpha (penalty/fit fraction):   {alpha:.2f}")
    logger.info(f"    >>> Tuned lambda = {lambda_opt:.4f}")
    logger.info(f"    At convergence: correct-LC penalty = {lambda_opt * mean_phi_correct:.6f} "
                f"({lambda_opt * mean_phi_correct / fit_conv * 100:.1f}% of fit)")
    logger.info(f"                    wrong-LC penalty   = {lambda_opt * mean_phi_wrong:.6f} "
                f"({lambda_opt * mean_phi_wrong / fit_conv * 100:.1f}% of fit)")

    tuning_diag = {
        "lambda_opt": lambda_opt,
        "mean_phi_correct": mean_phi_correct,
        "mean_phi_wrong": mean_phi_wrong,
        "discrimination_ratio": discrim_ratio,
        "convergence_pct": convergence_pct,
        "alpha": alpha,
        "fit_conv": fit_conv,
        "loo_probs": loo_probs,
        "phi_correct": phi_correct,
        "phi_wrong": phi_wrong,
    }
    return lambda_opt, tuning_diag


def auto_tune_beta(
    models: List[nn.Module], approach: str,
    df_metrics: pd.DataFrame,
    scaler_disp: StandardScaler, enc: OneHotEncoder, params: ScalingParams,
    logger: logging.Logger,
    fraction: float = 0.25
) -> float:
    """Auto-tune beta_robust so uncertainty penalty is ~fraction of fit_error at observed data.

    Evaluates ensemble on observed (angle, LC) pairs, computes the median
    fit_error and median (sigma_EA^2 + sigma_IPF^2), then sets:
        beta = fraction * median_fit_error / median_unc

    Fit error uses EA at ``D_COMMON`` compared to ``df_metrics['EA_common']`` so
    scaling matches the inverse-design objective (requires ``EA_common`` column).
    """
    if "EA_common" not in df_metrics.columns:
        raise ValueError("auto_tune_beta requires df_metrics['EA_common']; call enrich_df_metrics_ea_common first.")
    logger.info("  Auto-tuning beta_robust (ensemble uncertainty penalty weight)...")
    fit_errors = []
    unc_terms = []
    for _, row in df_metrics.iterrows():
        m = compute_ea_ipf_ensemble(models, approach, float(row["Angle"]), str(row["LC"]),
                                     scaler_disp, enc, params, d_eval=D_COMMON)
        ea_ref = float(row["EA_common"])
        w_ea = 1.0 / (ea_ref**2 + 1e-12)
        w_ipf = 1.0 / (row["IPF"]**2 + 1e-12)
        fe = w_ea * (m["EA"] - ea_ref) ** 2 + w_ipf * (m["IPF"] - row["IPF"]) ** 2
        unc = m.get("EA_std", 0)**2 + m.get("IPF_std", 0)**2
        fit_errors.append(fe)
        unc_terms.append(unc)

    med_fe = float(np.median(fit_errors))
    med_unc = float(np.median(unc_terms))
    if med_unc < 1e-12:
        logger.warning("    Ensemble uncertainty near zero, setting beta_robust=0")
        return 0.0

    beta = fraction * med_fe / med_unc
    logger.info(f"    beta_robust = {beta:.6f}")
    logger.info(f"    median fit_error = {med_fe:.6f}, median unc = {med_unc:.6f}, fraction = {fraction}")
    return beta


def run_lambda_sensitivity(
    models: List[nn.Module], approach: str,
    targets: List[Dict],
    scaler_disp: StandardScaler, enc: OneHotEncoder, params: ScalingParams,
    bo_cfg: BOConfig, cal_ens, feat_scaler: StandardScaler,
    output_dir: str, logger: logging.Logger
) -> pd.DataFrame:
    """
    Run inverse design at multiple lambda values and report how
    LC selection, prediction accuracy, and p(LC) vary.

    Tests lambda in {0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5}.
    For each lambda, runs the first target only (to keep runtime manageable)
    using joint GP-BO (skopt), same as the main inverse pipeline.
    """
    logger.info("  Running lambda sensitivity analysis...")
    lambda_values = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5]
    target = targets[0]  # first target only

    rows = []
    for lam in lambda_values:
        # Create a temporary BOConfig with this lambda
        _cfg = BOConfig(
            n_calls_total=bo_cfg.n_calls_total,
            n_init=bo_cfg.n_init,
            xi=bo_cfg.xi,
            n_candidates=bo_cfg.n_candidates,
            gp_restarts=bo_cfg.gp_restarts,
            seed=bo_cfg.seed,
            prob_weight=lam,
            beta_robust=getattr(bo_cfg, 'beta_robust', 0.0),
            run_classifier_ablation=False,
            lambda_sweep=False,
        )
        _cal = cal_ens if lam > 0 else None
        _fs = feat_scaler if lam > 0 else None
        res = run_inverse_design(
            models, approach, target["EA"], target["IPF"],
            scaler_disp, enc, params, _cfg, logger,
            cal_ens=_cal, feat_scaler=_fs)
        for method in ["gpbo"]:
            key = f"{method}_best"
            if key in res:
                b = res[key]
                lc_sel = b.get("lc", b.get("best_lc", ""))
                ang = b["x_best"]
                # Compute p(LC) at the selected point
                m = compute_ea_ipf_ensemble(
                    models, approach, ang, lc_sel, scaler_disp, enc, params,
                    d_eval=D_COMMON)
                _, p_lc = compute_lc_penalty(
                    cal_ens, feat_scaler,
                    m["EA"], m["IPF"],
                    lc_sel, prob_weight=0.0,  # just get p_lc
                    angle_deg=float(ang),
                )
                rows.append({
                    "lambda": lam,
                    "method": method.upper(),
                    "angle": f"{ang:.1f}",
                    "LC": lc_sel,
                    "EA_err%": f"{b.get('ea_error_pct', float('nan')):.2f}",
                    "IPF_err%": f"{b.get('ipf_error_pct', float('nan')):.2f}",
                    "p_LC": f"{p_lc:.4f}",
                    "y_best": f"{b['y_best']:.6f}",
                })

    df_sweep = pd.DataFrame(rows)
    df_sweep.to_csv(os.path.join(output_dir, "Table_lambda_sensitivity.csv"), index=False)
    logger.info("  Saved: Table_lambda_sensitivity.csv")
    logger.info(f"\n{df_sweep.to_string(index=False)}")
    return df_sweep


# =============================================================================
# [CHANGE D] ABLATION STUDY - Now on UNSEEN ANGLE protocol
# =============================================================================
def run_ablation_study(train_df: pd.DataFrame, val_df: pd.DataFrame,
                        scaler_disp: StandardScaler, scaler_out: StandardScaler,
                        enc: OneHotEncoder, params: ScalingParams,
                        protocol: str, logger: logging.Logger) -> pd.DataFrame:
    """Run ablation study on physics weight w_phys on unseen angle protocol."""
    logger.info(f"  Running ablation study on w_phys (protocol: {protocol})...")
    w_phys_values = [0.0, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    results = []
    for w in w_phys_values:
        model, hist, r2, meta = train_soft(train_df, val_df, scaler_disp, scaler_out, enc, params, CFG.seed_base, protocol, logger, w_phys_override=w)
        metrics = evaluate_model(model, "soft", val_df, scaler_disp, scaler_out, enc, params)
        results.append({"w_phys": w, "load_r2": metrics["load_r2"], "energy_r2": metrics["energy_r2"],
                       "load_rmse": metrics["load_rmse"], "energy_rmse": metrics["energy_rmse"],
                       "training_time": meta["training_time"], "protocol": protocol})
        logger.info(f"    w_phys={w:.1f}: R²_load={metrics['load_r2']:.4f}")
    return pd.DataFrame(results)


# =============================================================================
# OPTIMIZATION FUNCTIONS
# =============================================================================
def _norm_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal CDF."""
    try:
        from scipy.special import erf
        return 0.5 * (1 + erf(z / np.sqrt(2)))
    except ImportError:
        return 0.5 * (1 + np.tanh(z * 0.797884560802865))


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    """Expected Improvement acquisition function."""
    sigma = np.maximum(sigma, 1e-12)
    imp = best - mu - xi
    Z = imp / sigma
    pdf = np.exp(-0.5 * Z**2) / np.sqrt(2 * np.pi)
    return imp * _norm_cdf(Z) + sigma * pdf


def gp_bo_minimize(f: Callable, bounds: Tuple[float, float], bo_cfg: BOConfig, logger: logging.Logger) -> Dict:
    """GP-BO minimization with tracking (single LC version, kept for compatibility)."""
    if not HAS_SKLEARN_GP:
        raise RuntimeError("sklearn GP not available")
    rng = np.random.default_rng(bo_cfg.seed)
    lo, hi = bounds
    X = rng.uniform(lo, hi, (bo_cfg.n_init, 1))
    y = np.array([f(float(x)) for x in X[:, 0]])
    x_history, y_history = list(X[:, 0]), list(y)
    best_y_history = [float(np.min(y))]
    posterior_snapshots = []
    kernel = ConstantKernel(1.0) * Matern(5.0, nu=2.5) + WhiteKernel(1e-5)
    theta_dense = np.linspace(lo, hi, 200).reshape(-1, 1)
    
    for it in range(bo_cfg.n_calls_total - bo_cfg.n_init):
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=bo_cfg.gp_restarts, random_state=bo_cfg.seed + it, alpha=1e-10)
        gp.fit(X, y)
        mu_dense, sigma_dense = gp.predict(theta_dense, return_std=True)
        posterior_snapshots.append({"iteration": it + 1, "theta_grid": theta_dense.flatten().copy(), "mu": mu_dense.copy(), "sigma": sigma_dense.copy(), "X_obs": X[:, 0].copy(), "y_obs": y.copy()})
        candidates = np.vstack([rng.uniform(lo, hi, size=(bo_cfg.n_candidates, 1)), theta_dense])
        mu_cand, sigma_cand = gp.predict(candidates, return_std=True)
        ei = expected_improvement(mu_cand, sigma_cand, np.min(y), bo_cfg.xi)
        order = np.argsort(-ei)
        x_next = None
        for idx in order[:500]:
            cand = float(candidates[idx, 0])
            if np.min(np.abs(X[:, 0] - cand)) > 0.1:
                x_next = cand
                break
        if x_next is None:
            x_next = float(rng.uniform(lo, hi))
        y_next = f(x_next)
        X = np.vstack([X, [[x_next]]])
        y = np.append(y, y_next)
        x_history.append(x_next)
        y_history.append(y_next)
        best_y_history.append(float(np.min(y)))
    
    i_best = int(np.argmin(y))
    return {"x_best": float(X[i_best, 0]), "y_best": float(y[i_best]), "x_history": np.array(x_history),
            "y_history": np.array(y_history), "best_y_history": np.array(best_y_history),
            "posterior_snapshots": posterior_snapshots, "n_evals": len(y_history)}


def gp_bo_minimize_joint(objective_funcs: Dict[str, Callable], bounds: Tuple[float, float], 
                          bo_cfg: BOConfig, logger: logging.Logger) -> Dict:
    """
    Joint GP-BO minimization over (theta, LC) using skopt.gp_minimize.
    
    [V8] Uses scikit-optimize's gp_minimize with a joint search space:
      - theta: Real dimension (continuous angle)
      - LC: Categorical dimension (LC1 or LC2)
    
    A single GP with Matern kernel shares information across both LCs,
    producing well-resolved posteriors for both loading conditions. This
    solves the LC starvation problem where independent GPs caused all
    sequential evaluations to go to the "interesting" LC while ignoring
    the other.
    
    skopt stores the GP surrogate at every iteration (res.models), enabling
    clean posterior evolution visualizations.
    """
    if not HAS_SKOPT:
        raise RuntimeError("scikit-optimize (skopt) not available. "
                           "Install with: pip install scikit-optimize")
    
    lo, hi = bounds
    lc_list = sorted(objective_funcs.keys())
    lc_to_idx = {lc: i for i, lc in enumerate(lc_list)}
    idx_to_lc = {i: lc for lc, i in lc_to_idx.items()}
    
    # Build joint objective: skopt passes [theta, lc_index] as arguments
    def joint_objective(params):
        theta = float(params[0])
        lc_idx = int(params[1])
        lc = idx_to_lc[lc_idx]
        return float(objective_funcs[lc](theta))
    
    # Search space: continuous theta + categorical LC index
    space = [Real(lo, hi, name="theta"),
             Categorical(list(range(len(lc_list))), name="lc")]
    
    # Run skopt GP-BO (stores models at each iteration)
    res = skopt_gp_minimize(
        joint_objective, space,
        n_calls=bo_cfg.n_calls_total,
        n_initial_points=bo_cfg.n_init,
        random_state=bo_cfg.seed,
        acq_func="EI",
        xi=bo_cfg.xi,
    )
    
    # --- Reconstruct posterior snapshots from res.models ---
    theta_dense = np.linspace(lo, hi, 200)
    n_models = len(getattr(res, "models", []))
    
    # Select snapshot iterations (evenly spaced, always include first and last)
    if n_models >= 8:
        step = (n_models - 1) / 7.0
        snap_model_indices = [int(round(i * step)) for i in range(8)]
        snap_model_indices[-1] = n_models - 1
    else:
        snap_model_indices = list(range(n_models))
    
    posterior_snapshots = []
    for k in snap_model_indices:
        if k >= n_models:
            continue
        model = res.models[k]
        # Model k was fitted on (n_init + k) observations
        n_obs_k = min(bo_cfg.n_init + k, len(res.x_iters))
        
        snapshot = {"iteration": n_obs_k, "theta_grid": theta_dense.copy()}
        
        # Get observations up to what this model was fitted on
        xs_fit = np.array(res.x_iters[:n_obs_k], dtype=object)
        ys_fit = np.array(res.func_vals[:n_obs_k], dtype=float)
        
        for lc in lc_list:
            lc_idx = lc_to_idx[lc]
            
            # Predict posterior for this LC across theta_dense
            X_pred = [[float(t), lc_idx] for t in theta_dense]
            X_pred_t = res.space.transform(X_pred)
            mu, sigma = model.predict(X_pred_t, return_std=True)
            snapshot[f"mu_{lc}"] = mu.copy()
            snapshot[f"sigma_{lc}"] = sigma.copy()
            
            # Extract observations for this LC up to iteration k
            mask = np.array([int(x[1]) == lc_idx for x in xs_fit], dtype=bool)
            if np.any(mask):
                snapshot[f"X_obs_{lc}"] = np.array(
                    [float(x[0]) for x in xs_fit[mask]], dtype=float)
                snapshot[f"y_obs_{lc}"] = ys_fit[mask].copy()
            else:
                snapshot[f"X_obs_{lc}"] = np.array([], dtype=float)
                snapshot[f"y_obs_{lc}"] = np.array([], dtype=float)
        
        posterior_snapshots.append(snapshot)
    
    # --- Compute true objective landscapes ---
    true_landscapes = {}
    for lc in lc_list:
        true_landscapes[lc] = np.array(
            [objective_funcs[lc](float(t)) for t in theta_dense])
    
    # --- Extract per-LC results ---
    all_x = np.array(res.x_iters, dtype=object)
    all_y = np.array(res.func_vals, dtype=float)
    
    results_by_lc = {}
    for lc in lc_list:
        lc_idx = lc_to_idx[lc]
        mask = np.array([int(x[1]) == lc_idx for x in all_x], dtype=bool)
        if np.any(mask):
            thetas_lc = np.array([float(x[0]) for x in all_x[mask]])
            ys_lc = all_y[mask]
            ib = int(np.argmin(ys_lc))
            results_by_lc[lc] = {
                "x_best": float(thetas_lc[ib]),
                "y_best": float(ys_lc[ib]),
                "x_history": thetas_lc,
                "y_history": ys_lc,
                "lc": lc,
            }
    
    # --- Flat history ---
    all_x_hist = [(float(x[0]), int(x[1])) for x in all_x]
    
    # --- Best overall ---
    i_best = int(np.argmin(all_y))
    best_theta = float(all_x[i_best][0])
    best_lc = idx_to_lc[int(all_x[i_best][1])]
    
    best_y_history = list(np.minimum.accumulate(all_y))
    
    # Count per-LC evaluations for logging
    for lc in lc_list:
        n_lc = sum(1 for x in all_x if int(x[1]) == lc_to_idx[lc])
        logger.info(f"    GP-BO (skopt): {lc} received {n_lc}/{len(all_x)} evaluations")
    
    return {
        "x_best": best_theta,
        "y_best": float(all_y[i_best]),
        "best_lc": best_lc,
        "x_history": all_x_hist,
        "y_history": all_y,
        "best_y_history": np.array(best_y_history),
        "posterior_snapshots": posterior_snapshots,
        "n_evals": len(all_y),
        "lc_list": lc_list,
        "lc_to_idx": lc_to_idx,
        "results_by_lc": results_by_lc,
        "true_landscapes": true_landscapes,
        "theta_dense": theta_dense,
    }









def compute_bo_convergence_diagnostics(
    gpbo_result: Dict, bo_cfg: BOConfig, logger: logging.Logger
) -> Dict:
    """Compute convergence diagnostics for GP-BO optimization.

    Metrics:
    1. Relative improvement over last N evaluations (stagnation check)
    2. Budget utilization: evaluations to reach 95% of final improvement
    3. GP posterior std at optimum (confidence in solution)
    """
    best_y_hist = np.array(gpbo_result.get("best_y_history", []))
    n_evals = len(best_y_hist)

    if n_evals < 5:
        return {"converged": False, "reason": "too_few_evaluations"}

    diag = {}

    # 1. Relative improvement over last N=10 evaluations
    N_tail = min(10, n_evals // 3)
    y_final = best_y_hist[-1]
    y_at_tail_start = best_y_hist[-N_tail]
    if abs(y_at_tail_start) > 1e-12:
        rel_improvement_tail = (y_at_tail_start - y_final) / abs(y_at_tail_start)
    else:
        rel_improvement_tail = 0.0
    diag["rel_improvement_last_N"] = float(rel_improvement_tail)
    diag["N_tail"] = N_tail
    diag["stagnated"] = rel_improvement_tail < 0.01

    # 2. Budget utilization: eval index where 95% of total improvement is reached
    y_init = best_y_hist[0]
    total_improvement = y_init - y_final
    if total_improvement > 1e-12:
        threshold_95 = y_init - 0.95 * total_improvement
        meets = best_y_hist <= threshold_95
        if np.any(meets):
            idx_95 = int(np.argmax(meets))
        else:
            idx_95 = n_evals - 1
        diag["evals_to_95pct"] = idx_95 + 1
        diag["budget_utilization"] = (idx_95 + 1) / n_evals
    else:
        diag["evals_to_95pct"] = 1
        diag["budget_utilization"] = 1.0 / n_evals

    # 3. GP posterior std at optimum (from last snapshot)
    snapshots = gpbo_result.get("posterior_snapshots", [])
    if snapshots:
        last_snap = snapshots[-1]
        best_theta = gpbo_result.get("x_best", 0)
        best_lc = gpbo_result.get("best_lc", gpbo_result.get("lc", ""))
        theta_grid = last_snap.get("theta_grid", np.array([]))
        sigma_key = f"sigma_{best_lc}"
        if sigma_key in last_snap and len(theta_grid) > 0:
            idx_nearest = int(np.argmin(np.abs(theta_grid - best_theta)))
            diag["gp_std_at_optimum"] = float(last_snap[sigma_key][idx_nearest])

    logger.info(f"    Convergence: rel_improve_last{N_tail}={rel_improvement_tail:.4f} "
                f"({'stagnated' if diag['stagnated'] else 'improving'}), "
                f"95% at eval {diag.get('evals_to_95pct', '?')}/{n_evals} "
                f"(util={diag.get('budget_utilization', 0):.0%})"
                + (f", GP_std@opt={diag['gp_std_at_optimum']:.6f}" if 'gp_std_at_optimum' in diag else ""))

    return diag


def _inverse_coarse_grid_fallback(
    objective_funcs: Dict[str, Callable[[float], float]],
    bounds: Tuple[float, float],
    lc_categories: List[str],
    logger: logging.Logger,
) -> Tuple[str, Dict]:
    """Cheap inverse for ``CFG.dry_run`` when GP-BO is disabled (no skopt required)."""
    lo, hi = bounds
    thetas = np.linspace(lo, hi, max(9, int(round(hi - lo)) + 1))
    best_y = float("inf")
    best_theta = float(lo)
    best_lc = lc_categories[0]
    for lc in lc_categories:
        fn = objective_funcs[lc]
        for th in thetas:
            yv = float(fn(float(th)))
            if yv < best_y:
                best_y, best_theta, best_lc = yv, float(th), lc
    res = {
        "x_best": best_theta,
        "y_best": best_y,
        "func_vals": np.array([best_y]),
        "x_iters": [],
        "posterior_snapshots": [],
        "lc": best_lc,
        "best_lc": best_lc,
    }
    logger.info(
        f"    Coarse-grid inverse (dry run): LC={best_lc}, theta={best_theta:.2f} deg, J={best_y:.4g}"
    )
    return best_lc, res


def _make_objective(models, approach, lc, scaler_disp, enc, params, target_ea, target_ipf, w_ea, w_ipf,
                    cal_ens=None, feat_scaler=None, prob_weight=0.0, d_eval=None,
                    beta_robust=0.0, gamma_tikhonov=0.0, theta_center=57.5):
    """Factory function for objective closure with optional LC plausibility and robustness penalties.

    Objective:
        J = w_ea*(EA@d_eval - EA_target)^2 + w_ipf*(IPF - IPF_target)^2
            + prob_weight*(-log p_LC)
            + beta_robust*(sigma_EA^2 + sigma_IPF^2)
            + gamma_tikhonov*(theta - theta_center)^2

    The Tikhonov regularization term addresses ill-posedness by preferring
    solutions near the center of the design space (where data density is highest).
    """
    def objective(angle: float) -> float:
        m = compute_ea_ipf_ensemble(models, approach, float(angle), lc, scaler_disp, enc, params,
                                     d_eval=d_eval)
        fit_error = float(w_ea * (m["EA"] - target_ea)**2 + w_ipf * (m["IPF"] - target_ipf)**2)

        # Ensemble uncertainty penalty (robust design)
        unc_penalty = 0.0
        if beta_robust > 0:
            unc_penalty = float(beta_robust * (m.get("EA_std", 0)**2 + m.get("IPF_std", 0)**2))

        # Tikhonov regularization (ill-posedness)
        tikh_penalty = 0.0
        if gamma_tikhonov > 0:
            tikh_penalty = float(gamma_tikhonov * (float(angle) - theta_center) ** 2)

        if cal_ens is not None and prob_weight > 0:
            penalty, _ = compute_lc_penalty(
                cal_ens, feat_scaler, m["EA"], m["IPF"], lc, prob_weight,
                angle_deg=float(angle),
            )
            return fit_error + penalty + unc_penalty + tikh_penalty
        return fit_error + unc_penalty + tikh_penalty
    return objective



def run_inverse_design(models: List[nn.Module], approach: str, target_ea: float, target_ipf: float,
                       scaler_disp: StandardScaler, enc: OneHotEncoder, params: ScalingParams,
                       bo_cfg: BOConfig, logger: logging.Logger,
                       cal_ens=None, feat_scaler=None,
                       gamma_tikhonov_override: Optional[float] = None,
                       prob_weight_override: Optional[float] = None,
                       beta_robust_override: Optional[float] = None) -> Dict:
    """Run inverse design comparison with automatic LC selection.
    
    Uses EA evaluated at ``D_COMMON`` for both LCs to ensure
    displacement-fair comparison. ``target_ea`` must be EA absorbed to
    ``D_COMMON`` for both LC1 and LC2.

    Parameters
    ----------
    target_ea : float
        Target energy absorption (J) to ``D_COMMON`` for both LCs.
    target_ipf : float
        Target initial peak force (kN).
    gamma_tikhonov_override, prob_weight_override, beta_robust_override : optional
        When set, override auto-tuned Tikhonov / classifier / robustness weights (ablations).
    """
    lc_categories = [str(x) for x in enc.categories_[0].tolist()]
    bounds = (CFG.angle_opt_min, CFG.angle_opt_max)
    w_ea = 1.0 / (target_ea**2 + 1e-12)
    w_ipf = 1.0 / (target_ipf**2 + 1e-12)
    prob_weight = bo_cfg.prob_weight if cal_ens is not None else 0.0
    if prob_weight_override is not None:
        prob_weight = float(prob_weight_override) if cal_ens is not None else 0.0
    beta_robust = getattr(bo_cfg, 'beta_robust', 0.0)
    if beta_robust_override is not None:
        beta_robust = float(beta_robust_override)
    results = {"gpbo": {}, "gpbo_joint": None,
               "target_ea": target_ea, "target_ipf": target_ipf,
               "prob_weight": prob_weight, "beta_robust": beta_robust}
    # Note: Only GP-BO is used for inverse design (grid/DE/DA removed in v18)
    
    def _enrich(res_dict, lc):
        """Populate prediction metrics on an optimizer result."""
        m = compute_ea_ipf_ensemble(models, approach, res_dict["x_best"], lc, scaler_disp, enc, params,
                                     d_eval=D_COMMON)
        res_dict["pred_ea"], res_dict["pred_ipf"] = m["EA"], m["IPF"]
        res_dict["pred_ea_std"], res_dict["pred_ipf_std"] = m.get("EA_std", 0), m.get("IPF_std", 0)
        res_dict["ea_error_pct"] = abs(m["EA"] - target_ea) / target_ea * 100
        res_dict["ipf_error_pct"] = abs(m["IPF"] - target_ipf) / target_ipf * 100
        # Also compute full-range EA for informational reporting
        m_full = compute_ea_ipf_ensemble(models, approach, res_dict["x_best"], lc, scaler_disp, enc, params)
        res_dict["pred_ea_full"] = m_full["EA"]
        res_dict["d_eval"] = D_COMMON
    
    # Auto-tune Tikhonov regularization (unless caller overrides, e.g. ablation gamma=0)
    gamma_tikh = getattr(bo_cfg, 'gamma_tikhonov', 0.0)
    theta_ctr = getattr(bo_cfg, 'theta_center', 57.5)
    if gamma_tikhonov_override is not None:
        gamma_tikh = float(gamma_tikhonov_override)
    elif gamma_tikh <= 0:
        # Set so penalty at angle boundary is ~10% of typical fit error (normalized objectives)
        gamma_tikh = 0.1 / max(1.0, (CFG.angle_opt_max - theta_ctr) ** 2)
    logger.info(f"    Tikhonov regularization: gamma={gamma_tikh:.6f}, center={theta_ctr:.1f}")

    # Create objective functions for each LC (evaluated at D_COMMON)
    objective_funcs = {}
    for lc in lc_categories:
        objective_funcs[lc] = _make_objective(
            models, approach, lc, scaler_disp, enc, params,
            target_ea, target_ipf, w_ea, w_ipf,
            cal_ens=cal_ens, feat_scaler=feat_scaler, prob_weight=prob_weight,
            d_eval=D_COMMON, beta_robust=beta_robust,
            gamma_tikhonov=gamma_tikh, theta_center=theta_ctr
        )

    # Run JOINT GP-BO over [theta, LC] space (multi-start for global optimality)
    if CFG.run_gpbo and HAS_SKOPT:
        t0 = time.time()
        n_restarts = getattr(bo_cfg, 'n_bo_restarts', 5)
        try:
            gpbo_joint_res = gp_bo_minimize_joint_multistart(
                objective_funcs, bounds, bo_cfg, logger, n_restarts=n_restarts)
            gpbo_joint_res["wall_time"] = time.time() - t0

            # Convergence diagnostics
            conv_diag = compute_bo_convergence_diagnostics(gpbo_joint_res, bo_cfg, logger)
            gpbo_joint_res["convergence_diagnostics"] = conv_diag

            # Compute metrics for the best result
            best_lc = gpbo_joint_res["best_lc"]
            gpbo_joint_res["lc"] = best_lc
            _enrich(gpbo_joint_res, best_lc)
            
            results["gpbo_joint"] = gpbo_joint_res

            # --- ILL-POSEDNESS ANALYSIS ---
            # Solution landscape mapping + multiplicity index
            try:
                landscape = compute_solution_landscape(
                    objective_funcs, bounds, gpbo_joint_res["y_best"], logger)
                results["solution_landscape"] = landscape
                # Local sensitivity at optimum
                sensitivity = compute_local_sensitivity(
                    objective_funcs[best_lc], gpbo_joint_res["x_best"])
                results["local_sensitivity"] = sensitivity
                logger.info(f"    Sensitivity at optimum: dJ/dtheta={sensitivity['dJ_dtheta']:.6f}, "
                            f"d2J/dtheta2={sensitivity['d2J_dtheta2']:.6f}")
                # Inverse posterior
                posterior = compute_inverse_posterior(landscape, best_lc, logger)
                results["inverse_posterior"] = posterior
            except Exception as e:
                logger.warning(f"    Ill-posedness analysis failed: {e}")

            # Also populate per-LC results for compatibility with existing code
            for lc, lc_res in gpbo_joint_res.get("results_by_lc", {}).items():
                _enrich(lc_res, lc)
                lc_res["posterior_snapshots"] = []
                results["gpbo"][lc] = lc_res
                
        except Exception as e:
            logger.warning(f"    Joint GP-BO (skopt) failed: {e}")
            logger.warning(f"    Falling back to separate per-LC GP-BO (sklearn)")
            # Fallback to separate GP-BO per LC using sklearn
            if HAS_SKLEARN_GP:
                for lc in lc_categories:
                    try:
                        t0 = time.time()
                        gpbo_res = gp_bo_minimize(objective_funcs[lc], bounds, bo_cfg, logger)
                        gpbo_res["lc"] = lc
                        gpbo_res["wall_time"] = time.time() - t0
                        _enrich(gpbo_res, lc)
                        results["gpbo"][lc] = gpbo_res
                    except Exception as e2:
                        logger.warning(f"    GP-BO fallback failed for {lc}: {e2}")
    elif CFG.run_gpbo and HAS_SKLEARN_GP:
        # skopt not available; use per-LC sklearn GP-BO as fallback
        logger.info("    skopt not available; using per-LC sklearn GP-BO")
        for lc in lc_categories:
            try:
                t0 = time.time()
                gpbo_res = gp_bo_minimize(objective_funcs[lc], bounds, bo_cfg, logger)
                gpbo_res["lc"] = lc
                gpbo_res["wall_time"] = time.time() - t0
                _enrich(gpbo_res, lc)
                results["gpbo"][lc] = gpbo_res
            except Exception as e:
                logger.warning(f"    GP-BO (sklearn) failed for {lc}: {e}")

    if CFG.dry_run and not results["gpbo"] and results.get("gpbo_joint") is None:
        blc, stub = _inverse_coarse_grid_fallback(objective_funcs, bounds, lc_categories, logger)
        _enrich(stub, blc)
        results["gpbo"][blc] = stub

    for method in ["gpbo"]:
        if results[method]:
            best_lc = min(results[method].keys(), key=lambda l: results[method][l]["y_best"])
            results[f"{method}_best"] = results[method][best_lc]
    
    # Set gpbo_best from joint result if available
    if results.get("gpbo_joint"):
        results["gpbo_best"] = results["gpbo_joint"]
        if "lc" not in results["gpbo_best"] and "best_lc" in results["gpbo_best"]:
            results["gpbo_best"]["lc"] = results["gpbo_best"]["best_lc"]

    if CFG.run_gpbo and not CFG.dry_run and results.get("gpbo_best") is None:
        raise RuntimeError(
            "Inverse GP-BO produced no solution (gpbo_best is missing). "
            "Install scikit-optimize (pip install scikit-optimize) for skopt GP-BO, "
            "or ensure scikit-learn Gaussian-process fallback is available. "
            "Alternatively use --dry_run for a coarse-grid stub, or disable inverse GP-BO "
            "(CFG.run_gpbo = False)."
        )

    results["gamma_tikhonov_effective"] = float(gamma_tikh)
    results["theta_center"] = float(theta_ctr)
    
    return results


# =============================================================================
# ILL-POSEDNESS ANALYSIS: MULTI-START BO, LANDSCAPE, SENSITIVITY, POSTERIOR
# =============================================================================
def gp_bo_minimize_joint_multistart(objective_funcs, bounds, bo_cfg, logger, n_restarts=5):
    """Multi-start GP-BO: run gp_bo_minimize_joint N times with different seeds, keep best."""
    if n_restarts <= 1:
        return gp_bo_minimize_joint(objective_funcs, bounds, bo_cfg, logger)
    all_restarts = []
    original_seed = bo_cfg.seed
    for i in range(n_restarts):
        logger.info(f"    Multi-start BO restart {i+1}/{n_restarts} (seed={original_seed + i * 1000})")
        bo_cfg.seed = original_seed + i * 1000
        try:
            res = gp_bo_minimize_joint(objective_funcs, bounds, bo_cfg, logger)
            res["restart_id"] = i
            all_restarts.append(res)
        except Exception as e:
            logger.warning(f"    Restart {i+1} failed: {e}")
    bo_cfg.seed = original_seed
    if not all_restarts:
        raise RuntimeError("All multi-start BO restarts failed")
    best = min(all_restarts, key=lambda r: r["y_best"])
    best["all_restarts"] = all_restarts
    best["n_restarts_completed"] = len(all_restarts)
    logger.info(f"    Multi-start BO: best y={best['y_best']:.6f} from restart {best['restart_id']+1}")
    return best


def compute_solution_landscape(objective_funcs, bounds, best_objective, logger, n_grid=201):
    """Evaluate J(theta) on a dense grid for both LCs; count local minima -> multiplicity index."""
    theta_grid = np.linspace(bounds[0], bounds[1], n_grid)
    result = {"theta_grid": theta_grid}
    total_minima = 0
    threshold = max(best_objective * 1.5, best_objective + 0.01)
    for lc, obj_fn in objective_funcs.items():
        J_vals = np.array([obj_fn(float(t)) for t in theta_grid])
        result[f"J_{lc}"] = J_vals
        minima = []
        for i in range(1, len(J_vals) - 1):
            if J_vals[i] < J_vals[i - 1] and J_vals[i] < J_vals[i + 1] and J_vals[i] < threshold:
                minima.append({"theta": float(theta_grid[i]), "J": float(J_vals[i])})
        result[f"local_minima_{lc}"] = minima
        total_minima += len(minima)
    result["multiplicity_index"] = total_minima
    logger.info(f"    Solution landscape: {total_minima} local minima below threshold {threshold:.4f}")
    return result


def compute_local_sensitivity(objective_func, theta_opt, eps=0.1):
    """Compute dJ/dtheta and d2J/dtheta2 at the optimum via central finite differences."""
    J_plus = objective_func(theta_opt + eps)
    J_minus = objective_func(theta_opt - eps)
    J_center = objective_func(theta_opt)
    dJ_dtheta = (J_plus - J_minus) / (2 * eps)
    d2J_dtheta2 = (J_plus - 2 * J_center + J_minus) / (eps ** 2)
    return {"dJ_dtheta": float(dJ_dtheta), "d2J_dtheta2": float(d2J_dtheta2), "eps": eps}


def compute_forward_map_jacobian(models, approach, scaler_disp, enc, params, bounds, logger,
                                 n_grid=101, eps=0.1):
    """Compute dEA/dtheta and dIPF/dtheta via finite differences for both LCs.

    Identifies bifurcation points where derivatives change sign (non-uniqueness sources).
    """
    theta_grid = np.linspace(bounds[0], bounds[1], n_grid)
    lc_categories = [str(x) for x in enc.categories_[0].tolist()]
    result = {"theta_grid": theta_grid}
    for lc in lc_categories:
        dea_arr, dipf_arr = [], []
        for theta in theta_grid:
            m_plus = compute_ea_ipf_ensemble(models, approach, float(theta + eps), lc,
                                             scaler_disp, enc, params, d_eval=D_COMMON)
            m_minus = compute_ea_ipf_ensemble(models, approach, float(theta - eps), lc,
                                              scaler_disp, enc, params, d_eval=D_COMMON)
            dea_arr.append((m_plus["EA"] - m_minus["EA"]) / (2 * eps))
            dipf_arr.append((m_plus["IPF"] - m_minus["IPF"]) / (2 * eps))
        dea = np.array(dea_arr)
        dipf = np.array(dipf_arr)
        result[f"dEA_dtheta_{lc}"] = dea
        result[f"dIPF_dtheta_{lc}"] = dipf
        ea_signs = np.diff(np.sign(dea))
        ipf_signs = np.diff(np.sign(dipf))
        result[f"ea_bifurcations_{lc}"] = theta_grid[:-1][ea_signs != 0]
        result[f"ipf_bifurcations_{lc}"] = theta_grid[:-1][ipf_signs != 0]
        logger.info(f"    {lc}: {len(result[f'ea_bifurcations_{lc}'])} EA sign changes, "
                    f"{len(result[f'ipf_bifurcations_{lc}'])} IPF sign changes")
    return result


def compute_inverse_posterior(landscape, best_lc, logger):
    """Compute P(theta|target) proportional to exp(-J(theta)/T) as approximate inverse posterior.

    Temperature T = median(J) ensures the posterior covers a meaningful range.
    """
    J_vals = landscape.get(f"J_{best_lc}")
    theta_grid = landscape["theta_grid"]
    if J_vals is None or len(J_vals) == 0:
        return {}
    T = float(np.median(J_vals))
    if T < 1e-12:
        T = 1.0
    log_P = -J_vals / T
    log_P = log_P - np.max(log_P)
    P_unnorm = np.exp(log_P)
    Z = float(np.trapezoid(P_unnorm, theta_grid))
    if Z < 1e-20:
        return {}
    P = P_unnorm / Z
    mean = float(np.trapezoid(theta_grid * P, theta_grid))
    var = float(np.trapezoid((theta_grid - mean) ** 2 * P, theta_grid))
    std = float(np.sqrt(max(var, 0.0)))
    cdf = np.cumsum(P) * np.diff(np.concatenate([[theta_grid[0]], theta_grid]))
    cdf = cdf / cdf[-1]
    ci_lo = float(np.interp(0.025, cdf, theta_grid))
    ci_hi = float(np.interp(0.975, cdf, theta_grid))
    logger.info(f"    Inverse posterior ({best_lc}): mean={mean:.2f}, std={std:.2f}, "
                f"95% CI=[{ci_lo:.2f}, {ci_hi:.2f}], T={T:.4f}")
    return {"theta_grid": theta_grid, "posterior": P, "mean": mean, "std": std,
            "ci_95_lower": ci_lo, "ci_95_upper": ci_hi, "temperature": T}


def fig_solution_landscape(all_inverse_results, output_dir, logger):
    """Plot J(theta) solution landscape per target with local minima and multiplicity index."""
    targets_with_landscape = [r for r in all_inverse_results if "solution_landscape" in r]
    if not targets_with_landscape:
        return
    n = len(targets_with_landscape)
    fig_w = min(PRINT_WIDTH_IN, 3.5 * n)
    fonts = scaled_fonts(fig_w)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 3.0), squeeze=False)
    for i, res in enumerate(targets_with_landscape):
        ax = axes[0, i]
        sl = res["solution_landscape"]
        tid = res.get("target_info", {}).get("id", f"T{i+1}")
        for lc in sorted(k.replace("J_", "") for k in sl if k.startswith("J_")):
            J = sl[f"J_{lc}"]
            ax.plot(sl["theta_grid"], J, label=lc, linewidth=0.8)
            for m_pt in sl.get(f"local_minima_{lc}", []):
                ax.plot(m_pt["theta"], m_pt["J"], "v", markersize=5, color="red")
        ax.set_title(f"{tid} (SMI={sl['multiplicity_index']})", fontsize=fonts["title"])
        ax.set_xlabel(r"Angle ($^\circ$)", fontsize=fonts["label"])
        if i == 0:
            ax.set_ylabel(r"Objective $J(\theta)$", fontsize=fonts["label"])
        ax.legend(fontsize=fonts["legend"])
        ax.tick_params(labelsize=fonts["tick"])
    fig.tight_layout()
    path = os.path.join(output_dir, "Fig_solution_landscape.png")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: Fig_solution_landscape.png")


def fig_forward_map_jacobian(jacobian_results, output_dir, logger):
    """Plot dEA/dtheta and dIPF/dtheta for both LCs to show local invertibility."""
    theta = jacobian_results["theta_grid"]
    lcs = sorted(k.replace("dEA_dtheta_", "") for k in jacobian_results if k.startswith("dEA_dtheta_"))
    if not lcs:
        return
    fig_w = PRINT_WIDTH_IN
    fonts = scaled_fonts(fig_w)
    fig, axes = plt.subplots(2, len(lcs), figsize=(fig_w, 4.5), squeeze=False, sharex=True)
    for j, lc in enumerate(lcs):
        dea = jacobian_results[f"dEA_dtheta_{lc}"]
        dipf = jacobian_results[f"dIPF_dtheta_{lc}"]
        axes[0, j].plot(theta, dea, linewidth=0.8, color="C0")
        axes[0, j].axhline(0, color="gray", linewidth=0.5, linestyle="--")
        for bf in jacobian_results.get(f"ea_bifurcations_{lc}", []):
            axes[0, j].axvline(bf, color="red", linewidth=0.5, linestyle=":", alpha=0.7)
        axes[0, j].set_title(f"dEA/d$\\theta$ - {lc}", fontsize=fonts["title"])
        axes[0, j].set_ylabel("J/(kN$\\cdot$deg)" if j == 0 else "", fontsize=fonts["label"])
        axes[1, j].plot(theta, dipf, linewidth=0.8, color="C1")
        axes[1, j].axhline(0, color="gray", linewidth=0.5, linestyle="--")
        for bf in jacobian_results.get(f"ipf_bifurcations_{lc}", []):
            axes[1, j].axvline(bf, color="red", linewidth=0.5, linestyle=":", alpha=0.7)
        axes[1, j].set_title(f"dIPF/d$\\theta$ - {lc}", fontsize=fonts["title"])
        axes[1, j].set_xlabel(r"Angle ($^\circ$)", fontsize=fonts["label"])
        axes[1, j].set_ylabel("kN/deg" if j == 0 else "", fontsize=fonts["label"])
    for ax in axes.flat:
        ax.tick_params(labelsize=fonts["tick"])
    fig.tight_layout()
    path = os.path.join(output_dir, "Fig_forward_map_jacobian.png")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: Fig_forward_map_jacobian.png")


def fig_inverse_posterior(all_inverse_results, output_dir, logger):
    """Plot inverse posterior P(theta|target) for targets that have it."""
    targets_with_post = [r for r in all_inverse_results if "inverse_posterior" in r and r["inverse_posterior"]]
    if not targets_with_post:
        return
    n = len(targets_with_post)
    fig_w = min(PRINT_WIDTH_IN, 3.5 * n)
    fonts = scaled_fonts(fig_w)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 2.8), squeeze=False)
    for i, res in enumerate(targets_with_post):
        ax = axes[0, i]
        post = res["inverse_posterior"]
        tid = res.get("target_info", {}).get("id", f"T{i+1}")
        ax.plot(post["theta_grid"], post["posterior"], linewidth=0.8, color="C0")
        ax.axvline(post["mean"], color="C1", linewidth=0.7, linestyle="--",
                   label=f"mean={post['mean']:.1f}$^\\circ$")
        ax.axvspan(post["ci_95_lower"], post["ci_95_upper"], alpha=0.15, color="C0", label="95% CI")
        bo_best = res.get("gpbo_best", res.get("gpbo_joint", {}))
        if bo_best and "x_best" in bo_best:
            ax.axvline(bo_best["x_best"], color="red", linewidth=0.7, linestyle=":",
                       label=f"BO opt={bo_best['x_best']:.1f}$^\\circ$")
        ax.set_title(tid, fontsize=fonts["title"])
        ax.set_xlabel(r"Angle ($^\circ$)", fontsize=fonts["label"])
        if i == 0:
            ax.set_ylabel(r"$P(\theta \mid$ target$)$", fontsize=fonts["label"])
        ax.legend(fontsize=max(fonts["legend"] - 1, 5))
        ax.tick_params(labelsize=fonts["tick"])
    fig.tight_layout()
    path = os.path.join(output_dir, "Fig_inverse_posterior.png")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: Fig_inverse_posterior.png")


# =============================================================================
# INVERSE / ILL-POSEDNESS: PUBLICATION TABLES & EXTENDED POSTERIORS
# =============================================================================
def _two_objective_nondominated_max_ea_min_ipf(df: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated rows for maximize EA, minimize IPF (2D)."""
    if df is None or df.empty or "EA" not in df.columns or "IPF" not in df.columns:
        return pd.DataFrame(columns=["EA", "IPF"])
    pts = df[["EA", "IPF"]].values.astype(float)
    keep: List[int] = []
    for i in range(len(pts)):
        dominated = False
        for j in range(len(pts)):
            if i == j:
                continue
            if (pts[j, 0] >= pts[i, 0] and pts[j, 1] <= pts[i, 1]
                    and (pts[j, 0] > pts[i, 0] or pts[j, 1] < pts[i, 1])):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return df.iloc[keep].reset_index(drop=True)


def write_table_inverse_illposedness(all_inverse: List[Dict], output_dir: str, logger: logging.Logger) -> None:
    """One row per inverse target: multiplicity, sensitivity, posterior, multi-start, regularizers."""
    if not all_inverse:
        return
    rows = []
    for res in all_inverse:
        tid = res.get("target_info", {}).get("id", "?")
        sl = res.get("solution_landscape") or {}
        post = res.get("inverse_posterior") or {}
        loc = res.get("local_sensitivity") or {}
        best = res.get("gpbo_best") or {}
        gj = res.get("gpbo_joint") or {}
        rows.append({
            "Target_ID": tid,
            f"Target_EA@{EA_COMMON_MM_TAG}_J": res.get("target_ea", ""),
            "Target_IPF_kN": res.get("target_ipf", ""),
            "multiplicity_index": sl.get("multiplicity_index", ""),
            "posterior_mean_theta": post.get("mean", ""),
            "posterior_std_theta": post.get("std", ""),
            "posterior_ci95_theta_lo": post.get("ci_95_lower", ""),
            "posterior_ci95_theta_hi": post.get("ci_95_upper", ""),
            "dJ_dtheta_at_opt": loc.get("dJ_dtheta", ""),
            "d2J_dtheta2_at_opt": loc.get("d2J_dtheta2", ""),
            "n_bo_restarts_completed": gj.get("n_restarts_completed", ""),
            "best_bo_restart_id": gj.get("restart_id", "") if gj else "",
            "gamma_tikhonov_effective": res.get("gamma_tikhonov_effective", ""),
            "theta_center": res.get("theta_center", ""),
            "prob_weight": res.get("prob_weight", ""),
            "beta_robust": res.get("beta_robust", ""),
            "gpbo_y_best": best.get("y_best", ""),
            "gpbo_theta_best": best.get("x_best", ""),
            "gpbo_lc_best": best.get("lc", best.get("best_lc", "")),
        })
    path = os.path.join(output_dir, "Table3_inverse_illposedness.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info(f"  Saved: {os.path.basename(path)}")


def write_table_inverse_local_minima(all_inverse: List[Dict], output_dir: str, logger: logging.Logger) -> None:
    """Long-format local minima from solution landscapes (per LC)."""
    rows = []
    for res in all_inverse:
        tid = res.get("target_info", {}).get("id", "?")
        sl = res.get("solution_landscape")
        if not sl:
            continue
        for key in sl:
            if not key.startswith("local_minima_"):
                continue
            lc = key.replace("local_minima_", "")
            for m in sl.get(key, []) or []:
                rows.append({
                    "Target_ID": tid,
                    "LC": lc,
                    "theta_deg": m.get("theta", ""),
                    "J": m.get("J", ""),
                    "multiplicity_index": sl.get("multiplicity_index", ""),
                })
    if not rows:
        return
    path = os.path.join(output_dir, "Table_inverse_local_minima.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info(f"  Saved: {os.path.basename(path)}")


def write_table_inverse_topk_basins(all_inverse: List[Dict], output_dir: str, logger: logging.Logger, k: int = 12) -> None:
    """Ranked approximate basins: grid local minima + multi-start BO terminal designs."""
    rows = []
    for res in all_inverse:
        tid = res.get("target_info", {}).get("id", "?")
        basins: List[Dict[str, Any]] = []
        sl = res.get("solution_landscape")
        if sl:
            for key in sl:
                if not key.startswith("local_minima_"):
                    continue
                lc = key.replace("local_minima_", "")
                for m in sl.get(key, []) or []:
                    basins.append({
                        "source": "landscape_local_min",
                        "LC": lc,
                        "theta_deg": float(m["theta"]),
                        "J": float(m["J"]),
                    })
        gj = res.get("gpbo_joint") or {}
        for ar in gj.get("all_restarts", []) or []:
            basins.append({
                "source": f"bo_restart_{int(ar.get('restart_id', -1)) + 1}",
                "LC": ar.get("best_lc", ar.get("lc", "")),
                "theta_deg": float(ar.get("x_best", float("nan"))),
                "J": float(ar.get("y_best", float("nan"))),
            })
        basins.sort(key=lambda b: b["J"])
        seen = set()
        rank = 0
        for b in basins:
            sig = (round(b["theta_deg"], 3), str(b["LC"]), round(b["J"], 8))
            if sig in seen:
                continue
            seen.add(sig)
            rank += 1
            rows.append({"Target_ID": tid, "rank": rank, **b})
            if rank >= k:
                break
    if not rows:
        return
    path = os.path.join(output_dir, "Table_inverse_topk_basins.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info(f"  Saved: {os.path.basename(path)}")


def write_table_forward_jacobian_summary(jacobian_results: Dict[str, Any], output_dir: str, logger: logging.Logger) -> None:
    """Per-LC counts and derivative magnitudes for Jacobian / bifurcation diagnostics."""
    if not jacobian_results:
        return
    rows = []
    theta = jacobian_results.get("theta_grid")
    if theta is None:
        return
    dtheta = float(np.median(np.diff(theta))) if len(theta) > 1 else 1.0
    lcs = sorted(k.replace("dEA_dtheta_", "") for k in jacobian_results if k.startswith("dEA_dtheta_"))
    for lc in lcs:
        dea = jacobian_results.get(f"dEA_dtheta_{lc}")
        dipf = jacobian_results.get(f"dIPF_dtheta_{lc}")
        if dea is None or dipf is None:
            continue
        dea = np.asarray(dea, dtype=float)
        dipf = np.asarray(dipf, dtype=float)
        ea_bf = jacobian_results.get(f"ea_bifurcations_{lc}", np.array([]))
        ipf_bf = jacobian_results.get(f"ipf_bifurcations_{lc}", np.array([]))
        rows.append({
            "LC": lc,
            "n_theta_samples": len(theta),
            "finite_diff_step_deg": dtheta,
            "n_EA_derivative_sign_changes": len(np.asarray(ea_bf).reshape(-1)),
            "n_IPF_derivative_sign_changes": len(np.asarray(ipf_bf).reshape(-1)),
            "mean_abs_dEA_dtheta": float(np.mean(np.abs(dea))),
            "mean_abs_dIPF_dtheta": float(np.mean(np.abs(dipf))),
            "max_abs_dEA_dtheta": float(np.max(np.abs(dea))),
            "max_abs_dIPF_dtheta": float(np.max(np.abs(dipf))),
        })
    path = os.path.join(output_dir, "Table_forward_jacobian_summary.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info(f"  Saved: {os.path.basename(path)}")


def write_table_inverse_vs_calibration(
    all_inverse: List[Dict],
    calibration: Optional[Dict],
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Join inverse errors with random-protocol Hard-PINN conformal factors (forward calibration)."""
    if not all_inverse:
        return
    cal_h = None
    if calibration and "random" in calibration and "hard" in calibration["random"]:
        cal_h = calibration["random"]["hard"]
    rows = []
    for res in all_inverse:
        tid = res.get("target_info", {}).get("id", "?")
        best = res.get("gpbo_best") or {}
        row = {
            "Target_ID": tid,
            "EA_error_pct": best.get("ea_error_pct", ""),
            "IPF_error_pct": best.get("ipf_error_pct", ""),
            "pred_EA_std": best.get("pred_ea_std", ""),
            "pred_IPF_std": best.get("pred_ipf_std", ""),
        }
        if cal_h:
            row["conformal_factor_load"] = cal_h.get("conformal_factor", "")
            row["conformal_factor_energy"] = cal_h.get("energy_conformal_factor", "")
        rows.append(row)
    path = os.path.join(output_dir, "Table_inverse_vs_calibration.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info(f"  Saved: {os.path.basename(path)}")


def compute_inverse_posterior_likelihood(
    models: List[nn.Module],
    approach: str,
    scaler_disp: StandardScaler,
    enc: OneHotEncoder,
    params: ScalingParams,
    target_ea: float,
    target_ipf: float,
    lc: str,
    theta_grid: np.ndarray,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Discrete ``P(theta)`` ∝ ∏_metrics Gaussian(EA, IPF | ensemble mean, ensemble std) on a 1D grid."""
    log_l = []
    for t in theta_grid:
        m = compute_ea_ipf_ensemble(
            models, approach, float(t), lc, scaler_disp, enc, params, d_eval=D_COMMON)
        se = max(float(m.get("EA_std", 0.0)), 1e-6)
        sf = max(float(m.get("IPF_std", 0.0)), 1e-6)
        log_l.append(
            -0.5 * (((m["EA"] - target_ea) / se) ** 2 + ((m["IPF"] - target_ipf) / sf) ** 2)
        )
    log_l = np.asarray(log_l, dtype=float)
    log_l = log_l - np.max(log_l)
    P_un = np.exp(log_l)
    Z = float(np.trapezoid(P_un, theta_grid))
    if Z < 1e-30:
        return {}
    P = P_un / Z
    mean = float(np.trapezoid(theta_grid * P, theta_grid))
    var = float(np.trapezoid((theta_grid - mean) ** 2 * P, theta_grid))
    std = float(np.sqrt(max(var, 0.0)))
    cdf = np.cumsum(P) * np.diff(np.concatenate([[theta_grid[0]], theta_grid]))
    cdf = cdf / cdf[-1]
    ci_lo = float(np.interp(0.025, cdf, theta_grid))
    ci_hi = float(np.interp(0.975, cdf, theta_grid))
    logger.info(f"    Likelihood posterior ({lc}): mean={mean:.2f}, std={std:.2f}, 95% CI=[{ci_lo:.2f}, {ci_hi:.2f}]")
    return {
        "theta_grid": theta_grid,
        "posterior": P,
        "mean": mean,
        "std": std,
        "ci_95_lower": ci_lo,
        "ci_95_upper": ci_hi,
        "lc": lc,
    }


def augment_inverse_results_likelihood_posterior(
    all_inverse: List[Dict],
    models: List[nn.Module],
    approach: str,
    scaler_disp: StandardScaler,
    enc: OneHotEncoder,
    params: ScalingParams,
    logger: logging.Logger,
) -> None:
    for res in all_inverse:
        best = res.get("gpbo_best") or {}
        lc = best.get("lc") or best.get("best_lc")
        sl = res.get("solution_landscape")
        if not lc or not sl or "theta_grid" not in sl:
            continue
        theta_grid = np.asarray(sl["theta_grid"], dtype=float)
        post = compute_inverse_posterior_likelihood(
            models, approach, scaler_disp, enc, params,
            float(res["target_ea"]), float(res["target_ipf"]), str(lc), theta_grid, logger,
        )
        if post:
            res["inverse_posterior_likelihood"] = post


def fig_inverse_posterior_likelihood(all_inverse: List[Dict], output_dir: str, logger: logging.Logger) -> None:
    targets = [r for r in all_inverse if r.get("inverse_posterior_likelihood")]
    if not targets:
        return
    n = len(targets)
    fig_w = min(PRINT_WIDTH_IN, 3.5 * n)
    fonts = scaled_fonts(fig_w)
    fig, axes = plt.subplots(2, n, figsize=(fig_w, 4.2), squeeze=False, sharex="col")
    for i, res in enumerate(targets):
        tid = res.get("target_info", {}).get("id", f"T{i+1}")
        post_j = res.get("inverse_posterior", {})
        post_l = res.get("inverse_posterior_likelihood", {})
        th = np.asarray(post_l["theta_grid"], dtype=float)
        pj = post_j.get("posterior") if post_j else None
        if pj is None or len(np.asarray(pj).reshape(-1)) != len(th):
            pj = np.zeros_like(th, dtype=float)
        else:
            pj = np.asarray(pj, dtype=float).reshape(-1)
        axes[0, i].plot(th, pj, color="C0", linewidth=0.8, label="exp(-J/T)")
        axes[0, i].set_title(f"{tid}: J-posterior", fontsize=fonts["title"])
        axes[0, i].tick_params(labelsize=fonts["tick"])
        if i == 0:
            axes[0, i].set_ylabel(r"$P(\theta)$", fontsize=fonts["label"])
        axes[1, i].plot(th, post_l["posterior"], color="C2", linewidth=0.8, label="Gaussian lik.")
        axes[1, i].axvline(post_l["mean"], color="C1", linestyle="--", linewidth=0.7)
        axes[1, i].set_title(f"Likelihood ({post_l.get('lc', '')})", fontsize=fonts["title"])
        axes[1, i].set_xlabel(r"Angle ($^\circ$)", fontsize=fonts["label"])
        if i == 0:
            axes[1, i].set_ylabel(r"$P(\theta)$", fontsize=fonts["label"])
        bo_best = res.get("gpbo_best", {})
        if bo_best and "x_best" in bo_best:
            for axr in (axes[0, i], axes[1, i]):
                axr.axvline(bo_best["x_best"], color="red", linewidth=0.6, linestyle=":", alpha=0.85)
    fig.tight_layout()
    path = os.path.join(output_dir, "Fig_inverse_posterior_likelihood.png")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: Fig_inverse_posterior_likelihood.png")


def write_table_inverse_posterior_likelihood_summary(all_inverse: List[Dict], output_dir: str, logger: logging.Logger) -> None:
    rows = []
    for res in all_inverse:
        pl = res.get("inverse_posterior_likelihood")
        if not pl:
            continue
        tid = res.get("target_info", {}).get("id", "?")
        rows.append({
            "Target_ID": tid,
            "LC": pl.get("lc", ""),
            "likelihood_mean_theta": pl.get("mean", ""),
            "likelihood_std_theta": pl.get("std", ""),
            "likelihood_ci95_lo": pl.get("ci_95_lower", ""),
            "likelihood_ci95_hi": pl.get("ci_95_upper", ""),
        })
    if not rows:
        return
    path = os.path.join(output_dir, "Table_inverse_posterior_likelihood.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info(f"  Saved: {os.path.basename(path)}")


def write_table_inverse_member_spread(
    models: List[nn.Module],
    approach: str,
    all_inverse: List[Dict],
    scaler_disp: StandardScaler,
    enc: OneHotEncoder,
    params: ScalingParams,
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Per-member EA/IPF at the ensemble-reported optimum (forward epistemic spread)."""
    if not models or not all_inverse:
        return
    rows = []
    for res in all_inverse:
        best = res.get("gpbo_best") or {}
        lc = best.get("lc") or best.get("best_lc")
        th = best.get("x_best")
        tid = res.get("target_info", {}).get("id", "?")
        if lc is None or th is None:
            continue
        m = compute_ea_ipf_ensemble(
            models, approach, float(th), str(lc), scaler_disp, enc, params, d_eval=D_COMMON)
        ea_pm = m.get("ea_per_member")
        ipf_pm = m.get("ipf_per_member")
        if ea_pm is None or ipf_pm is None:
            continue
        ea_pm = np.asarray(ea_pm, dtype=float).ravel()
        ipf_pm = np.asarray(ipf_pm, dtype=float).ravel()
        rows.append({
            "Target_ID": tid,
            "theta_star_deg": float(th),
            "LC": str(lc),
            "EA_member_min": float(np.min(ea_pm)),
            "EA_member_max": float(np.max(ea_pm)),
            "EA_member_std": float(np.std(ea_pm, ddof=1)) if len(ea_pm) > 1 else 0.0,
            "IPF_member_min": float(np.min(ipf_pm)),
            "IPF_member_max": float(np.max(ipf_pm)),
            "IPF_member_std": float(np.std(ipf_pm, ddof=1)) if len(ipf_pm) > 1 else 0.0,
            "n_members": len(ea_pm),
        })
    if not rows:
        return
    path = os.path.join(output_dir, "Table_inverse_theta_member_spread.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info(f"  Saved: {os.path.basename(path)}")


def pick_validation_inverse_stress_targets(
    dual_results: Dict[str, Any],
    df_metrics: pd.DataFrame,
    df_all: pd.DataFrame,
    logger: logging.Logger,
    max_n: int = 5,
) -> List[Dict]:
    """Experimental (EA@D_COMMON, IPF) from random-protocol validation rows absent from train (Angle, LC)."""
    dr = dual_results.get("random")
    if dr is None or "train_df" not in dr or "val_df" not in dr:
        return []
    train_df, val_df = dr["train_df"], dr["val_df"]
    pairs_train = set(zip(train_df["Angle"].astype(float), train_df["LC"].astype(str)))
    cand = val_df.copy()
    mask = [tuple((float(r["Angle"]), str(r["LC"]))) not in pairs_train for _, r in cand.iterrows()]
    cand = cand.loc[mask]
    if cand.empty:
        cand = val_df.drop_duplicates(subset=["Angle", "LC"])
    dm = df_metrics.copy()
    if "EA_common" not in dm.columns:
        enrich_df_metrics_ea_common(dm, df_all, logger=logger)
    seen = set()
    targets: List[Dict] = []
    for _, vr in cand.iterrows():
        key = (float(vr["Angle"]), str(vr["LC"]))
        if key in seen:
            continue
        msub = dm[(dm["Angle"].astype(float) == key[0]) & (dm["LC"].astype(str) == key[1])]
        if msub.empty:
            continue
        row = msub.iloc[0]
        ea = float(row["EA_common"])
        ipf = float(row["IPF"])
        seen.add(key)
        targets.append({
            "id": f"S{len(targets)+1}",
            "EA": ea,
            "IPF": ipf,
            "d_eval": D_COMMON,
            "val_angle_deg": key[0],
            "val_LC": key[1],
            "rationale": "Random-protocol validation (Angle, LC) stress inverse",
        })
        if len(targets) >= max_n:
            break
    if targets:
        logger.info(f"  Validation stress inverse: {len(targets)} target(s) from val⊄train (Angle,LC) pairs")
    return targets


def run_validation_inverse_stress_and_save(
    inv_models: List[nn.Module],
    stress_targets: List[Dict],
    inv_scaler_disp: StandardScaler,
    inv_enc: OneHotEncoder,
    inv_params: ScalingParams,
    bo_cfg: BOConfig,
    cal_ens,
    clf_feat_scaler,
    output_dir: str,
    logger: logging.Logger,
) -> None:
    if not stress_targets or CFG.dry_run:
        return
    rows = []
    for t in stress_targets:
        logger.info(f"  Stress inverse {t['id']}: EA@{int(D_COMMON)}mm={t['EA']:.2f}J, IPF={t['IPF']:.3f}kN")
        res = run_inverse_design(
            inv_models, "hard", t["EA"], t["IPF"],
            inv_scaler_disp, inv_enc, inv_params, bo_cfg, logger,
            cal_ens=cal_ens, feat_scaler=clf_feat_scaler,
        )
        best = res.get("gpbo_best") or {}
        rows.append({
            "Stress_ID": t["id"],
            "source_val_angle_deg": t.get("val_angle_deg", ""),
            "source_val_LC": t.get("val_LC", ""),
            "target_EA_J": t["EA"],
            "target_IPF_kN": t["IPF"],
            "recovered_theta_deg": best.get("x_best", ""),
            "recovered_LC": best.get("lc", best.get("best_lc", "")),
            "EA_error_pct": best.get("ea_error_pct", ""),
            "IPF_error_pct": best.get("ipf_error_pct", ""),
            "objective_J": best.get("y_best", ""),
        })
    path = os.path.join(output_dir, "Table_inverse_stress_protocol.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info(f"  Saved: {os.path.basename(path)}")


def run_inverse_regularization_ablation_and_save(
    inv_models: List[nn.Module],
    targets: List[Dict],
    inv_scaler_disp: StandardScaler,
    inv_enc: OneHotEncoder,
    inv_params: ScalingParams,
    bo_cfg: BOConfig,
    cal_ens,
    clf_feat_scaler,
    baseline_results: List[Dict],
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Extra GP-BO runs: no Tikhonov, no classifier penalty, no robustness penalty."""
    if not targets or CFG.dry_run or not HAS_SKOPT:
        return
    baseline_by_id = {
        r["target_info"]["id"]: r for r in baseline_results
        if r.get("target_info") and "id" in r["target_info"]
    }
    rows = []
    variants = [
        ("no_tikhonov", {"gamma_tikhonov_override": 0.0}),
        ("no_classifier_penalty", {"prob_weight_override": 0.0}),
        ("no_robustness_penalty", {"beta_robust_override": 0.0}),
    ]
    for target in targets:
        tid = target["id"]
        te, tp = target["EA"], target["IPF"]
        base = baseline_by_id.get(tid) or {}
        bbest = base.get("gpbo_best") or {}
        for vname, kwargs in variants:
            r = run_inverse_design(
                inv_models, "hard", te, tp, inv_scaler_disp, inv_enc, inv_params, bo_cfg, logger,
                cal_ens=cal_ens, feat_scaler=clf_feat_scaler, **kwargs,
            )
            bb = r.get("gpbo_best") or {}
            rows.append({
                "Target_ID": tid,
                "variant": vname,
                "theta_deg": bb.get("x_best", ""),
                "LC": bb.get("lc", bb.get("best_lc", "")),
                "J": bb.get("y_best", ""),
                "EA_err_pct": bb.get("ea_error_pct", ""),
                "IPF_err_pct": bb.get("ipf_error_pct", ""),
                "delta_theta_vs_full_deg": (
                    float(bb.get("x_best", 0.0)) - float(bbest.get("x_best", 0.0))
                    if bb.get("x_best") is not None and bbest.get("x_best") is not None else ""
                ),
                "delta_J_vs_full": (
                    float(bb.get("y_best", 0.0)) - float(bbest.get("y_best", 0.0))
                    if bb.get("y_best") is not None and bbest.get("y_best") is not None else ""
                ),
            })
    path = os.path.join(output_dir, "Table_inverse_ablation.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info(f"  Saved: {os.path.basename(path)}")


def write_table_mobo_vs_landscape_pareto_distance(
    mobo_result: Optional[Dict[str, Any]],
    landscape_df: pd.DataFrame,
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Hypervolume gap + mean nearest-neighbour distance (MOBO ND vs dense landscape ND)."""
    if mobo_result is None or landscape_df is None or landscape_df.empty:
        return
    mobo_pf = mobo_result.get("pareto_approx_df")
    if mobo_pf is None or mobo_pf.empty:
        return
    nd_land = _two_objective_nondominated_max_ea_min_ipf(landscape_df)
    mobo_df = mobo_pf.rename(columns={"EA_J": "EA", "IPF_kN": "IPF"})
    nd_mobo = _two_objective_nondominated_max_ea_min_ipf(mobo_df)
    ref_ea = float(landscape_df["EA"].min())
    ref_ipf = float(landscape_df["IPF"].max())
    hv_land = compute_hypervolume_2d(nd_land, ref_ea, ref_ipf, logger, label="Landscape-ND")
    hv_mobo = compute_hypervolume_2d(nd_mobo, ref_ea, ref_ipf, logger, label="MOBO-ND")
    se = max(float(nd_land["EA"].std()), 1e-9)
    sf = max(float(nd_land["IPF"].std()), 1e-9)
    Lpts = nd_land[["EA", "IPF"]].values.astype(float)
    dists = []
    for _, r in nd_mobo.iterrows():
        p = np.array([r["EA"], r["IPF"]], dtype=float)
        dmin = min(float(np.hypot((p[0] - q[0]) / se, (p[1] - q[1]) / sf)) for q in Lpts) if len(Lpts) else float("nan")
        dists.append(dmin)
    rows = [{
        "n_landscape_ND": len(nd_land),
        "n_mobo_ND": len(nd_mobo),
        "hypervolume_landscape_ND": hv_land,
        "hypervolume_mobo_ND": hv_mobo,
        "hypervolume_ratio_mobo_over_land": (hv_mobo / hv_land) if hv_land and hv_land > 0 else "",
        "mean_norm_dist_mobo_to_landscape_ND": float(np.mean(dists)) if dists else "",
        "ref_EA_J": ref_ea,
        "ref_IPF_kN": ref_ipf,
    }]
    path = os.path.join(output_dir, "Table_mobo_vs_landscape_pareto_distance.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info(f"  Saved: {os.path.basename(path)}")


def generate_inverse_publication_artifacts(
    output_dir: str,
    logger: logging.Logger,
    all_inverse: List[Dict],
    calibration: Optional[Dict] = None,
    jacobian_results: Optional[Dict[str, Any]] = None,
    inv_models: Optional[List[nn.Module]] = None,
    inv_scaler_disp=None,
    inv_enc=None,
    inv_params=None,
    dual_results: Optional[Dict] = None,
    df_metrics: Optional[pd.DataFrame] = None,
    df_all: Optional[pd.DataFrame] = None,
    bo_cfg: Optional[BOConfig] = None,
    cal_ens=None,
    clf_feat_scaler=None,
    mobo_result=None,
    landscape_df: Optional[pd.DataFrame] = None,
) -> None:
    """Emit Q1-oriented inverse / MOBO CSVs and likelihood figure."""
    write_table_inverse_illposedness(all_inverse, output_dir, logger)
    write_table_inverse_local_minima(all_inverse, output_dir, logger)
    write_table_inverse_topk_basins(all_inverse, output_dir, logger)
    if jacobian_results:
        write_table_forward_jacobian_summary(jacobian_results, output_dir, logger)
    if calibration is not None:
        write_table_inverse_vs_calibration(all_inverse, calibration, output_dir, logger)
    if inv_models and inv_scaler_disp is not None and inv_enc is not None and inv_params is not None:
        augment_inverse_results_likelihood_posterior(
            all_inverse, inv_models, "hard", inv_scaler_disp, inv_enc, inv_params, logger,
        )
        fig_inverse_posterior_likelihood(all_inverse, output_dir, logger)
        write_table_inverse_posterior_likelihood_summary(all_inverse, output_dir, logger)
        if getattr(CFG, "run_inverse_member_spread", True):
            write_table_inverse_member_spread(
                inv_models, "hard", all_inverse, inv_scaler_disp, inv_enc, inv_params, output_dir, logger,
            )
    if (
        getattr(CFG, "run_reviewer_proof", True)
        and getattr(CFG, "run_inverse_stress_validation", True)
        and dual_results is not None
        and df_metrics is not None
        and df_all is not None
        and bo_cfg is not None
        and inv_models
        and inv_scaler_disp is not None
    ):
        nmax = int(getattr(CFG, "inverse_stress_max_targets", 5))
        st = pick_validation_inverse_stress_targets(dual_results, df_metrics, df_all, logger, max_n=nmax)
        run_validation_inverse_stress_and_save(
            inv_models, st, inv_scaler_disp, inv_enc, inv_params, bo_cfg,
            cal_ens, clf_feat_scaler, output_dir, logger,
        )
    if (
        getattr(CFG, "run_inverse_ablation", False)
        and bo_cfg is not None
        and inv_models
        and inv_scaler_disp is not None
    ):
        inv_targets = [r["target_info"] for r in all_inverse if r.get("target_info")]
        nmax = int(getattr(CFG, "inverse_ablation_max_targets", 2))
        run_inverse_regularization_ablation_and_save(
            inv_models, inv_targets[:nmax], inv_scaler_disp, inv_enc, inv_params, bo_cfg,
            cal_ens, clf_feat_scaler, all_inverse, output_dir, logger,
        )
    if mobo_result is not None and landscape_df is not None:
        write_table_mobo_vs_landscape_pareto_distance(mobo_result, landscape_df, output_dir, logger)


# =============================================================================
# VALIDATION: LEAVE-ONE-ANGLE-OUT CV & CALIBRATION VS ENSEMBLE SIZE
# =============================================================================
def run_leave_one_angle_out_cv(df_all, logger, n_ensemble_cv=3):
    """Leave-one-angle-out cross-validation for forward model robustness."""
    angles = sorted(df_all["Angle"].unique())
    logger.info(f"  LOAO-CV: {len(angles)} folds (angles: {angles})")
    fold_results = []
    for theta_held in angles:
        train_df = df_all[df_all["Angle"] != theta_held].copy().reset_index(drop=True)
        val_df = df_all[df_all["Angle"] == theta_held].copy().reset_index(drop=True)
        if len(val_df) == 0:
            continue
        scaler_d, scaler_o, enc_cv, par_cv = create_preprocessors(train_df, logger)
        prev_n = CFG.n_ensemble
        CFG.n_ensemble = n_ensemble_cv
        try:
            ens = train_ensemble("hard", train_df, val_df, scaler_d, scaler_o, enc_cv, par_cv,
                                 "unseen", logger)
        finally:
            CFG.n_ensemble = prev_n
        metrics = ens["metrics"]
        fold_results.append({
            "Angle": float(theta_held),
            "R2_Load": float(metrics["load_r2"]),
            "R2_Energy": float(metrics["energy_r2"]),
            "RMSE_Load": float(metrics["load_rmse"]),
            "N_Train": len(train_df),
            "N_Val": len(val_df),
        })
        logger.info(f"    theta={theta_held:.0f}: R2_load={metrics['load_r2']:.4f}, "
                    f"R2_energy={metrics['energy_r2']:.4f}")
    r2_loads = [f["R2_Load"] for f in fold_results]
    r2_energies = [f["R2_Energy"] for f in fold_results]
    return {
        "per_fold_r2": fold_results,
        "mean_r2_load": float(np.mean(r2_loads)) if r2_loads else float('nan'),
        "std_r2_load": float(np.std(r2_loads, ddof=1)) if len(r2_loads) > 1 else 0.0,
        "mean_r2_energy": float(np.mean(r2_energies)) if r2_energies else float('nan'),
        "std_r2_energy": float(np.std(r2_energies, ddof=1)) if len(r2_energies) > 1 else 0.0,
    }


def fig_loao_cv_results(loao_results, output_dir, logger):
    """Bar chart of per-angle R2 from leave-one-angle-out CV."""
    folds = loao_results["per_fold_r2"]
    if not folds:
        return
    fig_w = min(PRINT_WIDTH_IN, 5.0)
    fonts = scaled_fonts(fig_w)
    fig, ax = plt.subplots(figsize=(fig_w, 3.0))
    angles = [f["Angle"] for f in folds]
    r2s = [f["R2_Load"] for f in folds]
    ax.bar(range(len(angles)), r2s, color="C0", alpha=0.8, edgecolor="black", linewidth=0.4)
    ax.axhline(loao_results["mean_r2_load"], color="red", linewidth=0.8, linestyle="--",
               label=f"Mean={loao_results['mean_r2_load']:.3f}")
    ax.set_xticks(range(len(angles)))
    ax.set_xticklabels([f"{a:.0f}" for a in angles], fontsize=fonts["tick"])
    ax.set_ylabel(r"Load $R^2$", fontsize=fonts["label"])
    ax.set_xlabel("Held-out angle (deg)", fontsize=fonts["label"])
    ax.set_title("Leave-One-Angle-Out CV", fontsize=fonts["title"])
    ax.legend(fontsize=fonts["legend"])
    ax.tick_params(labelsize=fonts["tick"])
    fig.tight_layout()
    path = os.path.join(output_dir, "Fig_loao_cv.png")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: Fig_loao_cv.png")


def compute_calibration_vs_M(dual_results, logger, M_values=None):
    """Compute calibration metrics (conformal factor, coverage) at varying ensemble sizes."""
    if M_values is None:
        M_values = [2, 3, 5, 8, 10, 15, 20]
    results_by_M = {}
    for protocol in ["random", "unseen"]:
        if protocol not in dual_results:
            continue
        for approach in ["ddns", "soft", "hard"]:
            if approach not in dual_results[protocol]:
                continue
            ens_data = dual_results[protocol][approach]
            models = ens_data.get("models", [])
            M_total = len(models)
            key = f"{protocol}_{approach}"
            results_by_M[key] = {"M_values": [], "conformal_factors": [],
                                 "coverage_1sig": [], "coverage_2sig": []}
            y_val = ens_data["metrics"]["true_values"]
            member_preds_all = []
            for mm in ens_data.get("member_metrics", []):
                if "predictions" in mm and "load" in mm["predictions"]:
                    member_preds_all.append(mm["predictions"]["load"])
            for M_prime in M_values:
                if M_prime > len(member_preds_all) or M_prime < 2:
                    continue
                sub = member_preds_all[:M_prime]
                Fm = np.mean(sub, axis=0)
                Fs = np.std(sub, axis=0, ddof=1)
                residuals = np.abs(y_val["load"] - Fm)
                sigma_safe = np.maximum(Fs, 1e-12)
                norm_res = residuals / sigma_safe
                cf = float(np.percentile(norm_res, 68.3))
                cov_1 = float(np.mean(residuals < 1.0 * sigma_safe))
                cov_2 = float(np.mean(residuals < 2.0 * sigma_safe))
                results_by_M[key]["M_values"].append(M_prime)
                results_by_M[key]["conformal_factors"].append(cf)
                results_by_M[key]["coverage_1sig"].append(cov_1)
                results_by_M[key]["coverage_2sig"].append(cov_2)
    logger.info(f"  Calibration vs M: computed for {len(results_by_M)} protocol/approach combos")
    return results_by_M


def fig_calibration_vs_M(cal_vs_M, output_dir, logger):
    """Plot conformal factor and coverage vs ensemble size M."""
    if not cal_vs_M:
        return
    fig_w = PRINT_WIDTH_IN
    fonts = scaled_fonts(fig_w)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, 3.0))
    for key, data in cal_vs_M.items():
        if not data["M_values"]:
            continue
        ax1.plot(data["M_values"], data["conformal_factors"], "o-", label=key, markersize=3, linewidth=0.8)
        ax2.plot(data["M_values"], data["coverage_1sig"], "o-", label=f"{key} 1sig", markersize=3, linewidth=0.8)
    ax1.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_xlabel("Ensemble size M", fontsize=fonts["label"])
    ax1.set_ylabel("Conformal factor", fontsize=fonts["label"])
    ax1.set_title("Calibration factor", fontsize=fonts["title"])
    ax1.legend(fontsize=max(fonts["legend"] - 1, 5))
    ax2.axhline(0.683, color="gray", linewidth=0.5, linestyle="--", label="Expected 1sig")
    ax2.set_xlabel("Ensemble size M", fontsize=fonts["label"])
    ax2.set_ylabel("Coverage", fontsize=fonts["label"])
    ax2.set_title("Coverage vs M", fontsize=fonts["title"])
    ax2.legend(fontsize=max(fonts["legend"] - 1, 5))
    for ax in [ax1, ax2]:
        ax.tick_params(labelsize=fonts["tick"])
    fig.tight_layout()
    path = os.path.join(output_dir, "Fig_calibration_vs_M.png")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: Fig_calibration_vs_M.png")


def compute_hypervolume_2d(
    pareto_front_df: pd.DataFrame,
    ref_ea: float, ref_ipf: float,
    logger: logging.Logger,
    label: str = "Dominance"
) -> float:
    """Compute 2D hypervolume of Pareto front relative to reference (nadir) point.

    Objectives: maximize EA, minimize IPF.
    The hypervolume measures the area in (EA, IPF) space dominated by the
    Pareto front and bounded by the reference point.

    Returns hypervolume in J*kN units.
    """
    if pareto_front_df.empty:
        return 0.0

    pts = pareto_front_df[["EA", "IPF"]].values.copy()
    # Keep only points better than reference in both objectives
    mask = (pts[:, 0] > ref_ea) & (pts[:, 1] < ref_ipf)
    pts = pts[mask]
    if len(pts) == 0:
        return 0.0

    # Sort by EA ascending
    pts = pts[pts[:, 0].argsort()]

    # Sweep left to right: each point contributes a rectangle
    hv = 0.0
    prev_ea = ref_ea
    # Need to track the running minimum IPF from right-to-left for correct HV
    # Actually for maximize-EA/minimize-IPF, sweep by EA ascending:
    # The IPF at each step is the minimum IPF seen so far from current point rightward
    # Simpler: compute as sum of rectangles using the "staircase" method
    for i in range(len(pts)):
        ea_i, ipf_i = pts[i]
        width = ea_i - prev_ea
        height = ref_ipf - ipf_i
        if width > 0 and height > 0:
            hv += width * height
        prev_ea = ea_i

    logger.info(f"  {label} front hypervolume: {hv:.4f} J*kN "
                f"(ref: EA={ref_ea:.2f}J, IPF={ref_ipf:.2f}kN, "
                f"{len(pts)} non-dominated points)")
    return hv


def build_qnehvi_reference_point_from_landscape(
    landscape_df: pd.DataFrame,
    margin_frac: float,
    logger: logging.Logger,
    tkwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Build BoTorch ``ref_point`` for q(NE)HVI on **maximization** objectives.

    Objectives passed to BoTorch: ``y0 = EA`` (J, absorbed to ``D_COMMON``),
    ``y1 = -IPF`` (kN, so maximizing ``y1`` minimizes IPF).

    For hypervolume under maximization, ``ref_point`` must be **strictly worse**
    than any objective vector we intend to retain—here we anchor it slightly
    beyond the **nadir of the dense surrogate landscape** (min EA, worst IPF),
    following BoTorch guidance (slightly worse than a lower bound on each
    objective of interest).

    Returns
    -------
    ref_point : Tensor (2,)
        ``[r_EA, r_{-IPF}]`` with ``r_EA < min(EA_land)`` and
        ``r_{-IPF} < min(-IPF_land) = -max(IPF_land)``.
    meta : dict
        JSON-serializable documentation (ranges, margins, rule text).
    """
    ea_min = float(landscape_df["EA"].min())
    ea_max = float(landscape_df["EA"].max())
    ipf_min = float(landscape_df["IPF"].min())
    ipf_max = float(landscape_df["IPF"].max())
    span_ea = max(ea_max - ea_min, 1e-9)
    span_ipf = max(ipf_max - ipf_min, 1e-9)
    m = float(max(margin_frac, 1e-6))
    r0 = ea_min - m * span_ea
    r1 = (-ipf_max) - m * span_ipf
    ref_point = torch.tensor([r0, r1], **tkwargs)
    meta = {
        "objectives_botorch": [
            f"maximize EA (J) at d={D_COMMON:.0f} mm (displacement-fair)",
            "maximize -IPF (kN)  [equivalent to minimize IPF]",
        ],
        "reference_point_y": [r0, r1],
        "rule": (
            "r_EA = min(EA_landscape) - margin_frac * span(EA); "
            "r_negIPF = -max(IPF_landscape) - margin_frac * span(IPF). "
            "Both strictly worse than the landscape nadir in maximization form."
        ),
        "landscape_min_EA_J": ea_min,
        "landscape_max_EA_J": ea_max,
        "landscape_min_IPF_kN": ipf_min,
        "landscape_max_IPF_kN": ipf_max,
        "margin_frac": m,
        "n_landscape_points": int(len(landscape_df)),
    }
    logger.info(
        "  MOBO qNEHVI reference point (BoTorch maximization, y=[EA, -IPF]): "
        f"r=({r0:.4f}, {r1:.4f}); landscape EA∈[{ea_min:.2f},{ea_max:.2f}] J, "
        f"IPF∈[{ipf_min:.3f},{ipf_max:.3f}] kN, margin_frac={m:.4f}"
    )
    return ref_point, meta


def _mobo_evaluate_xy(
    xy_phys: torch.Tensor,
    models: List[nn.Module],
    approach: str,
    scaler_disp: StandardScaler,
    enc: OneHotEncoder,
    params: ScalingParams,
    lc_categories: List[str],
) -> torch.Tensor:
    """Evaluate surrogate objectives y=[EA, -IPF] at physical (angle, z_lc) rows.

    ``z_lc ∈ [0,1]`` maps to ``LC1`` if ``z < 0.5`` else ``LC2`` (first/second
    category order from the encoder).
    """
    xy = xy_phys.detach().cpu().double().numpy()
    rows = []
    for i in range(xy.shape[0]):
        ang = float(xy[i, 0])
        z = float(xy[i, 1])
        lc = str(lc_categories[1] if z >= 0.5 else lc_categories[0])
        m = compute_ea_ipf_ensemble(
            models, approach, ang, lc, scaler_disp, enc, params, d_eval=D_COMMON
        )
        rows.append([m["EA"], -float(m["IPF"])])
    return torch.tensor(rows, dtype=xy_phys.dtype, device=xy_phys.device)


def _mobo_init_model_list_gp(
    train_x_phys: torch.Tensor,
    train_y: torch.Tensor,
    bounds_phys: torch.Tensor,
) -> Tuple[Any, Any]:
    """Independent GPs per objective on normalized ``train_x`` (ModelListGP)."""
    train_xn = normalize(train_x_phys, bounds_phys)
    models = []
    for i in range(train_y.shape[-1]):
        yi = train_y[..., i : i + 1]
        models.append(SingleTaskGP(train_xn, yi))
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model


def run_multiobjective_mobo_qnehvi(
    models: List[nn.Module],
    approach: str,
    scaler_disp: StandardScaler,
    enc: OneHotEncoder,
    params: ScalingParams,
    landscape_df: pd.DataFrame,
    logger: logging.Logger,
    output_dir: str,
    cfg: Optional[MOBOQNEHVICfg] = None,
) -> Optional[Dict[str, Any]]:
    """Multi-objective Bayesian optimization with **qLogNEHVI** (BoTorch).

    Optimizes the surrogate vector objective
    ``(EA@D_COMMON, -IPF)`` over ``(θ, z_LC) ∈ [θ_min, θ_max] × [0,1]``,
    with ``z_LC < 0.5 → LC1`` else ``LC2`` (encoder category order).

    Acquisition uses **qLogNoisyExpectedHypervolumeImprovement** (qLogNEHVI),
    BoTorch's numerically stabilized noisy hypervolume improvement for batched
    queries; with deterministic PINN ensemble evaluations it reduces to the usual
    noiseless hypervolume improvement up to MC sampling noise.

    Requires ``landscape_df`` from :func:`run_multiobjective_sweep` to define a
    **documented** hypervolume reference point (nadir of dense grid + margin).
    """
    if not HAS_BOTORCH:
        logger.warning("  MOBO qNEHVI skipped: install botorch and gpytorch (pip install botorch).")
        return None
    if landscape_df is None or len(landscape_df) < 4:
        logger.warning("  MOBO qNEHVI skipped: landscape_df too small for reference point.")
        return None

    cfg = cfg or MOBO_QNEHVI_CFG
    bo_device = torch.device("cpu")
    tkwargs = {"dtype": torch.double, "device": bo_device}

    lc_categories = [str(x) for x in enc.categories_[0].tolist()]
    if len(lc_categories) < 2:
        logger.warning("  MOBO qNEHVI skipped: need at least two LC categories.")
        return None

    bounds_phys = torch.stack(
        [
            torch.tensor([CFG.angle_opt_min, 0.0], **tkwargs),
            torch.tensor([CFG.angle_opt_max, 1.0], **tkwargs),
        ],
        dim=0,
    )
    standard_bounds = torch.zeros(2, 2, **tkwargs)
    standard_bounds[1, :] = 1.0

    ref_point, ref_meta = build_qnehvi_reference_point_from_landscape(
        landscape_df, cfg.ref_margin_frac, logger, tkwargs
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=BadInitialCandidatesWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)

        init_x = draw_sobol_samples(
            bounds=standard_bounds, n=cfg.n_init, q=1, seed=CFG.seed
        ).squeeze(1).to(**tkwargs)
        train_x = unnormalize(init_x, bounds_phys)
        train_y = _mobo_evaluate_xy(train_x, models, approach, scaler_disp, enc, params, lc_categories)

        hv_trace: List[float] = []
        obs_rows: List[Dict[str, Any]] = []

        def _log_obs(batch: int, xy: torch.Tensor, y: torch.Tensor) -> None:
            xy_np = xy.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            for i in range(xy_np.shape[0]):
                z = float(xy_np[i, 1])
                lc = str(lc_categories[1] if z >= 0.5 else lc_categories[0])
                obs_rows.append({
                    "batch": batch,
                    "angle_deg": float(xy_np[i, 0]),
                    "lc_z": z,
                    "LC": lc,
                    "EA_J": float(y_np[i, 0]),
                    "IPF_kN": float(-y_np[i, 1]),
                    "y0_EA": float(y_np[i, 0]),
                    "y1_negIPF": float(y_np[i, 1]),
                })

        _log_obs(0, train_x, train_y)

        try:
            bd0 = DominatedPartitioning(ref_point=ref_point, Y=train_y)
            hv_trace.append(float(bd0.compute_hypervolume().item()))
        except Exception as ex:
            logger.warning(f"  MOBO qNEHVI: initial hypervolume computation failed ({ex}); HV trace may start at 0.")
            hv_trace.append(0.0)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([cfg.mc_samples]), seed=CFG.seed)

        for it in range(1, cfg.n_batches + 1):
            try:
                mll, model = _mobo_init_model_list_gp(train_x, train_y, bounds_phys)
                fit_gpytorch_mll(mll)
            except Exception as ex:
                logger.warning(f"  MOBO qNEHVI: GP fit failed at batch {it}: {ex}")
                break

            train_xn = normalize(train_x, bounds_phys)
            acq = qLogNoisyExpectedHypervolumeImprovement(
                model=model,
                ref_point=ref_point,
                X_baseline=train_xn,
                sampler=sampler,
                prune_baseline=True,
            )
            try:
                candidates_norm, _ = optimize_acqf(
                    acq_function=acq,
                    bounds=standard_bounds,
                    q=cfg.q,
                    num_restarts=cfg.num_restarts,
                    raw_samples=cfg.raw_samples,
                    options={"batch_limit": 5, "maxiter": 200},
                    sequential=True,
                )
            except Exception as ex:
                logger.warning(f"  MOBO qNEHVI: optimize_acqf failed at batch {it}: {ex}")
                break

            new_x = unnormalize(candidates_norm.detach(), bounds_phys)
            new_y = _mobo_evaluate_xy(new_x, models, approach, scaler_disp, enc, params, lc_categories)
            train_x = torch.cat([train_x, new_x], dim=0)
            train_y = torch.cat([train_y, new_y], dim=0)
            _log_obs(it, new_x, new_y)

            try:
                bd = DominatedPartitioning(ref_point=ref_point, Y=train_y)
                hv_trace.append(float(bd.compute_hypervolume().item()))
            except Exception as ex:
                logger.warning(f"  MOBO qNEHVI: hypervolume at batch {it} failed ({ex}).")
                hv_trace.append(hv_trace[-1] if hv_trace else 0.0)
            logger.info(
                f"    MOBO qNEHVI batch {it}/{cfg.n_batches}: "
                f"HV={hv_trace[-1]:.6f}, n_obs={train_x.shape[0]}"
            )

    obs_df = pd.DataFrame(obs_rows)
    obs_path = os.path.join(output_dir, "Table_mobo_qnehvi_observations.csv")
    obs_df.to_csv(obs_path, index=False)
    logger.info(f"  Saved: Table_mobo_qnehvi_observations.csv ({len(obs_df)} evaluations)")

    nd_mask = is_non_dominated(train_y)
    nd_x = train_x[nd_mask].detach().cpu().numpy()
    nd_y = train_y[nd_mask].detach().cpu().numpy()
    pareto_rows = []
    for i in range(nd_x.shape[0]):
        z = float(nd_x[i, 1])
        lc = str(lc_categories[1] if z >= 0.5 else lc_categories[0])
        pareto_rows.append({
            "angle_deg": float(nd_x[i, 0]),
            "LC": lc,
            "EA_J": float(nd_y[i, 0]),
            "IPF_kN": float(-nd_y[i, 1]),
        })
    pareto_df = pd.DataFrame(pareto_rows).sort_values("EA_J")
    pareto_path = os.path.join(output_dir, "Table_mobo_qnehvi_pareto_approx.csv")
    pareto_df.to_csv(pareto_path, index=False)
    logger.info(f"  Saved: Table_mobo_qnehvi_pareto_approx.csv ({len(pareto_df)} non-dominated points)")

    hv_df = pd.DataFrame({"batch": list(range(len(hv_trace))), "hypervolume_botorch": hv_trace})
    hv_path = os.path.join(output_dir, "Table_mobo_qnehvi_hypervolume_trace.csv")
    hv_df.to_csv(hv_path, index=False)
    logger.info(f"  Saved: Table_mobo_qnehvi_hypervolume_trace.csv")

    ref_path = os.path.join(output_dir, "mobo_qnehvi_reference_point.json")
    with open(ref_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **ref_meta,
                "mobo_config": asdict(cfg),
                "angle_bounds_deg": [CFG.angle_opt_min, CFG.angle_opt_max],
                "lc_encoding": "z in [0,1]; z<0.5 -> first encoder LC else second",
                "lc_order": lc_categories,
                "n_evaluations": int(train_x.shape[0]),
                "final_hypervolume": hv_trace[-1] if hv_trace else None,
            },
            f,
            indent=2,
        )
    logger.info(f"  Saved: mobo_qnehvi_reference_point.json")

    return {
        "train_x": train_x,
        "train_y": train_y,
        "hypervolume_trace": hv_trace,
        "pareto_approx_df": pareto_df,
        "observations_df": obs_df,
        "ref_point": ref_point,
        "ref_meta": ref_meta,
    }


def run_multiobjective_sweep(models: List[nn.Module], approach: str, scaler_disp: StandardScaler,
                              enc: OneHotEncoder, params: ScalingParams,
                              df_metrics: pd.DataFrame, logger: logging.Logger,
                              output_dir: str = ".",
                              df_all: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run multiobjective Pareto sweep and compute full landscape.
    
    Uses EA evaluated at ``D_COMMON`` for both LCs to ensure displacement-fair
    comparison. This isolates the intrinsic crashworthiness performance from
    the geometric crush-path advantage of LC2.

    Returns:
        pareto_df: DataFrame with Pareto-optimal solutions for each alpha
        landscape_df: DataFrame with (EA at ``D_COMMON``, IPF) predictions for
            all (angle, LC) combinations
    """
    # EA@D_COMMON ranges from experimental data for normalization (aligned with EA_common column if present)
    if "EA_common" in df_metrics.columns:
        ea_at_common = df_metrics["EA_common"].values.astype(float)
    else:
        if df_all is None:
            raise ValueError(
                "run_multiobjective_sweep: df_metrics has no EA_common column; "
                "pass df_all so EA@D_COMMON can be computed from raw curves."
            )
        ea_at_common = np.array([
            ea_at_common_from_row(row, df_all, logger=logger) for _, row in df_metrics.iterrows()])
    ea_min, ea_max = ea_at_common.min(), ea_at_common.max()
    ipf_min, ipf_max = df_metrics["IPF"].min(), df_metrics["IPF"].max()
    lc_categories = [str(x) for x in enc.categories_[0].tolist()]
    n_ang = 501 if not CFG.dry_run else 41
    angles = np.linspace(CFG.angle_opt_min, CFG.angle_opt_max, n_ang)

    # Compute full landscape ONCE (cache results to avoid redundant ensemble calls)
    # Both LCs evaluated at D_COMMON for displacement-fair comparison
    # Dense grid (501 x 2 = 1002 evaluations) enables smooth Pareto front characterization
    cache = {}
    landscape = []
    logger.info(f"  Evaluating design landscape: {len(angles)} angles x {len(lc_categories)} LCs "
                f"= {len(angles)*len(lc_categories)} evaluations at d={D_COMMON:.0f}mm")
    for lc in lc_categories:
        d_end_lc = disp_end_mm(lc)
        for ang in angles:
            m = compute_ea_ipf_ensemble(models, approach, ang, lc, scaler_disp, enc, params,
                                         d_eval=D_COMMON)
            # Full-range EA: only call separately for LCs where d_end > D_COMMON
            if d_end_lc > D_COMMON:
                m_full = compute_ea_ipf_ensemble(models, approach, ang, lc, scaler_disp, enc, params)
                ea_full = m_full["EA"]
            else:
                ea_full = m["EA"]  # d_end == D_COMMON, no difference
            ea_norm = (m["EA"] - ea_min) / (ea_max - ea_min + 1e-12)
            ipf_norm = (m["IPF"] - ipf_min) / (ipf_max - ipf_min + 1e-12)
            cache[(ang, lc)] = {"EA": m["EA"], "IPF": m["IPF"],
                                "EA_full": ea_full,
                                "ea_norm": ea_norm, "ipf_norm": ipf_norm,
                                "EA_std": m.get("EA_std", 0), "IPF_std": m.get("IPF_std", 0)}
            landscape.append({
                "angle": ang, "lc": lc,
                "EA": m["EA"], "IPF": m["IPF"],
                "EA_full": ea_full,
                "EA_norm": ea_norm, "IPF_norm": ipf_norm,
                "EA_std": m.get("EA_std", 0), "IPF_std": m.get("IPF_std", 0)
            })
    
    landscape_df = pd.DataFrame(landscape)
    
    # Pareto sweep (uses cache, no redundant ensemble calls)
    pareto_results = []
    logger.info(f"  Multi-objective Pareto sweep (EA evaluated at d={D_COMMON:.0f}mm for displacement-fair comparison):")
    logger.info("    alpha=0 -> minimize IPF (stable crushing)")
    logger.info(f"    alpha=1 -> maximize EA@{EA_COMMON_MM_TAG} (energy absorption rate)")
    
    for alpha in np.linspace(0.0, 1.0, 11):
        best_J, best_result = float('inf'), None
        for lc in lc_categories:
            for ang in angles:
                c = cache[(ang, lc)]
                J = -alpha * c["ea_norm"] + (1 - alpha) * c["ipf_norm"]
                if J < best_J:
                    best_J = J
                    best_result = {
                        "alpha": alpha, "angle": ang, "lc": lc,
                        "EA": c["EA"], "IPF": c["IPF"],
                        "EA_full": c["EA_full"],
                        "EA_norm": c["ea_norm"], "IPF_norm": c["ipf_norm"], "J": J,
                        "crushing_mode": "Stable" if lc == "LC1" else "Progressive/Catastrophic"
                    }
        if best_result:
            pareto_results.append(best_result)
            logger.info(f"    alpha={alpha:.1f}: theta={best_result['angle']:.1f} deg, {best_result['lc']}, "
                       f"EA@{EA_COMMON_MM_TAG}={best_result['EA']:.1f}J, EA_full={best_result['EA_full']:.1f}J, "
                       f"IPF={best_result['IPF']:.2f}kN ({best_result['crushing_mode']})")
    
    pareto_df = pd.DataFrame(pareto_results)
    
    # Per-LC conditional Pareto fronts (uses cache)
    pareto_by_lc = {}
    for lc in lc_categories:
        lc_pareto = []
        for alpha in np.linspace(0.0, 1.0, 11):
            best_J_lc, best_result_lc = float('inf'), None
            for ang in angles:
                c = cache[(ang, lc)]
                J = -alpha * c["ea_norm"] + (1 - alpha) * c["ipf_norm"]
                if J < best_J_lc:
                    best_J_lc = J
                    best_result_lc = {
                        "alpha": alpha, "angle": ang, "lc": lc,
                        "EA": c["EA"], "IPF": c["IPF"],
                        "EA_full": c["EA_full"],
                        "EA_norm": c["ea_norm"], "IPF_norm": c["ipf_norm"], "J": J,
                    }
            if best_result_lc:
                lc_pareto.append(best_result_lc)
        pareto_by_lc[lc] = pd.DataFrame(lc_pareto)
    
    # =========================================================================
    # PARETO DOMINANCE FILTERING (convexity-independent)
    # =========================================================================
    # The weighted-sum sweep above can only recover solutions on the convex hull
    # of the objective space. Pareto dominance filtering identifies ALL non-dominated
    # solutions directly from the 1,002-point landscape, regardless of front shape.
    # Objective: maximize EA (higher is better), minimize IPF (lower is better).
    # Point i is dominated if there exists any point j with EA_j >= EA_i AND IPF_j <= IPF_i
    # (with at least one strict inequality).
    logger.info(f"\n  Pareto dominance filtering on full landscape ({len(landscape)} points)...")
    
    ea_vals = np.array([pt["EA"] for pt in landscape], dtype=np.float64)
    ipf_vals = np.array([pt["IPF"] for pt in landscape], dtype=np.float64)
    n_pts = len(landscape)
    # Vectorised dominance (max EA, min IPF): j dominates i iff
    # EA[j]>=EA[i], IPF[j]<=IPF[i], and at least one strict inequality.
    ge_ea = ea_vals[:, None] >= ea_vals[None, :]
    le_ipf = ipf_vals[:, None] <= ipf_vals[None, :]
    strict = (ea_vals[:, None] > ea_vals[None, :]) | (ipf_vals[:, None] < ipf_vals[None, :])
    dominates = ge_ea & le_ipf & strict
    np.fill_diagonal(dominates, False)
    is_dominated = np.any(dominates, axis=0)
    
    pareto_indices = np.where(~is_dominated)[0]
    pareto_front = [landscape[i] for i in pareto_indices]
    pareto_front_df = pd.DataFrame(pareto_front).sort_values("EA").reset_index(drop=True)
    
    n_pareto = len(pareto_front_df)
    n_ws = len(pareto_df.drop_duplicates(subset=["angle", "lc"]))
    lc_on_front = pareto_front_df["lc"].unique().tolist()
    
    logger.info(f"    Non-dominated solutions: {n_pareto} (from {n_pts} candidates)")
    logger.info(f"    Weighted-sum unique solutions: {n_ws}")
    logger.info(f"    LCs on Pareto front: {lc_on_front}")
    
    if n_pareto == n_ws:
        logger.info(f"    Pareto dominance confirms weighted-sum front is COMPLETE (no hidden solutions)")
    elif n_pareto > n_ws:
        logger.info(f"    ** {n_pareto - n_ws} additional non-dominated solutions found beyond weighted-sum **")
        logger.info(f"    (These lie in non-convex regions inaccessible to linear scalarization)")
    
    # Log the full Pareto front
    logger.info(f"\n    Pareto-optimal designs (sorted by EA@{D_COMMON:.0f}mm):")
    for _, row in pareto_front_df.iterrows():
        logger.info(f"      theta={row['angle']:.1f} deg, {row['lc']}, "
                    f"EA@{EA_COMMON_MM_TAG}={row['EA']:.1f}J, IPF={row['IPF']:.3f}kN")
    
    # Save Pareto front table
    pareto_front_df.to_csv(os.path.join(output_dir, "Table_pareto_dominance.csv"), index=False)
    logger.info(f"  Saved: Table_pareto_dominance.csv")
    
    # Store in pareto_df attrs for downstream access
    pareto_df.attrs["pareto_dominance"] = pareto_front_df
    
    # LC dominance analysis
    lc_counts = pareto_df["lc"].value_counts()
    dominant_lc = lc_counts.idxmax() if len(lc_counts) > 0 else "N/A"
    logger.info(f"\n  LC selection in Pareto sweep (EA@{D_COMMON:.0f}mm): {dict(lc_counts)}")
    if len(lc_counts) == 1:
        logger.info(f"  ** {dominant_lc} dominates the entire displacement-fair Pareto front **")
        logger.info(f"     This indicates {dominant_lc} has intrinsically superior force response,")
        logger.info(f"     independent of crush-path length differences.")
    else:
        logger.info(f"  Both LCs appear on the displacement-fair front, indicating genuinely")
        logger.info(f"  different crashworthiness trade-off characteristics.")
    
    pareto_df.attrs["pareto_by_lc"] = pareto_by_lc

    # =========================================================================
    # CHEBYSHEV (AUGMENTED TCHEBYCHEFF) SCALARIZATION
    # =========================================================================
    # Linear weighted-sum cannot reach non-convex Pareto regions. Chebyshev
    # scalarization minimizes the worst-case weighted deviation from the utopia
    # point, recovering solutions in non-convex pockets.
    ea_utopia = max(c["ea_norm"] for c in cache.values())   # best EA_norm (max)
    ipf_utopia = min(c["ipf_norm"] for c in cache.values())  # best IPF_norm (min)
    rho = 0.05  # augmentation coefficient (standard in MCDM literature)

    cheby_results = []
    logger.info(f"\n  Chebyshev scalarization (utopia: EA_norm={ea_utopia:.3f}, IPF_norm={ipf_utopia:.3f}):")

    for alpha in np.linspace(0.0, 1.0, 11):
        best_J_cheb, best_result_cheb = float('inf'), None
        w_ea_ch = max(alpha, 1e-6)
        w_ipf_ch = max(1.0 - alpha, 1e-6)
        for lc in lc_categories:
            for ang in angles:
                c = cache[(ang, lc)]
                d_ea = ea_utopia - c["ea_norm"]      # distance to utopia in EA (want max)
                d_ipf = c["ipf_norm"] - ipf_utopia    # distance to utopia in IPF (want min)
                J_cheb = max(w_ea_ch * d_ea, w_ipf_ch * d_ipf) + rho * (d_ea + d_ipf)
                if J_cheb < best_J_cheb:
                    best_J_cheb = J_cheb
                    best_result_cheb = {
                        "alpha": alpha, "angle": ang, "lc": lc,
                        "EA": c["EA"], "IPF": c["IPF"],
                        "EA_full": c.get("EA_full", c["EA"]),
                        "EA_norm": c["ea_norm"], "IPF_norm": c["ipf_norm"],
                        "J_cheb": J_cheb,
                        "scalarization": "chebyshev",
                    }
        if best_result_cheb:
            cheby_results.append(best_result_cheb)
            logger.info(f"    alpha={alpha:.1f}: theta={best_result_cheb['angle']:.1f} deg, "
                       f"{best_result_cheb['lc']}, EA@{EA_COMMON_MM_TAG}={best_result_cheb['EA']:.1f}J, "
                       f"IPF={best_result_cheb['IPF']:.2f}kN")

    cheby_df = pd.DataFrame(cheby_results)

    # Compare: unique solutions found by each method
    ws_unique = set(zip(pareto_df["angle"].round(1), pareto_df["lc"]))
    ch_unique = set(zip(cheby_df["angle"].round(1), cheby_df["lc"])) if not cheby_df.empty else set()
    only_cheby = ch_unique - ws_unique
    logger.info(f"\n  Scalarization comparison:")
    logger.info(f"    Linear weighted-sum unique designs: {len(ws_unique)}")
    logger.info(f"    Chebyshev unique designs: {len(ch_unique)}")
    logger.info(f"    Designs found ONLY by Chebyshev: {len(only_cheby)}")
    if only_cheby:
        for ang, lc in sorted(only_cheby):
            logger.info(f"      theta={ang:.1f} deg, {lc}")

    pareto_df.attrs["cheby_df"] = cheby_df
    cheby_df.to_csv(os.path.join(output_dir, "Table_pareto_chebyshev.csv"), index=False)
    logger.info(f"  Saved: Table_pareto_chebyshev.csv")

    # =========================================================================
    # HYPERVOLUME INDICATOR
    # =========================================================================
    ref_ea = landscape_df["EA"].min()    # nadir EA (worst)
    ref_ipf = landscape_df["IPF"].max()  # nadir IPF (worst)
    hv = compute_hypervolume_2d(pareto_front_df, ref_ea, ref_ipf, logger)
    pareto_df.attrs["hypervolume"] = hv
    pareto_df.attrs["hv_ref_point"] = (ref_ea, ref_ipf)

    # Compare HV for Chebyshev front
    if not cheby_df.empty:
        ch_pts = cheby_df[["EA", "IPF"]].drop_duplicates().values
        ch_nd = []
        for i in range(len(ch_pts)):
            dominated = any(
                (ch_pts[j, 0] >= ch_pts[i, 0] and ch_pts[j, 1] <= ch_pts[i, 1] and
                 (ch_pts[j, 0] > ch_pts[i, 0] or ch_pts[j, 1] < ch_pts[i, 1]))
                for j in range(len(ch_pts)) if j != i
            )
            if not dominated:
                ch_nd.append(ch_pts[i])
        if ch_nd:
            ch_nd_df = pd.DataFrame(ch_nd, columns=["EA", "IPF"])
            hv_cheb = compute_hypervolume_2d(ch_nd_df, ref_ea, ref_ipf, logger, label="Chebyshev")
            pareto_df.attrs["hypervolume_chebyshev"] = hv_cheb

    return pareto_df, landscape_df


def fig_multiobjective_heatmaps(pareto_df: pd.DataFrame, landscape_df: pd.DataFrame, 
                                  output_dir: str, logger: logging.Logger,
                                  calibration: Optional[Dict] = None):
    """
    Generate comprehensive multi-objective visualization with 2D heatmaps.
    
    Shows:
    1. EA landscape as function of angle for both LCs
    2. IPF landscape as function of angle for both LCs
    3. Trade-off heatmap (composite objective)
    4. Pareto front with crushing mode annotations
    
    Key insight to visualize:
    - Low IPF priority (stable crushing) -> low angles + LC1
    - High EA priority (max energy) -> high angles + LC2 (but unstable)
    
    If calibration dict is provided, the +/-2sigma bands on EA and IPF are
    scaled by the conformal factor from the random protocol Hard-PINN
    (best available proxy for the full-data inverse model).
    """
    fig = plt.figure(figsize=(7.48, 10.5))
    gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.42, left=0.07, right=0.98, top=0.90, bottom=0.06)
    
    # Retrieve conformal factor for Hard-PINN (used for full-data inverse model bands)
    # Use random protocol as best available proxy; fall back to 1.0 if unavailable
    cf_ea = 1.0
    cf_ipf = 1.0
    if calibration is not None:
        for proto in ["random", "unseen"]:
            if proto in calibration and "hard" in calibration[proto]:
                # Use EA conformal factor from energy calibration, IPF from load calibration
                cf_ea = calibration[proto]["hard"].get("energy_conformal_factor",
                        calibration[proto]["hard"].get("conformal_factor", 1.0))
                cf_ipf = calibration[proto]["hard"].get("conformal_factor", 1.0)
                logger.info(f"  Design-space bands using {proto} Hard-PINN conformal factors: "
                            f"EA cf={cf_ea:.3f}, IPF cf={cf_ipf:.3f}")
                break
    
    # Prepare data for heatmaps
    angles = landscape_df["angle"].unique()
    lc_list = sorted(landscape_df["lc"].unique())
    
    # Panel (a): EA vs Angle for both LCs
    ax1 = fig.add_subplot(gs[0, 0])
    for lc in lc_list:
        lc_data = landscape_df[landscape_df["lc"] == lc].sort_values("angle")
        color = "#1f77b4" if lc == "LC1" else "#ff7f0e"
        linestyle = "-" if lc == "LC1" else "--"
        marker = "o" if lc == "LC1" else "s"
        label = f"{lc} (Stable)" if lc == "LC1" else f"{lc} (Progressive)"
        ax1.plot(lc_data["angle"], lc_data["EA"], color=color, linestyle=linestyle, 
                marker=marker, markersize=4, markevery=50, linewidth=2, label=label)
        # Add conformal-calibrated uncertainty band
        if "EA_std" in lc_data.columns and lc_data["EA_std"].max() > 0:
            ax1.fill_between(lc_data["angle"], 
                            lc_data["EA"] - 2*cf_ea*lc_data["EA_std"],
                            lc_data["EA"] + 2*cf_ea*lc_data["EA_std"],
                            color=color, alpha=0.15)
    ax1.set_xlabel("Interior Angle θ (°)")
    ax1.set_ylabel("Energy Absorption EA (J)")
    ax1.set_title("(a) Energy Absorption Landscape", fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Panel (b): IPF vs Angle for both LCs
    ax2 = fig.add_subplot(gs[0, 1])
    for lc in lc_list:
        lc_data = landscape_df[landscape_df["lc"] == lc].sort_values("angle")
        color = "#1f77b4" if lc == "LC1" else "#ff7f0e"
        linestyle = "-" if lc == "LC1" else "--"
        marker = "o" if lc == "LC1" else "s"
        label = f"{lc} (Stable)" if lc == "LC1" else f"{lc} (Progressive)"
        ax2.plot(lc_data["angle"], lc_data["IPF"], color=color, linestyle=linestyle,
                marker=marker, markersize=4, markevery=50, linewidth=2, label=label)
        if "IPF_std" in lc_data.columns and lc_data["IPF_std"].max() > 0:
            ax2.fill_between(lc_data["angle"],
                            lc_data["IPF"] - 2*cf_ipf*lc_data["IPF_std"],
                            lc_data["IPF"] + 2*cf_ipf*lc_data["IPF_std"],
                            color=color, alpha=0.15)
    ax2.set_xlabel("Interior Angle θ (°)")
    ax2.set_ylabel("Initial Peak Force IPF (kN)")
    ax2.set_title("(b) Peak Force Landscape", fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Panel (c): 2D Heatmap - Trade-off surface for LC1
    ax3 = fig.add_subplot(gs[0, 2])
    lc1_data = landscape_df[landscape_df["lc"] == "LC1"].sort_values("angle")
    alphas = np.linspace(0, 1, 11)
    angles_lc1 = lc1_data["angle"].values
    # Create heatmap: rows = alpha, cols = angle, values = J
    Z1 = np.zeros((len(alphas), len(angles_lc1)))
    for i, alpha in enumerate(alphas):
        for j, (_, row) in enumerate(lc1_data.iterrows()):
            Z1[i, j] = -alpha * row["EA_norm"] + (1 - alpha) * row["IPF_norm"]
    
    im3 = ax3.imshow(Z1, aspect='auto', cmap='RdYlGn_r', origin='lower',
                     extent=[angles_lc1.min(), angles_lc1.max(), 0, 1])
    ax3.set_xlabel("Interior Angle θ (°)")
    ax3.set_ylabel("EA Weight α")
    ax3.set_title("(c) Trade-off Surface: LC1", fontweight='bold')
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.72, pad=0.02)
    cbar3.set_label("Objective J (lower=better)")
    
    # Panel (d): 2D Heatmap - Trade-off surface for LC2
    ax4 = fig.add_subplot(gs[1, 0])
    lc2_data = landscape_df[landscape_df["lc"] == "LC2"].sort_values("angle")
    angles_lc2 = lc2_data["angle"].values
    Z2 = np.zeros((len(alphas), len(angles_lc2)))
    for i, alpha in enumerate(alphas):
        for j, (_, row) in enumerate(lc2_data.iterrows()):
            Z2[i, j] = -alpha * row["EA_norm"] + (1 - alpha) * row["IPF_norm"]
    
    im4 = ax4.imshow(Z2, aspect='auto', cmap='RdYlGn_r', origin='lower',
                     extent=[angles_lc2.min(), angles_lc2.max(), 0, 1])
    ax4.set_xlabel("Interior Angle θ (°)")
    ax4.set_ylabel("EA Weight α")
    ax4.set_title("(d) Trade-off Surface: LC2", fontweight='bold')
    cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.72, pad=0.02)
    cbar4.set_label("Objective J (lower=better)")
    
    # Panel (e): Pareto front in EA-IPF space
    ax5 = fig.add_subplot(gs[1, 1])
    # Background: all design points
    for lc in lc_list:
        lc_data = landscape_df[landscape_df["lc"] == lc]
        color = "#1f77b4" if lc == "LC1" else "#ff7f0e"
        ax5.scatter(lc_data["EA"], lc_data["IPF"], c=color, s=5, alpha=0.15, label=f"{lc} designs")
    
    # Pareto points
    pareto_lc1 = pareto_df[pareto_df["lc"] == "LC1"]
    pareto_lc2 = pareto_df[pareto_df["lc"] == "LC2"]
    
    if len(pareto_lc1) > 0:
        ax5.scatter(pareto_lc1["EA"], pareto_lc1["IPF"], c="#1f77b4", s=150, marker='*',
                   edgecolors='black', linewidths=1.5, zorder=10, label="Pareto (LC1)")
    if len(pareto_lc2) > 0:
        ax5.scatter(pareto_lc2["EA"], pareto_lc2["IPF"], c="#ff7f0e", s=150, marker='D',
                   edgecolors='black', linewidths=1.5, zorder=10, label="Pareto (LC2)")
    
    # Connect Pareto points with line
    pareto_sorted = pareto_df.sort_values("EA")
    ax5.plot(pareto_sorted["EA"], pareto_sorted["IPF"], 'k--', linewidth=1.5, alpha=0.7, zorder=5)
    
    # Annotate key points (positioned to avoid overlap)
    if len(pareto_df) > 0:
        # Low alpha (IPF priority)
        low_alpha = pareto_df[pareto_df["alpha"] == pareto_df["alpha"].min()].iloc[0]
        ax5.annotate("α=0 (Stable)", (low_alpha["EA"], low_alpha["IPF"]),
                    xytext=(-40, 24), textcoords='offset points',
                    fontsize=7.5, ha='center',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.6))
        
        # High alpha (EA priority)
        high_alpha = pareto_df[pareto_df["alpha"] == pareto_df["alpha"].max()].iloc[0]
        ax5.annotate("α=1 (Max EA)", (high_alpha["EA"], high_alpha["IPF"]),
                    xytext=(36, -28), textcoords='offset points',
                    fontsize=7.5, ha='center',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.6))
    
    ax5.set_xlabel("Energy Absorption EA (J)")
    ax5.set_ylabel("Initial Peak Force IPF (kN)")
    ax5.set_title("(e) Pareto Front: EA vs IPF Trade-off", fontweight='bold')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3, linestyle='--')
    
    # Panel (f): Optimal angle and LC vs alpha
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Create dual y-axis for angle and LC
    ax6_twin = ax6.twinx()
    
    # Optimal angle vs alpha
    ax6.plot(pareto_df["alpha"], pareto_df["angle"], 'b-o', linewidth=2, markersize=6, label="Optimal θ")
    ax6.set_xlabel("EA Weight α")
    ax6.set_ylabel("Optimal Angle θ (°)", color='black')
    ax6.tick_params(axis='y', labelcolor='black')
    ax6.set_ylim(44, 71)
    
    # LC choice as bar colors
    lc_numeric = [1 if lc == "LC1" else 2 for lc in pareto_df["lc"]]
    colors = ["#1f77b4" if lc == "LC1" else "#ff7f0e" for lc in pareto_df["lc"]]
    ax6_twin.bar(pareto_df["alpha"], lc_numeric, width=0.08, alpha=0.4, color=colors)
    ax6_twin.set_ylabel("Loading (LC)", color='gray')
    ax6_twin.set_ylim(0.5, 2.5)
    ax6_twin.set_yticks([1, 2])
    ax6_twin.set_yticklabels(["LC1", "LC2"])
    ax6_twin.tick_params(axis='y', labelcolor='gray')
    
    ax6.set_title("(f) Optimal Design vs Priority Weight", fontweight='bold')
    ax6.grid(True, alpha=0.3, linestyle='--', axis='x')
    
    # Add annotation boxes
    ax6.axvspan(0, 0.3, alpha=0.1, color='blue', label='IPF priority zone')
    ax6.axvspan(0.7, 1.0, alpha=0.1, color='red', label='EA priority zone')
    
    fig.suptitle("Multi-Objective Crashworthiness Optimization: EA vs IPF Trade-off",
                 fontweight='bold', y=0.995)
    
    apply_fig_style(fig)
    fig.savefig(os.path.join(output_dir, "Fig_multiobjective_heatmaps.png"), 
               dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("  Saved: Fig_multiobjective_heatmaps.png")


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def add_subplot_label(ax, label, x=-0.05, y=1.02):
    """Add subplot label (a), (b), etc. — font matches figure width; position avoids title clash."""
    fw = float(ax.figure.get_figwidth())
    sf = scaled_fonts(fw)
    ax.text(x, y, f"({label})", transform=ax.transAxes, fontsize=sf["panel"],
            fontweight='bold', va='top', ha='left', clip_on=False)


def fig_residual_histograms(dual_results: Dict, output_dir: str, logger: logging.Logger):
    """Generate residual histograms for all models (unseen protocol only)."""
    for protocol in ["unseen"]:
        if protocol not in dual_results:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        labels = ['a', 'b', 'c', 'd', 'e', 'f']
        label_idx = 0
        for i, approach in enumerate(["ddns", "soft", "hard"]):
            if approach not in dual_results[protocol]:
                continue
            m = dual_results[protocol][approach]["metrics"]
            for j, (key, xlabel, unit) in enumerate([("load", "Load Residual", "kN"), ("energy", "Energy Residual", "J")]):
                ax = axes[j, i]
                errors = m[f"{key}_errors"]
                ax.hist(errors, bins=30, color=COLORS[approach], edgecolor='black', alpha=0.7, linewidth=0.8)
                ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
                ax.set_xlabel(f"{xlabel} ({unit})")
                ax.set_ylabel("Frequency")
                ax.set_title(f"{MODEL_LABELS[approach]}: μ={np.mean(errors):.3f}, σ={np.std(errors):.3f}")
                add_subplot_label(ax, labels[label_idx])
                label_idx += 1
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
        fig.suptitle(f"Residual Distributions ({protocol_label(protocol)})", fontweight='bold', y=0.995)
        apply_fig_style(fig)
        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
        fig.savefig(os.path.join(output_dir, f"Fig_residuals_{protocol}.png"), dpi=600, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.info(f"  Saved: Fig_residuals_{protocol}.png")


def fig_boxplot_comparison(dual_results: Dict, output_dir: str, logger: logging.Logger):
    """Generate boxplot comparison of R² across ensemble members (both protocols, load + energy)."""
    protocols_avail = [p for p in ["random", "unseen"] if p in dual_results]
    if not protocols_avail:
        return
    
    n_protos = len(protocols_avail)
    fig, axes = plt.subplots(n_protos, 2, figsize=(10, 4.5 * n_protos), squeeze=False)
    approaches = ["ddns", "soft", "hard"]
    
    for row, protocol in enumerate(protocols_avail):
        for col, (metric_key, ylabel) in enumerate([("load_r2", "Load $R^2$"), ("energy_r2", "Energy $R^2$")]):
            ax = axes[row, col]
            data_vals = []
            labels = []
            colors = []
            for approach in approaches:
                if approach in dual_results[protocol]:
                    r2_vals = [m[metric_key] for m in dual_results[protocol][approach]["member_metrics"]]
                    data_vals.append(r2_vals)
                    labels.append(MODEL_LABELS[approach])
                    colors.append(COLORS[approach])
            
            bp = ax.boxplot(data_vals, tick_labels=labels, patch_artist=True, widths=0.5)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(1.5)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{protocol_label(protocol)}", fontweight='bold')
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            add_subplot_label(ax, chr(ord('a') + row * 2 + col))
    
    fig.suptitle("Ensemble Performance Distribution", fontweight='bold', y=0.995)
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.04, 0.04, 0.98, 0.90])
    fig.savefig(os.path.join(output_dir, "Fig_boxplot_comparison.png"), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("  Saved: Fig_boxplot_comparison.png")


def fig_parity_plots(dual_results: Dict, output_dir: str, logger: logging.Logger):
    """Generate parity plots (predicted vs actual) for unseen protocol only."""
    for protocol in ["unseen"]:
        if protocol not in dual_results:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        labels = ['a', 'b', 'c', 'd', 'e', 'f']
        label_idx = 0
        for i, approach in enumerate(["ddns", "soft", "hard"]):
            if approach not in dual_results[protocol]:
                continue
            m = dual_results[protocol][approach]["metrics"]
            for j, (key, ylabel, unit) in enumerate([("load", "Load", "kN"), ("energy", "Energy", "J")]):
                ax = axes[j, i]
                y_true, y_pred = m["true_values"][key], m["predictions"][key]
                ax.scatter(y_true, y_pred, c=COLORS[approach], alpha=0.6, s=25, edgecolors='white', linewidths=0.5, marker=MARKERS[approach])
                lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
                margin = (lims[1] - lims[0]) * 0.05
                lims = [lims[0] - margin, lims[1] + margin]
                ax.plot(lims, lims, 'k--', lw=1.5, label='Perfect fit')
                ax.set_xlim(lims)
                ax.set_ylim(lims)
                ax.set_xlabel(f"Actual {ylabel} ({unit})")
                ax.set_ylabel(f"Predicted {ylabel} ({unit})")
                ax.set_title(f"{MODEL_LABELS[approach]}: R²={m[f'{key}_r2']:.4f}")
                ax.legend(loc='lower right')
                add_subplot_label(ax, labels[label_idx])
                label_idx += 1
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
        fig.suptitle(f"Parity Plots ({protocol_label(protocol)})", fontweight='bold', y=0.995)
        apply_fig_style(fig)
        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
        fig.savefig(os.path.join(output_dir, f"Fig_parity_{protocol}.png"), dpi=600, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.info(f"  Saved: Fig_parity_{protocol}.png")


def fig_cross_protocol_comparison(dual_results: Dict, output_dir: str, logger: logging.Logger):
    """Generate cross-protocol comparison bar chart. [CHANGE C] y-axis starts at 0.5"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(3)
    width = 0.35
    approaches = ["ddns", "soft", "hard"]
    for i, (key, ylabel, subplot_label) in enumerate([("load_r2", "Load $R^2$", 'a'), ("energy_r2", "Energy $R^2$", 'b')]):
        ax = axes[i]
        for p_idx, protocol in enumerate(["random", "unseen"]):
            if protocol not in dual_results:
                continue
            vals, errs, colors = [], [], []
            for approach in approaches:
                if approach in dual_results[protocol]:
                    member_vals = [m[key] for m in dual_results[protocol][approach]["member_metrics"]]
                    vals.append(np.mean(member_vals))
                    _dd = 1 if len(member_vals) > 1 else 0
                    errs.append(float(np.std(member_vals, ddof=_dd)))
                    colors.append(COLORS[approach])
                else:
                    vals.append(0)
                    errs.append(0)
                    colors.append('gray')
            hatch = '' if p_idx == 0 else '//'
            alpha = 1.0 if p_idx == 0 else 0.6
            for j, (val, err, color) in enumerate(zip(vals, errs, colors)):
                ax.bar(x[j] + p_idx * width, val, width, yerr=err, color=color, edgecolor='black', hatch=hatch, capsize=3, error_kw={'linewidth': 1.0}, alpha=alpha)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([MODEL_LABELS[a] for a in approaches])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.50, 1.02)  # [CHANGE C] Start at 0.5
        add_subplot_label(ax, subplot_label)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    legend_elements = [mpatches.Patch(facecolor=COLORS["ddns"], edgecolor='black', label='DDNS'),
                       mpatches.Patch(facecolor=COLORS["soft"], edgecolor='black', label='Soft-PINN'),
                       mpatches.Patch(facecolor=COLORS["hard"], edgecolor='black', label='Hard-PINN'),
                       mpatches.Patch(facecolor='white', edgecolor='black', hatch='', label='Random Split'),
                       mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', label='Unseen Angle')]
    fig.suptitle("Cross-Protocol Performance Comparison", fontweight='bold', y=1.01)
    fig.legend(handles=legend_elements, loc='upper center', ncol=5,
               bbox_to_anchor=(0.5, 1.005), frameon=True, columnspacing=0.8, handletextpad=0.35)
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.03, 0.06, 0.97, 0.82])
    fig.savefig(os.path.join(output_dir, "Fig_cross_protocol.png"), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("  Saved: Fig_cross_protocol.png")


def fig_unseen_curves(dual_results: Dict, df_all: pd.DataFrame, output_dir: str,
                      logger: logging.Logger, calibration: Optional[Dict] = None):
    """Generate curves for unseen angle using ensemble mean with conformal-calibrated bands.
    [CHANGE A & B] LC-specific displacement ranges.
    Uses the ensemble-mean prediction (statistically coherent with the ±2σ band).
    If calibration dict is provided, scales the band by the conformal factor.
    """
    if "unseen" not in dual_results:
        return
    scaler_disp = dual_results["unseen"]["scaler_disp"]
    enc = dual_results["unseen"]["enc"]
    params = dual_results["unseen"]["params"]
    val_df = dual_results["unseen"]["val_df"]
    lcs = sorted(val_df["LC"].unique())
    
    # Retrieve conformal factors (load and energy) per approach
    conformal_factors = {}
    for approach in ["ddns", "soft", "hard"]:
        cf_load = 1.0
        cf_energy = 1.0
        if (calibration is not None and "unseen" in calibration
                and approach in calibration["unseen"]):
            cal = calibration["unseen"][approach]
            cf_load = cal.get("conformal_factor", 1.0)
            cf_energy = cal.get("energy_conformal_factor", 1.0)
            logger.info(f"    {MODEL_LABELS.get(approach, approach)} conformal factors: "
                        f"load={cf_load:.3f}, energy={cf_energy:.3f}")
        conformal_factors[approach] = {"load": cf_load, "energy": cf_energy}
    
    # Load curves (ensemble mean + conformal-calibrated bands)
    fig, axes = plt.subplots(1, len(lcs), figsize=(5.5 * len(lcs), 4.5))
    if len(lcs) == 1:
        axes = [axes]
    for idx, lc in enumerate(lcs):
        ax = axes[idx]
        disp_end = disp_end_mm(lc)
        sub = val_df[val_df["LC"] == lc].sort_values("disp_mm")
        disps = sub["disp_mm"].values
        ax.plot(disps, sub["load_kN"].values, color='black', linestyle='-', linewidth=2.5, label="Experiment", zorder=10)
        for approach in ["ddns", "soft", "hard"]:
            if approach in dual_results["unseen"]:
                models = dual_results["unseen"][approach]["models"]
                Fm, Fs, _, _ = predict_curve_ensemble(
                    models, approach, 60.0, lc, disps, scaler_disp, enc, params)
                ens_r2 = dual_results["unseen"][approach]["metrics"]["load_r2"]
                cf = conformal_factors[approach]["load"]
                ax.plot(disps, Fm, color=COLORS[approach], linestyle=LINESTYLES[approach],
                        marker=MARKERS[approach], markevery=35, markersize=5, linewidth=1.5,
                        label=f"{MODEL_LABELS[approach]} (R\u00b2={ens_r2:.3f})")
                ax.fill_between(disps, Fm - 2*cf*Fs, Fm + 2*cf*Fs,
                                color=COLORS[approach], alpha=0.15, linewidth=0)
        ax.set_xlabel("Displacement (mm)")
        ax.set_ylabel("Load (kN)")
        ax.set_title(f"{lc}, $\\theta$ = 60° (Unseen Angle)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, framealpha=0.95)
        ax.set_xlim(0, disp_end)
        add_subplot_label(ax, chr(ord('a') + idx))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    fig.suptitle("Load Predictions for Unseen Angle (Ensemble Mean, Conformal ±2σ)", fontweight='bold', y=0.99)
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.02, 0.22, 0.98, 0.94])
    fig.savefig(os.path.join(output_dir, "Fig_unseen_load_curves.png"), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("  Saved: Fig_unseen_load_curves.png")
    
    # Energy curves (ensemble mean + conformal-calibrated bands)
    fig, axes = plt.subplots(1, len(lcs), figsize=(5.5 * len(lcs), 4.5))
    if len(lcs) == 1:
        axes = [axes]
    for idx, lc in enumerate(lcs):
        ax = axes[idx]
        disp_end = disp_end_mm(lc)
        sub = val_df[val_df["LC"] == lc].sort_values("disp_mm")
        disps = sub["disp_mm"].values
        ax.plot(disps, sub["energy_J"].values, color='black', linestyle='-', linewidth=2.5, label="Experiment", zorder=10)
        for approach in ["ddns", "soft", "hard"]:
            if approach in dual_results["unseen"]:
                models = dual_results["unseen"][approach]["models"]
                _, _, Em, Es = predict_curve_ensemble(
                    models, approach, 60.0, lc, disps, scaler_disp, enc, params)
                ens_r2 = dual_results["unseen"][approach]["metrics"]["energy_r2"]
                cf = conformal_factors[approach]["energy"]
                ax.plot(disps, Em, color=COLORS[approach], linestyle=LINESTYLES[approach],
                        marker=MARKERS[approach], markevery=35, markersize=5, linewidth=1.5,
                        label=f"{MODEL_LABELS[approach]} (R\u00b2={ens_r2:.3f})")
                ax.fill_between(disps, Em - 2*cf*Es, Em + 2*cf*Es,
                                color=COLORS[approach], alpha=0.15, linewidth=0)
        ax.set_xlabel("Displacement (mm)")
        ax.set_ylabel("Energy (J)")
        ax.set_title(f"{lc}, $\\theta$ = 60° (Unseen Angle)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, framealpha=0.95)
        ax.set_xlim(0, disp_end)
        add_subplot_label(ax, chr(ord('a') + idx))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    fig.suptitle("Energy Predictions for Unseen Angle (Ensemble Mean, Conformal ±2σ)", fontweight='bold', y=0.99)
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.02, 0.22, 0.98, 0.94])
    fig.savefig(os.path.join(output_dir, "Fig_unseen_energy_curves.png"), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("  Saved: Fig_unseen_energy_curves.png")


def fig_random_grid_curves(dual_results: Dict, df_all: pd.DataFrame, output_dir: str, logger: logging.Logger):
    """Generate grid of curves. [CHANGE A & B] LC-specific displacement ranges."""
    if "random" not in dual_results:
        return
    scaler_disp = dual_results["random"]["scaler_disp"]
    enc = dual_results["random"]["enc"]
    params = dual_results["random"]["params"]
    angles = sorted(df_all["Angle"].unique())
    lcs = sorted(df_all["LC"].unique())
    
    # Create figure at print width (190mm = 7.48in for Composite Structures full-page)
    fig, axes = plt.subplots(len(lcs), len(angles), figsize=(7.48, 2.5 * len(lcs) + 0.5))
    
    for i, lc in enumerate(lcs):
        disp_end = disp_end_mm(lc)  # [CHANGE B] Define for this LC
        for j, ang in enumerate(angles):
            ax = axes[i, j] if len(lcs) > 1 else axes[j]
            sub = df_all[(df_all["LC"] == lc) & (df_all["Angle"] == ang)].sort_values("disp_mm")
            if len(sub) == 0:
                ax.set_visible(False)
                continue
            disps = sub["disp_mm"].values
            ax.plot(disps, sub["load_kN"].values, 'k-', linewidth=1.2, label='Experiment')
            for approach in ["ddns", "soft", "hard"]:
                if approach in dual_results["random"]:
                    models = dual_results["random"][approach]["models"]
                    Fm, _, _, _ = predict_curve_ensemble(models, approach, ang, lc, disps, scaler_disp, enc, params)
                    ax.plot(disps, Fm, color=COLORS[approach], linestyle=LINESTYLES[approach], linewidth=0.8, alpha=0.85)
            ax.set_title(f"{lc}, {ang}°", fontsize=8, pad=2)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel("Load (kN)", fontsize=8)
            if i == len(lcs) - 1:
                ax.set_xlabel("Disp. (mm)", fontsize=8)
            ax.set_xlim(0, disp_end)  # [CHANGE A] LC-specific
    
    # Create legend elements
    legend_elements = [Line2D([0], [0], color='k', linewidth=1.2, label='Experiment')]
    legend_elements += [Line2D([0], [0], color=COLORS[a], linestyle=LINESTYLES[a], linewidth=1.0, label=MODEL_LABELS[a]) for a in ["ddns", "soft", "hard"]]
    
    # Place legend below the title, above the plots
    fig.legend(handles=legend_elements, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 0.995), frameon=True, framealpha=0.95)
    
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.86])
    fig.savefig(os.path.join(output_dir, "Fig_random_grid_curves.png"), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("  Saved: Fig_random_grid_curves.png")


def fig_ablation_study(df_ablation: pd.DataFrame, output_dir: str, logger: logging.Logger):
    """Generate ablation study figure. [CHANGE D] Updated for unseen protocol."""
    if df_ablation.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax = axes[0, 0]
    ax.semilogx(df_ablation["w_phys"] + 0.01, df_ablation["load_r2"], color=COLORS["soft"], marker='o', markersize=7, linewidth=1.5, label="Load $R^2$", markerfacecolor='white', markeredgecolor=COLORS["soft"], markeredgewidth=1.5)
    ax.semilogx(df_ablation["w_phys"] + 0.01, df_ablation["energy_r2"], color=COLORS["ddns"], marker='s', linestyle='--', markersize=7, linewidth=1.5, label="Energy $R^2$", markerfacecolor=COLORS["ddns"])
    ax.set_xlabel(r"Physics weight $w_{phys}$")
    ax.set_ylabel("$R^2$")
    ax.legend(loc='best')
    ax.axvline(1.0, color=COLORS["hard"], linestyle=':', linewidth=2.0)
    add_subplot_label(ax, 'a')
    ax = axes[0, 1]
    ax.semilogx(df_ablation["w_phys"] + 0.01, df_ablation["load_rmse"], color=COLORS["soft"], marker='o', markersize=7, linewidth=1.5, label="Load RMSE", markerfacecolor='white', markeredgecolor=COLORS["soft"], markeredgewidth=1.5)
    ax.semilogx(df_ablation["w_phys"] + 0.01, df_ablation["energy_rmse"], color=COLORS["ddns"], marker='s', linestyle='--', markersize=7, linewidth=1.5, label="Energy RMSE", markerfacecolor=COLORS["ddns"])
    ax.set_xlabel(r"Physics weight $w_{phys}$")
    ax.set_ylabel("RMSE")
    ax.legend(loc='best')
    ax.axvline(1.0, color=COLORS["hard"], linestyle=':', linewidth=2.0)
    add_subplot_label(ax, 'b')
    ax = axes[1, 0]
    ax.semilogx(df_ablation["w_phys"] + 0.01, df_ablation["training_time"], color=COLORS["hard"], marker='^', markersize=7, linewidth=1.5, markerfacecolor=COLORS["hard"])
    ax.set_xlabel(r"Physics weight $w_{phys}$")
    ax.set_ylabel("Training Time (s)")
    ax.axvline(1.0, color=COLORS["hard"], linestyle=':', linewidth=2.0)
    add_subplot_label(ax, 'c')
    ax = axes[1, 1]
    ax.axis('off')
    best_idx = df_ablation["load_r2"].idxmax()
    best_w = df_ablation.loc[best_idx, "w_phys"]
    protocol_str = df_ablation["protocol"].iloc[0] if "protocol" in df_ablation.columns else "unseen"
    summary_text = f"Ablation Study Summary:\n\n• Protocol: {protocol_str.upper()} angle θ=60°\n\n• Tested $w_{{phys}}$ values: {list(df_ablation['w_phys'].values)}\n\n• Best Load $R^2$ at $w_{{phys}}$ = {best_w:.1f}\n\n• Optimal range: 5-20"
    _sf_ab = scaled_fonts(10.0)
    ax.text(0.06, 0.92, summary_text, transform=ax.transAxes, fontsize=_sf_ab["annot"],
            verticalalignment='top', fontfamily='serif',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='black', pad=0.35))
    add_subplot_label(ax, 'd')
    fig.suptitle("Ablation Study: Effect of Physics Weight (Unseen θ=60°)", fontweight='bold', y=0.98)
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    fig.savefig(os.path.join(output_dir, "Fig_ablation_study.png"), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("  Saved: Fig_ablation_study.png")


def fig_bo_posterior_evaluation(opt_results: Dict, output_dir: str, logger: logging.Logger, tag: str = ""):
    """Generate GP-BO Posterior Evaluation figure (one file per inverse target).

    2x4 grid: each panel is a BO snapshot with both LC1 and LC2 — true objective
    (when available), GP mean +/- 2 sigma, and observations.
    
    Each subplot corresponds to a BO iteration and shows:
      - True objective landscape for each LC (solid line, serves as ground truth)
      - GP posterior mean +/- 2 sigma for each LC (dashed line + shaded band)
      - Observations so far for each LC (scatter)
    
    This demonstrates how GP-BO jointly searches over (theta, LC), reduces
    uncertainty for BOTH loading conditions, and discovers the optimal (theta*, LC*).
    """
    # Try to get joint results first (preferred)
    gpbo_joint = opt_results.get("gpbo_joint")
    
    # Colors and markers for each LC
    lc_colors = {"LC1": "#1f77b4", "LC2": "#ff7f0e"}  # Blue for LC1, Orange for LC2
    lc_markers = {"LC1": "*", "LC2": "D"}  # Star for LC1, Diamond for LC2
    lc_labels = {"LC1": "LC1", "LC2": "LC2"}
    
    if gpbo_joint and gpbo_joint.get("posterior_snapshots"):
        # Use joint posterior snapshots (correct approach)
        snapshots = gpbo_joint["posterior_snapshots"]
        lc_list = gpbo_joint.get("lc_list", ["LC1", "LC2"])
        true_landscapes = gpbo_joint.get("true_landscapes", {})
        theta_dense_ref = gpbo_joint.get("theta_dense", None)
        n_iters = len(snapshots)
        
        if n_iters == 0:
            logger.warning("  No joint posterior snapshots available")
            return
        
        # Select 8 iterations to show, spanning the FULL iteration range
        if n_iters >= 40:
            snapshot_indices = [0, 8, 14, 19, 26, 33, 36, n_iters - 1]
        elif n_iters >= 20:
            step = (n_iters - 1) / 7.0
            snapshot_indices = [int(round(i * step)) for i in range(8)]
            snapshot_indices[-1] = n_iters - 1
        elif n_iters >= 8:
            step = n_iters // 8
            snapshot_indices = [i * step for i in range(8)]
            snapshot_indices[-1] = n_iters - 1
        else:
            snapshot_indices = list(range(n_iters))
            while len(snapshot_indices) < 8:
                snapshot_indices.append(snapshot_indices[-1])
        
        # Clamp indices to valid range
        snapshot_indices = [min(idx, n_iters - 1) for idx in snapshot_indices]
        
        # Create 2x4 figure (extra height for multi-line subplot titles + bottom legend)
        fig, axes = plt.subplots(2, 4, figsize=(7.48, 5.35))
        axes = axes.flatten()
        
        for ax_idx, snap_idx in enumerate(snapshot_indices[:8]):
            ax = axes[ax_idx]
            snap = snapshots[snap_idx]
            theta_grid = snap["theta_grid"]
            
            # Plot each LC from the SAME snapshot (same GP model)
            for lc in lc_list:
                color = lc_colors.get(lc, "#1f77b4")
                marker = lc_markers.get(lc, "o")
                label_lc = lc_labels.get(lc, lc)
                
                # Plot TRUE objective landscape (solid, thin, as reference)
                if lc in true_landscapes and theta_dense_ref is not None:
                    ax.plot(theta_dense_ref, true_landscapes[lc], color=color,
                            linestyle='-', linewidth=1.0, alpha=0.5,
                            label=f"True obj. ({label_lc})" if ax_idx == 0 else "")
                
                # Get posterior mean and sigma for this LC from joint snapshot
                mu = snap.get(f"mu_{lc}")
                sigma = snap.get(f"sigma_{lc}")
                X_obs = snap.get(f"X_obs_{lc}")
                y_obs = snap.get(f"y_obs_{lc}")
                
                if mu is None or sigma is None:
                    continue
                
                # Plot posterior mean and uncertainty band
                ax.plot(theta_grid, mu, color=color, linestyle='--', linewidth=1.5, 
                       label=rf"$\mu(\theta)$ | {label_lc}" if ax_idx == 0 else "")
                ax.fill_between(theta_grid, mu - 2*sigma, mu + 2*sigma, 
                               color=color, alpha=0.15, linewidth=0,
                               label=rf"$\pm 2\sigma$ ({label_lc})" if ax_idx == 0 else "")
                
                # Plot observations
                if X_obs is not None and len(X_obs) > 0:
                    ax.scatter(X_obs, y_obs, c=color, s=60, marker=marker, 
                              edgecolors='black', linewidths=0.5, zorder=10,
                              label=f"Obs ({label_lc})" if ax_idx == 0 else "")
            
            # Count total observations across both LCs for this snapshot
            n_obs_total = sum(len(snap.get(f"X_obs_{lc}", [])) for lc in lc_list)
            n_obs_per_lc = {lc: len(snap.get(f"X_obs_{lc}", [])) for lc in lc_list}
            obs_str = ", ".join([f"{lc}={n_obs_per_lc[lc]}" for lc in lc_list])
            
            # Use actual iteration number from snapshot, not snapshot index
            iter_num = snap.get("iteration", snap_idx + 1)
            total_evals = gpbo_joint.get("n_evals", n_iters)
            
            ax.set_xlabel(r"Angle $\theta$ (deg.)")
            ax.set_ylabel("Objective")
            ax.set_title(f"Iter {iter_num}/{total_evals}\n({obs_str})", fontsize=7)
            ax.set_xlim(45, 70)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        # Create shared legend at the bottom
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3,
                   bbox_to_anchor=(0.5, 0.02), frameon=True, framealpha=0.95, columnspacing=0.6)
        
        title_suffix = f" ({tag})" if tag else ""
        fig.suptitle(f"GP-BO Posterior Evaluation{title_suffix}", fontweight='bold', y=0.99)
        apply_fig_style(fig)
        for _ax in axes:
            if _ax.get_title():
                _ax.title.set_fontsize(min(float(_ax.title.get_fontsize()), 7.2))
        plt.tight_layout(rect=[0.02, 0.10, 0.98, 0.94])
        suffix = f"_{tag}" if tag else ""
        out_name = f"Fig_gpbo_posterior_evaluation{suffix}.png"
        fig.savefig(os.path.join(output_dir, out_name), dpi=600, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.info(f"  Saved: {out_name}")
        return
    
    # Fallback: Use separate per-LC snapshots (legacy behavior)
    gpbo_results = opt_results.get("gpbo", {})
    if not gpbo_results:
        logger.warning("  No GP-BO results available for posterior figure")
        return
    
    lc_list = sorted(gpbo_results.keys())
    if len(lc_list) == 0:
        logger.warning("  No LC results in GP-BO for posterior figure")
        return
    
    # Find common iteration count (use minimum across LCs)
    min_iters = min(len(gpbo_results[lc].get("posterior_snapshots", [])) for lc in lc_list)
    if min_iters == 0:
        logger.warning("  No posterior snapshots available (fallback mode)")
        return
    
    # Select 8 iterations to show
    if min_iters >= 40:
        snapshot_indices = [0, 8, 14, 19, 26, 33, 36, min_iters - 1]
    elif min_iters >= 20:
        step = (min_iters - 1) / 7.0
        snapshot_indices = [int(round(i * step)) for i in range(8)]
        snapshot_indices[-1] = min_iters - 1
    elif min_iters >= 8:
        step = min_iters // 8
        snapshot_indices = [i * step for i in range(8)]
        snapshot_indices[-1] = min_iters - 1
    else:
        snapshot_indices = list(range(min_iters))
        while len(snapshot_indices) < 8:
            snapshot_indices.append(snapshot_indices[-1])
    
    snapshot_indices = [min(idx, min_iters - 1) for idx in snapshot_indices]
    
    fig, axes = plt.subplots(2, 4, figsize=(7.48, 5.35))
    axes = axes.flatten()
    
    for ax_idx, snap_idx in enumerate(snapshot_indices[:8]):
        ax = axes[ax_idx]
        
        for lc in lc_list:
            gpbo = gpbo_results[lc]
            snapshots = gpbo.get("posterior_snapshots", [])
            if snap_idx >= len(snapshots):
                continue
            
            snap = snapshots[snap_idx]
            theta_grid = snap["theta_grid"]
            mu = snap["mu"]
            sigma = snap["sigma"]
            X_obs = snap["X_obs"]
            y_obs = snap["y_obs"]
            
            color = lc_colors.get(lc, "#1f77b4")
            marker = lc_markers.get(lc, "o")
            label_lc = lc_labels.get(lc, lc)
            
            ax.plot(theta_grid, mu, color=color, linestyle='--', linewidth=1.5, 
                   label=rf"$\mu(\theta)$ | {label_lc}" if ax_idx == 0 else "")
            ax.fill_between(theta_grid, mu - 2*sigma, mu + 2*sigma, 
                           color=color, alpha=0.15, linewidth=0)
            ax.scatter(X_obs, y_obs, c=color, s=60, marker=marker, 
                      edgecolors='black', linewidths=0.5, zorder=10,
                      label=f"Obs ({label_lc})" if ax_idx == 0 else "")
        
        total_iters = min_iters
        ax.set_xlabel(r"Angle $\theta$ (deg.)")
        ax.set_ylabel("Objective")
        ax.set_title(f"Iter {snap_idx + 1}/{total_iters}", fontsize=7)
        ax.set_xlim(45, 70)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, 0.02), frameon=True, framealpha=0.95, columnspacing=0.6)
    
    title_suffix = f" ({tag})" if tag else ""
    fig.suptitle(f"GP-BO Posterior Evaluation{title_suffix}", fontweight='bold', y=0.99)
    apply_fig_style(fig)
    for _ax in axes:
        if _ax.get_title():
            _ax.title.set_fontsize(min(float(_ax.title.get_fontsize()), 7.2))
    plt.tight_layout(rect=[0.02, 0.10, 0.98, 0.94])
    suffix = f"_{tag}" if tag else ""
    out_name = f"Fig_gpbo_posterior_evaluation{suffix}.png"
    fig.savefig(os.path.join(output_dir, out_name), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"  Saved: {out_name}")


def select_best_optimizer(result: Dict, tol: float = 1e-8) -> Dict:
    """Select the best optimizer for a given target.
    
    Primary criterion: lowest final objective value (accuracy).
    Secondary criterion: fewest total evaluations (efficiency).
    Tertiary: lowest wall time.
    
    Returns a dict with keys 'name', 'method_key', 'objective', 'wall_time',
    'n_evals', and 'result' (the raw optimizer result dict), or {} if nothing found.
    """
    OPT_MAP = [
        ("gpbo_best",  "GP-BO"),
    ]
    candidates = []
    for key, label in OPT_MAP:
        if key in result:
            best = result[key]
            candidates.append({
                "name": label,
                "method_key": key,
                "objective": best["y_best"],
                "wall_time": best.get("wall_time", float('inf')),
                "n_evals": best.get("n_evals", 100),
                "result": best,
            })
    if not candidates:
        return {}
    # Primary: lowest objective.  Secondary: fewest evals.  Tertiary: fastest.
    candidates.sort(key=lambda c: (c["objective"], c["n_evals"], c["wall_time"]))
    return candidates[0]


def select_most_efficient_optimizer(result: Dict, error_threshold_pct: float = 3.0) -> Dict:
    """Among GP-BO runs meeting the error bar, prefer the fewest objective evaluations.

    Accuracy threshold: ``max(EA error %, IPF error %) < error_threshold_pct``.

    Returns the qualifying result with the smallest ``n_evals``, or {} if none qualify.
    """
    OPT_MAP = [
        ("gpbo_best",  "GP-BO"),
    ]
    candidates = []
    for key, label in OPT_MAP:
        if key in result:
            best = result[key]
            ea_err = best.get("ea_error_pct", 100)
            ipf_err = best.get("ipf_error_pct", 100)
            max_err = max(ea_err, ipf_err)
            if max_err <= error_threshold_pct:
                candidates.append({
                    "name": label,
                    "method_key": key,
                    "n_evals": best.get("n_evals", 100),
                    "max_error_pct": max_err,
                    "wall_time": best.get("wall_time", float('inf')),
                    "result": best,
                })
    if not candidates:
        return {}
    # Primary: fewest evals.  Secondary: lowest max error.
    candidates.sort(key=lambda c: (c["n_evals"], c["max_error_pct"]))
    return candidates[0]


def fig_optimizer_comparison(all_inverse_results: List[Dict], output_dir: str, logger: logging.Logger):
    """GP-BO inverse design: objective convergence and final loss for each target.

    Layout: one column per target, two rows (best-so-far objective vs evaluations,
    plus final objective bar). Only GP-BO is run in this pipeline (v18+).

    The lowest-objective run per target is highlighted with a gold border.
    """
    if not all_inverse_results:
        return
    
    # Dark, high-contrast optimizer colours
    OPT_COLORS = {"GP-BO": "#E64B35"}
    OPT_ORDER = ["GP-BO"]
    # Map internal keys to display labels
    KEY_TO_LABEL = {"gpbo_best": "GP-BO"}
    
    n = len(all_inverse_results)
    ncols = min(n, 5)
    fig, axes = plt.subplots(2, ncols, figsize=(3.6 * ncols, 7.5), squeeze=False)
    
    for col, result in enumerate(all_inverse_results[:ncols]):
        tid = result.get("target_info", {}).get("id", f"T{col + 1}")
        
        # ---- Row 0: convergence curves ----
        ax = axes[0, col]
        for mkey, label, ls in [("gpbo_best", "GP-BO", "-")]:
            if mkey in result and "best_y_history" in result[mkey]:
                hist = result[mkey]["best_y_history"]
                ax.plot(range(1, len(hist) + 1), hist,
                        color=OPT_COLORS.get(label, "black"), linestyle=ls,
                        linewidth=1.8, label=label)
        ax.set_xlabel("Evaluations", color="black")
        ax.set_ylabel("Best Objective", color="black")
        ax.set_title(f"{tid}", fontweight='bold', color="black")
        ax.tick_params(colors='black')
        if col == 0:
            ax.legend(framealpha=0.95, loc='upper right')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        add_subplot_label(ax, chr(ord('a') + col))
        
        # ---- Row 1: final-objective bar chart ----
        ax = axes[1, col]
        names, objs, wtimes, bcolors = [], [], [], []
        for label in OPT_ORDER:
            mkey = {"GP-BO": "gpbo_best"}[label]
            if mkey in result:
                names.append(label)
                objs.append(result[mkey]["y_best"])
                wtimes.append(result[mkey].get("wall_time", 0))
                bcolors.append(OPT_COLORS.get(label, "black"))
        
        if names:
            x = np.arange(len(names))
            bars = ax.bar(x, objs, color=bcolors, edgecolor='black', linewidth=1, width=0.65)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=28, ha='right', color="black")
            ax.set_ylabel("Final Objective", color="black")
            ax.tick_params(colors='black')

            y_max = max(objs) if objs else 1

            # Highlight the best optimizer with a gold border and star
            winner = select_best_optimizer(result)
            if winner:
                for i, name in enumerate(names):
                    if name == winner["name"]:
                        bars[i].set_edgecolor('#DAA520')
                        bars[i].set_linewidth(3.0)
                        ax.text(
                            bars[i].get_x() + bars[i].get_width() / 2,
                            bars[i].get_height() + 0.08 * max(y_max, 1e-6),
                            "[best]", ha='center', va='bottom',
                            fontsize=8.5, fontweight='bold', color='#DAA520'
                        )

            for bar, val in zip(bars, objs):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.03 * max(y_max, 1e-6),
                        f'{val:.1e}', ha='center', va='bottom',
                        fontsize=8, fontweight='bold', color="black")
            ax.set_ylim(0, y_max * 1.3 if y_max > 0 else 0.001)

            add_subplot_label(ax, chr(ord('a') + ncols + col))
    
    fig.suptitle("GP-BO inverse design: objective vs evaluations (all targets)",
                 fontweight='bold', y=0.98, color="black")
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.94])
    fig.savefig(os.path.join(output_dir, "Fig_optimizer_comparison.png"),
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("  Saved: Fig_optimizer_comparison.png")


def fig_inverse_optimizer_convergence(opt_results: Dict, output_dir: str, logger: logging.Logger,
                                     tag: str = ""):
    """Plot GP-BO convergence (best-so-far objective vs evaluations)."""
    methods = [
        ("gpbo_best", "GP-BO", COLORS.get("gpbo", COLORS.get("hard", "black")), "-"),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6))

    plotted = False
    for key, label, color, ls in methods:
        m = opt_results.get(key)
        if not m:
            continue
        yb = np.asarray(m.get("best_y_history", []), dtype=float)
        if yb.size == 0:
            continue
        x = np.arange(1, len(yb) + 1)
        ax.plot(x, yb, color=color, linestyle=ls, linewidth=1.8, label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Function Evaluations")
    ax.set_ylabel("Best Objective (lower is better)")
    ax.legend(loc='best', framealpha=0.95)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.set_title(f"Inverse Design Convergence (EA = {opt_results['target_ea']:.1f} J, IPF = {opt_results['target_ipf']:.2f} kN)")
    apply_fig_style(fig)
    plt.tight_layout()

    suffix = f"_{tag}" if tag else ""
    filepath = os.path.join(output_dir, f"Fig_inverse_convergence{suffix}.png")
    fig.savefig(filepath, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"  Saved: Fig_inverse_convergence{suffix}.png")


def fig_target_feasibility(df_metrics: pd.DataFrame, targets: List[Dict], output_dir: str, logger: logging.Logger):
    """Plot empirical EA-IPF design space and overlay chosen targets (feasibility justification)."""
    if df_metrics.empty or not targets:
        return

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.2))

    ea_col = "EA_common" if "EA_common" in df_metrics.columns else "EA"
    for lc in sorted(df_metrics["LC"].unique()):
        sub = df_metrics[df_metrics["LC"] == lc]
        ax.scatter(sub[ea_col], sub["IPF"], s=55, marker=LC_MARKERS.get(lc, 'o'),
                   facecolor='white', edgecolor=COLORS.get(lc, 'black'), linewidths=1.2,
                   label=f"Observed {lc}")

    # Targets
    for t in targets:
        ax.scatter([t["EA"]], [t["IPF"]], s=180, marker='*',
                   facecolor='black', edgecolor='black', linewidths=0.6, zorder=10)
        _sf_t = scaled_fonts(6.8)
        ax.annotate(t["id"], (t["EA"], t["IPF"]), textcoords="offset points", xytext=(4, 4),
                      fontsize=_sf_t["annot"])

    if ea_col == "EA_common":
        ax.set_xlabel(f"Energy absorbed to {D_COMMON:.0f} mm (J)")
    else:
        ax.set_xlabel("Energy Absorption, EA (J)")
    ax.set_ylabel("Initial Peak Force, IPF (kN)")
    ax.legend(loc='best', framealpha=0.95)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.set_title("Feasibility of Inverse Design Targets in Empirical EA-IPF Space")
    apply_fig_style(fig)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "Fig_inverse_target_feasibility.png")
    fig.savefig(filepath, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("  Saved: Fig_inverse_target_feasibility.png")


def fig_design_space(models: List[nn.Module], approach: str, scaler_disp: StandardScaler, enc: OneHotEncoder, params: ScalingParams, output_dir: str, logger: logging.Logger):
    """Generate design space prediction figure (EA at ``D_COMMON`` for LC-fair comparison; IPF unchanged)."""
    angles = np.linspace(CFG.angle_opt_min, CFG.angle_opt_max, 26)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for lc, marker, ls, color in [("LC1", "o", "-", COLORS["soft"]), ("LC2", "s", "--", COLORS["ddns"])]:
        EA_vals, IPF_vals = [], []
        for ang in angles:
            m = compute_ea_ipf_ensemble(
                models, approach, ang, lc, scaler_disp, enc, params, d_eval=D_COMMON)
            EA_vals.append(m["EA"])
            IPF_vals.append(m["IPF"])
        axes[0].plot(angles, EA_vals, linestyle=ls, marker=marker, markersize=5, linewidth=1.5, label=lc, color=color, markerfacecolor='white', markeredgecolor=color)
        axes[1].plot(angles, IPF_vals, linestyle=ls, marker=marker, markersize=5, linewidth=1.5, label=lc, color=color, markerfacecolor='white', markeredgecolor=color)
    axes[0].set_xlabel("Angle $\\theta$ (°)")
    axes[0].set_ylabel(f"Energy absorption (J) @ $d$={D_COMMON:.0f} mm")
    axes[0].legend(loc='best')
    add_subplot_label(axes[0], 'a')
    axes[1].set_xlabel("Angle $\\theta$ (°)")
    axes[1].set_ylabel("Initial Peak Force (kN)")
    axes[1].legend(loc='best')
    add_subplot_label(axes[1], 'b')
    fig.suptitle("Design Space Predictions (Hard-PINN)", fontweight='bold', y=0.98)
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.03, 0.06, 0.97, 0.90])
    fig.savefig(os.path.join(output_dir, "Fig_design_space.png"), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("  Saved: Fig_design_space.png")


def fig_pareto_tradeoff(df_pareto: pd.DataFrame, output_dir: str, logger: logging.Logger):
    """Generate Pareto trade-off figure with per-LC conditional fronts.
    
    Uses dark, high-contrast colours for all lines, labels, and axes.
    Avoids overlapping markers via offset and distinct styles.
    """
    if df_pareto.empty:
        return
    
    pareto_by_lc = df_pareto.attrs.get("pareto_by_lc", {})
    has_per_lc = len(pareto_by_lc) > 0
    
    # ---- colour palette (all fully opaque, dark, distinguishable) ----
    C_GLOBAL = "#000000"   # black for global front
    C_LC1    = "#3C5488"   # dark blue
    C_LC2    = "#E64B35"   # dark red/vermillion
    C_EA     = "#00A087"   # dark teal
    C_IPF    = "#B07AA1"   # dark plum (distinct from EA)
    
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    
    # ==================== Panel (a): Optimal angle vs alpha ====================
    ax = axes[0, 0]
    ax.plot(df_pareto["alpha"], df_pareto["angle"],
            color=C_GLOBAL, marker='o', markersize=7, linewidth=2,
            label="Global best", zorder=5)
    if has_per_lc:
        for lc, lc_df in pareto_by_lc.items():
            c = C_LC1 if lc == "LC1" else C_LC2
            ls = "--" if lc == "LC1" else "-."
            mk = "^" if lc == "LC1" else "s"
            ax.plot(lc_df["alpha"], lc_df["angle"], color=c, linestyle=ls,
                    marker=mk, markersize=5, markerfacecolor='white',
                    markeredgecolor=c, markeredgewidth=1.5,
                    linewidth=1.5, label=f"{lc} conditional", zorder=3)
    ax.set_xlabel(r"Trade-off weight $\alpha$ (EA priority)", color="black")
    ax.set_ylabel(r"Optimal Angle $\theta^*$ (°)", color="black")
    ax.tick_params(colors='black')
    ax.legend(framealpha=0.95, edgecolor='black')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    add_subplot_label(ax, 'a')
    
    # ==================== Panel (b): EA and IPF vs alpha ====================
    ax = axes[0, 1]
    ln1 = ax.plot(df_pareto["alpha"], df_pareto["EA"],
                  color=C_EA, marker='o', markersize=6, linewidth=2, label="EA")
    ax.set_xlabel(r"Trade-off weight $\alpha$", color="black")
    ax.set_ylabel(f"Energy Absorption, EA (J)", color=C_EA)
    ax.tick_params(axis='y', colors=C_EA)
    ax.tick_params(axis='x', colors='black')
    
    ax2 = ax.twinx()
    ln2 = ax2.plot(df_pareto["alpha"], df_pareto["IPF"],
                   color=C_IPF, marker='s', linestyle='--', markersize=6,
                   linewidth=2, label="IPF")
    ax2.set_ylabel("Initial Peak Force, IPF (kN)", color=C_IPF)
    ax2.tick_params(axis='y', colors=C_IPF)
    
    # Merged legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, framealpha=0.95, loc='center right',
              edgecolor='black')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    add_subplot_label(ax, 'b')
    
    # ==================== Panel (c): EA-IPF Pareto fronts ====================
    ax = axes[1, 0]
    if has_per_lc:
        for lc, lc_df in pareto_by_lc.items():
            c = C_LC1 if lc == "LC1" else C_LC2
            mk = "o" if lc == "LC1" else "s"
            label_lc = f"{lc} ({'Stable' if lc == 'LC1' else 'Progressive'})"
            lc_sorted = lc_df.sort_values("EA")
            ax.plot(lc_sorted["EA"], lc_sorted["IPF"],
                    color=c, marker=mk, markersize=7, markerfacecolor='white',
                    markeredgecolor=c, markeredgewidth=1.5,
                    linewidth=2, label=label_lc, zorder=3)
    # Global front (smaller markers, offset zorder to avoid overlap)
    pareto_sorted = df_pareto.sort_values("EA")
    ax.plot(pareto_sorted["EA"], pareto_sorted["IPF"],
            color=C_GLOBAL, linestyle='-', linewidth=1.5,
            marker='*', markersize=10, markerfacecolor=C_GLOBAL,
            label="Global front", zorder=5)
    ax.set_xlabel(f"Energy Absorption, EA (J)", color="black")
    ax.set_ylabel("Initial Peak Force, IPF (kN)", color="black")
    ax.tick_params(colors='black')
    ax.legend(framealpha=0.95, edgecolor='black')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    add_subplot_label(ax, 'c')
    
    # ==================== Panel (d): Scalarized objective vs alpha ====================
    ax = axes[1, 1]
    ax.plot(df_pareto["alpha"], df_pareto["J"],
            color=C_GLOBAL, marker='o', markersize=7, linewidth=2,
            label="Global", zorder=5)
    if has_per_lc:
        for lc, lc_df in pareto_by_lc.items():
            c = C_LC1 if lc == "LC1" else C_LC2
            ls = "--" if lc == "LC1" else "-."
            mk = "^" if lc == "LC1" else "s"
            ax.plot(lc_df["alpha"], lc_df["J"], color=c, linestyle=ls,
                    marker=mk, markersize=5, markerfacecolor='white',
                    markeredgecolor=c, markeredgewidth=1.5,
                    linewidth=1.5, label=f"{lc}", zorder=3)
    ax.set_xlabel(r"Trade-off weight $\alpha$", color="black")
    ax.set_ylabel("Scalarised Objective $J$", color="black")
    ax.tick_params(colors='black')
    ax.legend(framealpha=0.95, edgecolor='black')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    add_subplot_label(ax, 'd')
    
    fig.suptitle("Multi-Objective EA vs IPF Trade-off",
                 fontweight='bold', y=0.98, color="black")
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.93])
    fig.savefig(os.path.join(output_dir, "Fig_pareto_tradeoff.png"),
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("  Saved: Fig_pareto_tradeoff.png")


def fig_moo_objective_space_validation(
    pareto_df: pd.DataFrame,
    landscape_df: pd.DataFrame,
    output_dir: str,
    logger: logging.Logger,
    mobo_result: Optional[Dict[str, Any]] = None,
    df_metrics: Optional[pd.DataFrame] = None,
) -> None:
    """Validate MOO in **objective space** (EA@D_COMMON vs IPF).

    Panel (a): dense surrogate landscape (both LCs), non-dominated frontier
    from dominance on the grid, and weighted-sum sweep markers.

    Panel (b): optional **qLogNEHVI** observations (colour = BO batch),
    approximate Pareto set from MOBO samples, dominance frontier (faint),
    and the **hypervolume reference** as a marker in (EA, IPF) coordinates.

    When ``df_metrics`` provides ``EA_common``, overlays experimental design
    points for surrogate sanity-check (not used by the BO acquisition).
    """
    if landscape_df is None or len(landscape_df) < 4:
        logger.warning("  Skipping Fig_moo_objective_space_validation (empty landscape).")
        return

    pareto_dom = pareto_df.attrs.get("pareto_dominance", pd.DataFrame())
    C_LC1 = "#3C5488"
    C_LC2 = "#E64B35"
    C_DOM = "#222222"
    C_WS = "#009E73"
    C_REF = "#7E57C2"

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

    def _scatter_landscape(ax, alpha=0.22, s=14):
        for lc, c in [("LC1", C_LC1), ("LC2", C_LC2)]:
            sub = landscape_df[landscape_df["lc"] == lc]
            if len(sub) == 0:
                continue
            ax.scatter(sub["EA"], sub["IPF"], s=s, alpha=alpha, c=c, label=f"Surrogate ({lc})", edgecolors="none")

    def _plot_dom_front(ax, lw=2.2, z=6):
        if pareto_dom is None or pareto_dom.empty:
            return
        d = pareto_dom.sort_values("EA").reset_index(drop=True)
        ax.plot(d["EA"], d["IPF"], color=C_DOM, linewidth=lw, drawstyle="steps-post",
                label="Non-dominated (grid)", zorder=z)
        ax.scatter(d["EA"], d["IPF"], s=26, c=C_DOM, zorder=z + 1, edgecolors="white", linewidths=0.4)

    # --- (a) Surrogate + dominance + weighted-sum ---
    ax = axes[0]
    _scatter_landscape(ax, alpha=0.24, s=16)
    _plot_dom_front(ax)
    if not pareto_df.empty and {"EA", "IPF"}.issubset(pareto_df.columns):
        ax.scatter(pareto_df["EA"], pareto_df["IPF"], s=70, marker="D", facecolors="white",
                   edgecolors=C_WS, linewidths=1.8, zorder=8, label=r"Weighted-sum ($\alpha$ sweep)")
    if df_metrics is not None and len(df_metrics) > 0:
        ea_col = "EA_common" if "EA_common" in df_metrics.columns else "EA"
        if ea_col in df_metrics.columns and "IPF" in df_metrics.columns:
            ax.scatter(df_metrics[ea_col], df_metrics["IPF"], s=55, marker="P", c="#F0B429",
                       edgecolors="#333333", linewidths=0.6, zorder=7, label="Experiments")
    ax.set_xlabel(f"EA (J) at $d$={D_COMMON:.0f} mm")
    ax.set_ylabel("IPF (kN)")
    ax.set_title("(a) Surrogate landscape vs Pareto definitions")
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.28, linestyle="--")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    add_subplot_label(ax, "a")

    # --- (b) MOBO + reference point ---
    ax = axes[1]
    _scatter_landscape(ax, alpha=0.14, s=12)
    _plot_dom_front(ax, lw=1.4, z=4)
    if mobo_result and mobo_result.get("observations_df") is not None:
        obs = mobo_result["observations_df"]
        if len(obs) > 0 and "batch" in obs.columns:
            sc = ax.scatter(obs["EA_J"], obs["IPF_kN"], c=obs["batch"], cmap="viridis",
                            s=38, alpha=0.9, zorder=9, edgecolors="white", linewidths=0.35,
                            label="qLogNEHVI evaluations")
            try:
                plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label="BO batch")
            except Exception:
                pass
        pnd = mobo_result.get("pareto_approx_df")
        if pnd is not None and len(pnd) > 0:
            ax.scatter(pnd["EA_J"], pnd["IPF_kN"], s=120, marker="*", facecolors="#FF6D00",
                       edgecolors="black", linewidths=0.5, zorder=11, label="MOBO non-dominated")
        rm = mobo_result.get("ref_meta") or {}
        ry = rm.get("reference_point_y")
        if ry is not None and len(ry) >= 2:
            r_ea, r_neg = float(ry[0]), float(ry[1])
            ipf_ref = -r_neg
            ax.scatter([r_ea], [ipf_ref], s=200, marker="X", c=C_REF, zorder=12, linewidths=1.2,
                       edgecolors="white", label="HV ref. (BoTorch)")
            ax.annotate(
                "reference\n(worse than nadir)",
                xy=(r_ea, ipf_ref), xytext=(12, -14), textcoords="offset points",
                fontsize=7, color=C_REF,
                arrowprops=dict(arrowstyle="-|>", color=C_REF, lw=0.8, shrinkA=2, shrinkB=2),
            )
        ax.set_title("(b) MOBO (qLogNEHVI) vs grid Pareto + ref. point")
    else:
        ax.set_title("(b) MOBO not run (use default pipeline + botorch)")
    ax.set_xlabel(f"EA (J) at $d$={D_COMMON:.0f} mm")
    ax.set_ylabel("IPF (kN)")
    ax.legend(loc="best", framealpha=0.95)
    ax.grid(True, alpha=0.28, linestyle="--")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    add_subplot_label(ax, "b")

    fig.suptitle(
        "Multi-objective validation: objective space (EA vs IPF)",
        fontweight="bold", y=0.98,
    )
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.90])
    out = os.path.join(output_dir, "Fig_moo_objective_space_validation.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: Fig_moo_objective_space_validation.png")


def fig_mobo_qnehvi_diagnostics(
    mobo_result: Optional[Dict[str, Any]],
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Diagnostics for qLogNEHVI: hypervolume growth and design-space coverage."""
    if mobo_result is None:
        return
    hv = mobo_result.get("hypervolume_trace") or []
    obs = mobo_result.get("observations_df")
    if not hv and (obs is None or len(obs) == 0):
        logger.warning("  Skipping Fig_mobo_qnehvi_diagnostics (no MOBO data).")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))

    ax = axes[0]
    if hv:
        ax.plot(np.arange(len(hv)), hv, color="#222222", marker="o", markersize=5, linewidth=1.6)
        ax.set_xlabel("MOBO iteration (incl. initial Sobol)")
        ax.set_ylabel("Hypervolume (BoTorch, maximization)")
        ax.set_title("(a) Hypervolume vs iteration")
        ax.grid(True, alpha=0.28, linestyle="--")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    elif obs is not None and len(obs) > 0 and "batch" in obs.columns:
        bc = obs.groupby("batch").size()
        ax.bar(bc.index.astype(int), bc.values, color="#444444", edgecolor="black", linewidth=0.4)
        ax.set_xlabel("BO batch")
        ax.set_ylabel("Number of evaluations")
        ax.set_title("(a) Evaluations per batch")
        ax.grid(True, axis="y", alpha=0.28, linestyle="--")
    else:
        ax.text(0.5, 0.5, "No MOBO trace", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    add_subplot_label(ax, "a")

    ax = axes[1]
    if obs is not None and len(obs) > 0 and "batch" in obs.columns:
        for lc, mk, ec in [("LC1", "o", "#3C5488"), ("LC2", "s", "#E64B35")]:
            sub = obs[obs["LC"] == lc]
            if len(sub) == 0:
                continue
            ax.scatter(sub["angle_deg"], sub["batch"], c=sub["batch"], cmap="plasma",
                       marker=mk, s=42, label=lc, edgecolors=ec, linewidths=0.8, alpha=0.88)
        ax.set_xlabel(r"Interior angle $\theta$ (°)")
        ax.set_ylabel("BO batch index")
        ax.set_title(r"(b) MOBO samples in $\theta$ vs batch")
        ax.legend(loc="best", framealpha=0.95)
        ax.grid(True, alpha=0.28, linestyle="--")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    else:
        ax.set_visible(False)
    add_subplot_label(ax, "b")

    fig.suptitle("MOBO (qLogNEHVI): hypervolume and design coverage", fontweight="bold", y=0.98)
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.02, 0.05, 0.98, 0.90])
    out = os.path.join(output_dir, "Fig_mobo_qnehvi_diagnostics.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: Fig_mobo_qnehvi_diagnostics.png")


def fig_training_curves(dual_results: Dict, output_dir: str, logger: logging.Logger):
    """Generate training convergence curves."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    labels = ['a', 'b', 'c', 'd', 'e', 'f']
    label_idx = 0
    for p_idx, protocol in enumerate(["random", "unseen"]):
        if protocol not in dual_results:
            continue
        for a_idx, approach in enumerate(["ddns", "soft", "hard"]):
            ax = axes[p_idx, a_idx]
            if approach not in dual_results[protocol]:
                ax.set_visible(False)
                continue
            histories = dual_results[protocol][approach]["histories"]
            for hist in histories:
                ax.semilogy(hist["epoch"], hist["train_loss"], color=COLORS[approach], alpha=0.4, linewidth=1.0)
            min_len = min(len(h["epoch"]) for h in histories)
            if min_len > 0:
                epochs = histories[0]["epoch"][:min_len]
                aligned_losses = [h["train_loss"][:min_len] for h in histories]
                mean_loss = np.mean(aligned_losses, axis=0)
                ax.semilogy(epochs, mean_loss, color='black', linewidth=2.0, label='Mean')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Training Loss")
            ax.set_title(f"{MODEL_LABELS[approach]} ({protocol_label(protocol)})")
            add_subplot_label(ax, labels[label_idx])
            label_idx += 1
    fig.suptitle("Training Convergence", fontweight='bold', y=0.995)
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    fig.savefig(os.path.join(output_dir, "Fig_training_curves.png"), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("  Saved: Fig_training_curves.png")


def fig_model_complexity(dual_results: Dict, output_dir: str, logger: logging.Logger):
    """Generate model complexity comparison figure."""
    approaches = ["ddns", "soft", "hard"]
    n_params = [dual_results["random"][a]["n_params"] if a in dual_results["random"] else 0 for a in approaches]
    train_times = [dual_results["random"][a]["avg_training_time"] if a in dual_results["random"] else 0 for a in approaches]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    x = np.arange(len(approaches))
    colors_bar = [COLORS["ddns"], COLORS["soft"], COLORS["hard"]]
    ax = axes[0]
    bars = ax.bar(x, n_params, color=colors_bar, edgecolor='black', linewidth=1.2, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[a] for a in approaches])
    ax.set_ylabel("Number of Parameters")
    add_subplot_label(ax, 'a')
    _sf_mc = scaled_fonts(9.0)
    for bar, val in zip(bars, n_params):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, f'{val:,}', ha='center', va='bottom',
                fontsize=_sf_mc["annot"])
    ax = axes[1]
    bars = ax.bar(x, train_times, color=colors_bar, edgecolor='black', linewidth=1.2, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[a] for a in approaches])
    ax.set_ylabel("Training Time (s)")
    add_subplot_label(ax, 'b')
    for bar, val in zip(bars, train_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}', ha='center', va='bottom',
                fontsize=_sf_mc["annot"])
    fig.suptitle("Model Complexity Comparison", fontweight='bold', y=0.98)
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.03, 0.06, 0.97, 0.90])
    fig.savefig(os.path.join(output_dir, "Fig_model_complexity.png"), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("  Saved: Fig_model_complexity.png")


# =============================================================================
# Physics Verification, Baselines, Sensitivity
# =============================================================================

def compute_physics_residuals(models: List[nn.Module], approach: str, val_df: pd.DataFrame,
                               scaler_disp: StandardScaler, enc: OneHotEncoder, 
                               params: ScalingParams) -> Dict:
    """Compute thermodynamic consistency residuals |F_pred - dE/dd| for constraint verification.
    
    Physics consistency measures whether the model's own force prediction matches
    the gradient of its own energy prediction.  For Hard-PINN this is zero by
    construction (F IS dE/dd).  For Soft-PINN the soft penalty encourages but
    does not enforce the constraint.  For DDNS no constraint is applied.
    """
    Xv = to_tensor(build_features(val_df, scaler_disp, enc))
    F_actual = val_df["load_kN"].values  # Ground-truth experimental force
    
    all_violations = []
    all_dEdd = []
    all_Fpred = []
    
    n = int(Xv.shape[0])
    for model in models:
        model.eval()
        viol_parts: List[np.ndarray] = []
        dEdd_parts: List[np.ndarray] = []
        Fpred_parts: List[np.ndarray] = []
        for start in range(0, n, HARD_PINN_EVAL_BATCH):
            Xb = Xv[start : start + HARD_PINN_EVAL_BATCH].detach().clone().requires_grad_(True)
            if approach in ["ddns", "soft"]:
                # Model outputs [F_norm, E_norm] — two independent heads
                pv = model(Xb)
                F_pred = (pv[:, 0:1] * params.sig_F + params.mu_F)  # model's direct force output (kN)
                E_pred = pv[:, 1:2]  # normalized energy output
                dE_dX = torch.autograd.grad(E_pred.sum(), Xb, create_graph=False)[0]
                dE_dd = dE_dX[:, U_COL:U_COL+1] * params.grad_factor  # force from energy gradient (kN)
                F_pred_np = F_pred.detach().cpu().numpy().reshape(-1)
                dE_dd_np = dE_dd.detach().cpu().numpy().reshape(-1)
                violation = np.abs(F_pred_np - dE_dd_np)  # thermodynamic consistency
            else:  # Hard-PINN: F = dE/dd by construction
                E_n = model(Xb)
                dE_dX = torch.autograd.grad(E_n.sum(), Xb, create_graph=False)[0]
                dE_dd = dE_dX[:, U_COL:U_COL+1] * params.grad_factor
                dE_dd_np = dE_dd.detach().cpu().numpy().reshape(-1)
                F_pred_np = dE_dd_np.copy()
                violation = np.zeros_like(dE_dd_np)
            viol_parts.append(violation)
            dEdd_parts.append(dE_dd_np)
            Fpred_parts.append(F_pred_np)
        all_violations.append(np.concatenate(viol_parts))
        all_dEdd.append(np.concatenate(dEdd_parts))
        all_Fpred.append(np.concatenate(Fpred_parts))
    
    mean_violation = np.mean(all_violations, axis=0)
    mean_dEdd = np.mean(all_dEdd, axis=0)
    mean_Fpred = np.mean(all_Fpred, axis=0)
    
    # Force positivity and energy monotonicity verification
    mean_Fpred_arr = np.array(all_Fpred)  # (M, N)
    frac_negative_force = float(np.mean(mean_Fpred < 0))  # fraction of points with F < 0
    mean_min_force = float(np.min(mean_Fpred))
    # Per-member force negativity rate
    per_member_neg_frac = [float(np.mean(f < 0)) for f in all_Fpred]
    mean_neg_frac_members = float(np.mean(per_member_neg_frac))

    return {
        "residuals": mean_violation,
        "dE_dd": mean_dEdd,
        "F_pred": mean_Fpred,
        "F_actual": F_actual,
        "mean_abs_residual": float(np.mean(mean_violation)),
        "max_abs_residual": float(np.max(mean_violation)),
        "std_residual": float(np.std(mean_violation)),
        "frac_negative_force": frac_negative_force,
        "min_force_kN": mean_min_force,
        "mean_neg_frac_members": mean_neg_frac_members,
    }


def fig_physics_verification(dual_results: Dict, val_df: pd.DataFrame, scaler_disp: StandardScaler,
                              enc: OneHotEncoder, params: ScalingParams, output_dir: str, 
                              logger: logging.Logger):
    """Generate physics constraint verification figure showing dE/dd = F satisfaction."""
    logger.info("  Generating physics verification figure...")
    
    fig, axes = plt.subplots(1, 3, figsize=(7.48, 2.8))
    
    residual_data = {}
    colors = {"ddns": COLORS["ddns"], "soft": COLORS["soft"], "hard": COLORS["hard"]}
    
    for protocol in ["unseen"]:  # Use unseen protocol for verification
        if protocol not in dual_results:
            continue
        
        for approach in ["ddns", "soft", "hard"]:
            if approach not in dual_results[protocol]:
                continue
            
            models = dual_results[protocol][approach]["models"]
            res = compute_physics_residuals(models, approach, val_df, scaler_disp, enc, params)
            residual_data[approach] = res
            logger.info(f"    {MODEL_LABELS[approach]}: Mean |F_pred - dE/dd| = {res['mean_abs_residual']:.6f} kN, "
                        f"F<0 fraction = {res['frac_negative_force']:.4f}, "
                        f"min F = {res['min_force_kN']:.4f} kN")
    
    # Panel (a): Histogram of residuals - FIX: Ensure no overlap with y-axis
    ax = axes[0]
    
    # Determine bin range from DDNS data (largest residuals)
    if "ddns" in residual_data:
        max_residual = np.percentile(residual_data["ddns"]["residuals"], 99)
    else:
        max_residual = 1500
    
    # Create bins that start from 0, not negative
    bins = np.linspace(0, max_residual, 51)
    
    for approach in ["ddns", "soft", "hard"]:
        if approach in residual_data:
            res = np.abs(residual_data[approach]["residuals"])  # Ensure positive
            res_clipped = np.clip(res, 0, max_residual)
            ax.hist(res_clipped, bins=bins, alpha=0.6, color=colors[approach], 
                   label=f'{MODEL_LABELS[approach]} (μ={residual_data[approach]["mean_abs_residual"]:.4f})',
                   edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel("|F$_{pred}$ - dE/dd| (kN)")
    ax.set_ylabel("Frequency")
    ax.set_title("Physics Residual Distribution", fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_yscale('log')
    # FIX: Set x-axis limits with small padding to prevent overlap with y-axis
    ax.set_xlim(-max_residual * 0.02, max_residual * 1.05)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    add_subplot_label(ax, 'a')
    
    # Panel (b): dE/dd vs F_actual scatter for each model
    ax = axes[1]
    
    # Plot in reverse order so Hard-PINN (most important) is on top
    for approach in ["ddns", "soft", "hard"]:
        if approach in residual_data:
            dEdd = residual_data[approach]["dE_dd"]
            F = residual_data[approach]["F_pred"]
            # Subsample for visibility
            np.random.seed(42)
            idx = np.random.choice(len(dEdd), min(500, len(dEdd)), replace=False)
            ax.scatter(F[idx], dEdd[idx], alpha=0.5, s=20, c=colors[approach], 
                      label=MODEL_LABELS[approach], edgecolors='none', zorder=3 if approach == "hard" else 2)
    
    # Add perfect agreement line
    all_F = np.concatenate([residual_data[a]["F_pred"] for a in residual_data])
    all_dEdd = np.concatenate([residual_data[a]["dE_dd"] for a in residual_data])
    val_min = min(np.min(all_F), np.min(all_dEdd))
    val_max = max(np.max(all_F), np.max(all_dEdd))
    padding = (val_max - val_min) * 0.05
    lims = [val_min - padding, val_max + padding]
    ax.plot(lims, lims, 'k--', linewidth=2, label='Perfect: dE/dd = F$_{pred}$', zorder=1)
    ax.set_xlabel("Predicted F (kN)")
    ax.set_ylabel("Computed dE/dd (kN)")
    ax.set_title("Constraint Satisfaction: dE/dd vs F$_{pred}$", fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    add_subplot_label(ax, 'b')
    
    # Panel (c): Bar chart of mean absolute residuals - FIX: Better annotation positioning
    ax = axes[2]
    approaches = list(residual_data.keys())
    means = np.array([float(residual_data[a]["mean_abs_residual"]) for a in approaches], dtype=float)
    
    # For log-scale plotting, bars must be strictly > 0
    eps = 1e-12
    means_plot = np.clip(means, eps, None)
    
    bars = ax.bar(
        range(len(approaches)),
        means_plot,
        color=[colors[a] for a in approaches],
        edgecolor="black",
        linewidth=1.2,
        width=0.6
    )
    
    ax.set_xticks(range(len(approaches)))
    ax.set_xticklabels([MODEL_LABELS[a] for a in approaches])
    ax.set_ylabel("Mean |F$_{pred}$ - dE/dd| (kN)")
    ax.set_title("Physics Violation Magnitude", fontweight='bold')
    ax.set_yscale("log")
    
    # FIX: Set y-limits with MORE headroom to prevent annotation overlap with border
    ymin = 1e-13  # Small but visible on log scale
    ymax = max(means_plot.max(), 1) * 20  # Much more headroom for annotations
    ax.set_ylim(ymin, ymax)
    
    # FIX: Add value labels with better positioning
    _sf_ph = scaled_fonts(7.48)
    for bar, val in zip(bars, means):
        height = float(bar.get_height())
        # Position text INSIDE the bar for tall bars, ABOVE for short bars
        if val > 1:  # Tall bars (DDNS, Soft-PINN)
            # Position inside bar, near top
            y_text = height * 0.7
            va = 'top'
            text_color = 'white'
            fontweight = 'bold'
        else:  # Short bars (Hard-PINN)
            # Position above bar
            y_text = max(height * 2, ymin * 100)
            va = 'bottom'
            text_color = 'black'
            fontweight = 'bold'
        
        # Format the value appropriately
        if val < 1e-6:
            label_text = f"{val:.2e}"
        elif val < 1:
            label_text = f"{val:.4f}"
        else:
            label_text = f"{val:.2e}"
        
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_text,
            label_text,
            ha="center",
            va=va,
            fontsize=_sf_ph["annot"],
            fontweight=fontweight,
            color=text_color
        )
    
    add_subplot_label(ax, "c")
    
    fig.suptitle("Thermodynamic Consistency Verification: F$_{pred}$ = dE/dd", fontweight='bold', y=0.995)
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
    fig.savefig(os.path.join(output_dir, "Fig_physics_verification.png"), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("  Saved: Fig_physics_verification.png")
    
    # Save residual statistics to CSV
    rows = []
    for approach in residual_data:
        rows.append({
            "Model": MODEL_LABELS[approach],
            "Mean_Abs_Residual_kN": f"{residual_data[approach]['mean_abs_residual']:.6f}",
            "Max_Abs_Residual_kN": f"{residual_data[approach]['max_abs_residual']:.6f}",
            "Std_Residual_kN": f"{residual_data[approach]['std_residual']:.6f}",
            "Frac_Negative_Force": f"{residual_data[approach]['frac_negative_force']:.4f}",
            "Min_Force_kN": f"{residual_data[approach]['min_force_kN']:.4f}",
        })
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "Table_physics_verification.csv"), index=False)
    logger.info("  Saved: Table_physics_verification.csv")
    
    return residual_data


def train_baseline_models(train_df: pd.DataFrame, val_df: pd.DataFrame, 
                          scaler_disp: StandardScaler, enc: OneHotEncoder,
                          params: ScalingParams, logger: logging.Logger) -> Dict:
    """Train baseline ML models (Linear, RF, XGBoost, GP) for comparison."""
    logger.info("  Training baseline models...")
    
    X_train = build_features(train_df, scaler_disp, enc)
    y_train_load = train_df["load_kN"].values
    y_train_energy = train_df["energy_J"].values
    
    X_val = build_features(val_df, scaler_disp, enc)
    y_val_load = val_df["load_kN"].values
    y_val_energy = val_df["energy_J"].values
    
    results = {}
    
    # 1. Linear Regression
    try:
        from sklearn.linear_model import Ridge
        t0 = time.time()
        lr_load = Ridge(alpha=1.0).fit(X_train, y_train_load)
        lr_energy = Ridge(alpha=1.0).fit(X_train, y_train_energy)
        train_time = time.time() - t0
        
        pred_load = lr_load.predict(X_val)
        pred_energy = lr_energy.predict(X_val)
        
        results["Linear"] = {
            "load_r2": r2_score(y_val_load, pred_load),
            "energy_r2": r2_score(y_val_energy, pred_energy),
            "load_rmse": np.sqrt(mean_squared_error(y_val_load, pred_load)),
            "energy_rmse": np.sqrt(mean_squared_error(y_val_energy, pred_energy)),
            "load_mae": mean_absolute_error(y_val_load, pred_load),
            "energy_mae": mean_absolute_error(y_val_energy, pred_energy),
            "train_time": train_time,
            "n_params": X_train.shape[1] + 1
        }
        logger.info(f"    Linear: Load R²={results['Linear']['load_r2']:.4f}, Energy R²={results['Linear']['energy_r2']:.4f}")
    except Exception as e:
        logger.warning(f"    Linear regression failed: {e}")
    
    # 2. Random Forest
    try:
        from sklearn.ensemble import RandomForestRegressor
        t0 = time.time()
        rf_load = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=CFG.seed, n_jobs=-1).fit(X_train, y_train_load)
        rf_energy = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=CFG.seed, n_jobs=-1).fit(X_train, y_train_energy)
        train_time = time.time() - t0
        
        pred_load = rf_load.predict(X_val)
        pred_energy = rf_energy.predict(X_val)
        
        results["RandomForest"] = {
            "load_r2": r2_score(y_val_load, pred_load),
            "energy_r2": r2_score(y_val_energy, pred_energy),
            "load_rmse": np.sqrt(mean_squared_error(y_val_load, pred_load)),
            "energy_rmse": np.sqrt(mean_squared_error(y_val_energy, pred_energy)),
            "load_mae": mean_absolute_error(y_val_load, pred_load),
            "energy_mae": mean_absolute_error(y_val_energy, pred_energy),
            "train_time": train_time,
            "n_params": "100 trees"
        }
        logger.info(f"    RandomForest: Load R²={results['RandomForest']['load_r2']:.4f}, Energy R²={results['RandomForest']['energy_r2']:.4f}")
    except Exception as e:
        logger.warning(f"    Random Forest failed: {e}")
    
    # 3. XGBoost (if available)
    try:
        import xgboost as xgb
        t0 = time.time()
        xgb_load = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=CFG.seed, verbosity=0).fit(X_train, y_train_load)
        xgb_energy = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=CFG.seed, verbosity=0).fit(X_train, y_train_energy)
        train_time = time.time() - t0
        
        pred_load = xgb_load.predict(X_val)
        pred_energy = xgb_energy.predict(X_val)
        
        results["XGBoost"] = {
            "load_r2": r2_score(y_val_load, pred_load),
            "energy_r2": r2_score(y_val_energy, pred_energy),
            "load_rmse": np.sqrt(mean_squared_error(y_val_load, pred_load)),
            "energy_rmse": np.sqrt(mean_squared_error(y_val_energy, pred_energy)),
            "load_mae": mean_absolute_error(y_val_load, pred_load),
            "energy_mae": mean_absolute_error(y_val_energy, pred_energy),
            "train_time": train_time,
            "n_params": "100 trees"
        }
        logger.info(f"    XGBoost: Load R²={results['XGBoost']['load_r2']:.4f}, Energy R²={results['XGBoost']['energy_r2']:.4f}")
    except ImportError:
        logger.warning("    XGBoost not available, skipping")
    except Exception as e:
        logger.warning(f"    XGBoost failed: {e}")
    
    # 4. Gaussian Process (subsampled for tractability)
    if HAS_SKLEARN_GP:
        try:
            # Subsample for GP (O(N³) complexity)
            max_gp_samples = 2000
            if len(X_train) > max_gp_samples:
                idx = np.random.choice(len(X_train), max_gp_samples, replace=False)
                X_train_gp = X_train[idx]
                y_train_load_gp = y_train_load[idx]
                y_train_energy_gp = y_train_energy[idx]
            else:
                X_train_gp = X_train
                y_train_load_gp = y_train_load
                y_train_energy_gp = y_train_energy
            
            t0 = time.time()
            # Use separate well-configured kernels for load and energy
            # Matern 5/2 with adaptive length scales + noise term
            kernel_load = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * \
                          Matern(length_scale=np.ones(X_train_gp.shape[1]), nu=2.5, 
                                 length_scale_bounds=(1e-2, 1e2)) + \
                          WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
            kernel_energy = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * \
                            Matern(length_scale=np.ones(X_train_gp.shape[1]), nu=2.5,
                                   length_scale_bounds=(1e-2, 1e2)) + \
                            WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
            
            gp_load = GaussianProcessRegressor(
                kernel=kernel_load, normalize_y=True, 
                random_state=CFG.seed, n_restarts_optimizer=10, alpha=1e-6
            ).fit(X_train_gp, y_train_load_gp)
            gp_energy = GaussianProcessRegressor(
                kernel=kernel_energy, normalize_y=True,
                random_state=CFG.seed, n_restarts_optimizer=10, alpha=1e-6
            ).fit(X_train_gp, y_train_energy_gp)
            train_time = time.time() - t0
            
            pred_load = gp_load.predict(X_val)
            pred_energy = gp_energy.predict(X_val)
            
            results["GaussianProcess"] = {
                "load_r2": r2_score(y_val_load, pred_load),
                "energy_r2": r2_score(y_val_energy, pred_energy),
                "load_rmse": np.sqrt(mean_squared_error(y_val_load, pred_load)),
                "energy_rmse": np.sqrt(mean_squared_error(y_val_energy, pred_energy)),
                "load_mae": mean_absolute_error(y_val_load, pred_load),
                "energy_mae": mean_absolute_error(y_val_energy, pred_energy),
                "train_time": train_time,
                "n_params": f"N={len(X_train_gp)}"
            }
            logger.info(f"    GaussianProcess: Load R²={results['GaussianProcess']['load_r2']:.4f}, Energy R²={results['GaussianProcess']['energy_r2']:.4f}")
            logger.info(f"      GP Load kernel: {gp_load.kernel_}")
            logger.info(f"      GP Energy kernel: {gp_energy.kernel_}")
            logger.info(f"      normalize_y=True, n_restarts=10, alpha=1e-6, N_train={len(X_train_gp)}")
        except Exception as e:
            logger.warning(f"    Gaussian Process failed: {e}")
    
    return results


def fig_baseline_comparison(baseline_results: Dict, dual_results: Dict, output_dir: str, logger: logging.Logger,
                            protocol: str = "random"):
    """Generate baseline comparison figure and table for a given protocol."""
    if not baseline_results:
        logger.warning("  No baseline results to plot")
        return
    
    tag = "" if protocol == "random" else f"_{protocol}"
    ptitle = protocol_label(protocol)
    
    # Combine baseline and PINN results
    all_results = dict(baseline_results)
    for approach in ["ddns", "soft", "hard"]:
        if approach in dual_results.get(protocol, {}):
            m = dual_results[protocol][approach]["metrics"]
            all_results[MODEL_LABELS[approach]] = {
                "load_r2": m["load_r2"],
                "energy_r2": m["energy_r2"],
                "load_rmse": m["load_rmse"],
                "energy_rmse": m["energy_rmse"],
                "load_mae": m.get("load_mae", 0),
                "energy_mae": m.get("energy_mae", 0),
                "train_time": dual_results[protocol][approach]["avg_training_time"],
                "n_params": dual_results[protocol][approach]["n_params"]
            }
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(7.48, 3.0))
    
    models = list(all_results.keys())
    colors_list = []
    for m in models:
        if m == "Linear":
            colors_list.append("#808080")
        elif m == "RandomForest":
            colors_list.append("#228B22")
        elif m == "XGBoost":
            colors_list.append("#FF8C00")
        elif m == "GaussianProcess":
            colors_list.append("#8B008B")
        elif m == "DDNS":
            colors_list.append(COLORS["ddns"])
        elif m == "Soft-PINN":
            colors_list.append(COLORS["soft"])
        elif m == "Hard-PINN":
            colors_list.append(COLORS["hard"])
        else:
            colors_list.append("#000000")
    
    x = np.arange(len(models))
    
    # Panel (a): Load R²
    ax = axes[0]
    load_r2 = [all_results[m]["load_r2"] for m in models]
    bars = ax.bar(x, load_r2, color=colors_list, edgecolor='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=40, ha='right')
    ax.set_ylabel("Load R²")
    ax.set_ylim(0, 1.05)
    add_subplot_label(ax, 'a')
    
    # Panel (b): Energy R²
    ax = axes[1]
    energy_r2 = [all_results[m]["energy_r2"] for m in models]
    bars = ax.bar(x, energy_r2, color=colors_list, edgecolor='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=40, ha='right')
    ax.set_ylabel("Energy R²")
    ax.set_ylim(0, 1.05)
    add_subplot_label(ax, 'b')
    
    # Panel (c): Training time
    ax = axes[2]
    train_times = [all_results[m]["train_time"] for m in models]
    bars = ax.bar(x, train_times, color=colors_list, edgecolor='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=40, ha='right')
    ax.set_ylabel("Training Time (s)")
    ax.set_yscale('log')
    add_subplot_label(ax, 'c')
    
    fig.suptitle(f"Baseline Model Comparison ({ptitle})", fontweight='bold', y=0.98)
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.04, 0.18, 0.98, 0.92])
    fig.savefig(os.path.join(output_dir, f"Fig_baseline_comparison{tag}.png"), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"  Saved: Fig_baseline_comparison{tag}.png")
    
    # Save table
    rows = []
    for m in models:
        r = all_results[m]
        rows.append({
            "Model": m,
            "Load_R2": f"{r['load_r2']:.4f}",
            "Energy_R2": f"{r['energy_r2']:.4f}",
            "Load_RMSE": f"{r['load_rmse']:.4f}",
            "Energy_RMSE": f"{r['energy_rmse']:.4f}",
            "Load_MAE": f"{r.get('load_mae', 0):.4f}",
            "Energy_MAE": f"{r.get('energy_mae', 0):.4f}",
            "Train_Time_s": f"{r['train_time']:.1f}",
            "Parameters": r['n_params']
        })
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, f"Table_baseline_comparison{tag}.csv"), index=False)
    logger.info(f"  Saved: Table_baseline_comparison{tag}.csv")


def run_hyperparam_sensitivity(train_df: pd.DataFrame, val_df: pd.DataFrame,
                                scaler_disp: StandardScaler, scaler_out: StandardScaler,
                                enc: OneHotEncoder, params: ScalingParams,
                                protocol: str, logger: logging.Logger) -> pd.DataFrame:
    """Run hyperparameter sensitivity analysis on w_phys and learning rate."""
    logger.info("  Running hyperparameter sensitivity analysis...")
    
    w_phys_values = [0.0, 1.0, 5.0, 10.0, 20.0, 50.0]
    lr_values = [1e-4, 3e-4, 5e-4, 1e-3]
    
    results = []
    total = len(w_phys_values) * len(lr_values)
    count = 0
    
    for w_phys in w_phys_values:
        for lr in lr_values:
            count += 1
            logger.info(f"    [{count}/{total}] w_phys={w_phys}, lr={lr}")
            
            try:
                # Train single Soft-PINN with these hyperparameters
                set_seed(CFG.seed)
                cfg = get_model_config("soft", protocol, w_phys_override=w_phys)
                cfg["lr"] = lr
                cfg["epochs"] = min(cfg["epochs"], 500)  # Reduced for sensitivity analysis
                
                Xtr = to_tensor(build_features(train_df, scaler_disp, enc))
                ytr = to_tensor(build_targets(train_df, scaler_out))
                Xv = to_tensor(build_features(val_df, scaler_disp, enc))
                y_val = val_df[["load_kN", "energy_J"]].values
                
                model = SoftPINNNet(Xtr.shape[1], cfg["hidden_layers"], cfg["dropout"], cfg["softplus_beta"]).to(DEVICE)
                opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
                mse = nn.MSELoss()
                sl1 = nn.SmoothL1Loss(beta=cfg["smoothl1_beta"])
                colloc_sampler = create_collocation_sampler(train_df, scaler_disp, enc)
                rng = np.random.default_rng(CFG.seed)
                
                loader = DataLoader(
                    TensorDataset(Xtr, ytr), batch_size=cfg["batch_size"], shuffle=True, **_data_loader_kwargs()
                )
                
                for ep in range(1, cfg["epochs"] + 1):
                    model.train()
                    for Xb, yb in loader:
                        Xb.requires_grad_(True)
                        pred = model(Xb)
                        loss_data = cfg["w_data_load"] * sl1(pred[:, 0:1], yb[:, 0:1]) + cfg["w_data_energy"] * mse(pred[:, 1:2], yb[:, 1:2])
                        
                        loss_phys = torch.tensor(0.0, device=DEVICE)
                        if cfg["w_phys"] > 0 and cfg["colloc_ratio"] > 0:
                            n_colloc = int(Xb.shape[0] * cfg["colloc_ratio"])
                            Xc = colloc_sampler(n_colloc, rng).requires_grad_(True)
                            pc = model(Xc)
                            loss_phys = cfg["w_phys"] * physics_loss_soft(Xc, pc[:, 0:1], pc[:, 1:2], params)
                        
                        loss = loss_data + loss_phys
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                
                # Evaluate
                model.eval()
                with torch.no_grad():
                    pv = model(Xv)
                    Fv = (pv[:, 0] * params.sig_F + params.mu_F).cpu().numpy()
                    Ev = (pv[:, 1] * params.sig_E + params.mu_E).cpu().numpy()
                
                load_r2 = r2_score(y_val[:, 0], Fv)
                energy_r2 = r2_score(y_val[:, 1], Ev)
                
                results.append({
                    "w_phys": w_phys,
                    "lr": lr,
                    "load_r2": load_r2,
                    "energy_r2": energy_r2
                })
                
            except Exception as e:
                logger.warning(f"      Failed: {e}")
                results.append({"w_phys": w_phys, "lr": lr, "load_r2": 0, "energy_r2": 0})
    
    return pd.DataFrame(results)


def fig_hyperparam_sensitivity(sensitivity_df: pd.DataFrame, output_dir: str, logger: logging.Logger,
                                tag: str = ""):
    """Generate hyperparameter sensitivity heatmap figure."""
    if sensitivity_df.empty:
        logger.warning("  No sensitivity data to plot")
        return
    
    suffix = f"_{tag}" if tag else ""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pivot for heatmap
    w_phys_vals = sorted(sensitivity_df["w_phys"].unique())
    lr_vals = sorted(sensitivity_df["lr"].unique())
    
    for idx, (metric, title) in enumerate([("load_r2", "Load R²"), ("energy_r2", "Energy R²")]):
        ax = axes[idx]
        
        # Create matrix
        matrix = np.zeros((len(w_phys_vals), len(lr_vals)))
        for i, wp in enumerate(w_phys_vals):
            for j, lr in enumerate(lr_vals):
                row = sensitivity_df[(sensitivity_df["w_phys"] == wp) & (sensitivity_df["lr"] == lr)]
                if not row.empty:
                    matrix[i, j] = row[metric].values[0]
        
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
        ax.set_xticks(range(len(lr_vals)))
        ax.set_xticklabels([f"{lr:.0e}" for lr in lr_vals])
        ax.set_yticks(range(len(w_phys_vals)))
        ax.set_yticklabels([f"{wp:.0f}" for wp in w_phys_vals])
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Physics Weight (w_phys)")
        ax.set_title(title)
        
        # Add value annotations
        for i in range(len(w_phys_vals)):
            for j in range(len(lr_vals)):
                val = matrix[i, j]
                color = 'white' if val < 0.75 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7.5, color=color)
        
        plt.colorbar(im, ax=ax, shrink=0.8)
        add_subplot_label(ax, chr(ord('a') + idx))
    
    ptitle = f" ({tag})" if tag else ""
    fig.suptitle(f"Hyperparameter Sensitivity Analysis (Soft-PINN{ptitle})", fontweight='bold', y=0.98)
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.03, 0.05, 0.97, 0.91])
    fig.savefig(os.path.join(output_dir, f"Fig_hyperparam_sensitivity{suffix}.png"), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"  Saved: Fig_hyperparam_sensitivity{suffix}.png")
    
    # Save table
    sensitivity_df.to_csv(os.path.join(output_dir, f"Table_hyperparam_sensitivity{suffix}.csv"), index=False)
    logger.info(f"  Saved: Table_hyperparam_sensitivity{suffix}.csv")


def run_inverse_design_robust(models: List[nn.Module], approach: str, target_ea: float, target_ipf: float,
                               scaler_disp: StandardScaler, enc: OneHotEncoder, params: ScalingParams,
                               bo_cfg, logger: logging.Logger, n_seeds: int = 5,
                               cal_ens=None, feat_scaler=None) -> Dict:
    """Run inverse design with multiple seeds for robustness analysis.

    ``target_ea`` is EA at ``D_COMMON`` (displacement-fair vs. full-stroke EA).
    When ``cal_ens`` / ``feat_scaler`` are passed, each replicate uses the same
    classifier-augmented objective as :func:`run_inverse_design`.
    """
    logger.info(f"    Running multi-seed inverse design (n_seeds={n_seeds})...")
    
    results_by_method = {"gpbo": []}
    
    for seed_offset in range(n_seeds):
        seed = CFG.seed + seed_offset * 100

        # Preserve full BOConfig (Tikhonov, multi-start count, etc.); only vary seed.
        bo_cfg_seed = replace(
            bo_cfg,
            seed=seed,
            run_classifier_ablation=False,
            lambda_sweep=False,
        )

        res = run_inverse_design(models, approach, target_ea, target_ipf,
                                 scaler_disp, enc, params, bo_cfg_seed, logger,
                                 cal_ens=cal_ens, feat_scaler=feat_scaler)
        
        for method in ["gpbo"]:
            key = f"{method}_best"
            if key in res:
                results_by_method[method].append({
                    "x_best": res[key]["x_best"],
                    "y_best": res[key]["y_best"],
                    "ea_error_pct": res[key].get("ea_error_pct", float("nan")),
                    "ipf_error_pct": res[key]["ipf_error_pct"]
                })
    
    aggregated = {
        "target_ea": target_ea,
        "target_ipf": target_ipf,
        "n_seeds": n_seeds
    }
    
    for method in ["gpbo"]:
        if results_by_method[method]:
            x_vals = [r["x_best"] for r in results_by_method[method]]
            y_vals = [r["y_best"] for r in results_by_method[method]]
            ea_errs = [r["ea_error_pct"] for r in results_by_method[method]]
            ipf_errs = [r["ipf_error_pct"] for r in results_by_method[method]]
            
            aggregated[f"{method}_x_mean"] = np.mean(x_vals)
            aggregated[f"{method}_x_std"] = np.std(x_vals, ddof=1)
            aggregated[f"{method}_y_mean"] = np.mean(y_vals)
            aggregated[f"{method}_y_std"] = np.std(y_vals, ddof=1)
            aggregated[f"{method}_ea_err_mean"] = np.mean(ea_errs)
            aggregated[f"{method}_ea_err_std"] = np.std(ea_errs, ddof=1)
            aggregated[f"{method}_ipf_err_mean"] = np.mean(ipf_errs)
            aggregated[f"{method}_ipf_err_std"] = np.std(ipf_errs, ddof=1)
    
    return aggregated


def generate_inverse_robustness_table(robust_results: List[Dict], output_dir: str, logger: logging.Logger):
    """Generate inverse design robustness table."""
    if not robust_results:
        return
    
    rows = []
    for res in robust_results:
        for method in ["gpbo"]:
            if f"{method}_x_mean" in res:
                rows.append({
                    f"Target_EA@{EA_COMMON_MM_TAG}": f"{res.get('target_ea', 0):.3f}",
                    "Target_IPF": f"{res['target_ipf']:.3f}",
                    "Method": method.upper(),
                    "Angle_mean_std": f"{res[f'{method}_x_mean']:.1f} ± {res[f'{method}_x_std']:.1f}",
                    "Objective_mean_std": f"{res[f'{method}_y_mean']:.4f} ± {res[f'{method}_y_std']:.4f}",
                    "EA_err_mean_std": f"{res[f'{method}_ea_err_mean']:.1f} ± {res[f'{method}_ea_err_std']:.1f}%",
                    "IPF_err_mean_std": f"{res[f'{method}_ipf_err_mean']:.1f} ± {res[f'{method}_ipf_err_std']:.1f}%",
                    "N_seeds": res["n_seeds"]
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "Table_inverse_robustness.csv"), index=False)
    logger.info("  Saved: Table_inverse_robustness.csv")


def compute_uncertainty_calibration(dual_results: Dict, logger: logging.Logger) -> Dict:
    """Compute uncertainty calibration metrics at multiple confidence levels.
    
    Also computes post-hoc conformal calibration factors to correct overconfident
    ensemble intervals. The conformal factor scales sigma so that the observed
    coverage matches the expected coverage at the median confidence level.
    
    Returns:
        Dict with per-protocol, per-approach calibration data including:
        - observed coverage at multiple sigma levels
        - conformal calibration factor
        - corrected coverage after calibration
    """
    calibration = {}
    
    # Confidence levels to evaluate
    sigma_levels = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    # Expected coverage under Gaussian assumption
    expected_coverage = np.array([2 * stats.norm.cdf(s) - 1 for s in sigma_levels]) if HAS_SCIPY else \
                        np.array([0.383, 0.683, 0.866, 0.954, 0.988, 0.997])
    
    for protocol in ["random", "unseen"]:
        if protocol not in dual_results:
            continue
        
        calibration[protocol] = {}
        
        for approach in ["ddns", "soft", "hard"]:
            if approach not in dual_results[protocol]:
                continue
            
            m = dual_results[protocol][approach]["metrics"]
            preds = m["predictions"]
            true_vals = m["true_values"]
            
            if "load_std" not in preds:
                continue
            
            residuals = np.abs(true_vals["load"] - preds["load"])
            sigma = preds["load_std"]
            
            # Avoid division by zero for zero-variance predictions
            sigma_safe = np.maximum(sigma, 1e-12)
            
            # Compute observed coverage at each level
            observed_coverage = np.array([
                float(np.mean(residuals < s * sigma_safe)) for s in sigma_levels
            ])
            
            # Post-hoc conformal calibration:
            # Find the factor c such that c * sigma gives correct 68.3% coverage
            # i.e., find c where P(|residual| < c * sigma) = 0.683
            # This is equivalent to finding the quantile of |residual| / sigma
            normalized_residuals = residuals / sigma_safe
            
            # Conformal factor: the 68.3th percentile of |residual|/sigma
            # If calibrated, this should be ~1.0 (i.e., 1-sigma = 68.3% coverage)
            conformal_factor = float(np.percentile(normalized_residuals, 68.3))
            
            # Corrected coverage using conformal factor
            corrected_coverage = np.array([
                float(np.mean(residuals < s * conformal_factor * sigma_safe)) for s in sigma_levels
            ])
            
            # Also compute energy calibration if available
            energy_cal = {}
            if "energy_std" in preds:
                e_residuals = np.abs(true_vals["energy"] - preds["energy"])
                e_sigma = np.maximum(preds["energy_std"], 1e-12)
                e_observed = np.array([float(np.mean(e_residuals < s * e_sigma)) for s in sigma_levels])
                e_norm_res = e_residuals / e_sigma
                e_conformal = float(np.percentile(e_norm_res, 68.3))
                e_corrected = np.array([float(np.mean(e_residuals < s * e_conformal * e_sigma)) for s in sigma_levels])
                energy_cal = {
                    "energy_observed_coverage": e_observed,
                    "energy_conformal_factor": e_conformal,
                    "energy_corrected_coverage": e_corrected,
                }
            
            calibration[protocol][approach] = {
                "sigma_levels": sigma_levels,
                "expected_coverage": expected_coverage,
                "observed_coverage": observed_coverage,
                "conformal_factor": conformal_factor,
                "corrected_coverage": corrected_coverage,
                # Legacy keys for backward compatibility
                "load_within_1sigma": float(observed_coverage[1]),  # index 1 = 1.0 sigma
                "load_within_2sigma": float(observed_coverage[3]),  # index 3 = 2.0 sigma
                "expected_1sigma": 0.683,
                "expected_2sigma": 0.954,
                **energy_cal,
            }
            
            logger.info(f"  {protocol} {approach}: {observed_coverage[1]:.1%} within +/-1sigma "
                       f"(expected: 68.3%), {observed_coverage[3]:.1%} within +/-2sigma (expected: 95.4%)")
            logger.info(f"    Conformal factor: {conformal_factor:.3f} "
                       f"(1.0=perfectly calibrated, >{1.0}=overconfident)")
            logger.info(f"    After conformal correction: {corrected_coverage[1]:.1%} within +/-1sigma, "
                       f"{corrected_coverage[3]:.1%} within +/-2sigma")
    
    return calibration


def fig_reliability_diagram(calibration: Dict, output_dir: str, logger: logging.Logger):
    """Generate reliability diagram showing observed vs. expected coverage.
    
    Includes raw ensemble intervals and post-hoc conformal-corrected intervals.
    The diagonal represents perfect calibration; points below indicate overconfidence.
    """
    n_protocols = sum(1 for p in ["random", "unseen"] if p in calibration and calibration[p])
    if n_protocols == 0:
        logger.warning("  No calibration data for reliability diagram")
        return
    
    fig, axes = plt.subplots(1, n_protocols, figsize=(5.5 * n_protocols, 5), squeeze=False)
    axes = axes.flatten()
    
    approach_colors = {"ddns": COLORS["ddns"], "soft": COLORS["soft"], "hard": COLORS["hard"]}
    approach_markers = {"ddns": "o", "soft": "D", "hard": "^"}
    
    ax_idx = 0
    for protocol in ["random", "unseen"]:
        if protocol not in calibration or not calibration[protocol]:
            continue
        
        ax = axes[ax_idx]
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Perfect calibration')
        ax.fill_between([0, 1], [0, 1], [0, 0], alpha=0.05, color='red', label='Overconfident')
        ax.fill_between([0, 1], [1, 1], [0, 1], alpha=0.05, color='blue', label='Underconfident')
        
        for approach in ["ddns", "soft", "hard"]:
            if approach not in calibration[protocol]:
                continue
            
            cal = calibration[protocol][approach]
            expected = cal["expected_coverage"]
            observed = cal["observed_coverage"]
            corrected = cal["corrected_coverage"]
            color = approach_colors.get(approach, "gray")
            marker = approach_markers.get(approach, "o")
            label = MODEL_LABELS.get(approach, approach)
            
            # Raw (uncalibrated)
            ax.plot(expected, observed, color=color, marker=marker, markersize=8,
                   linewidth=1.5, label=f"{label} (raw)", alpha=0.9)
            
            # Corrected (conformal)
            ax.plot(expected, corrected, color=color, marker=marker, markersize=6,
                   linewidth=1.2, linestyle=':', alpha=0.6,
                   label=f"{label} (conformal, c={cal['conformal_factor']:.2f})")
        
        ax.set_xlabel("Expected Coverage")
        ax.set_ylabel("Observed Coverage")
        ax.set_title(f"{protocol_label(protocol)}", fontweight='bold')
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_aspect('equal')
        ax.legend(loc='lower right', framealpha=0.95, ncol=1,
                  borderpad=0.35, labelspacing=0.35, handletextpad=0.45,
                  fontsize=7.5)
        ax.grid(True, alpha=0.3)
        add_subplot_label(ax, chr(ord('a') + ax_idx))
        ax_idx += 1
    
    fig.suptitle("Uncertainty Calibration: Reliability Diagram", fontweight='bold', y=0.98)
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.03, 0.05, 0.97, 0.91])
    fig.savefig(os.path.join(output_dir, "Fig_reliability_diagram.png"), 
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("  Saved: Fig_reliability_diagram.png")
    
    # Save calibration table
    rows = []
    for protocol in ["random", "unseen"]:
        if protocol not in calibration:
            continue
        for approach in ["ddns", "soft", "hard"]:
            if approach not in calibration[protocol]:
                continue
            cal = calibration[protocol][approach]
            rows.append({
                "Protocol": protocol,
                "Model": MODEL_LABELS.get(approach, approach),
                "Coverage_1sigma_raw": f"{cal['observed_coverage'][1]:.3f}",
                "Coverage_2sigma_raw": f"{cal['observed_coverage'][3]:.3f}",
                "Conformal_factor": f"{cal['conformal_factor']:.3f}",
                "Coverage_1sigma_corrected": f"{cal['corrected_coverage'][1]:.3f}",
                "Coverage_2sigma_corrected": f"{cal['corrected_coverage'][3]:.3f}",
            })
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, "Table_uncertainty_calibration.csv"), index=False)
        logger.info("  Saved: Table_uncertainty_calibration.csv")


def run_same_capacity_experiment(train_df: pd.DataFrame, val_df: pd.DataFrame,
                                  scaler_disp: StandardScaler, scaler_out: StandardScaler,
                                  enc: OneHotEncoder, params: ScalingParams,
                                  protocol: str, logger: logging.Logger) -> Dict:
    """Run same-capacity experiment: Hard-PINN [64,64,64] vs Soft-PINN [64,64,64]."""
    logger.info("  Running same-capacity experiment (Hard vs Soft with [64,64,64])...")
    
    results = {}
    architecture = [64, 64, 64]
    
    for approach in ["soft", "hard"]:
        logger.info(f"    Training {approach.upper()} with {architecture} architecture...")
        
        set_seed(CFG.seed)
        cfg = get_model_config(approach, protocol)
        original_arch = cfg["hidden_layers"]
        cfg["hidden_layers"] = architecture  # Override architecture
        
        # Scale learning rate for architecture change (Hard-PINN is gradient-sensitive)
        if approach == "hard":
            # Use a more conservative LR for the larger Hard-PINN to avoid gradient explosion
            cfg["lr"] = cfg["lr"] * 0.5
            logger.info(f"      LR scaled to {cfg['lr']:.6f} for larger architecture")
        
        Xtr = to_tensor(build_features(train_df, scaler_disp, enc))
        ytr = to_tensor(build_targets(train_df, scaler_out))
        Xv = to_tensor(build_features(val_df, scaler_disp, enc))
        y_val = val_df[["load_kN", "energy_J"]].values
        
        if approach == "soft":
            model = SoftPINNNet(Xtr.shape[1], architecture, cfg["dropout"], cfg["softplus_beta"]).to(DEVICE)
        else:
            model = HardEnergyNet(Xtr.shape[1], architecture, cfg.get("dropout", 0.0), cfg["softplus_beta"]).to(DEVICE)
        
        n_params = model.count_parameters()
        
        opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        mse = nn.MSELoss()
        sl1 = nn.SmoothL1Loss(beta=cfg["smoothl1_beta"])
        
        if approach == "soft":
            colloc_sampler = create_collocation_sampler(train_df, scaler_disp, enc)
            rng = np.random.default_rng(CFG.seed)
        
        loader = DataLoader(
            TensorDataset(Xtr, ytr), batch_size=cfg["batch_size"], shuffle=True, **_data_loader_kwargs()
        )
        
        t0 = time.time()
        for ep in range(1, min(cfg["epochs"], 800) + 1):  # Reduced epochs for experiment
            model.train()
            for Xb, yb in loader:
                if approach == "soft":
                    Xb.requires_grad_(True)
                    pred = model(Xb)
                    loss_data = cfg["w_data_load"] * sl1(pred[:, 0:1], yb[:, 0:1]) + cfg["w_data_energy"] * mse(pred[:, 1:2], yb[:, 1:2])
                    
                    loss_phys = torch.tensor(0.0, device=DEVICE)
                    if cfg["w_phys"] > 0 and cfg.get("colloc_ratio", 0) > 0:
                        n_colloc = int(Xb.shape[0] * cfg["colloc_ratio"])
                        Xc = colloc_sampler(n_colloc, rng).requires_grad_(True)
                        pc = model(Xc)
                        loss_phys = cfg["w_phys"] * physics_loss_soft(Xc, pc[:, 0:1], pc[:, 1:2], params)
                    
                    loss = loss_data + loss_phys
                else:  # Hard-PINN
                    Xb_g = Xb.requires_grad_(True)
                    E_n = model(Xb_g)  # normalized energy output
                    dE = torch.autograd.grad(E_n, Xb_g, torch.ones_like(E_n), create_graph=True)[0]
                    F_phys = dE[:, U_COL:U_COL+1] * params.grad_factor  # physical force (kN)
                    F_n = (F_phys - params.mu_F) / params.sig_F  # normalize to match targets

                    # yb[:, 0:1] = normalized load, yb[:, 1:2] = normalized energy
                    loss = cfg.get("w_load", 1.0) * sl1(F_n, yb[:, 0:1]) + \
                           cfg.get("w_energy", 1.0) * mse(E_n, yb[:, 1:2])
                
                opt.zero_grad()
                loss.backward()
                # Gradient clipping for Hard-PINN stability with larger architecture
                if approach == "hard":
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
        
        train_time = time.time() - t0
        
        # Evaluate
        model.eval()
        if approach == "soft":
            with torch.no_grad():
                pv = model(Xv)
                Fv = (pv[:, 0] * params.sig_F + params.mu_F).cpu().numpy()
                Ev = (pv[:, 1] * params.sig_E + params.mu_E).cpu().numpy()
        else:
            Fv, Ev = hard_pinn_predict_load_energy(model, Xv, params)
        
        load_r2 = r2_score(y_val[:, 0], Fv)
        energy_r2 = r2_score(y_val[:, 1], Ev)
        
        results[approach] = {
            "load_r2": load_r2,
            "energy_r2": energy_r2,
            "load_rmse": np.sqrt(mean_squared_error(y_val[:, 0], Fv)),
            "energy_rmse": np.sqrt(mean_squared_error(y_val[:, 1], Ev)),
            "n_params": n_params,
            "train_time": train_time,
            "architecture": architecture
        }
        
        logger.info(f"      {approach.upper()} [64,64,64]: Load R²={load_r2:.4f}, Energy R²={energy_r2:.4f}, Params={n_params}")
    
    return results


def fig_same_capacity_comparison(same_cap_results: Dict, dual_results: Dict, output_dir: str, logger: logging.Logger):
    """Generate same-capacity comparison figure."""
    if not same_cap_results:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # Get original results for comparison
    models_data = []
    
    # Original architectures
    for approach in ["soft", "hard"]:
        if approach in dual_results.get("random", {}):
            m = dual_results["random"][approach]["metrics"]
            models_data.append({
                "name": f"{MODEL_LABELS[approach]}\n(original)",
                "load_r2": m["load_r2"],
                "energy_r2": m["energy_r2"],
                "n_params": dual_results["random"][approach]["n_params"],
                "color": COLORS[approach],
                "hatch": ""
            })
    
    # Same-capacity
    for approach in ["soft", "hard"]:
        if approach in same_cap_results:
            r = same_cap_results[approach]
            models_data.append({
                "name": f"{MODEL_LABELS[approach]}\n[64,64,64]",
                "load_r2": r["load_r2"],
                "energy_r2": r["energy_r2"],
                "n_params": r["n_params"],
                "color": COLORS[approach],
                "hatch": "///"
            })
    
    x = np.arange(len(models_data))
    
    # Panel (a): Load R²
    ax = axes[0]
    load_r2_values = [m["load_r2"] for m in models_data]
    bar_colors = [m["color"] for m in models_data]
    bars = ax.bar(x, load_r2_values, 
                  color=bar_colors,
                  hatch=[m["hatch"] for m in models_data],
                  edgecolor='black', linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([m["name"] for m in models_data])
    ax.set_ylabel("Load R²")
    # Dynamic y-axis: handle negative R² gracefully
    y_min_load = min(0.0, min(load_r2_values) - 0.05)
    ax.set_ylim(y_min_load, 1.05)
    if y_min_load < 0:
        ax.axhline(0.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    add_subplot_label(ax, 'a')
    
    # Add value labels
    for bar, m in zip(bars, models_data):
        va = 'bottom' if m["load_r2"] >= 0 else 'top'
        offset = 0.02 if m["load_r2"] >= 0 else -0.02
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset, 
               f'{m["load_r2"]:.3f}', ha='center', va=va, fontsize=8.5, fontweight='bold')
    
    # Panel (b): Energy R²
    ax = axes[1]
    energy_r2_values = [m["energy_r2"] for m in models_data]
    bars = ax.bar(x, energy_r2_values, 
                  color=bar_colors,
                  hatch=[m["hatch"] for m in models_data],
                  edgecolor='black', linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([m["name"] for m in models_data])
    ax.set_ylabel("Energy R²")
    y_min_energy = min(0.0, min(energy_r2_values) - 0.05)
    ax.set_ylim(y_min_energy, 1.05)
    if y_min_energy < 0:
        ax.axhline(0.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    add_subplot_label(ax, 'b')
    
    for bar, m in zip(bars, models_data):
        va = 'bottom' if m["energy_r2"] >= 0 else 'top'
        offset = 0.02 if m["energy_r2"] >= 0 else -0.02
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset, 
               f'{m["energy_r2"]:.3f}', ha='center', va=va, fontsize=8.5, fontweight='bold')
    
    fig.suptitle("Same-Capacity Experiment: Hard-PINN vs Soft-PINN with Identical Architecture", 
                 fontweight='bold', y=0.98)
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.04, 0.12, 0.98, 0.90])
    fig.savefig(os.path.join(output_dir, "Fig_same_capacity_comparison.png"), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("  Saved: Fig_same_capacity_comparison.png")
    
    # Save table
    rows = []
    for m in models_data:
        rows.append({
            "Model": m["name"].replace("\n", " "),
            "Load_R2": f"{m['load_r2']:.4f}",
            "Energy_R2": f"{m['energy_r2']:.4f}",
            "Parameters": m["n_params"]
        })
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "Table_same_capacity.csv"), index=False)
    logger.info("  Saved: Table_same_capacity.csv")


def run_extended_ablation(train_df: pd.DataFrame, val_df: pd.DataFrame,
                           scaler_disp: StandardScaler, scaler_out: StandardScaler,
                           enc: OneHotEncoder, params: ScalingParams,
                           protocol: str, logger: logging.Logger) -> pd.DataFrame:
    """Run extended ablation study on multiple components."""
    logger.info("  Running extended ablation study...")
    
    results = []
    
    # Baseline configuration
    base_cfg = get_model_config("soft", protocol)
    
    ablation_configs = [
        {"name": "Baseline", "changes": {}},
        {"name": "No physics (w_phys=0)", "changes": {"w_phys": 0.0}},
        {"name": "No BC (w_bc=0)", "changes": {"w_bc": 0.0}},
        {"name": "No collocation", "changes": {"colloc_ratio": 0.0}},
        {"name": "No dropout", "changes": {"dropout": 0.0}},
        {"name": "High physics (w_phys=50)", "changes": {"w_phys": 50.0}},
        {"name": "MSE loss only", "changes": {"smoothl1_beta": 1e6}},  # Large beta approximates MSE
    ]
    
    for ablation in ablation_configs:
        logger.info(f"    {ablation['name']}...")
        
        try:
            set_seed(CFG.seed)
            cfg = copy.deepcopy(base_cfg)
            cfg.update(ablation["changes"])
            cfg["epochs"] = min(cfg["epochs"], 500)  # Reduced for ablation
            
            Xtr = to_tensor(build_features(train_df, scaler_disp, enc))
            ytr = to_tensor(build_targets(train_df, scaler_out))
            Xv = to_tensor(build_features(val_df, scaler_disp, enc))
            y_val = val_df[["load_kN", "energy_J"]].values
            
            model = SoftPINNNet(Xtr.shape[1], cfg["hidden_layers"], cfg["dropout"], cfg["softplus_beta"]).to(DEVICE)
            opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            mse = nn.MSELoss()
            sl1 = nn.SmoothL1Loss(beta=cfg["smoothl1_beta"])
            
            colloc_sampler = create_collocation_sampler(train_df, scaler_disp, enc) if cfg.get("colloc_ratio", 0) > 0 else None
            rng = np.random.default_rng(CFG.seed)
            
            loader = DataLoader(
                TensorDataset(Xtr, ytr), batch_size=cfg["batch_size"], shuffle=True, **_data_loader_kwargs()
            )
            
            for ep in range(1, cfg["epochs"] + 1):
                model.train()
                for Xb, yb in loader:
                    Xb.requires_grad_(True)
                    pred = model(Xb)
                    loss_data = cfg.get("w_data_load", 1.0) * sl1(pred[:, 0:1], yb[:, 0:1]) + \
                               cfg.get("w_data_energy", 1.0) * mse(pred[:, 1:2], yb[:, 1:2])
                    
                    loss_phys = torch.tensor(0.0, device=DEVICE)
                    if cfg.get("w_phys", 0) > 0 and colloc_sampler is not None:
                        n_colloc = int(Xb.shape[0] * cfg["colloc_ratio"])
                        if n_colloc > 0:
                            Xc = colloc_sampler(n_colloc, rng).requires_grad_(True)
                            pc = model(Xc)
                            loss_phys = cfg["w_phys"] * physics_loss_soft(Xc, pc[:, 0:1], pc[:, 1:2], params)
                    
                    loss_bc = torch.tensor(0.0, device=DEVICE)
                    if cfg.get("w_bc", 0) > 0:
                        # BC: energy should be ~0 at zero displacement
                        combos = train_df[["Angle", "LC"]].drop_duplicates()
                        bc_df = combos.copy()
                        bc_df["disp_mm"] = 0.0
                        X_bc = to_tensor(build_features(bc_df, scaler_disp, enc))
                        pred_bc = model(X_bc)
                        E_bc_phys = pred_bc[:, 1:2] * params.sig_E + params.mu_E
                        loss_bc = cfg["w_bc"] * nn.MSELoss()(E_bc_phys, torch.zeros_like(E_bc_phys))

                    loss = loss_data + loss_phys + loss_bc
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                pv = model(Xv)
                Fv = (pv[:, 0] * params.sig_F + params.mu_F).cpu().numpy()
                Ev = (pv[:, 1] * params.sig_E + params.mu_E).cpu().numpy()
            
            load_r2 = r2_score(y_val[:, 0], Fv)
            energy_r2 = r2_score(y_val[:, 1], Ev)
            
            results.append({
                "Ablation": ablation["name"],
                "Load_R2": load_r2,
                "Energy_R2": energy_r2,
                "Delta_Load_R2": 0.0,  # Will compute below
                "Delta_Energy_R2": 0.0
            })
            
        except Exception as e:
            logger.warning(f"      Failed: {e}")
            results.append({
                "Ablation": ablation["name"],
                "Load_R2": 0.0,
                "Energy_R2": 0.0,
                "Delta_Load_R2": 0.0,
                "Delta_Energy_R2": 0.0
            })
    
    # Compute deltas relative to baseline
    if results and results[0]["Load_R2"] > 0:
        baseline_load = results[0]["Load_R2"]
        baseline_energy = results[0]["Energy_R2"]
        for r in results:
            r["Delta_Load_R2"] = r["Load_R2"] - baseline_load
            r["Delta_Energy_R2"] = r["Energy_R2"] - baseline_energy
    
    return pd.DataFrame(results)


def fig_extended_ablation(ablation_df: pd.DataFrame, output_dir: str, logger: logging.Logger,
                          tag: str = ""):
    """Generate extended ablation figure."""
    if ablation_df.empty:
        return
    
    suffix = f"_{tag}" if tag else ""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    x = np.arange(len(ablation_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ablation_df["Load_R2"], width, label='Load R²', 
                   color=COLORS["soft"], edgecolor='black')
    bars2 = ax.bar(x + width/2, ablation_df["Energy_R2"], width, label='Energy R²', 
                   color=COLORS["hard"], edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(ablation_df["Ablation"], rotation=40, ha='right')
    ax.set_ylabel("R²")
    all_r2 = list(ablation_df["Load_R2"]) + list(ablation_df["Energy_R2"])
    y_min = min(0.5, min(all_r2) - 0.05) if all_r2 else 0.0
    ax.set_ylim(y_min, 1.05)
    ax.legend(loc='lower right')
    ax.axhline(ablation_df["Load_R2"].iloc[0], color='gray', linestyle='--', alpha=0.5, label='Baseline')
    
    ptitle = f" ({tag})" if tag else ""
    fig.suptitle(f"Extended Ablation Study: Component Contributions{ptitle}", fontweight='bold', y=0.98)
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.04, 0.18, 0.98, 0.90])
    fig.savefig(os.path.join(output_dir, f"Fig_extended_ablation{suffix}.png"), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"  Saved: Fig_extended_ablation{suffix}.png")
    
    ablation_df.to_csv(os.path.join(output_dir, f"Table_extended_ablation{suffix}.csv"), index=False)
    logger.info(f"  Saved: Table_extended_ablation{suffix}.csv")


# =============================================================================
# SUMMARY TABLES
# =============================================================================
def generate_summary_tables(dual_results: Dict, df_metrics: pd.DataFrame, all_inverse: List[Dict],
                            stat_tests: Dict, output_dir: str, logger: logging.Logger,
                            calibration: Optional[Dict] = None):
    """Generate summary tables in CSV format."""
    # Table 1: Forward model results
    rows = []
    for protocol in ["random", "unseen"]:
        if protocol not in dual_results:
            continue
        for approach in ["ddns", "soft", "hard"]:
            if approach not in dual_results[protocol]:
                continue
            m = dual_results[protocol][approach]["metrics"]
            mm = dual_results[protocol][approach]["member_metrics"]
            ci_load = compute_confidence_intervals([x["load_r2"] for x in mm])
            ci_energy = compute_confidence_intervals([x["energy_r2"] for x in mm])
            row = {"Protocol": protocol_label(protocol), "Model": MODEL_LABELS[approach],
                        "M_total": dual_results[protocol][approach].get("M_total", len(mm)),
                        "M_eff": dual_results[protocol][approach].get("M_eff", len(mm)),
                        "Load_R2": f"{m['load_r2']:.4f}", "Load_R2_CI": f"[{ci_load['ci_lower']:.4f}, {ci_load['ci_upper']:.4f}]",
                        "Energy_R2": f"{m['energy_r2']:.4f}", "Energy_R2_CI": f"[{ci_energy['ci_lower']:.4f}, {ci_energy['ci_upper']:.4f}]",
                        "Load_RMSE": f"{m['load_rmse']:.4f}", "Energy_RMSE": f"{m['energy_rmse']:.4f}",
                        "Load_MAE": f"{m.get('load_mae', 0):.4f}", "Energy_MAE": f"{m.get('energy_mae', 0):.4f}",
                        "Avg_Time_s": f"{dual_results[protocol][approach]['avg_training_time']:.1f}"}
            # Add conformal factors if calibration data is available
            if (calibration is not None and protocol in calibration
                    and approach in calibration[protocol]):
                cal = calibration[protocol][approach]
                row["Conformal_Factor_Load"] = f"{cal.get('conformal_factor', 1.0):.3f}"
                row["Conformal_Factor_Energy"] = f"{cal.get('energy_conformal_factor', 1.0):.3f}"
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "Table1_forward_results.csv"), index=False)
    logger.info("  Saved: Table1_forward_results.csv")
    
    # Table 2: Statistical tests
    if stat_tests:
        rows = []
        for protocol in ["random", "unseen"]:
            if protocol not in stat_tests:
                continue
            for comp, vals in stat_tests[protocol].items():
                p_bonf = vals.get("p_bonferroni", vals["t_pvalue"])
                rows.append({"Protocol": protocol_label(protocol), "Comparison": comp.replace("_vs_", " vs ").upper(),
                            "t_statistic": f"{vals['t_statistic']:.4f}", "p_value": f"{vals['t_pvalue']:.4f}",
                            "p_bonferroni": f"{p_bonf:.4f}",
                            "Cohens_d": f"{vals['cohens_d']:.3f}",
                            "Significant_raw": "Yes" if vals['t_pvalue'] < 0.05 else "No",
                            "Significant_Bonferroni": "Yes" if p_bonf < 0.05 else "No"})
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, "Table2_statistical_tests.csv"), index=False)
        logger.info("  Saved: Table2_statistical_tests.csv")
    
    # Table 3: Inverse design results
    if all_inverse:
        rows = []
        for res in all_inverse:
            row = {f"Target_EA@{EA_COMMON_MM_TAG}_J": f"{res.get('target_ea', 0):.3f}",
                   "Target_IPF_kN": f"{res['target_ipf']:.3f}"}
            for method in ["gpbo"]:
                key = f"{method}_best"
                if key in res:
                    best = res[key]
                    prefix = method.upper()
                    row[f"{prefix}_Angle"] = f"{best['x_best']:.1f}"
                    row[f"{prefix}_LC"] = best.get("lc", "")
                    row[f"{prefix}_EA_err%"] = f"{best.get('ea_error_pct', float('nan')):.1f}"
                    row[f"{prefix}_IPF_err%"] = f"{best['ipf_error_pct']:.1f}"
                    # Delivered crashworthiness metrics (the engineering outcome)
                    row[f"{prefix}_EA@{EA_COMMON_MM_TAG}_J"] = f"{best.get('pred_ea', 0):.2f}"
                    row[f"{prefix}_EA@{EA_COMMON_MM_TAG}_std"] = f"{best.get('pred_ea_std', 0):.3f}"
                    row[f"{prefix}_EA_full_J"] = f"{best.get('pred_ea_full', best.get('pred_ea', 0)):.2f}"
                    row[f"{prefix}_IPF_kN"] = f"{best.get('pred_ipf', 0):.3f}"
                    row[f"{prefix}_IPF_std"] = f"{best.get('pred_ipf_std', 0):.4f}"
                    row[f"{prefix}_Objective"] = f"{best['y_best']:.6f}"
                    if "wall_time" in best:
                        row[f"{prefix}_Time_s"] = f"{best['wall_time']:.2f}"
            rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, "Table3_inverse_design.csv"), index=False)
        logger.info("  Saved: Table3_inverse_design.csv")
    
    # Table 4: Model complexity
    rows = []
    ref_protocol = "random" if "random" in dual_results else next(iter(dual_results), None)
    if ref_protocol:
        for approach in ["ddns", "soft", "hard"]:
            if approach in dual_results[ref_protocol]:
                rows.append({"Model": MODEL_LABELS[approach], "Parameters": dual_results[ref_protocol][approach]["n_params"],
                            "Avg_Training_Time_s": f"{dual_results[ref_protocol][approach]['avg_training_time']:.1f}"})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, "Table4_model_complexity.csv"), index=False)
        logger.info("  Saved: Table4_model_complexity.csv")

    # Table 5: Per-LC performance breakdown (addresses reviewer concern about aggregated R²)
    rows = []
    for protocol in ["random", "unseen"]:
        if protocol not in dual_results:
            continue
        val_df = dual_results[protocol].get("val_df")
        if val_df is None:
            continue
        scaler_disp_p = dual_results[protocol]["scaler_disp"]
        scaler_out_p = dual_results[protocol]["scaler_out"]
        enc_p = dual_results[protocol]["enc"]
        params_p = dual_results[protocol]["params"]
        for approach in ["ddns", "soft", "hard"]:
            if approach not in dual_results[protocol]:
                continue
            models_list = dual_results[protocol][approach]["models"]
            for lc in sorted(val_df["LC"].unique()):
                sub = val_df[val_df["LC"] == lc]
                if len(sub) < 2:
                    continue
                ens_m = evaluate_ensemble(models_list, approach, sub, scaler_disp_p, scaler_out_p, enc_p, params_p)
                rows.append({
                    "Protocol": protocol_label(protocol),
                    "Model": MODEL_LABELS.get(approach, approach),
                    "LC": lc,
                    "N_points": len(sub),
                    "Load_R2": f"{ens_m['load_r2']:.4f}",
                    "Energy_R2": f"{ens_m['energy_r2']:.4f}",
                    "Load_RMSE": f"{ens_m['load_rmse']:.4f}",
                    "Load_MAE": f"{ens_m['load_mae']:.4f}",
                })
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(output_dir, "Table5_per_LC_breakdown.csv"), index=False)
        logger.info("  Saved: Table5_per_LC_breakdown.csv")


def save_reproducibility_artifacts(output_dir: str, protocol: str, train_df: pd.DataFrame, scaler_disp: StandardScaler, scaler_out: StandardScaler, enc: OneHotEncoder, params: ScalingParams, logger: logging.Logger):
    """Save artifacts for reproducibility."""
    artifact_dir = os.path.join(output_dir, f"artifacts_{protocol}")
    os.makedirs(artifact_dir, exist_ok=True)
    train_df.to_csv(os.path.join(artifact_dir, "train_data.csv"), index=False)
    with open(os.path.join(artifact_dir, "scalers.pkl"), "wb") as f:
        pickle.dump({"scaler_disp": scaler_disp, "scaler_out": scaler_out, "enc": enc}, f)
    with open(os.path.join(artifact_dir, "params.json"), "w") as f:
        json.dump(asdict(params), f, indent=2)
    logger.info(f"  Saved reproducibility artifacts to {artifact_dir}")


def generate_optimizer_comparison_table(all_inverse_results: List[Dict], output_dir: str, logger: logging.Logger):
    """Summarize GP-BO inverse runs per target (single optimizer; dual *criteria*).
    
    Reports two labels per target when applicable:
      [acc] lowest final objective (accuracy proxy on the BO loss)
      [eff] fewest evaluations among runs with max(EA%, IPF%) error < 3%

    Also reports iterations-to-99% convergence on the best-so-far trace.
    """
    def iters_to_convergence(best_y_history, threshold_frac=0.99):
        """Return the iteration index at which 99% of the final improvement is achieved."""
        if len(best_y_history) < 2:
            return len(best_y_history)
        y0 = best_y_history[0]
        y_final = best_y_history[-1]
        if abs(y0 - y_final) < 1e-15:
            return 1
        target = y0 - threshold_frac * (y0 - y_final)
        for i, y in enumerate(best_y_history):
            if y <= target:
                return i + 1  # 1-indexed
        return len(best_y_history)

    rows = []
    for result in all_inverse_results:
        tid = result.get("target_info", {}).get("id", "?")
        target_id = (f"{tid} (EA@{EA_COMMON_MM_TAG}={result.get('target_ea', result.get('target_ea', 0)):.3f}, "
                      f"IPF={result['target_ipf']:.2f})")
        acc_winner = select_best_optimizer(result)
        eff_winner = select_most_efficient_optimizer(result, error_threshold_pct=3.0)
        acc_name = acc_winner.get("name", "") if acc_winner else ""
        eff_name = eff_winner.get("name", "") if eff_winner else ""
        for method in ["gpbo"]:
            key = f"{method}_best"
            if key in result:
                best = result[key]
                display = method.upper()
                # Mark accuracy and efficiency winners
                markers = []
                if display == acc_name:
                    markers.append("[acc]")
                if display == eff_name:
                    markers.append("[eff]")
                badge = "".join(markers)
                # Compute iterations to 99% convergence
                byh = best.get("best_y_history", np.array([]))
                conv_iter = iters_to_convergence(byh) if len(byh) > 0 else ""
                rows.append({"Target": target_id,
                            "Optimizer": display,
                            "Best_Angle": f"{best['x_best']:.2f}",
                            "Selected_LC": best.get("lc", ""),
                            "EA_Error_%": f"{best.get('ea_error_pct', float('nan')):.2f}",
                            "IPF_Error_%": f"{best['ipf_error_pct']:.2f}",
                            f"Delivered_EA@{EA_COMMON_MM_TAG}_J": f"{best.get('pred_ea', 0):.2f}",
                            "Delivered_EA_full_J": f"{best.get('pred_ea_full', best.get('pred_ea', 0)):.2f}",
                            "Delivered_IPF_kN": f"{best.get('pred_ipf', 0):.3f}",
                            "Objective": f"{best['y_best']:.6f}",
                            "N_Evals": best.get("n_evals", ""),
                            "Iters_to_99pct": conv_iter,
                            "Wall_Time_s": f"{best.get('wall_time', 0):.2f}",
                            "Winner": badge})
    if rows:
        df_comp = pd.DataFrame(rows)
        df_comp.to_csv(os.path.join(output_dir, "Table_optimizer_comparison.csv"), index=False)
        logger.info("  Saved: Table_optimizer_comparison.csv")

        # Log summary: accuracy and efficiency win rates
        acc_wins_gpbo = sum(1 for r in all_inverse_results
                           if select_best_optimizer(r).get("name") == "GP-BO")
        eff_wins_gpbo = sum(1 for r in all_inverse_results
                           if select_most_efficient_optimizer(r).get("name") == "GP-BO")
        n = len(all_inverse_results)
        logger.info(f"  GP-BO [acc] (lowest objective): {acc_wins_gpbo}/{n} targets")
        logger.info(f"  GP-BO [eff] (<3% error, fewest evals): {eff_wins_gpbo}/{n} targets")


# =============================================================================
# REPRODUCIBILITY & Q1 PUBLICATION HELPERS
# =============================================================================
def log_runtime_environment(output_dir: str, logger: logging.Logger) -> Dict[str, Any]:
    """Record library versions and flags for reproducibility (supplementary / Zenodo)."""
    import platform

    def _pkg_ver(name: str) -> Optional[str]:
        try:
            import importlib.metadata as imd
            return imd.version(name)
        except Exception:
            return None

    scipy_ver: Optional[str] = None
    if HAS_SCIPY:
        try:
            import scipy
            scipy_ver = scipy.__version__
        except Exception:
            scipy_ver = None

    cuda_dev_name: Optional[str] = None
    if torch.cuda.is_available():
        try:
            cuda_dev_name = torch.cuda.get_device_name(0)
        except Exception:
            cuda_dev_name = None

    info: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "python_short": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": cuda_dev_name,
        "sklearn": _pkg_ver("scikit-learn"),
        "scipy": scipy_ver,
        "HAS_SCIPY": bool(HAS_SCIPY),
        "HAS_SKOPT": bool(HAS_SKOPT),
        "HAS_BOTORCH": bool(HAS_BOTORCH),
        "HAS_SKLEARN_GP": bool(HAS_SKLEARN_GP),
        "CFG_seed": CFG.seed,
        "CFG_n_ensemble": CFG.n_ensemble,
        "D_COMMON_mm": D_COMMON,
    }
    if HAS_BOTORCH:
        info["botorch"] = _pkg_ver("botorch")
        info["gpytorch"] = _pkg_ver("gpytorch")
    path = os.path.join(output_dir, "runtime_environment.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)
    logger.info("  Saved: runtime_environment.json")
    logger.info("  Runtime stack (see JSON for full strings):")
    logger.info(f"    Python {info['python_short']} | numpy {info['numpy']} | torch {info['torch']}")
    logger.info(f"    sklearn {info.get('sklearn')} | skopt={HAS_SKOPT} | botorch={HAS_BOTORCH}")
    return info


def apply_dry_run_settings(logger: logging.Logger) -> None:
    """Tighten global budgets for ``CFG.dry_run`` (CI smoke; not for publication numbers)."""
    if not getattr(CFG, "dry_run", False):
        return
    CFG.n_ensemble = min(int(CFG.n_ensemble), 2)
    CFG.run_reviewer_proof = False
    CFG.run_ablation = False
    CFG.run_mobo_qnehvi = False
    CFG._skip_mobo = True
    CFG.run_gpbo = False
    CFG.run_loao_cv = False
    CFG.run_rar = False
    BO_CFG.lambda_sweep = False
    BO_CFG.run_classifier_ablation = False
    CFG.run_inverse_ablation = False
    CFG.run_inverse_stress_validation = False
    logger.info(
        "DRY RUN MODE: M<=2, short epochs, coarser MO landscape, "
        "no MOBO / reviewer-proof / ablation / GP-BO; inverse uses coarse angle grid."
    )


def check_publication_dependencies(logger: logging.Logger) -> None:
    """Fail fast when ``CFG.strict_paper_deps`` and a required stack piece is missing."""
    if not getattr(CFG, "strict_paper_deps", False):
        return
    missing: List[str] = []
    if CFG.run_gpbo and not HAS_SKOPT:
        missing.append("scikit-optimize (skopt) required for GP-BO inverse design")
    if not HAS_BOTORCH:
        missing.append("botorch+gpytorch required for MOBO (qLogNEHVI) — primary multi-objective method")
    if HAS_BOTORCH and (not CFG.run_mobo_qnehvi or getattr(CFG, "_skip_mobo", False)):
        missing.append(
            "strict_paper_deps requires MOBO to run (omit --no_mobo_qnehvi) for a complete submission bundle"
        )
    if missing:
        for m in missing:
            logger.error(f"  strict_paper_deps: {m}")
        raise RuntimeError(
            "strict_paper_deps is enabled but optional dependencies are missing: "
            + "; ".join(missing)
            + ". Install requirements.txt (skopt + botorch). strict_paper requires MOBO to run "
            "(omit --no_mobo_qnehvi). If inverse GP-BO is not required, set CFG.run_gpbo = False before run."
        )


def write_statistical_testing_policy(output_dir: str, logger: logging.Logger) -> None:
    """Document multiplicity and primary vs exploratory tests for reviewers."""
    text = """STATISTICAL TESTING POLICY (MANUSCRIPT / SUPPLEMENT)
============================================================

Primary protocol-level comparisons
------------------------------------
- Pre-specified comparisons between modelling approaches (e.g. Hard-PINN vs
  Soft-PINN vs DDNS) are reported per train/validation protocol (random split
  vs unseen angle). Bonferroni-adjusted p-values correct multiplicity **within
  each protocol** over the pairwise model comparisons reported for that
  protocol—not pooled across protocols.

- Welch t-tests on validation load R² across ensemble members are a
  standardized summary; members are not independent replicates (shared split,
  same architecture). Treat p-values descriptively alongside Cohen's d.

Exploratory / descriptive analyses
----------------------------------
- Inverse design, multi-objective sweeps, MOBO (qLogNEHVI), classifier
  ablations, and hyperparameter sensitivity analyses are **descriptive** unless
  explicitly framed as confirmatory with a pre-specified alpha budget.

Recommendations for the paper text
----------------------------------
- State which tests are primary confirmatory vs exploratory.
- Report effect sizes (Cohen's d is already in outputs where applicable) alongside p-values.
- For multiple inverse targets, distinguish **per-target** error reporting from
  global claims about optimiser performance.

This file is generated automatically for reproducibility bundles (Zenodo / SI).
"""
    path = os.path.join(output_dir, "STATISTICAL_TESTING_POLICY.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info("  Saved: STATISTICAL_TESTING_POLICY.txt")


def _fmt_float_for_csv(x: Any, nd: int = 2) -> str:
    """Format numeric cell for CSV; tolerate missing / non-numeric values."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    if np.isnan(v):
        return "nan"
    return f"{v:.{nd}f}"


def generate_compute_budget_summary(
    dual_results: Dict[str, Any],
    all_inverse_results: List[Dict],
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Summarise parameter counts, training times, and optimisation budgets for SI tables."""
    rows: List[Dict[str, Any]] = []
    for proto in ["random", "unseen"]:
        if proto not in dual_results:
            continue
        for app in ["ddns", "soft", "hard"]:
            if app not in dual_results[proto]:
                continue
            dr = dual_results[proto][app]
            rows.append({
                "stage": f"forward_{proto}_{app}",
                "ensemble_M_total": dr.get("M_total", ""),
                "ensemble_M_effective": dr.get("M_eff", ""),
                "n_parameters": dr.get("n_params", ""),
                "avg_train_time_s_per_member": _fmt_float_for_csv(dr.get("avg_training_time")),
            })
    rows.append({
        "stage": "inverse_gpbo_config",
        "ensemble_M_total": "",
        "ensemble_M_effective": "",
        "n_parameters": "",
        "avg_train_time_s_per_member": "",
        "n_gpbo_calls_default": BO_CFG.n_calls_total,
        "n_gpbo_init_default": BO_CFG.n_init,
    })
    for res in all_inverse_results:
        best = res.get("gpbo_best")
        if not best:
            continue
        tid = res.get("target_info", {}).get("id", "?")
        rows.append({
            "stage": f"inverse_gpbo_target_{tid}",
            "ensemble_M_total": "",
            "ensemble_M_effective": "",
            "n_parameters": "",
            "avg_train_time_s_per_member": _fmt_float_for_csv(best.get("wall_time")),
            "n_gpbo_calls": best.get("n_evals", ""),
        })
    mobo_csv = os.path.join(output_dir, "Table_mobo_qnehvi_observations.csv")
    if os.path.isfile(mobo_csv):
        try:
            n_m = len(pd.read_csv(mobo_csv))
            rows.append({
                "stage": "mobo_qnehvi_total_forward_evals",
                "ensemble_M_total": "",
                "ensemble_M_effective": "",
                "n_parameters": "",
                "avg_train_time_s_per_member": "",
                "n_mobo_forward_evaluations": n_m,
            })
        except Exception as ex:
            logger.warning(f"  Could not read MOBO observation count: {ex}")
    out = os.path.join(output_dir, "Table_compute_reproducibility_budget.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("  Saved: Table_compute_reproducibility_budget.csv")


def fig_inverse_parity_uncertainty(
    all_inverse_results: List[Dict],
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Parity plots for inverse GP-BO with ensemble ±2σ bars (publication validation)."""
    if not all_inverse_results:
        return
    t_ea, p_ea, s_ea = [], [], []
    t_ipf, p_ipf, s_ipf = [], [], []
    labels: List[str] = []
    ang: List[float] = []
    for res in all_inverse_results:
        b = res.get("gpbo_best")
        if not b:
            continue
        labels.append(str(res.get("target_info", {}).get("id", "?")))
        t_ea.append(float(res.get("target_ea", float("nan"))))
        p_ea.append(float(b.get("pred_ea", float("nan"))))
        s_ea.append(max(float(b.get("pred_ea_std", 0.0)), 1e-9))
        t_ipf.append(float(res.get("target_ipf", float("nan"))))
        p_ipf.append(float(b.get("pred_ipf", float("nan"))))
        s_ipf.append(max(float(b.get("pred_ipf_std", 0.0)), 1e-6))
        ang.append(float(b.get("x_best", float("nan"))))
    if not t_ea:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    for ax, tt, pp, ss, xlab, ylab in [
        (axes[0], t_ea, p_ea, s_ea, f"Target EA (J) @ {D_COMMON:.0f} mm", f"Predicted EA (J) @ {D_COMMON:.0f} mm"),
        (axes[1], t_ipf, p_ipf, s_ipf, "Target IPF (kN)", "Predicted IPF (kN)"),
    ]:
        arr = np.asarray(tt + pp, dtype=np.float64)
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        pad = 0.04 * max(hi - lo, 1e-6)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1.0, alpha=0.7, label="Identity")
        ax.errorbar(tt, pp, yerr=2 * np.array(ss), fmt="none", ecolor="#555555", capsize=3, alpha=0.85)
        ax.scatter(tt, pp, s=85, c="#0072B2", edgecolors="black", linewidths=0.6, zorder=5)
        for i, lab in enumerate(labels):
            xi, yi = float(tt[i]), float(pp[i])
            if np.isfinite(xi) and np.isfinite(yi):
                ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.28, linestyle="--")
        ax.legend(loc="upper left", framealpha=0.95)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    add_subplot_label(axes[0], "a")
    add_subplot_label(axes[1], "b")
    fig.suptitle("Inverse design validation: parity with ensemble uncertainty", fontweight="bold", y=0.98)
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.90])
    fig.savefig(os.path.join(output_dir, "Fig_inverse_parity_uncertainty.png"),
                dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: Fig_inverse_parity_uncertainty.png")

    # Second small figure: error vs recovered angle (interpolation stress)
    fig2, ax2 = plt.subplots(figsize=(5.2, 4.0))
    ang_a = np.asarray(ang, dtype=np.float64)
    err_pct = np.abs(np.asarray(p_ea, dtype=np.float64) - np.asarray(t_ea, dtype=np.float64)) / (
        np.asarray(t_ea, dtype=np.float64) + 1e-12
    ) * 100.0
    ok = np.isfinite(ang_a) & np.isfinite(err_pct)
    if np.any(ok):
        ax2.scatter(ang_a[ok], err_pct[ok], s=80, c="#D55E00", edgecolors="black", linewidths=0.5)
    for i, lab in enumerate(labels):
        if ok[i]:
            ax2.annotate(lab, (float(ang_a[i]), float(err_pct[i])),
                         textcoords="offset points", xytext=(3, 3), fontsize=7)
    ax2.set_xlabel(r"Recovered angle $\theta$ (°)")
    ax2.set_ylabel("Relative EA error (%)")
    ax2.set_title("Inverse error vs optimised angle")
    ax2.grid(True, alpha=0.28, linestyle="--")
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    add_subplot_label(ax2, "a")
    apply_fig_style(fig2)
    plt.tight_layout(rect=[0.08, 0.08, 0.95, 0.92])
    fig2.savefig(os.path.join(output_dir, "Fig_inverse_error_vs_angle.png"),
                 dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    logger.info("  Saved: Fig_inverse_error_vs_angle.png")


def fig_validation_error_maps(
    dual_results: Dict, output_dir: str, logger: logging.Logger, approach: str = "hard"
) -> None:
    """Hexbin maps of pointwise |load| / |energy| errors vs (angle, displacement) on validation rows."""
    protocols = [p for p in ("random", "unseen") if p in dual_results and approach in dual_results[p]]
    if not protocols:
        return
    fig, axes = plt.subplots(2, len(protocols), figsize=(5.2 * len(protocols), 7.2), squeeze=False)
    for j, protocol in enumerate(protocols):
        dr = dual_results[protocol]
        ens = evaluate_ensemble(
            dr[approach]["models"], approach, dr["val_df"],
            dr["scaler_disp"], dr["scaler_out"], dr["enc"], dr["params"],
        )
        ang = dr["val_df"]["Angle"].values.astype(float)
        disp = dr["val_df"]["disp_mm"].values.astype(float)
        abs_le = np.abs(ens["load_errors"])
        abs_ee = np.abs(ens["energy_errors"])
        for i, (errs, title) in enumerate([(abs_le, "|load error| (kN)"), (abs_ee, "|energy error| (J)")]):
            ax = axes[i, j]
            hb = ax.hexbin(
                ang, disp, C=errs, reduce_C_function=np.mean,
                gridsize=(28, 24), mincnt=1, cmap="magma", linewidths=0,
            )
            plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.02, label=title)
            ax.set_xlabel(r"Angle $\theta$ (°)")
            ax.set_ylabel("Displacement (mm)")
            ax.set_title(f"{protocol_label(protocol)} — {MODEL_LABELS.get(approach, approach)}")
            ax.grid(True, alpha=0.22, linestyle="--")
            add_subplot_label(ax, chr(ord("a") + i * len(protocols) + j))
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
    fig.savefig(
        os.path.join(output_dir, "Fig_validation_error_maps_angle_disp.png"),
        dpi=600, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)
    logger.info("  Saved: Fig_validation_error_maps_angle_disp.png")


def table_validation_errors_by_angle_bin(
    dual_results: Dict, output_dir: str, logger: logging.Logger,
    approach: str = "hard", bin_deg: float = 5.0,
) -> None:
    """Aggregate mean absolute load/energy errors over angle bins (validation set)."""
    rows = []
    edges = np.arange(40.0, 75.0 + bin_deg, bin_deg)
    for protocol in ("random", "unseen"):
        if protocol not in dual_results or approach not in dual_results[protocol]:
            continue
        dr = dual_results[protocol]
        ens = evaluate_ensemble(
            dr[approach]["models"], approach, dr["val_df"],
            dr["scaler_disp"], dr["scaler_out"], dr["enc"], dr["params"],
        )
        ang = dr["val_df"]["Angle"].values.astype(float)
        abs_le = np.abs(ens["load_errors"])
        abs_ee = np.abs(ens["energy_errors"])
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (ang >= lo) & (ang < hi)
            if not np.any(m):
                continue
            rows.append({
                "protocol": protocol,
                "angle_bin_deg": f"[{lo:g},{hi:g})",
                "n_points": int(np.sum(m)),
                "mean_abs_load_err_kN": float(np.mean(abs_le[m])),
                "mean_abs_energy_err_J": float(np.mean(abs_ee[m])),
            })
    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, "Table_validation_errors_by_angle_bin.csv"), index=False,
        )
        logger.info("  Saved: Table_validation_errors_by_angle_bin.csv")


def fig_qq_load_residuals(
    dual_results: Dict, output_dir: str, logger: logging.Logger, approach: str = "hard",
) -> None:
    """Normal Q–Q plot of validation load residuals (Hard-PINN ensemble), unseen protocol."""
    if "unseen" not in dual_results or approach not in dual_results["unseen"]:
        return
    dr = dual_results["unseen"]
    ens = evaluate_ensemble(
        dr[approach]["models"], approach, dr["val_df"],
        dr["scaler_disp"], dr["scaler_out"], dr["enc"], dr["params"],
    )
    r = np.asarray(ens["load_errors"], dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    if len(r) < 8:
        return
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    if HAS_SCIPY:
        stats.probplot(r, dist="norm", plot=ax)
    else:
        r_sorted = np.sort(r)
        n = len(r_sorted)
        p = (np.arange(1, n + 1) - 0.5) / n
        q = np.sqrt(2) * np.erfinv(2 * np.clip(p, 1e-6, 1 - 1e-6) - 1)
        ax.scatter(q, r_sorted, s=12, alpha=0.75)
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Ordered residuals")
    ax.set_title(
        f"Q–Q (load residuals): unseen θ*={CFG.theta_star}°, "
        f"{MODEL_LABELS.get(approach, approach)}",
        fontsize=10,
    )
    ax.grid(True, alpha=0.28, linestyle="--")
    add_subplot_label(ax, "a")
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.12, 0.10, 0.96, 0.90])
    fig.savefig(
        os.path.join(output_dir, "Fig_qq_load_residuals_unseen.png"),
        dpi=600, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)
    logger.info("  Saved: Fig_qq_load_residuals_unseen.png")


def fig_lc_classifier_diagnostics(
    clf_diag: Dict, output_dir: str, logger: logging.Logger,
) -> None:
    """Confusion matrix, ROC, PR, and calibration from stored CV predictions (no new fits)."""
    if not clf_diag or "cv_y" not in clf_diag:
        return
    y = np.asarray(clf_diag["cv_y"], dtype=int)
    pred = np.asarray(clf_diag["cv_pred"], dtype=int)
    pr = np.asarray(clf_diag["cv_prob_lc2"], dtype=np.float64)
    if len(y) < 4 or len(np.unique(y)) < 2:
        logger.warning("  LC classifier diagnostics skipped: need both classes in CV labels.")
        return
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2))
    ax00, ax01, ax10, ax11 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    cm = confusion_matrix(y, pred, labels=[0, 1])
    im = ax00.imshow(cm, cmap="Blues")
    ax00.set_xticks([0, 1])
    ax00.set_yticks([0, 1])
    ax00.set_xticklabels(["Pred LC1", "Pred LC2"])
    ax00.set_yticklabels(["True LC1", "True LC2"])
    for (i, j), v in np.ndenumerate(cm):
        ax00.text(j, i, int(v), ha="center", va="center", color="black", fontsize=11)
    plt.colorbar(im, ax=ax00, fraction=0.046)
    ax00.set_title("CV confusion")
    fpr, tpr, _ = roc_curve(y, pr)
    ax01.plot(fpr, tpr, color=COLORS.get("hard", "#0072B2"), lw=2.0, label=f"AUC = {auc(fpr, tpr):.3f}")
    ax01.plot([0, 1], [0, 1], "k--", alpha=0.45)
    ax01.set_xlabel("FPR")
    ax01.set_ylabel("TPR")
    ax01.set_title("ROC (score = P(LC2))")
    ax01.legend(loc="lower right", fontsize=8)
    ax01.grid(True, alpha=0.25)
    prec, rec, _ = precision_recall_curve(y, pr)
    ap = average_precision_score(y, pr)
    ax10.plot(rec, prec, color=COLORS.get("soft", "#4DBBD5"), lw=2.0, label=f"AP = {ap:.3f}")
    ax10.set_xlabel("Recall (LC2)")
    ax10.set_ylabel("Precision (LC2)")
    ax10.set_title("Precision–recall (CV)")
    ax10.set_xlim(0, 1.02)
    ax10.set_ylim(0, 1.02)
    ax10.legend(loc="upper right", fontsize=8)
    ax10.grid(True, alpha=0.25)
    n_bins = max(3, min(8, len(y) // 3))
    prob_true, prob_pred = calibration_curve(y, pr, n_bins=n_bins, strategy="uniform")
    ax11.plot(prob_pred, prob_true, "s-", color=COLORS.get("ddns", "#E69F00"), lw=1.5, markersize=6)
    ax11.plot([0, 1], [0, 1], "k--", alpha=0.45)
    ax11.set_xlabel("Mean predicted P(LC2)")
    ax11.set_ylabel("Fraction positives (LC2)")
    ax11.set_title("Calibration (CV)")
    ax11.grid(True, alpha=0.25)
    for ax, lab in zip([ax00, ax01, ax10, ax11], "abcd"):
        add_subplot_label(ax, lab)
    fig.suptitle(
        f"LC plausibility classifier — {clf_diag.get('cv_method', 'CV')} on design metrics",
        fontweight="bold", y=0.98,
    )
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    fig.savefig(
        os.path.join(output_dir, "Fig_lc_classifier_cv_diagnostics.png"),
        dpi=600, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)
    logger.info("  Saved: Fig_lc_classifier_cv_diagnostics.png")
    pred_rows = []
    for i in range(len(y)):
        pred_rows.append({
            "row_index": i,
            "y_true_lc": "LC1" if y[i] == 0 else "LC2",
            "y_pred_lc": "LC1" if pred[i] == 0 else "LC2",
            "p_lc2_cv": float(pr[i]),
        })
    pd.DataFrame(pred_rows).to_csv(
        os.path.join(output_dir, "Table_lc_classifier_cv_predictions.csv"), index=False,
    )
    logger.info("  Saved: Table_lc_classifier_cv_predictions.csv")


def fig_landscape_ensemble_disagreement(
    landscape_df: pd.DataFrame, output_dir: str, logger: logging.Logger,
) -> None:
    """EA_std and IPF_std across the dense surrogate landscape (ensemble disagreement)."""
    if landscape_df is None or len(landscape_df) < 10:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    for ax, col, ylab in zip(
        axes,
        ("EA_std", "IPF_std"),
        (f"EA std (J) @ {D_COMMON:.0f} mm", "IPF std (kN)"),
    ):
        if col not in landscape_df.columns:
            continue
        for lc, sty in zip(sorted(landscape_df["lc"].unique()), ["-", "--"]):
            sub = landscape_df[landscape_df["lc"] == lc].sort_values("angle")
            ax.plot(sub["angle"], sub[col], linestyle=sty, lw=1.2, label=str(lc))
        ax.set_xlabel(r"Angle $\theta$ (°)")
        ax.set_ylabel(ylab)
        ax.set_title(f"Ensemble disagreement: {col.replace('_', ' ')}")
        ax.legend(title="LC", fontsize=8)
        ax.grid(True, alpha=0.26, linestyle="--")
    add_subplot_label(axes[0], "a")
    add_subplot_label(axes[1], "b")
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.03, 0.06, 0.98, 0.92])
    fig.savefig(
        os.path.join(output_dir, "Fig_landscape_ensemble_disagreement.png"),
        dpi=600, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)
    logger.info("  Saved: Fig_landscape_ensemble_disagreement.png")


def fig_d_common_sensitivity_ea(
    models: List[nn.Module], approach: str,
    scaler_disp: StandardScaler, enc: OneHotEncoder, params: ScalingParams,
    landscape_df: pd.DataFrame, output_dir: str, logger: logging.Logger,
) -> None:
    """EA@d vs displacement endpoint d (re-evaluating the trained ensemble only; no new experiments)."""
    if landscape_df is None or len(landscape_df) < 4:
        return
    lc_list = sorted(landscape_df["lc"].unique())
    sens_rows = []
    fig, axes = plt.subplots(1, len(lc_list), figsize=(5.0 * len(lc_list), 4.2), squeeze=False)
    axes = axes.flatten()
    for ax, lc in zip(axes, lc_list):
        d_end = disp_end_mm(str(lc))
        d_grid = np.linspace(max(1.0, D_COMMON - 35.0), min(d_end, D_COMMON + 35.0), 24)
        sub = landscape_df[landscape_df["lc"] == lc]
        q_angles = np.percentile(sub["angle"].values, [20, 45, 65, 85])
        line_colors = matplotlib.cm.viridis(np.linspace(0.15, 0.92, len(q_angles)))
        for k, ang in enumerate(q_angles):
            ea_list = []
            for d_ev in d_grid:
                m = compute_ea_ipf_ensemble(
                    models, approach, float(ang), str(lc), scaler_disp, enc, params, d_eval=float(d_ev),
                )
                ea_list.append(m["EA"])
                sens_rows.append({
                    "lc": lc, "angle_deg": float(ang), "d_eval_mm": float(d_ev),
                    "EA_J": float(m["EA"]), "EA_std": float(m.get("EA_std", 0.0)),
                })
            ax.plot(d_grid, ea_list, color=line_colors[k], lw=1.6, label=f"θ={ang:.1f}°")
        ax.axvline(D_COMMON, color="gray", ls=":", lw=1.2, alpha=0.85)
        ax.set_xlabel(r"Displacement endpoint $d$ (mm) for EA")
        ax.set_ylabel(r"EA (J) to $d$")
        ax.set_title(f"{lc} (d_end={d_end:.0f} mm)")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.26, linestyle="--")
    for ax, lab in zip(axes, "ab"):
        add_subplot_label(ax, lab)
    fig.suptitle(
        "Sensitivity of EA metric to displacement endpoint (ensemble mean; same trained models)",
        fontweight="bold", y=0.98,
    )
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.03, 0.05, 0.97, 0.88])
    fig.savefig(
        os.path.join(output_dir, "Fig_d_common_sensitivity_EA_vs_disp_endpoint.png"),
        dpi=600, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)
    logger.info("  Saved: Fig_d_common_sensitivity_EA_vs_disp_endpoint.png")
    if sens_rows:
        pd.DataFrame(sens_rows).to_csv(
            os.path.join(output_dir, "Table_d_common_sensitivity_EA_grid.csv"), index=False,
        )
        logger.info("  Saved: Table_d_common_sensitivity_EA_grid.csv")


def fig_inverse_vs_nearest_experimental_curve(
    df_all: pd.DataFrame,
    all_inverse_results: List[Dict],
    inv_models: List[nn.Module],
    inv_scaler_disp: StandardScaler,
    inv_enc: OneHotEncoder,
    inv_params: ScalingParams,
    output_dir: str,
    logger: logging.Logger,
) -> None:
    """Predicted load–disp at inverse optimum vs nearest experimental curve (same LC)."""
    if not all_inverse_results:
        return
    panels = []
    for res in all_inverse_results:
        b = res.get("gpbo_best")
        if not b:
            continue
        pred_lc = str(b.get("lc") or b.get("best_lc") or "")
        if not pred_lc:
            continue
        pred_angle = float(b.get("x_best", float("nan")))
        if not np.isfinite(pred_angle):
            continue
        tid = str(res.get("target_info", {}).get("id", "?"))
        panels.append((tid, pred_angle, pred_lc))
    if not panels:
        return
    n = len(panels)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.9 * ncols, 3.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    for idx, (tid, pred_angle, pred_lc) in enumerate(panels):
        ax = axes_flat[idx]
        cand = df_all[df_all["LC"].astype(str) == pred_lc]
        if cand.empty:
            ax.set_visible(False)
            continue
        nearest_ang = float(min(cand["Angle"].unique(), key=lambda a: abs(float(a) - pred_angle)))
        exp_df = cand[np.isclose(cand["Angle"].astype(float), nearest_ang)].sort_values("disp_mm")
        m_pred = compute_ea_ipf_ensemble(
            inv_models, "hard", pred_angle, pred_lc, inv_scaler_disp, inv_enc, inv_params,
        )
        ax.plot(exp_df["disp_mm"], exp_df["load_kN"], "o-", ms=3, lw=1.2, label=f"Exp θ={nearest_ang:.0f}°", color="#009E73")
        ax.plot(m_pred["disps"], m_pred["loads"], "-", lw=1.6, label=f"PINN θ={pred_angle:.1f}°", color="#0072B2")
        ax.set_xlabel("Displacement (mm)")
        ax.set_ylabel("Load (kN)")
        ax.set_title(f"Target {tid} ({pred_lc})")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.26, linestyle="--")
        add_subplot_label(ax, chr(ord("a") + idx))
    for j in range(len(panels), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(
        "Inverse optimum vs nearest experimental load–displacement (same LC)",
        fontweight="bold", y=0.98,
    )
    apply_fig_style(fig)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
    fig.savefig(
        os.path.join(output_dir, "Fig_inverse_vs_nearest_experimental_curve.png"),
        dpi=600, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)
    logger.info("  Saved: Fig_inverse_vs_nearest_experimental_curve.png")


def generate_output_manifest(output_dir: str, logger: logging.Logger) -> None:
    """List generated figures, tables, and key sidecar files (no extra experiments)."""
    exts = (".png", ".csv", ".json", ".txt", ".pdf")
    rows = []
    for name in sorted(os.listdir(output_dir)):
        path = os.path.join(output_dir, name)
        if not os.path.isfile(path):
            continue
        low = name.lower()
        if not low.endswith(exts):
            continue
        kind = "figure" if low.endswith(".png") or low.endswith(".pdf") else "other"
        if low.startswith("fig_") or name.startswith("Fig_"):
            kind = "figure"
        elif low.startswith("table") or name.startswith("Table"):
            kind = "table"
        elif low.endswith(".json"):
            kind = "json"
        elif low.endswith(".txt"):
            kind = "text"
        try:
            sz = os.path.getsize(path)
        except OSError:
            sz = -1
        rows.append({"filename": name, "kind": kind, "size_bytes": int(sz)})
    if rows:
        mf = os.path.join(output_dir, "MANIFEST_outputs.csv")
        pd.DataFrame(rows).to_csv(mf, index=False)
        logger.info(f"  Saved: {os.path.basename(mf)} ({len(rows)} entries)")


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run_pipeline(data_dir: str, output_dir: str):
    """Main pipeline orchestrating all computations."""
    refresh_device()
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)
    apply_dry_run_settings(logger)
    set_publication_style()
    log_runtime_environment(output_dir, logger)
    check_publication_dependencies(logger)
    
    logger.info("=" * 80)
    logger.info("PINN CRASHWORTHINESS FRAMEWORK - VERSION 5 (Reviewer-Proof Edition)")
    logger.info("=" * 80)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Ensemble size: {CFG.n_ensemble}")
    logger.info(f"Random seed: {CFG.seed_base}")
    logger.info("")
    logger.info("VERSION 6 CHANGES:")
    logger.info("  [P] Ensemble size increased from M=5 to M=20 for tighter uncertainty bands")
    logger.info("  [Q] Unseen-angle curves now use ensemble mean (not best member) as center line")
    logger.info("  [R] Conformal calibration applied to all plotted +/-2sigma confidence bands")
    logger.info("  [S] Conformal factors reported in Table1_forward_results.csv")
    logger.info("  [T] On-manifold inverse targets with grid-midpoint placement")
    logger.info(
        f"  [U] GP-BO budget (defaults): {BO_CFG.n_calls_total} total "
        f"(n_init={BO_CFG.n_init}, joint theta+LC search via skopt)"
    )
    logger.info("  [V] Convergence filter: Tukey fence (Q1-1.5*IQR) on training-set R2")
    logger.info("  [W] Hard-PINN unseen architecture widened from [32,32] to [64,64]")
    logger.info("  [X] Dual optimizer selection: accuracy (star) and efficiency (diamond)")
    logger.info("  [Y] Target uniqueness scan: detect multimodal objective landscapes")
    logger.info("  [Z] M_total and M_eff reported in Table1_forward_results.csv")
    logger.info(f"       Tukey fence multiplier: {CFG.convergence_filter_iqr}")
    logger.info("")
    logger.info("VERSION 7 CHANGES:")
    logger.info("  [AA] Hard-PINN unseen architecture reverted to [32,32]")
    logger.info("  [AB] Epochs and early-stopping patience increased across all configs")
    logger.info("  [AC] ttest_rel replaced with ttest_ind (Welch) to handle unequal M_eff")
    logger.info("  [AD] GP-BO Posterior Evolution figure: true objective landscape overlay")
    logger.info("  [AE] GP-BO Posterior Evaluation: Fig_gpbo_posterior_evaluation_<target>.png (one per target)")
    logger.info("")
    logger.info("VERSION 8 CHANGES:")
    logger.info("  [AF] Hard-PINN unseen: stabilized training (warmup + cosine + SWA)")
    logger.info("       LR warmup (80 epochs) prevents catastrophic early gradient updates")
    logger.info("       Cosine annealing replaces ReduceLROnPlateau for deterministic decay")
    logger.info("       SWA (last 20%) averages weights for smoother loss landscape")
    logger.info("  [AG] Hard-PINN unseen: HPO v3 config, arch [128,64] (was [32,32])")
    logger.info("       154 Optuna TPE trials, best val R²=0.851 (was 0.850)")
    logger.info("  [AH] Soft-PINN unseen: HPO v3 config, arch [256,128] (was [256,128,64])")
    logger.info("       95 Optuna TPE trials, best val R²=0.805 (was 0.801)")
    logger.info("  [AI] Full-data Hard-PINN (inverse design) also uses stabilized training")
    logger.info("  [AJ] GP-BO switched to skopt.gp_minimize with joint (theta, LC) space")
    logger.info("       Single GP shares information across LCs (fixes LC1 starvation)")
    logger.info(
        f"       Default budget: {BO_CFG.n_calls_total} calls (n_init={BO_CFG.n_init}); "
        "override via BO_CFG / code if needed"
    )
    logger.info("       Posteriors reconstructed from res.models at each iteration")
    logger.info("")
    logger.info("VERSION 9 (Q1 inverse / MOBO artifacts):")
    logger.info("  [AK] Table3_inverse_illposedness, Table_inverse_local_minima, Table_inverse_topk_basins")
    logger.info("  [AL] Table_forward_jacobian_summary, Table_inverse_vs_calibration")
    logger.info("  [AM] Likelihood posterior: Fig_inverse_posterior_likelihood, Table_inverse_posterior_likelihood")
    logger.info("  [AN] Table_inverse_stress_protocol (reviewer_proof), Table_inverse_theta_member_spread")
    logger.info("  [AO] Table_inverse_ablation (--inverse_ablation); Table_mobo_vs_landscape_pareto_distance")
    logger.info("  [AP] Multi-seed BO uses full BOConfig (dataclasses.replace); strict_paper requires MOBO run")
    logger.info("")
    logger.info("VERSION 5 REVIEWER-PROOF ADDITIONS:")
    logger.info("  [H] Physics verification figure: dE/dd = F proof")
    logger.info("  [I] Baseline comparison: Linear, RF, XGBoost, GP")
    logger.info("  [K] Hyperparameter sensitivity: w_phys × lr heatmap")
    logger.info("  [L] Multi-seed inverse design robustness")
    logger.info("  [M] Uncertainty calibration analysis")
    logger.info("  [N] MAE added to all tables")
    logger.info("  [O] Extended ablation table")
    logger.info("")
    logger.info("VERSION 4 CHANGES:")
    logger.info("  [A] LC-specific displacement ranges (LC1=80mm, LC2=130mm)")
    logger.info("  [B] Fixed NameError with disp_end variable")
    logger.info("  [C] Cross-protocol bar plot y-axis starts at 0.5")
    logger.info("  [D] Ablation study on unseen angle θ=60° protocol")
    logger.info("  [E] Feasible inverse design targets from empirical data")
    logger.info("  [F] GP-BO Posterior Evaluation (2x4 grid, one PNG per inverse target)")
    logger.info("  [G] GP-BO inverse: convergence diagnostics and objective traces per target")
    logger.info("")
    
    # Load data
    df_all = load_data(data_dir, logger)
    dual_results = {}
    
    # Protocol 1: Random 80-20 split
    logger.info("\n" + "=" * 70)
    logger.info("PROTOCOL 1: RANDOM 80-20 SPLIT (INTERPOLATION)")
    logger.info("=" * 70)
    train_df_r, val_df_r = split_random_80_20(df_all, CFG.split_seed, logger)
    scaler_disp_r, scaler_out_r, enc_r, params_r = create_preprocessors(train_df_r, logger)
    save_reproducibility_artifacts(output_dir, "random", train_df_r, scaler_disp_r, scaler_out_r, enc_r, params_r, logger)
    
    results_random = {}
    for approach in ["ddns", "soft", "hard"]:
        results_random[approach] = train_ensemble(approach, train_df_r, val_df_r, scaler_disp_r, scaler_out_r, enc_r, params_r, "random", logger)
    
    dual_results["random"] = results_random
    dual_results["random"]["train_df"] = train_df_r
    dual_results["random"]["val_df"] = val_df_r
    dual_results["random"]["scaler_disp"] = scaler_disp_r
    dual_results["random"]["scaler_out"] = scaler_out_r
    dual_results["random"]["enc"] = enc_r
    dual_results["random"]["params"] = params_r
    
    # Protocol 2: Unseen angle
    logger.info("\n" + "=" * 70)
    logger.info(f"PROTOCOL 2: UNSEEN-ANGLE HOLDOUT (θ*={CFG.theta_star}°)")
    logger.info("=" * 70)
    train_df_u, val_df_u = split_unseen_angle(df_all, CFG.theta_star, logger)
    scaler_disp_u, scaler_out_u, enc_u, params_u = create_preprocessors(train_df_u, logger)
    save_reproducibility_artifacts(output_dir, "unseen", train_df_u, scaler_disp_u, scaler_out_u, enc_u, params_u, logger)
    
    results_unseen = {}
    for approach in ["ddns", "soft", "hard"]:
        results_unseen[approach] = train_ensemble(approach, train_df_u, val_df_u, scaler_disp_u, scaler_out_u, enc_u, params_u, "unseen", logger)
    
    dual_results["unseen"] = results_unseen
    dual_results["unseen"]["train_df"] = train_df_u
    dual_results["unseen"]["val_df"] = val_df_u
    dual_results["unseen"]["scaler_disp"] = scaler_disp_u
    dual_results["unseen"]["scaler_out"] = scaler_out_u
    dual_results["unseen"]["enc"] = enc_u
    dual_results["unseen"]["params"] = params_u
    
    # Statistical tests
    logger.info("\n" + "=" * 70)
    logger.info("STATISTICAL SIGNIFICANCE TESTS")
    logger.info("=" * 70)
    stat_tests = compute_statistical_tests(dual_results, logger)
    
    # Design space metrics
    logger.info("\n" + "=" * 70)
    logger.info("DESIGN SPACE METRICS (FROM DATA)")
    logger.info("=" * 70)
    df_metrics = compute_design_space_metrics(df_all, logger)
    logger.info(f"Computed metrics for {len(df_metrics)} configurations")
    enrich_df_metrics_ea_common(df_metrics, df_all, logger)
    
    # [CHANGE D] Ablation study on unseen angle protocol
    df_ablation = pd.DataFrame()
    if CFG.run_ablation:
        logger.info("\n" + "=" * 70)
        logger.info("ABLATION STUDY: PHYSICS WEIGHT (UNSEEN ANGLE θ=60°)")
        logger.info("=" * 70)
        df_ablation = run_ablation_study(train_df_u, val_df_u, scaler_disp_u, scaler_out_u, enc_u, params_u, "unseen", logger)
        df_ablation.to_csv(os.path.join(output_dir, "Table5_ablation.csv"), index=False)
        logger.info("  Saved: Table5_ablation.csv")
    
    # Forward model figures
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING FORWARD MODEL FIGURES")
    logger.info("=" * 70)
    
    # Compute uncertainty calibration BEFORE figures so conformal factors
    # can be applied to plotted confidence bands
    logger.info("  Computing uncertainty calibration (conformal factors)...")
    calibration = compute_uncertainty_calibration(dual_results, logger)
    
    fig_parity_plots(dual_results, output_dir, logger)
    fig_residual_histograms(dual_results, output_dir, logger)
    fig_boxplot_comparison(dual_results, output_dir, logger)
    fig_training_curves(dual_results, output_dir, logger)
    fig_cross_protocol_comparison(dual_results, output_dir, logger)
    fig_unseen_curves(dual_results, df_all, output_dir, logger, calibration=calibration)
    fig_random_grid_curves(dual_results, df_all, output_dir, logger)
    fig_validation_error_maps(dual_results, output_dir, logger)
    table_validation_errors_by_angle_bin(dual_results, output_dir, logger)
    fig_qq_load_residuals(dual_results, output_dir, logger)
    # Ablation figure removed per user request
    # if not df_ablation.empty:
    #     fig_ablation_study(df_ablation, output_dir, logger)
    
    # =========================================================================
    # REVIEWER-PROOF ADDITIONS (V5)
    # =========================================================================
    if CFG.run_reviewer_proof:
        logger.info("\n" + "=" * 70)
        logger.info("REVIEWER-PROOF ANALYSIS: PHYSICS VERIFICATION")
        logger.info("=" * 70)
        physics_residuals = fig_physics_verification(dual_results, val_df_u, scaler_disp_u, 
                                                       enc_u, params_u, output_dir, logger)
        
        # --- BASELINE COMPARISON: UNSEEN PROTOCOL ONLY ---
        logger.info("\n" + "=" * 70)
        logger.info("REVIEWER-PROOF ANALYSIS: BASELINE MODEL COMPARISON (UNSEEN θ=60°)")
        logger.info("=" * 70)
        baseline_results_u = train_baseline_models(train_df_u, val_df_u, scaler_disp_u, enc_u, params_u, logger)
        fig_baseline_comparison(baseline_results_u, dual_results, output_dir, logger, protocol="unseen")
        
        # --- HYPERPARAMETER SENSITIVITY: UNSEEN PROTOCOL ---
        logger.info("\n" + "=" * 70)
        logger.info("REVIEWER-PROOF ANALYSIS: HYPERPARAMETER SENSITIVITY (UNSEEN θ=60°)")
        logger.info("=" * 70)
        sensitivity_df_u = run_hyperparam_sensitivity(train_df_u, val_df_u, scaler_disp_u, scaler_out_u, enc_u, params_u, "unseen", logger)
        fig_hyperparam_sensitivity(sensitivity_df_u, output_dir, logger, tag="unseen")
        
        logger.info("\n" + "=" * 70)
        logger.info("REVIEWER-PROOF ANALYSIS: UNCERTAINTY CALIBRATION")
        logger.info("=" * 70)
        # calibration already computed before figure generation; reuse it
        fig_reliability_diagram(calibration, output_dir, logger)

        # Calibration vs ensemble size study
        logger.info("  Computing calibration sensitivity to ensemble size...")
        try:
            cal_vs_M = compute_calibration_vs_M(dual_results, logger)
            fig_calibration_vs_M(cal_vs_M, output_dir, logger)
        except Exception as e:
            logger.warning(f"  Calibration vs M analysis failed: {e}")

    else:
        logger.info("\n  Skipping reviewer-proof analyses (--no_reviewer_proof flag set)")

    # Leave-one-angle-out cross-validation
    if getattr(CFG, 'run_loao_cv', True) and not CFG.dry_run:
        logger.info("\n" + "=" * 70)
        logger.info("LEAVE-ONE-ANGLE-OUT CROSS-VALIDATION")
        logger.info("=" * 70)
        try:
            loao_results = run_leave_one_angle_out_cv(df_all, logger)
            fig_loao_cv_results(loao_results, output_dir, logger)
            pd.DataFrame(loao_results["per_fold_r2"]).to_csv(
                os.path.join(output_dir, "Table_loao_cv.csv"), index=False)
            logger.info("  Saved: Table_loao_cv.csv")
        except Exception as e:
            logger.warning(f"  LOAO-CV failed: {e}")

    # [CHANGE E] Generate feasible inverse design targets
    logger.info("\n" + "=" * 70)
    logger.info("INVERSE DESIGN: FULL-DATA HARD-PINN TRAINING")
    logger.info("=" * 70)
    logger.info("  Training Hard-PINN on 100% of data for maximum inverse design accuracy")
    logger.info("  (No holdout: forward model already validated via dual protocols)")
    
    # Train full-data Hard-PINN ensemble for inverse design
    inv_models, inv_scaler_disp, inv_scaler_out, inv_enc, inv_params = train_full_data_hard_pinn(df_all, logger)
    
    # Generate inverse design targets from empirical data quantiles
    logger.info("\n" + "=" * 70)
    logger.info("INVERSE DESIGN TARGETS")
    logger.info("=" * 70)
    inverse_targets = generate_feasible_targets(df_metrics, logger, df_all=df_all)
    
    # =========================================================================
    # ENSEMBLE CLASSIFIER FOR LOADING-CONDITION PLAUSIBILITY
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("ENSEMBLE CLASSIFIER: LOADING-CONDITION PLAUSIBILITY")
    logger.info("=" * 70)
    cal_ens, clf_feat_scaler, clf_diag = train_lc_plausibility_classifier(df_metrics, logger)
    fig_lc_classifier_diagnostics(clf_diag, output_dir, logger)
    generate_classifier_diagnostics_table(cal_ens, clf_feat_scaler, clf_diag, output_dir, logger)
    fig_classifier_decision_boundary(cal_ens, clf_feat_scaler, df_metrics, output_dir, logger)
    
    # Auto-tune lambda (classifier penalty weight)
    lambda_opt, lambda_diag = auto_tune_lambda(cal_ens, clf_feat_scaler, df_metrics, logger)
    BO_CFG.prob_weight = lambda_opt

    # Auto-tune robust uncertainty penalty
    beta_robust = auto_tune_beta(inv_models, "hard", df_metrics,
                                  inv_scaler_disp, inv_enc, inv_params, logger)
    BO_CFG.beta_robust = beta_robust

    # Inverse design optimization using full-data models WITH classifier + robustness penalties
    logger.info("\n" + "=" * 70)
    logger.info("INVERSE DESIGN: GP-BO (WITH CLASSIFIER + ROBUSTNESS PENALTIES)")
    logger.info("=" * 70)
    logger.info(f"  Using full-data Hard-PINN models + classifier penalty (w={BO_CFG.prob_weight:.4f})"
                f" + robustness penalty (beta={BO_CFG.beta_robust:.6f})")
    
    all_inverse_results = []
    for target in inverse_targets:
        logger.info(f"\n  Target {target['id']}: EA@{D_COMMON:.0f}mm={target['EA']:.2f}J, IPF={target['IPF']:.3f}kN")
        logger.info(f"    Expected: {target.get('expected_lc', 'N/A')} at {target.get('expected_angle_range', 'N/A')}")
        res = run_inverse_design(inv_models, "hard", target["EA"], target["IPF"], 
                                 inv_scaler_disp, inv_enc, inv_params, BO_CFG, logger,
                                 cal_ens=cal_ens, feat_scaler=clf_feat_scaler)
        res["target_info"] = target  # Store target info for analysis
        all_inverse_results.append(res)
        
        for method in ["gpbo"]:
            key = f"{method}_best"
            if key in res:
                best = res[key]
                method_name = method.upper()
                pred_lc = best.get('lc', best.get('best_lc', ''))
                pred_angle = best['x_best']
                # Check if angle is non-training angle
                training_angles = [45, 50, 55, 60, 65, 70]
                is_interpolated = not any(abs(pred_angle - ta) < 0.5 for ta in training_angles)
                interp_marker = "✓" if is_interpolated else "≈trained"
                logger.info(f"    {method_name}: θ={pred_angle:.1f}° {interp_marker}, LC={pred_lc}, "
                           f"EA@{EA_COMMON_MM_TAG} err={best.get('ea_error_pct', float('nan')):.1f}%, "
                           f"IPF_err={best['ipf_error_pct']:.1f}% "
                           f"→ EA@{EA_COMMON_MM_TAG}={best.get('pred_ea', 0):.2f}J, "
                           f"EA_full={best.get('pred_ea_full', best.get('pred_ea', 0)):.2f}J, "
                           f"IPF={best.get('pred_ipf', 0):.3f}kN")
        
        # One convergence curve + one GP-BO Posterior Evaluation figure per target
        # (avoid duplicate BO diagnostic figures; see fig_bo_posterior_evaluation docstring).
        fig_inverse_optimizer_convergence(res, output_dir, logger, tag=target['id'])
        fig_bo_posterior_evaluation(res, output_dir, logger, tag=target['id'])
    
    # Analyze LC distribution in results
    logger.info("\n  --- Inverse Design LC Distribution Summary ---")
    for method in ["gpbo"]:
        lc1_count = sum(1 for r in all_inverse_results if f"{method}_best" in r and r[f"{method}_best"].get("lc") == "LC1")
        lc2_count = sum(1 for r in all_inverse_results if f"{method}_best" in r and r[f"{method}_best"].get("lc") == "LC2")
        logger.info(f"    {method.upper()}: LC1={lc1_count}, LC2={lc2_count} (target: 2 LC1, 3 LC2)")
    
    # Inverse design figures (combined)
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING INVERSE DESIGN FIGURES")
    logger.info("=" * 70)
    if all_inverse_results:
        generate_optimizer_comparison_table(all_inverse_results, output_dir, logger)
    fig_design_space(inv_models, "hard", inv_scaler_disp, inv_enc, inv_params, output_dir, logger)
    if all_inverse_results:
        fig_inverse_parity_uncertainty(all_inverse_results, output_dir, logger)
        fig_inverse_vs_nearest_experimental_curve(
            df_all, all_inverse_results, inv_models,
            inv_scaler_disp, inv_enc, inv_params, output_dir, logger,
        )
        # Ill-posedness analysis figures
        fig_solution_landscape(all_inverse_results, output_dir, logger)
        fig_inverse_posterior(all_inverse_results, output_dir, logger)

    # Forward-map Jacobian analysis
    logger.info("\n" + "=" * 70)
    logger.info("FORWARD-MAP JACOBIAN ANALYSIS (ILL-POSEDNESS CHARACTERIZATION)")
    logger.info("=" * 70)
    jacobian_results = None
    try:
        jacobian_results = compute_forward_map_jacobian(
            inv_models, "hard", inv_scaler_disp, inv_enc, inv_params,
            (CFG.angle_opt_min, CFG.angle_opt_max), logger)
        fig_forward_map_jacobian(jacobian_results, output_dir, logger)
    except Exception as e:
        logger.warning(f"  Jacobian analysis failed: {e}")

    # =========================================================================
    # LAMBDA SENSITIVITY SWEEP
    # =========================================================================
    if BO_CFG.lambda_sweep:
        logger.info("\n" + "=" * 70)
        logger.info("LAMBDA SENSITIVITY ANALYSIS")
        logger.info("=" * 70)
        run_lambda_sensitivity(
            inv_models, "hard", inverse_targets,
            inv_scaler_disp, inv_enc, inv_params,
            BO_CFG, cal_ens, clf_feat_scaler,
            output_dir, logger
        )
    
    # =========================================================================
    # CLASSIFIER ABLATION: WITH vs WITHOUT PLAUSIBILITY PENALTY
    # =========================================================================
    if BO_CFG.run_classifier_ablation:
        logger.info("\n" + "=" * 70)
        logger.info("CLASSIFIER ABLATION: WITH vs WITHOUT PENALTY")
        logger.info("=" * 70)
        logger.info("  Running inverse design WITHOUT classifier penalty for comparison...")
        
        all_inverse_results_no_clf = []
        for target in inverse_targets:
            res_no = run_inverse_design(
                inv_models, "hard", target["EA"], target["IPF"],
                inv_scaler_disp, inv_enc, inv_params, BO_CFG, logger,
                cal_ens=None, feat_scaler=None)  # no classifier
            res_no["target_info"] = target
            all_inverse_results_no_clf.append(res_no)
        
        # Build comparison table
        ablation_rows = []
        for i, target in enumerate(inverse_targets):
            tid = target["id"]
            for method in ["gpbo"]:
                key = f"{method}_best"
                # WITH penalty
                if key in all_inverse_results[i]:
                    bw = all_inverse_results[i][key]
                    ea_err_w = bw.get("ea_error_pct", float("nan"))
                    ipf_err_w = bw.get("ipf_error_pct", float("nan"))
                    lc_w = bw.get("lc", bw.get("best_lc", ""))
                    angle_w = bw.get("x_best", float("nan"))
                    # Compute p_LC for the WITH-penalty result
                    m_w = compute_ea_ipf_ensemble(
                        inv_models, "hard", angle_w, lc_w,
                        inv_scaler_disp, inv_enc, inv_params, d_eval=D_COMMON
                    )
                    _, p_lc_w = compute_lc_penalty(
                        cal_ens, clf_feat_scaler,
                        m_w["EA"], m_w["IPF"],
                        lc_w, prob_weight=0.0,  # just get p_lc
                        angle_deg=float(angle_w),
                    )
                else:
                    ea_err_w = ipf_err_w = p_lc_w = float("nan")
                    lc_w = angle_w = ""
                
                # WITHOUT penalty
                if key in all_inverse_results_no_clf[i]:
                    bwo = all_inverse_results_no_clf[i][key]
                    ea_err_wo = bwo.get("ea_error_pct", float("nan"))
                    ipf_err_wo = bwo.get("ipf_error_pct", float("nan"))
                    lc_wo = bwo.get("lc", bwo.get("best_lc", ""))
                    angle_wo = bwo.get("x_best", float("nan"))
                    m_wo = compute_ea_ipf_ensemble(
                        inv_models, "hard", angle_wo, lc_wo,
                        inv_scaler_disp, inv_enc, inv_params, d_eval=D_COMMON
                    )
                    _, p_lc_wo = compute_lc_penalty(
                        cal_ens, clf_feat_scaler,
                        m_wo["EA"], m_wo["IPF"],
                        lc_wo, prob_weight=0.0,
                        angle_deg=float(angle_wo),
                    )
                else:
                    ea_err_wo = ipf_err_wo = p_lc_wo = float("nan")
                    lc_wo = angle_wo = ""
                
                ablation_rows.append({
                    "Target": tid,
                    "Method": method.upper(),
                    "With_Penalty_Angle": f"{angle_w:.1f}" if isinstance(angle_w, float) else "",
                    "With_Penalty_LC": lc_w,
                    "With_Penalty_EA_err%": f"{ea_err_w:.2f}",
                    "With_Penalty_IPF_err%": f"{ipf_err_w:.2f}",
                    "With_Penalty_p_LC": f"{p_lc_w:.4f}" if not np.isnan(p_lc_w) else "",
                    "No_Penalty_Angle": f"{angle_wo:.1f}" if isinstance(angle_wo, float) else "",
                    "No_Penalty_LC": lc_wo,
                    "No_Penalty_EA_err%": f"{ea_err_wo:.2f}",
                    "No_Penalty_IPF_err%": f"{ipf_err_wo:.2f}",
                    "No_Penalty_p_LC": f"{p_lc_wo:.4f}" if not np.isnan(p_lc_wo) else "",
                })
        
        df_clf_ablation = pd.DataFrame(ablation_rows)
        df_clf_ablation.to_csv(os.path.join(output_dir, "Table_classifier_ablation.csv"), index=False)
        logger.info("  Saved: Table_classifier_ablation.csv")
        
        # Summary statistics
        with_plc = [float(r["With_Penalty_p_LC"]) for r in ablation_rows 
                     if r["With_Penalty_p_LC"] and r["With_Penalty_p_LC"] != "nan"]
        without_plc = [float(r["No_Penalty_p_LC"]) for r in ablation_rows 
                        if r["No_Penalty_p_LC"] and r["No_Penalty_p_LC"] != "nan"]
        if with_plc and without_plc:
            logger.info(f"  Mean p_LC WITH penalty:    {np.mean(with_plc):.4f} (±{np.std(with_plc):.4f})")
            logger.info(f"  Mean p_LC WITHOUT penalty: {np.mean(without_plc):.4f} (±{np.std(without_plc):.4f})")
    
    # =========================================================================
    # REVIEWER-PROOF: MULTI-SEED INVERSE DESIGN ROBUSTNESS
    # =========================================================================
    if CFG.run_reviewer_proof:
        logger.info("\n" + "=" * 70)
        logger.info("REVIEWER-PROOF: INVERSE DESIGN ROBUSTNESS (MULTI-SEED)")
        logger.info("=" * 70)
        robust_inverse_results = []
        for target in inverse_targets[:3]:  # First 3 targets for robustness analysis
            logger.info(f"\n  Robustness analysis for Target {target['id']}")
            robust_res = run_inverse_design_robust(
                inv_models, "hard", target["EA"], target["IPF"],
                inv_scaler_disp, inv_enc, inv_params, BO_CFG, logger, n_seeds=5,
                cal_ens=cal_ens, feat_scaler=clf_feat_scaler)
            robust_inverse_results.append(robust_res)
        generate_inverse_robustness_table(robust_inverse_results, output_dir, logger)
    
    # =========================================================================
    # MULTIOBJECTIVE OPTIMIZATION WITH ENHANCED VISUALIZATION
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info(f"MULTIOBJECTIVE EA@{D_COMMON:.0f}mm vs IPF TRADE-OFF ANALYSIS (DISPLACEMENT-FAIR)")
    logger.info("=" * 70)
    logger.info(f"  Both LCs evaluated at d={D_COMMON:.0f}mm to isolate intrinsic force response")
    logger.info("  Key insight: α=0 (IPF priority) → low angles + LC1 (stable)")
    logger.info("              α=1 (EA priority) → high angles + LC2 (catastrophic risk)")
    
    pareto_df, landscape_df = run_multiobjective_sweep(inv_models, "hard", inv_scaler_disp, inv_enc, 
                                                        inv_params, df_metrics, logger,
                                                        output_dir=output_dir, df_all=df_all)
    
    # Traditional Pareto figure
    fig_pareto_tradeoff(pareto_df, output_dir, logger)
    
    # Enhanced multi-objective heatmaps
    fig_multiobjective_heatmaps(pareto_df, landscape_df, output_dir, logger, calibration=calibration)
    
    # Save tables
    pareto_df.to_csv(os.path.join(output_dir, "Table6_pareto_sweep.csv"), index=False)
    landscape_df.to_csv(os.path.join(output_dir, "Table_design_landscape.csv"), index=False)
    logger.info("  Saved: Table6_pareto_sweep.csv")
    logger.info("  Saved: Table_design_landscape.csv")
    fig_landscape_ensemble_disagreement(landscape_df, output_dir, logger)
    fig_d_common_sensitivity_ea(
        inv_models, "hard", inv_scaler_disp, inv_enc, inv_params,
        landscape_df, output_dir, logger,
    )

    # -------------------------------------------------------------------------
    # Multi-objective Bayesian optimization (qLogNEHVI / BoTorch)
    # -------------------------------------------------------------------------
    mobo_result = None
    if HAS_BOTORCH and CFG.run_mobo_qnehvi and not getattr(CFG, '_skip_mobo', False):
        logger.info("\n" + "=" * 70)
        logger.info("MULTI-OBJECTIVE BO: qLogNEHVI (EXPECTED HYPERVOLUME, BOTORCH)")
        logger.info("=" * 70)
        logger.info("  Acquisition: qLogNoisyExpectedHypervolumeImprovement (stable qNEHVI; Daulton et al., 2023)")
        logger.info("  Reference point: documented in mobo_qnehvi_reference_point.json (landscape nadir + margin)")
        mobo_result = run_multiobjective_mobo_qnehvi(
            inv_models, "hard", inv_scaler_disp, inv_enc, inv_params,
            landscape_df, logger, output_dir=output_dir, cfg=MOBO_QNEHVI_CFG,
        )

    fig_moo_objective_space_validation(
        pareto_df, landscape_df, output_dir, logger,
        mobo_result=mobo_result, df_metrics=df_metrics,
    )
    fig_mobo_qnehvi_diagnostics(mobo_result, output_dir, logger)

    # Save per-LC conditional Pareto fronts
    pareto_by_lc = pareto_df.attrs.get("pareto_by_lc", {})
    for lc, lc_df in pareto_by_lc.items():
        fname = f"Table_pareto_conditional_{lc}.csv"
        lc_df.to_csv(os.path.join(output_dir, fname), index=False)
        logger.info(f"  Saved: {fname}")

    # =========================================================================
    # PARETO-OPTIMAL INVERSE DESIGN TARGETS
    # =========================================================================
    # Run inverse design on targets drawn from the actual Pareto front to
    # verify that GP-BO can recover known-feasible designs.
    if CFG.run_reviewer_proof:
        pareto_dominance_df = pareto_df.attrs.get("pareto_dominance", pd.DataFrame())
        if not pareto_dominance_df.empty and len(pareto_dominance_df) >= 5:
            logger.info("\n" + "=" * 70)
            logger.info("INVERSE DESIGN WITH PARETO-OPTIMAL TARGETS")
            logger.info("=" * 70)
            pareto_targets = generate_pareto_targets(pareto_dominance_df, logger, n_targets=5)

            if pareto_targets:
                pareto_inverse_results = []
                for target in pareto_targets:
                    logger.info(f"\n  Pareto Target {target['id']}: EA@{EA_COMMON_MM_TAG}={target['EA']:.2f}J, IPF={target['IPF']:.3f}kN")
                    res = run_inverse_design(inv_models, "hard", target["EA"], target["IPF"],
                                             inv_scaler_disp, inv_enc, inv_params, BO_CFG, logger,
                                             cal_ens=cal_ens, feat_scaler=clf_feat_scaler)
                    res["target_info"] = target
                    pareto_inverse_results.append(res)

                    best = res.get("gpbo_best", {})
                    if best:
                        angle_err = abs(best["x_best"] - target["angle_hint"])
                        lc_match = best.get("lc", "") == target["lc_hint"]
                        logger.info(f"    Result: theta={best['x_best']:.1f} deg ({'+' if angle_err < 1 else ''}{angle_err:.1f} deg from hint), "
                                   f"LC={best.get('lc','')}{' MATCH' if lc_match else ' MISMATCH'}, "
                                   f"EA_err={best.get('ea_error_pct',0):.1f}%, IPF_err={best.get('ipf_error_pct',0):.1f}%")

                # Save recovery table
                rows = []
                for target, res in zip(pareto_targets, pareto_inverse_results):
                    best = res.get("gpbo_best", {})
                    rows.append({
                        "Target": target["id"],
                        "EA_target": f"{target['EA']:.2f}",
                        "IPF_target": f"{target['IPF']:.3f}",
                        "Hint_angle": f"{target['angle_hint']:.1f}",
                        "Hint_LC": target["lc_hint"],
                        "Recovered_angle": f"{best.get('x_best', float('nan')):.1f}",
                        "Recovered_LC": best.get("lc", ""),
                        "EA_err_pct": f"{best.get('ea_error_pct', float('nan')):.2f}",
                        "IPF_err_pct": f"{best.get('ipf_error_pct', float('nan')):.2f}",
                        "Angle_error_deg": f"{abs(best.get('x_best', 0) - target['angle_hint']):.1f}",
                        "LC_match": best.get("lc", "") == target["lc_hint"],
                    })
                pd.DataFrame(rows).to_csv(os.path.join(output_dir, "Table_pareto_target_recovery.csv"), index=False)
                logger.info("  Saved: Table_pareto_target_recovery.csv")

    # Summary tables
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING SUMMARY TABLES")
    logger.info("=" * 70)
    generate_summary_tables(dual_results, df_metrics, all_inverse_results, stat_tests, output_dir, logger, calibration=calibration)

    logger.info("\n" + "=" * 70)
    logger.info("INVERSE / MOBO PUBLICATION ARTIFACTS (TABLES)")
    logger.info("=" * 70)
    generate_inverse_publication_artifacts(
        output_dir,
        logger,
        all_inverse_results,
        calibration=calibration,
        jacobian_results=jacobian_results,
        inv_models=inv_models,
        inv_scaler_disp=inv_scaler_disp,
        inv_enc=inv_enc,
        inv_params=inv_params,
        dual_results=dual_results,
        df_metrics=df_metrics,
        df_all=df_all,
        bo_cfg=BO_CFG,
        cal_ens=cal_ens,
        clf_feat_scaler=clf_feat_scaler,
        mobo_result=mobo_result,
        landscape_df=landscape_df,
    )

    logger.info("\n" + "=" * 70)
    logger.info("REPRODUCIBILITY & COMPUTE SUMMARY (PUBLICATION)")
    logger.info("=" * 70)
    write_statistical_testing_policy(output_dir, logger)
    generate_compute_budget_summary(dual_results, all_inverse_results, output_dir, logger)
    generate_output_manifest(output_dir, logger)

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"All results saved to: {output_dir}")
    logger.info("=" * 80)
    
    return dual_results, all_inverse_results, pareto_df, stat_tests, df_ablation



# =============================================================================
# CLI ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="PINN Crashworthiness Framework - Version 5 (Reviewer-Proof Edition)")
    parser.add_argument("--data_dir", type=str, default=".", help="Directory containing data files")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--n_ensemble", type=int, default=20, help="Ensemble size (default: 20)")
    parser.add_argument("--show_plots", action="store_true", help="Display plots on screen")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed base")
    parser.add_argument("--no_ablation", action="store_true", help="Skip ablation study")

    parser.add_argument("--no_reviewer_proof", action="store_true", help="Skip reviewer-proof analyses (baselines, sensitivity, robustness)")
    parser.add_argument("--force_cpu", action="store_true", help="Use CPU even if CUDA is available")
    parser.add_argument("--no_mobo_qnehvi", action="store_true",
                        help="Skip multi-objective BO (qLogNEHVI / BoTorch) after landscape sweep")
    parser.add_argument(
        "--strict_paper",
        action="store_true",
        help="Abort if optional deps missing (skopt for GP-BO, botorch for MOBO). Use for submission runs.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="CI/smoke: tiny training budgets, M<=2, no MOBO/reviewer extras; inverse uses coarse grid (no skopt).",
    )
    parser.add_argument(
        "--inverse_ablation",
        action="store_true",
        help="Run extra GP-BO inverse ablations (no Tikhonov / no classifier / no robustness) on first targets; expensive.",
    )
    parser.add_argument(
        "--no_inverse_stress",
        action="store_true",
        help="Skip validation-row inverse stress targets (Table_inverse_stress_protocol.csv).",
    )
    parser.add_argument(
        "--no_inverse_member_spread",
        action="store_true",
        help="Skip Table_inverse_theta_member_spread.csv.",
    )
    parser.add_argument("--no_loao_cv", action="store_true", help="Skip leave-one-angle-out cross-validation")
    parser.add_argument("--no_rar", action="store_true", help="Skip residual-based adaptive refinement for collocation")
    args = parser.parse_args()

    CFG.dry_run = bool(args.dry_run)
    CFG.n_ensemble = args.n_ensemble
    CFG.seed = args.seed
    CFG.seed_base = args.seed
    CFG.split_seed = args.seed
    BO_CFG.seed = args.seed
    CFG.show_plots = args.show_plots
    CFG.run_ablation = not args.no_ablation
    CFG.run_reviewer_proof = not args.no_reviewer_proof
    CFG.run_mobo_qnehvi = not args.no_mobo_qnehvi
    CFG._skip_mobo = args.no_mobo_qnehvi
    CFG.strict_paper_deps = bool(args.strict_paper)
    CFG.force_cpu = args.force_cpu
    CFG.run_loao_cv = not args.no_loao_cv
    CFG.run_rar = not args.no_rar
    CFG.run_inverse_ablation = bool(args.inverse_ablation)
    CFG.run_inverse_stress_validation = not args.no_inverse_stress
    CFG.run_inverse_member_spread = not args.no_inverse_member_spread
    refresh_device()
    
    run_pipeline(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()