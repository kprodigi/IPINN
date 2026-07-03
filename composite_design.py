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
# Matplotlib emits a UserWarning per text element when a fallback font in the
# rcParams family list is not installed. The fallback behaviour is correct
# (it picks the next available family) but the messages are noisy on systems
# where, e.g., Liberation Sans isn't installed alongside Arial.
warnings.filterwarnings("ignore", message=r"findfont: Font family.*not found")
import logging as _logging
_logging.getLogger("matplotlib.font_manager").setLevel(_logging.ERROR)

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

# =============================================================================
# OPTIONAL DISK CACHE (resume long training across interruptions / reduced runs)
# =============================================================================
# Opt-in via the env var IPINN_CACHE_DIR.  ``_explicit_disk_cache`` memoises a
# unit of work (e.g. one ensemble member) under an explicit, order-independent
# key, so an interrupted run keeps every finished unit and only recomputes the
# rest on re-run.  Unset IPINN_CACHE_DIR (all production/HPC runs) => transparent
# pass-through with zero behavioural change.
import functools as _functools  # noqa: E402
_CACHE_DIR = os.environ.get("IPINN_CACHE_DIR") or None


def _explicit_disk_cache(key: str, compute_fn):
    """Memoise ``compute_fn()`` to ``$IPINN_CACHE_DIR/<key>.pkl`` (resume-friendly)."""
    if not _CACHE_DIR:
        return compute_fn()
    path = os.path.join(_CACHE_DIR, f"{key}.pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception:
            pass
    result = compute_fn()
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:
        pass
    return result


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


# =============================================================================
# ATOMIC CSV WRITES (prevents truncated CSVs on SLURM preemption)
# =============================================================================
# ``DataFrame.to_csv`` is patched so every CSV write the pipeline performs goes
# through tmp-file + os.replace.  os.replace is atomic on POSIX and Windows
# NTFS, so a SLURM SIGTERM mid-write either leaves the prior version intact
# or replaces it with the complete new file — never a half-written CSV.
# Only triggers when the destination is a string/path; pass-through otherwise
# (StringIO/buffer cases used in tests are unaffected).
_ORIG_DATAFRAME_TO_CSV = pd.DataFrame.to_csv


def _atomic_to_csv(self, path_or_buf=None, *args, **kwargs):
    if isinstance(path_or_buf, (str, os.PathLike)):
        target = os.fspath(path_or_buf)
        # Unique tmp name (pid + monotonic counter): two concurrent jobs
        # writing the same CSV into a shared output_dir must not race on a
        # single deterministic "<target>.tmp".
        _atomic_to_csv._n = getattr(_atomic_to_csv, "_n", 0) + 1
        tmp = f"{target}.{os.getpid()}.{_atomic_to_csv._n}.tmp"
        result = _ORIG_DATAFRAME_TO_CSV(self, tmp, *args, **kwargs)
        os.replace(tmp, target)
        return result
    return _ORIG_DATAFRAME_TO_CSV(self, path_or_buf, *args, **kwargs)


pd.DataFrame.to_csv = _atomic_to_csv


def setup_logging(output_dir: str, tag: str = "") -> logging.Logger:
    """Configure logging to both file and console."""
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger("PINN_Crashworthiness")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fname = f"run_log_{tag}.txt" if tag else "run_log.txt"
    fh = logging.FileHandler(os.path.join(output_dir, fname), mode="w", encoding="utf-8")
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
# Font sizes are capped at the single-column reference size so they never
# exceed the reference even when figures are wider than PRINT_WIDTH_IN.
#
# Target sizes (reference at fig_width >= PRINT_WIDTH_IN):
#   axis labels:    9 pt
#   subplot titles: 9 pt (bold)
#   tick labels:    8 pt
#   legend:         8 pt
#   panel labels:   9.5 pt (bold)
#   line width:     0.8 pt
# =============================================================================

# Composite Structures (Elsevier) figure standards
# - Full-page width (2-col):  190 mm = 7.48 in   -> PRINT_WIDTH_IN below
# - 1.5-column:                140 mm = 5.51 in
# - Single column:              90 mm = 3.54 in
# - Body text in published paper: ~9 pt (regular)
# - Figure text minimum at final scale: 7 pt (Elsevier rule)
# - Figures saved at 600 DPI; lines >= 0.5 pt at final scale
PRINT_WIDTH_IN = 7.48
SINGLE_COL_IN = 3.54
DOUBLE_COL_IN = PRINT_WIDTH_IN
ONE_AND_HALF_COL_IN = 5.51

def set_publication_style():
    """Matplotlib defaults for publication-grade figures.

    Modest font sizes (10-12 pt) so multi-panel layouts fit cleanly at
    journal page width without per-figure adjustments.  Layout is
    handled by matplotlib's constrained_layout engine (enabled
    globally via figure.constrained_layout.use=True).
    """
    plt.rcParams.update({
        # Output geometry
        "figure.dpi":          150,
        "figure.facecolor":    "white",
        "savefig.dpi":         600,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.05,
        "savefig.facecolor":   "white",
        "pdf.fonttype":        42,
        "ps.fonttype":         42,
        # Layout — constrained_layout solves text-overlap problems
        # automatically; enable it as the default for every figure.
        "figure.constrained_layout.use": True,
        # Fonts — Arial with sensible publication sizes (smaller than
        # body text so panels stay legible at thumbnail scale).
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size":        11,
        "mathtext.default": "regular",
        "mathtext.fontset": "dejavusans",
        # Axes
        "axes.titlesize":   12,
        "axes.titleweight": "bold",
        "axes.labelsize":   11,
        "axes.labelweight": "bold",
        "axes.linewidth":   1.0,
        "axes.grid":        True,
        "axes.axisbelow":   True,
        # Ticks
        "xtick.labelsize":     10,
        "ytick.labelsize":     10,
        "xtick.direction":     "in",
        "ytick.direction":     "in",
        "xtick.major.size":    4,
        "ytick.major.size":    4,
        "xtick.minor.size":    2.5,
        "ytick.minor.size":    2.5,
        "xtick.major.width":   1.0,
        "ytick.major.width":   1.0,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        # Legend
        "legend.fontsize":     9,
        "legend.frameon":      True,
        "legend.framealpha":   0.92,
        "legend.edgecolor":    "0.7",
        # Lines / markers
        "lines.linewidth":  1.4,
        "lines.markersize": 5,
        # Grid
        "grid.alpha":      0.25,
        "grid.linewidth":  0.5,
        "grid.linestyle":  "--",
        "errorbar.capsize": 3,
    })


# =============================================================================
# COLOR AND STYLE DEFINITIONS
# =============================================================================
# Wong palette (Nature Methods, 2011): colourblind-safe and reasonably distinct
# in greyscale.  Pair with line-style + marker so figures remain legible in
# black-and-white print proofs.
#
#   #000000  black           (Hard-PINN, max contrast)
#   #0072B2  deep blue       (Soft-PINN, LC1; mid-dark grey in B/W)
#   #D55E00  vermillion      (DDNS, LC2; mid-light grey in B/W)
#   #009E73  bluish green    (auxiliary, e.g. weighted-sum sweep)
#   #CC79A7  reddish purple  (auxiliary, e.g. ill-posedness markers)
#   #56B4E9  sky blue        (auxiliary)
#   #E69F00  orange          (auxiliary)
#   #F0E442  yellow          (highlight; pair only with dark edge)

# Models — distinct in colour AND lightness; line styles add B/W safety.
COLORS = {
    "ddns": "#D55E00",    # vermillion
    "soft": "#0072B2",    # deep blue
    "hard": "#000000",    # black (anchor)
    "LC1":  "#0072B2",    # deep blue (matches Soft hue but used in different plots)
    "LC2":  "#D55E00",    # vermillion
    "data": "#000000",
    "experiment": "#000000",
    "gpbo": "#009E73",    # bluish green (clearly distinct from model colours)
}

LINESTYLES = {
    "ddns": "--",  "soft": "-.", "hard": "-",
    "data": "-",   "gpbo": "-",
    "LC1":  "-",   "LC2":  "--",
}

MARKERS = {
    "ddns": "o",   "soft": "s",   "hard": "D",
    "gpbo": "^",
    "LC1":  "o",   "LC2":  "s",
}

LC_MARKERS = {"LC1": MARKERS["LC1"], "LC2": MARKERS["LC2"]}

# Hatch patterns for bar plots (B/W safe, in addition to colour fill).
HATCHES = {"ddns": "//", "soft": "\\\\", "hard": "", "random": "", "unseen": "//"}
FILLSTYLES = {"ddns": "none", "soft": "full", "hard": "full"}

MODEL_LABELS = {"ddns": "DDNS", "soft": "Soft-PINN", "hard": "Hard-PINN"}


# =============================================================================
# REPRODUCIBILITY
# =============================================================================
def set_seed(seed: int) -> None:
    """Set all RNG seeds and request the strongest determinism PyTorch can give.

    Notes
    -----
    Bit-identical reproducibility on GPU is *not* fully achievable when autograd
    accumulates gradients across CUDA atomics (used by the second-order
    ``create_graph=True`` path in Hard-PINN training). Every controllable knob
    is set — including the cuBLAS workspace and
    ``use_deterministic_algorithms`` in warn-only mode — so reruns on the same
    hardware/PyTorch version produce near-identical numbers (within 1e-6
    typical drift).
    """
    # PYTHONHASHSEED only honoured if set BEFORE Python starts — set it here
    # for the env, but the authoritative place is the launch script. Document in main().
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuBLAS deterministic algorithms require this env var (CUDA >= 10.2).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only=True: log a warning if a non-deterministic kernel is hit but do
    # not abort. Hard-PINN double-backward triggers a few of these; they bias
    # numbers by <1e-6, well below the conformal-band width.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (TypeError, RuntimeError):
        # Older PyTorch: warn_only kwarg not available; fall back silently.
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def make_torch_generator(seed: int) -> torch.Generator:
    """A seeded ``torch.Generator`` for DataLoader determinism."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def r2_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Robust R² that returns NaN (not -inf) on a constant-target slice.

    sklearn's ``r2_score`` divides by ``var(y_true)``; when a held-out slice has
    near-zero variance (e.g. an LC×θ subset where energy is briefly flat) the
    metric blows up to ``-inf`` and contaminates the Tukey fence. Such slices
    are treated as "undefined" so the caller can decide how to aggregate.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0 or y_pred.size == 0 or y_true.size != y_pred.size:
        return float("nan")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        return float("nan")
    var_y = float(np.var(y_true, ddof=0))
    if var_y < 1e-12:
        return float("nan")
    return float(r2_score(y_true, y_pred))


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
    # Bootstrap resampling unit: "row" (published behaviour) or "curve"
    # (cluster bootstrap over whole (Angle, LC) curves — the exchangeable
    # unit; see bootstrap_indices()).
    bootstrap_unit: str = "row"
    # Minimum surviving ensemble members before a production run aborts
    # (dry runs only warn).  See train_ensemble().
    min_ensemble_members: int = 3
    # SHA-256 (short) of the loaded dataset rows; set by load_data() and
    # stamped into every saved bundle for provenance cross-checks.
    data_fingerprint: str = ""
    # Hard-PINN architecture override: "" (use cfg / default "mlp"),
    # "monotone" (F>=0 + E(0)=0 by construction), "separable" (low-order
    # theta-dependence), or "monotone_separable" (both).  Used by the
    # theta-generalization ablation harness; production default unchanged.
    hard_architecture: str = ""
    # Curvature-penalty window (mm): 0 = penalize everywhere (published
    # behaviour); >0 = exclude the pre-peak region below this displacement
    # from the d²E/dd² penalty so IPF sharpness is not smoothed away (see
    # curvature_regularization_hard()).
    curvature_window_after_mm: float = 0.0
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
    run_robustness_analyses: bool = True  # extended robustness analyses
    # When True, missing optional deps (skopt for GP-BO) abort at startup.
    strict_paper_deps: bool = False
    # Expensive inverse ablations (extra GP-BO runs per target); enable with --inverse_ablation.
    run_inverse_ablation: bool = False
    inverse_ablation_max_targets: int = 2
    # Validation-row inverse stress test (uses random-protocol val rows not in train).
    run_inverse_stress_validation: bool = True
    inverse_stress_max_targets: int = 5
    # Per-member forward spread at reported optimum (cheap).
    run_inverse_member_spread: bool = True
    # CI / smoke: tiny budgets, robustness extras skipped, GP-BO replaced by coarse grid inverse.
    dry_run: bool = False
    # Cap every training config's epochs at this value when > 0 (feasible
    # reduced/local "real" runs; 0 = full tuned budgets).
    max_epochs: int = 0
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


CFG = Config()


def protocol_label(protocol: str) -> str:
    """Human-readable protocol caption (``CFG.theta_star`` for unseen holdout).

    The random split is labelled for what it measures: rows are split WITHIN
    curves (stratified by Angle×LC), so validation points sit between training
    points on the same smooth curve.  That is reconstruction / within-curve
    interpolation fidelity — NOT generalization to new designs, which is what
    the unseen-angle protocol tests.  Labelling it "Random 80/20 Split" invited
    readers to mistake ~0.99 R² for design-level generalization.
    """
    if protocol == "random":
        return "Within-Curve Interpolation (random 80/20)"
    if protocol == "unseen":
        return rf"Unseen Angle $\theta^*={CFG.theta_star:g}^\circ$"
    return str(protocol)


def bootstrap_indices(train_df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap row indices honouring ``CFG.bootstrap_unit``.

    - ``"row"`` (default, paper behaviour): iid row resampling.  With densely
      sampled curves every member sees ~63% of EVERY curve, so members are
      near-identical and ensemble σ under-represents uncertainty (the root
      cause of the ~3× conformal factors).
    - ``"curve"``: cluster bootstrap at the exchangeable unit — whole
      (Angle, LC) curves are resampled with replacement, so member-to-member
      spread reflects which DESIGNS were seen, the statistically defensible
      unit for design-level uncertainty.

    Kept opt-in (default preserves published behaviour); switch via
    ``CFG.bootstrap_unit = "curve"`` for reruns and report the comparison.
    """
    n = len(train_df)
    if getattr(CFG, "bootstrap_unit", "row") != "curve":
        return rng.integers(0, n, n)
    keys = list(zip(train_df["Angle"].astype(float), train_df["LC"].astype(str)))
    uniq = sorted(set(keys))
    rows_by_curve = {k: np.flatnonzero([kk == k for kk in keys]) for k in uniq}
    picked = rng.integers(0, len(uniq), len(uniq))
    return np.concatenate([rows_by_curve[uniq[i]] for i in picked])


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


def _data_loader_kwargs(seed: Optional[int] = None) -> Dict[str, Any]:
    """DataLoader options.

    pin_memory is disabled because ``to_tensor`` already places data on
    ``DEVICE`` directly (no host→device copy needed).

    When ``seed`` is provided, a seeded ``torch.Generator`` is attached so the
    shuffle order at every epoch is deterministic given the seed. This is
    essential for reproducibility under the determinism push (`set_seed`).
    """
    kw: Dict[str, Any] = {"pin_memory": False}
    if seed is not None:
        kw["generator"] = make_torch_generator(int(seed))
    return kw


@dataclass
class BOConfig:
    """GP-BO configuration.

    Uses skopt.gp_minimize with joint (theta, LC) search space.  A single
    GP with Matern kernel over theta and a categorical LC dimension shares
    information across loading conditions, producing well-resolved
    posteriors for both LCs.

    Budget is set tight: empirically the optimum is found in ~7 / 20 EI
    evaluations on this problem, with cross-restart theta-spread of
    +/- 0.0 deg on most targets.  Total budget is B=20 (5 random initial
    + 15 EI) with an early-stop after ``stagnation_patience`` consecutive
    non-improving evaluations.  Total surrogate calls per target are
    ``n_bo_restarts * effective_calls`` where ``effective_calls <=
    n_calls_total``.
    """
    n_calls_total: int = 20
    n_init: int = 5
    xi: float = 0.01
    n_candidates: int = 500
    gp_restarts: int = 3
    seed: int = 2026
    prob_weight: float = 0.02          # ensemble classifier penalty weight (auto-tuned if 'auto')
    run_classifier_ablation: bool = True  # run with vs without penalty comparison
    lambda_sweep: bool = True           # run lambda sensitivity analysis
    # Multi-start BO: run GP-BO n_bo_restarts times with different seeds, keep best
    n_bo_restarts: int = 5
    # Early-stop guard: terminate a single restart once the best-so-far has
    # not improved by ``stagnation_min_delta`` for ``stagnation_patience``
    # consecutive evaluations beyond ``n_init``.  Keeps the worst case at
    # ``n_calls_total`` but typically halves it on this problem.
    stagnation_patience: int = 6
    stagnation_min_delta: float = 1.0e-5


BO_CFG = BOConfig()


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
    """Get model configuration with protocol-specific hyperparameters.

    Unseen-θ=60° configs are hardcoded HPO-best.  DDNS runs with BC disabled
    inside train_ddns since it has no physics losses.

    To reproduce the reported R² numbers, run:
        python composite_design.py --mode forward --data_dir ./data \\
            --output_dir ./results --n_ensemble 20 --strict_paper
    """

    if protocol == "unseen":
        # ---- DDNS cfg ----
        # Adam optimizer, no physics terms (DDNS = data-driven baseline).
        cfg_ddns = {
            "optimizer": "adam", "lr": 4.2123162503e-05,
            "weight_decay": 3.1582563297e-05, "batch_size": 64,
            "hidden_layers": [128, 64, 32], "dropout": 0.016401,
            "softplus_beta": 18.9027, "smoothl1_beta": 1.0838,
            "w_data_load": 3.568932, "w_data_energy": 3.451798,
            "w_phys": 0.0, "w_bc": 0.0, "colloc_ratio": 0.0,
            "epochs": 800, "eval_every": 25,
            "earlystop_patience_evals": 15, "earlystop_min_delta": 1e-5,
            "sched_patience": 58, "sched_factor": 0.4589,
        }

        # ---- Soft-PINN cfg ----
        # Loss = data + work-energy residual (w_phys) + paired E(0)/F(0)
        # soft BC penalty (w_bc) + three auxiliary soft regularisers
        # (monotonicity F>=0, angle smoothness dF/dθ, curvature d²E/dd²).
        # The auxiliary regularisers collectively encode the smoothness
        # priors the unseen-angle protocol relies on.
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

        # ---- Hard-PINN cfg ----
        # Architecture: HardEnergyNet (single-output energy MLP), force
        # F = dE/dd computed by autograd at both training and inference, so
        # the work-energy identity is enforced by construction.  The boundary
        # conditions E(0)=0 and F(0)=0 are encouraged through the three
        # auxiliary soft regularisers (monotonicity, angle smoothness, energy
        # curvature) which collectively shape F and E near d=0.  Training
        # schedule: warmup + cosine LR + SWA over the final ``swa_pct`` of
        # epochs.  Adam optimizer.
        cfg_hard = {
            "optimizer": "adam", "lr": 9.9507487403e-05,
            "weight_decay": 3.7459350574e-03, "batch_size": 8,
            "hidden_layers": [128, 64], "dropout": 0.005504,
            "softplus_beta": 11.6712, "smoothl1_beta": 0.1176,
            "w_load": 6.8031, "w_energy": 8.6549,
            "grad_clip": 0.9834,
            "w_monotonicity": 7.719974,
            "w_angle_smooth": 0.016094,
            "w_curvature": 0.001285,
            "smooth_delta_deg": 1.9329,
            "colloc_ratio": 3.5795,
            "extrapolate_angles": True,
            "epochs": 800, "eval_every": 20,
            "earlystop_patience_evals": 20, "earlystop_min_delta": 1e-5,
            "sched_patience": 73, "sched_factor": 0.37,
            # Stabilization params: warmup + cosine + SWA.
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
    # Reduced-run mode (CFG.max_epochs>0): cap epochs and, for the tiny-batch
    # Hard config, enlarge the batch so a feasible local run trains quickly.
    _cap = int(getattr(CFG, "max_epochs", 0) or 0)
    if _cap > 0:
        if int(cfg.get("epochs", 0)) > _cap:
            cfg["epochs"] = _cap
            cfg["eval_every"] = max(1, min(int(cfg.get("eval_every", 20)), max(1, _cap // 8)))
            if "warmup_epochs" in cfg:
                cfg["warmup_epochs"] = max(2, min(int(cfg["warmup_epochs"]), max(2, _cap // 3)))
        if int(cfg.get("batch_size", 32)) < 16:
            cfg["batch_size"] = 64
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
        # Record the source design (the experimental row this target came
        # from) so downstream tables can report the ground-truth recovery
        # error Δθ = |θ_recovered − θ_source| and LC agreement — the
        # independent check that the inverse round-trip actually works,
        # rather than only the surrogate's self-consistency at the optimum.
        src_row = df_metrics.iloc[i]
        targets.append({
            "id": tid,
            "EA": float(ea_c[i]),
            "IPF": float(ipf[i]),
            "d_eval": D_COMMON,
            "rationale": rat,
            "source_angle": float(src_row["Angle"]) if "Angle" in src_row else None,
            "source_lc": str(src_row["LC"]) if "LC" in src_row else None,
        })
    logger.info(f"  Generated paired feasible targets (EA to d={D_COMMON:.0f} mm + matching IPF):")
    for t in targets:
        logger.info(
            f"    {t['id']}: EA@{D_COMMON:.0f}mm={t['EA']:.2f}J, IPF={t['IPF']:.3f}kN "
            f"[source: θ={t['source_angle']}°, {t['source_lc']}] - {t['rationale']}")
    return targets


def generate_verification_targets(models: List, approach: str,
                                  scaler_disp, enc, params,
                                  df_metrics: pd.DataFrame,
                                  logger: logging.Logger) -> List[Dict]:
    """Targets that stress the inverse map beyond trivially-recoverable rows.

    The T1-T5 targets are observed experimental rows, so their recovery only
    exercises in-distribution lookups.  This adds two harder families:

    - **Off-grid targets (V1, V2):** the surrogate's own (EA, IPF) prediction
      at non-grid angles (52.5°, 62.5°) between the measured 5°-spaced anchors.
      The true θ is known BY CONSTRUCTION, so ``Δθ = |θ_recovered − θ_source|``
      is an exact, non-circular test of the surrogate map's invertibility at
      angles never present in any data row.
    - **Infeasible target (V3):** an (EA, IPF) combination outside the
      attainable set (EA 30% above the observed maximum at an IPF 30% below
      the observed minimum).  A correct inverse pipeline must *flag* this as
      non-attainable (large residual objective) rather than silently return a
      confident but wrong design.
    """
    targets: List[Dict] = []
    lc_cats = [str(x) for x in enc.categories_[0].tolist()]
    offgrid = [(52.5, lc_cats[0]), (62.5, lc_cats[-1])]
    for k, (ang, lc) in enumerate(offgrid, start=1):
        try:
            m = compute_ea_ipf_ensemble(models, approach, float(ang), lc,
                                        scaler_disp, enc, params, d_eval=D_COMMON)
        except Exception as exc:
            logger.warning(f"  Verification target V{k} skipped (forward eval failed): {exc}")
            continue
        targets.append({
            "id": f"V{k}",
            "EA": float(m["EA"]),
            "IPF": float(m["IPF"]),
            "d_eval": D_COMMON,
            "source_angle": float(ang),
            "source_lc": lc,
            "expect_feasible": True,
            "rationale": f"Off-grid surrogate round-trip (true θ={ang}°, {lc} by construction)",
        })
    # Infeasible probe from the empirical attainable set.
    if "EA_common" in df_metrics.columns and "IPF" in df_metrics.columns and len(df_metrics):
        ea_max = float(df_metrics["EA_common"].max())
        ipf_min = float(df_metrics["IPF"].min())
        targets.append({
            "id": "V3",
            "EA": ea_max * 1.30,
            "IPF": ipf_min * 0.70,
            "d_eval": D_COMMON,
            "source_angle": None,
            "source_lc": None,
            "expect_feasible": False,
            "rationale": "Deliberately infeasible (EA 30% above max at IPF 30% below min) — must be flagged non-attainable",
        })
    logger.info(f"  Generated {len(targets)} verification targets (off-grid round-trip + infeasibility probe)")
    return targets


def run_verification_inverse_and_save(models: List, approach: str,
                                      scaler_disp, enc, params,
                                      df_metrics: pd.DataFrame,
                                      bo_cfg, cal_ens, feat_scaler,
                                      output_dir: str, logger: logging.Logger,
                                      feas_rel_tol: float = 0.05) -> Optional[pd.DataFrame]:
    """Run the verification targets and write Table_inverse_verification.csv.

    ``Attainable`` is decided from the residual objective: with the 1/target²
    weighting, sqrt(J_fit) is the combined relative (EA, IPF) error, so a
    solution counts as attainable when that error is below ``feas_rel_tol``
    (default 5%).  The table's headline columns are Δθ for the off-grid
    round-trips and the Attainable verdict for the infeasibility probe.
    """
    v_targets = generate_verification_targets(models, approach, scaler_disp, enc,
                                              params, df_metrics, logger)
    if not v_targets:
        return None
    rows = []
    for t in v_targets:
        logger.info(f"  Verification inverse {t['id']}: EA@{int(D_COMMON)}mm={t['EA']:.2f}J, "
                    f"IPF={t['IPF']:.3f}kN — {t['rationale']}")
        res = run_inverse_design(models, approach, t["EA"], t["IPF"],
                                 scaler_disp, enc, params, bo_cfg, logger,
                                 cal_ens=cal_ens, feat_scaler=feat_scaler)
        res["target_info"] = t
        best = res.get("gpbo_best") or {}
        rec_theta = best.get("x_best")
        rec_lc = best.get("lc", best.get("best_lc", ""))
        # Combined relative error from the fit term (classifier penalty excluded
        # is not separable here, so this slightly overestimates — conservative).
        ea_err = float(best.get("ea_error_pct", float("nan")))
        ipf_err = float(best.get("ipf_error_pct", float("nan")))
        combined_rel = float(np.hypot(ea_err / 100.0, ipf_err / 100.0)) if np.isfinite(ea_err) else float("nan")
        attainable = bool(np.isfinite(combined_rel) and combined_rel <= feas_rel_tol)
        src = t.get("source_angle")
        rows.append({
            "Verification_ID": t["id"],
            "Family": "off-grid round-trip" if t.get("expect_feasible") else "infeasibility probe",
            "target_EA_J": f"{t['EA']:.3f}",
            "target_IPF_kN": f"{t['IPF']:.3f}",
            "true_theta_deg": f"{src:.1f}" if src is not None else "n/a (infeasible)",
            "true_LC": t.get("source_lc") or "n/a",
            "recovered_theta_deg": f"{float(rec_theta):.2f}" if rec_theta is not None else "",
            "recovered_LC": rec_lc,
            "delta_theta_deg": (f"{abs(float(rec_theta) - float(src)):.2f}"
                                if (rec_theta is not None and src is not None) else ""),
            "LC_match": (("Yes" if str(rec_lc) == str(t.get("source_lc")) else "No")
                         if (rec_lc and t.get("source_lc")) else ""),
            "EA_error_pct": f"{ea_err:.2f}" if np.isfinite(ea_err) else "",
            "IPF_error_pct": f"{ipf_err:.2f}" if np.isfinite(ipf_err) else "",
            "combined_rel_error": f"{combined_rel:.4f}" if np.isfinite(combined_rel) else "",
            "Attainable": "Yes" if attainable else "No",
            "Expected_attainable": "Yes" if t.get("expect_feasible") else "No",
            "Verdict_correct": ("Yes" if attainable == bool(t.get("expect_feasible")) else "No"),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Table_inverse_verification.csv")
    df.to_csv(path, index=False)
    logger.info(f"  Saved: {os.path.basename(path)}")
    return df


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
    2. The inverse-design surrogate should use the most accurate fit available
    3. Inverse-design validation comes from comparing predicted vs actual (EA, IPF)

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
    
    # Use unseen-protocol Hard cfg for full-data training: same architecture,
    # dropout, loss weights, optimizer, and stabilization (warmup + cosine
    # + SWA), so the inverse-design surrogate inherits the same physics
    # treatment as the validated forward model.
    cfg = get_model_config("hard", protocol="unseen")
    cfg["epochs"] = 1500  # Full data (incl. 60°) converges faster than unseen protocol
    cfg["batch_size"] = 128  # Larger batch for the full-data sample
    cfg["warmup_epochs"] = 150  # Scale warmup proportionally with total epochs
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
    w_curv = cfg.get("w_curvature", 0.0)
    smooth_delta = cfg.get("smooth_delta_deg", 1.5)
    colloc_ratio = cfg.get("colloc_ratio", 0.0)
    extrapolate = cfg.get("extrapolate_angles", True)

    colloc_sampler = None
    if w_mono > 0 or w_smooth > 0 or w_curv > 0:
        colloc_sampler = create_collocation_sampler(df_all, scaler_disp, enc, extrapolate_angles=extrapolate)
        logger.info(f"    Collocation regularizers active: w_mono={w_mono}, w_smooth={w_smooth}, "
                    f"w_curv={w_curv}, colloc_ratio={colloc_ratio}, extrapolate={extrapolate}")
    
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
            indices = bootstrap_indices(df_all, rng)
            X_train = X_tensor[indices]
            Y_train = Y_tensor[indices]
        else:
            X_train = X_tensor
            Y_train = Y_tensor
        
        # Create and train model (architecture via CFG.hard_architecture /
        # cfg["architecture"]; default "mlp" = published production model).
        model = make_hard_energy_net(X_full.shape[1], cfg, params).to(DEVICE)

        if cfg["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        
        # Stabilized training for inverse design Hard-PINN
        warmup_ep = cfg.get("warmup_epochs", 80)
        total_ep = cfg["epochs"]
        scheduler = WarmupCosineScheduler(optimizer, warmup_ep, total_ep, eta_min=cfg.get("eta_min", 1e-6))
        swa_start = int(total_ep * (1.0 - cfg.get("swa_pct", 0.20)))
        swa_model = AveragedModel(model, device=DEVICE)
        swa_active = False
        
        # Training loop
        dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=False, **_data_loader_kwargs(seed=seed)
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

                    if w_curv > 0:
                        loss_curv = curvature_regularization_hard(Xc, model, params)
                        loss = loss + w_curv * loss_curv

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.get("grad_clip", 1.0))
                optimizer.step()
                epoch_loss += loss.item()
            
            # Stabilized schedule + SWA
            scheduler.step()
            if epoch >= swa_start:
                swa_active = True
                swa_model.update_parameters(model)
        
        # Use SWA model for final evaluation if active
        eval_model = swa_model.module if swa_active else model
        
        # Compute training-set R² for convergence check (batched: full tensor grad is VRAM-heavy)
        eval_model.eval()
        Fv, _ = hard_pinn_predict_load_energy(eval_model, X_tensor, params)
        tr_r2 = float(r2_safe(y_full_np[:, 0], Fv))
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
    """Validate input data structure and physical sanity rigorously.

    Checks:
      - required columns present, no NaNs
      - disp_mm is non-negative
      - within each (Angle, LC) group, disp_mm is monotonically non-decreasing
        when sorted by row order (the curves must be strictly causal: a quasi-
        static crushing experiment cannot have the crosshead retract)
      - LC is one of the expected categories (warn-only)
    """
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

    # Physical sanity: displacement must be non-negative.
    if (df["disp_mm"] < 0).any():
        issues.append(f"disp_mm has {int((df['disp_mm'] < 0).sum())} negative values")

    # Within each (Angle, LC) group, disp_mm must be monotonically non-decreasing.
    # We assume rows are stored in acquisition order; if not, downstream code
    # sorts before resampling, so this check applies to raw input only.
    bad_groups = []
    for (lc, ang), grp in df.groupby(["LC", "Angle"]):
        d = grp["disp_mm"].values
        if len(d) >= 2 and np.any(np.diff(d) < -1e-9):
            n_bad = int(np.sum(np.diff(d) < -1e-9))
            bad_groups.append(f"({lc}, θ={ang}): {n_bad} non-monotone disp steps")
    if bad_groups:
        issues.append("Non-monotone displacement: " + "; ".join(bad_groups[:3])
                      + (f" ... +{len(bad_groups)-3} more groups" if len(bad_groups) > 3 else ""))

    # LC categories — expect LC1/LC2; warn but do not fail on others.
    lcs_seen = set(df["LC"].astype(str).str.upper().unique())
    expected = {"LC1", "LC2"}
    unexpected = lcs_seen - expected
    if unexpected:
        logger.warning(f"  validate_input_data: unexpected LC labels seen: {sorted(unexpected)} "
                       f"(expected {sorted(expected)}); proceeding.")

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
                n_before = len(df)
                df = df[req].dropna()
                if len(df) < n_before:
                    logger.warning(f"  {os.path.basename(f)}: dropped {n_before - len(df)} "
                                   f"rows with missing values")
                df_list.append(df)
                logger.info(f"  Loaded {len(df)} rows from {os.path.basename(f)}")
            else:
                missing = [r for r in req if r not in df.columns]
                raise ValueError(f"missing required columns after mapping: {missing}")
        except Exception as e:
            # A per-file failure must be FATAL: silently continuing yields a
            # complete-looking single-LC run whose every downstream number is
            # wrong-but-plausible.
            raise RuntimeError(f"Failed to load data file {f}: {e}") from e

    if not df_list:
        raise ValueError("No valid data files loaded")

    df_all = pd.concat(df_list, ignore_index=True)
    df_all["disp_mm"] = df_all["disp_mm"].astype(float)
    df_all["load_kN"] = df_all["load_kN"].astype(float)
    df_all["energy_J"] = df_all["energy_J"].astype(float)
    df_all["Angle"] = df_all["Angle"].astype(float)
    df_all["LC"] = df_all["LC"].astype(str)

    # ---- Data QA (previously dead code — now enforced) ---------------------
    # 1. Non-monotone displacement rows are corrupted (e.g. an out-of-order
    #    logger row): they inject a false (d, F, E) sample and a force
    #    discontinuity into training.  Auto-drop with a logged count.
    n_dropped_nonmono = 0
    cleaned = []
    for (lc, ang), g in df_all.groupby(["LC", "Angle"], sort=False):
        g = g.reset_index(drop=True)
        d = g["disp_mm"].values
        keep = np.ones(len(g), dtype=bool)
        run_max = -np.inf
        for i, dv in enumerate(d):
            if dv < run_max - 1e-9:
                keep[i] = False
            else:
                run_max = max(run_max, dv)
        if (~keep).any():
            n_dropped_nonmono += int((~keep).sum())
            logger.warning(f"  Data QA: dropped {int((~keep).sum())} non-monotone-displacement "
                           f"row(s) in ({lc}, θ={ang:g}°) — corrupted logger rows")
        cleaned.append(g.loc[keep])
    if n_dropped_nonmono:
        df_all = pd.concat(cleaned, ignore_index=True)

    # 2. Structural validation (both LCs present, per-curve sanity).
    ok, issues = validate_input_data(df_all, logger)
    if not ok:
        for iss in issues:
            logger.warning(f"  Data QA issue: {iss}")
    lcs_present = set(df_all["LC"].unique())
    if not {"LC1", "LC2"} <= lcs_present:
        raise ValueError(f"Expected both LC1 and LC2 in the dataset; found {sorted(lcs_present)}. "
                         f"A single-LC run would silently invalidate every cross-LC analysis.")

    # Dataset fingerprint for bundle provenance (see _bundle_meta()).
    try:
        import hashlib as _hl
        CFG.data_fingerprint = _hl.sha256(
            pd.util.hash_pandas_object(df_all, index=False).values.tobytes()).hexdigest()[:12]
        logger.info(f"  Data fingerprint: {CFG.data_fingerprint}")
    except Exception:
        pass

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
    """MLP for DDNS and Soft-PINN (outputs both F and E).

    Optionally enforces ``F(d=0) ≡ 0`` and ``E(d=0) ≡ 0`` in raw
    (un-normalized) space via an architectural correction; activate by calling
    :meth:`configure_zero_bc` after construction.  The correction is
    autograd-safe and does not modify the d-derivative of E (so a downstream
    ``F = dE/dd`` computation is unaffected).
    """

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
        # Architectural F(d=0)=0, E(d=0)=0 correction (inactive by default).
        # Activated via configure_zero_bc(params).
        self._zero_bc_active = False
        self.register_buffer("_d_scaled_at_zero", torch.zeros(1))
        self.register_buffer("_c0_F", torch.zeros(1))
        self.register_buffer("_c0_E", torch.zeros(1))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def configure_zero_bc(self, params: "ScalingParams", enabled: bool = True) -> None:
        """Activate (or disable) the architectural F(d=0)=E(d=0)=0 correction.

        After ``configure_zero_bc(params)`` returns, ``forward(x)`` produces
        ``[F_n, E_n]`` whose un-normalized values are exactly zero whenever
        the displacement column of ``x`` equals ``-mu_d/sig_d`` (i.e. raw
        d = 0), for any (θ, LC).  Implementation: subtract the network's
        prediction at d=0 and add the constant offset that maps normalized
        zero to raw zero.
        """
        self._zero_bc_active = bool(enabled)
        if not enabled:
            return
        self._d_scaled_at_zero.fill_(-params.mu_d / max(1e-12, params.sig_d))
        self._c0_F.fill_(-params.mu_F / max(1e-12, params.sig_F))
        self._c0_E.fill_(-params.mu_E / max(1e-12, params.sig_E))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if not self._zero_bc_active:
            return out
        # Build x|d=0 = same (θ, LC) but displacement column replaced by the
        # scaled value of raw d=0.  No in-place ops — autograd-safe so the
        # d-derivative of out remains intact.
        d0 = self._d_scaled_at_zero.view(1, 1).expand(x.size(0), 1)
        x0 = torch.cat([d0, x[:, U_COL + 1:]], dim=1)
        out0 = self.net(x0)
        F_corr = out[:, 0:1] - out0[:, 0:1] + self._c0_F
        E_corr = out[:, 1:2] - out0[:, 1:2] + self._c0_E
        return torch.cat([F_corr, E_corr], dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HardEnergyNet(nn.Module):
    """MLP for Hard-PINN (outputs only E; F derived by differentiation).

    Optionally enforces BOTH ``E(d=0) ≡ 0`` AND ``F(d=0) ≡ dE/dd|_{d=0} ≡ 0``
    in raw (un-normalized) space via an architectural correction; activate by
    calling :meth:`configure_zero_bc` after construction.

    The correction is slope-subtraction in scaled space::

        E_corrected(x) = E_net(x) − E_net(x|d=0)
                         − (d_s − d_s0) · (∂E_net/∂d_s)|_{x=x|d=0}
                         + c_{0,E}

    where ``d_s`` is the scaled displacement column of ``x``, ``d_s0`` is its
    scaled-zero value ``-mu_d/sig_d``, and ``c_{0,E} = -mu_E/sig_E``.  At
    ``x|d=0`` the second and third terms cancel exactly: the value reduces to
    ``c_{0,E}`` (raw E = 0) and ``∂E_corrected/∂d_s|_{x=x|d=0} = 0`` (raw F =
    0).  Both BCs are enforced for every (θ, LC).

    Cost vs a value-only correction (subtract ``E_net(x|d=0)`` only):
    one additional inner ``autograd.grad`` call per forward pass to obtain
    ``∂E_net/∂d_s|_{x|d=0}``, plus a second-order graph during training (so
    the outer physics-loss backward can flow back to the net's weights).
    Roughly 2–3× the value-only forward/backward cost.
    """

    def __init__(self, in_d: int, hidden_layers: List[int], dropout: float,
                 softplus_beta: float):
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
        self._init_weights()
        # Architectural E(d=0)=0 AND F(d=0)=0 correction (inactive by default).
        self._zero_bc_active = False
        self.register_buffer("_d_scaled_at_zero", torch.zeros(1))
        self.register_buffer("_c0_E", torch.zeros(1))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def configure_zero_bc(self, params: "ScalingParams", enabled: bool = True) -> None:
        """Activate (or disable) the architectural E(d=0)=0 AND F(d=0)=0 correction.

        After this call, ``forward(x)`` produces a normalized energy whose
        un-normalized value is exactly zero whenever the displacement column
        of ``x`` equals ``-mu_d/sig_d`` (i.e. raw d = 0), for any (θ, LC).
        Additionally, the d-derivative ``∂E_corrected/∂d_s|_{x|d=0} = 0``,
        which by the chain rule gives raw ``F(d=0) = dE/dd|_{d=0} = 0``.
        Both BCs hold at every batch element and for every (θ, LC).
        """
        self._zero_bc_active = bool(enabled)
        if not enabled:
            return
        self._d_scaled_at_zero.fill_(-params.mu_d / max(1e-12, params.sig_d))
        self._c0_E.fill_(-params.mu_E / max(1e-12, params.sig_E))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if not self._zero_bc_active:
            return out
        # Build x|d=0 = same (θ, LC) but displacement column replaced by the
        # scaled value of raw d=0.  No in-place ops anywhere — autograd-safe.
        d0 = self._d_scaled_at_zero.view(1, 1).expand(x.size(0), 1)
        x0 = torch.cat([d0, x[:, U_COL + 1:]], dim=1)
        # (1) Network value at the boundary.
        out0 = self.net(x0)
        # (2) Network d-slope at the boundary.  Use a separate detached input
        # with ``requires_grad=True`` so the local sensitivity can be pulled
        # out without entangling the outer-grad graph.  ``create_graph=self.training``
        # so backward through the loss can reach the net's weights via
        # ``dE_dd_at_zero`` during training; at eval/inference the
        # double-backward cost is saved.  ``retain_graph`` left at its default
        # (= the value of ``create_graph``); explicitly passing ``False`` here
        # would free intermediates that the outer ``loss.backward()`` needs
        # when the loss depends on both ``dE/dd`` and ``E_n`` (physics + data terms).
        x0_grad = x0.detach().clone().requires_grad_(True)
        out0_grad = self.net(x0_grad)
        dE_dd_at_zero = torch.autograd.grad(
            out0_grad.sum(),
            x0_grad,
            create_graph=self.training,
        )[0][:, U_COL:U_COL + 1]
        # (3) Slope-subtraction correction in scaled space.  ``d_delta`` is a
        # function of ``x[:, U_COL]`` so the outer ``F = ∂E/∂d_s`` autograd
        # still works; at ``x = x|d=0`` it is zero, killing the slope term.
        d_delta = x[:, U_COL:U_COL + 1] - d0
        return out - out0 - d_delta * dE_dd_at_zero + self._c0_E

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# HARD-PINN ARCHITECTURE VARIANTS (Tier 1: monotone-in-d; Tier 2: separable-θ)
# =============================================================================
# Motivation (design-generalization study): F = ∂E/∂d constrains the
# DISPLACEMENT direction only — it is satisfied by any smooth E(d, θ) and adds
# no structure across θ, which is where generalization fails with only six
# measured angles.  These variants attack that directly:
#
#   Tier 1 (``MonotoneHardEnergyNet``): E is non-decreasing in d BY
#   CONSTRUCTION (positive-weight d-path + non-decreasing activations), so
#   F = ∂E/∂d ≥ 0 exactly at every input, including angles never seen in
#   training; combined with the intrinsic E(0)=0 shift this also gives
#   E ≥ 0 and EA ≥ 0 everywhere.  Physics plausibility becomes a theorem,
#   not a soft penalty.
#
#   Tier 2 (``SeparableHardEnergyNet``): E(d, θ, LC) = Σ_k φ_k(z)·B_k(d)
#   with the θ-dependence φ_k restricted to a SINGLE linear map over
#   z = [sinθ, cosθ, LC one-hot] (first-order Fourier in θ + LC intercepts,
#   ~4 dof per basis function).  A free MLP over θ is unidentifiable from
#   5 training angles; this matches model capacity in θ to the data.
#   ``monotone=True`` additionally constrains φ_k ≥ 0 and B_k monotone,
#   giving the combined Tier 1+2 model.
#
# All variants are drop-in replacements for ``HardEnergyNet`` (same
# constructor signature prefix, ``forward(x)→[N,1]`` normalized energy, force
# recovered by autograd downstream).  Selected via ``cfg["architecture"]`` or
# the ``CFG.hard_architecture`` override; default remains the published MLP.

class PositiveLinear(nn.Module):
    """Linear layer with weights constrained positive via softplus reparameterization.

    Used on every d-influenced path of the monotone variants: positive weights
    composed with non-decreasing activations give a non-decreasing map, which
    autograd differentiates to F ≥ 0 exactly.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        w = torch.empty(out_features, in_features)
        nn.init.kaiming_normal_(w, nonlinearity="relu")
        # Inverse-softplus of |kaiming| so effective weights start at a
        # sensible positive scale: softplus(raw) == |w| at init.
        w_abs = w.abs().clamp_min(1e-4)
        self.weight_raw = nn.Parameter(torch.log(torch.expm1(w_abs)))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @property
    def effective_weight(self) -> torch.Tensor:
        return nn.functional.softplus(self.weight_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.effective_weight, self.bias)


class MonotoneHardEnergyNet(nn.Module):
    """Hard-PINN energy net with F = ∂E/∂d ≥ 0 and E(0) = 0 by construction.

    Partial-monotone architecture: a *monotone* channel group (m) carries the
    displacement through positive-weight layers; a *free* channel group (f)
    processes only z = (θ features, LC) and modulates the monotone path
    additively.  Since f never sees d, ∂E/∂d flows exclusively through
    positive-weight compositions of non-decreasing softplus activations —
    hence F ≥ 0 everywhere, for every (θ, LC), trained or not.

    The E(0)=0 boundary condition is intrinsic: the forward pass returns
    ``N(x) − N(x|d=0) + c0_E`` (value-subtraction in scaled space), so raw
    E(d=0) = 0 exactly and — with monotonicity — raw E(d) ≥ 0 for all d ≥ 0.
    ``configure_zero_bc(params)`` must be called once after construction to
    stamp the scaler-dependent buffers; its ``enabled`` argument is ignored
    (the BC is part of the architecture, not an option).
    """

    def __init__(self, in_d: int, hidden_layers: List[int], dropout: float,
                 softplus_beta: float):
        super().__init__()
        if in_d < 2:
            raise ValueError("MonotoneHardEnergyNet expects displacement + feature columns")
        self.z_dim = in_d - 1
        act = lambda: nn.Softplus(beta=softplus_beta)
        m_widths = [max(1, (h + 1) // 2) for h in hidden_layers]
        f_widths = [max(1, h - mw) for h, mw in zip(hidden_layers, m_widths)]
        self.m_d_in = PositiveLinear(1, m_widths[0])
        self.m_z_in = nn.Linear(self.z_dim, m_widths[0])
        self.f_in = nn.Linear(self.z_dim, f_widths[0])
        self.m_layers = nn.ModuleList()
        self.mf_layers = nn.ModuleList()
        self.f_layers = nn.ModuleList()
        for i in range(1, len(hidden_layers)):
            self.m_layers.append(PositiveLinear(m_widths[i - 1], m_widths[i]))
            self.mf_layers.append(nn.Linear(f_widths[i - 1], m_widths[i]))
            self.f_layers.append(nn.Linear(f_widths[i - 1], f_widths[i]))
        self.m_out = PositiveLinear(m_widths[-1], 1)
        self.f_out = nn.Linear(f_widths[-1], 1)
        self.act = act()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.register_buffer("_d_scaled_at_zero", torch.zeros(1))
        self.register_buffer("_c0_E", torch.zeros(1))

    def configure_zero_bc(self, params: "ScalingParams", enabled: bool = True) -> None:
        """Stamp scaler-dependent buffers.  ``enabled`` is ignored: E(0)=0 is intrinsic."""
        del enabled
        self._d_scaled_at_zero.fill_(-params.mu_d / max(1e-12, params.sig_d))
        self._c0_E.fill_(-params.mu_E / max(1e-12, params.sig_E))

    def _raw(self, d: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        m = self.drop(self.act(self.m_d_in(d) + self.m_z_in(z)))
        f = self.drop(self.act(self.f_in(z)))
        for m_lin, mf_lin, f_lin in zip(self.m_layers, self.mf_layers, self.f_layers):
            m_next = self.drop(self.act(m_lin(m) + mf_lin(f)))
            f = self.drop(self.act(f_lin(f)))
            m = m_next
        return self.m_out(m) + self.f_out(f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x[:, U_COL:U_COL + 1]
        z = x[:, U_COL + 1:]
        d0 = self._d_scaled_at_zero.view(1, 1).expand(x.size(0), 1)
        return self._raw(d, z) - self._raw(d0, z) + self._c0_E

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SeparableHardEnergyNet(nn.Module):
    """Hard-PINN energy net with low-order θ-dependence: E = Σ_k φ_k(z)·B_k(d).

    ``B_k(d)``: K learned displacement basis functions (small MLP over d).
    ``φ_k(z)``: a SINGLE linear map over z = [sinθ, cosθ, LC one-hot] —
    first-order Fourier in θ plus per-LC intercepts, ~4 dof per basis
    function.  This is the Tier-2 inductive bias: with only five training
    angles, restricting θ-capacity is the difference between an identifiable
    design trend and an unconstrained interpolant.

    ``monotone=True`` (Tier 1+2): φ_k = softplus(linear(z)) ≥ 0 and the basis
    MLP uses positive weights, so ∂E/∂d = Σ φ_k·B'_k ≥ 0 exactly.

    E(0)=0 is intrinsic via basis-value subtraction:
    ``E = Σ φ_k(z)·(B_k(d) − B_k(d0)) + c0_E``.
    """

    def __init__(self, in_d: int, hidden_layers: List[int], dropout: float,
                 softplus_beta: float, n_basis: int = 4, monotone: bool = True):
        super().__init__()
        if in_d < 2:
            raise ValueError("SeparableHardEnergyNet expects displacement + feature columns")
        self.z_dim = in_d - 1
        self.n_basis = int(n_basis)
        self.monotone = bool(monotone)
        Lin = PositiveLinear if self.monotone else nn.Linear
        layers: List[nn.Module] = []
        d_in = 1
        for h in hidden_layers:
            layers.append(Lin(d_in, h))
            layers.append(nn.Softplus(beta=softplus_beta))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = h
        layers.append(Lin(d_in, self.n_basis))
        self.basis = nn.Sequential(*layers)
        self.phi = nn.Linear(self.z_dim, self.n_basis)
        self.register_buffer("_d_scaled_at_zero", torch.zeros(1))
        self.register_buffer("_c0_E", torch.zeros(1))

    def configure_zero_bc(self, params: "ScalingParams", enabled: bool = True) -> None:
        """Stamp scaler-dependent buffers.  ``enabled`` is ignored: E(0)=0 is intrinsic."""
        del enabled
        self._d_scaled_at_zero.fill_(-params.mu_d / max(1e-12, params.sig_d))
        self._c0_E.fill_(-params.mu_E / max(1e-12, params.sig_E))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x[:, U_COL:U_COL + 1]
        z = x[:, U_COL + 1:]
        d0 = self._d_scaled_at_zero.view(1, 1).expand(x.size(0), 1)
        phi = self.phi(z)
        if self.monotone:
            phi = nn.functional.softplus(phi)
        basis_delta = self.basis(d) - self.basis(d0)
        return (phi * basis_delta).sum(dim=1, keepdim=True) + self._c0_E

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


HARD_ARCHITECTURES = ("mlp", "monotone", "separable", "monotone_separable")


def make_hard_energy_net(in_d: int, cfg: Dict, params: Optional["ScalingParams"] = None) -> nn.Module:
    """Construct the Hard-PINN energy network for the requested architecture.

    Selection precedence: ``CFG.hard_architecture`` (global override, used by
    the θ-generalization ablation harness) > ``cfg["architecture"]`` >
    ``"mlp"`` (the published production architecture — behaviour unchanged).
    When ``params`` is given, boundary-condition buffers are stamped: for the
    baseline MLP this keeps the production ``configure_zero_bc(enabled=False)``
    convention; for the variants E(0)=0 is intrinsic.
    """
    arch = str(getattr(CFG, "hard_architecture", "") or cfg.get("architecture", "") or "mlp").lower()
    if arch not in HARD_ARCHITECTURES:
        raise ValueError(f"Unknown hard architecture {arch!r}; expected one of {HARD_ARCHITECTURES}")
    if arch == "monotone":
        model: nn.Module = MonotoneHardEnergyNet(
            in_d, cfg["hidden_layers"], cfg["dropout"], cfg["softplus_beta"])
    elif arch in ("separable", "monotone_separable"):
        model = SeparableHardEnergyNet(
            in_d, cfg["hidden_layers"], cfg["dropout"], cfg["softplus_beta"],
            n_basis=int(cfg.get("n_basis", 4)), monotone=(arch == "monotone_separable"))
    else:
        model = HardEnergyNet(in_d, cfg["hidden_layers"], cfg["dropout"], cfg["softplus_beta"])
    if params is not None:
        model.configure_zero_bc(params, enabled=False)
    return model


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
    """Monotonicity constraint for Soft-PINN: energy must be non-decreasing in disp.

    The penalty is ``mean(relu(-dE/du_scaled)^2)`` evaluated in the network's
    **normalized** displacement coordinate. Sign-only constraint: the sign of
    ``dE/du_scaled`` and of physical ``dE/dd`` are identical (because the
    StandardScaler scale on disp is strictly positive), so this enforces the
    same thermodynamic statement as :func:`monotonicity_loss_hard`.

    Why not work in physical units here?
    ------------------------------------
    Normalising the gradient to physical kN (matching Hard-PINN's
    monotonicity loss) would scale the penalty magnitude by
    ``(sig_E/sig_d)^2 / sig_E^2 = 1/sig_d^2``, shrinking the HPO-tuned
    ``w_monotonicity`` to a negligible effective weight.  The HPO was
    performed against the formulation below, so it is preserved as written.

    The Soft-PINN and Hard-PINN ``w_monotonicity`` values therefore live on
    different magnitude scales by design — each was independently tuned to
    its own loss formulation.  They are NOT directly comparable.
    """
    Xin_g = Xin.requires_grad_(True) if not Xin.requires_grad else Xin
    pred = model(Xin_g)
    # Recover physical E so the gradient sign is unambiguously thermodynamic;
    # magnitude is then ``sig_E * (dE_n/du_scaled)`` ∈ [J] (NOT kN/mm).
    E_pred = pred[:, 1:2] * params.sig_E + params.mu_E
    dE_dX = torch.autograd.grad(E_pred.sum(), Xin_g, create_graph=True)[0]
    dE_du_scaled = dE_dX[:, U_COL:U_COL + 1]
    return torch.mean(F.relu(-dE_du_scaled) ** 2)


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


def curvature_regularization_hard(Xin: torch.Tensor, model: nn.Module,
                                   params: ScalingParams) -> torch.Tensor:
    """Second-order curvature regularization for Hard-PINN: penalize |d²E/dd²|.

    Smooths the force curve F = dE/dd by penalizing its derivative dF/dd = d²E/dd².
    This helps Hard-PINN avoid oscillatory force predictions in data-sparse regions
    (like the unseen 60° angle), encouraging physically plausible smooth curves.

    Interaction with IPF (documented, opt-in mitigation): the penalty acts on
    ALL collocation points, including the pre-peak region whose sharp rise
    defines the initial peak force — one of the two inverse-design objectives.
    With the tuned weight (~1.3e-3) the bias is small, but it is untested by
    default.  Set ``CFG.curvature_window_after_mm > 0`` to exclude collocation
    points with displacement below that value (e.g. 10 mm) from the penalty,
    and compare the resulting IPF predictions as an ablation.
    """
    Xin_g = Xin.requires_grad_(True) if not Xin.requires_grad else Xin
    E_n = model(Xin_g)
    dE = torch.autograd.grad(E_n.sum(), Xin_g, create_graph=True)[0]
    F_col = dE[:, U_COL:U_COL + 1]
    d2E = torch.autograd.grad(F_col.sum(), Xin_g, create_graph=True)[0]
    d2E_dd = d2E[:, U_COL:U_COL + 1]
    window_mm = float(getattr(CFG, "curvature_window_after_mm", 0.0))
    if window_mm > 0.0:
        # The displacement column of Xin is standardized: d_scaled = (d-μ)/σ.
        d_scaled_cut = (window_mm - float(params.mu_d)) / max(float(params.sig_d), 1e-12)
        mask = (Xin_g[:, U_COL:U_COL + 1].detach() > d_scaled_cut).float()
        denom = mask.sum().clamp(min=1.0)
        return (mask * d2E_dd ** 2).sum() / denom
    return torch.mean(d2E_dd ** 2)


def _val_checkpoint_score(r2_load: float, r2_energy: float, approach: str = "soft") -> float:
    """Validation score for checkpointing, LR schedule, and early stopping.

    By convention, **load R² is the sole checkpointing metric** for
    all three approaches (DDNS, Soft-PINN, Hard-PINN).  ``r2_energy`` and
    ``approach`` are kept in the signature for call-site compatibility but
    are unused.  NaN-safe: returns -inf if ``r2_load`` is NaN so the
    checkpointer skips that epoch instead of marking it as best.
    """
    del r2_energy, approach  # unused; kept for call-site compatibility
    r_l = float(r2_load)
    return r_l if np.isfinite(r_l) else float("-inf")


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
    # DDNS is the data-driven baseline — no architectural physics priors.
    # Calling configure_zero_bc(..., enabled=False) is explicit (the network
    # default is also False, but this documents the choice).
    model.configure_zero_bc(params, enabled=False)
    opt = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]) if cfg.get("optimizer", "adamw").lower() == "adam" else optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=cfg["sched_patience"], factor=cfg["sched_factor"], mode='max')
    mse = nn.MSELoss()
    sl1 = nn.SmoothL1Loss(beta=cfg["smoothl1_beta"])

    rng = np.random.default_rng(seed)
    idx = bootstrap_indices(train_df, rng) if CFG.bootstrap else np.arange(Xtr.shape[0])
    loader = DataLoader(
        TensorDataset(Xtr[idx], ytr[idx]), batch_size=cfg["batch_size"], shuffle=True, **_data_loader_kwargs(seed=seed)
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
            r2_l = r2_safe(y_val[:, 0], Fv)
            r2_e = r2_safe(y_val[:, 1], Ev)
            val_score = _val_checkpoint_score(r2_l, r2_e, approach="ddns")
            logger.info(
                f"    [ddns] ep={ep:4d}  train_loss={loss_sum/nb:.4f}  "
                f"val_R2_load={r2_l:.4f}  val_R2_energy={r2_e:.4f}"
            )
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
    # Soft-PINN by name should use SOFT (penalty) constraints throughout.
    # The architectural F(0)=E(0)=0 correction is reserved for Hard-PINN; for
    # Soft-PINN the BC is enforced via the ``w_bc * (E(d=0))²`` penalty term
    # below (the standard Soft-PINN loss formulation).  This keeps
    # the methodological ladder clean:
    #   DDNS       — no physics
    #   Soft-PINN  — soft physics + soft BC (penalties in loss)
    #   Hard-PINN  — hard physics (F = dE/dd by autograd) + hard BC (architectural)
    model.configure_zero_bc(params, enabled=False)
    opt = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]) if cfg.get("optimizer", "adamw").lower() == "adam" else optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=cfg["sched_patience"], factor=cfg["sched_factor"], mode='max')
    mse = nn.MSELoss()
    sl1 = nn.SmoothL1Loss(beta=cfg["smoothl1_beta"])

    # Use extrapolating collocation for unseen protocol
    extrapolate = cfg.get("extrapolate_angles", False)
    colloc_sampler = create_collocation_sampler(train_df, scaler_disp, enc, extrapolate_angles=extrapolate)

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
    idx = bootstrap_indices(train_df, rng_b) if CFG.bootstrap else np.arange(Xtr.shape[0])
    loader = DataLoader(
        TensorDataset(Xtr[idx], ytr[idx]), batch_size=cfg["batch_size"], shuffle=True, **_data_loader_kwargs(seed=seed)
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
            Xc = colloc_sampler(n_colloc, rng).requires_grad_(True)
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
                # Soft BC: paired penalty on E(0)=0 AND F(0)=0.  Each
                # penalty is computed on raw-units predictions so the two
                # terms are dimensionally commensurate after the trained
                # scalers are inverted.
                pred_bc = model(X_bc_t)
                E_bc_phys = pred_bc[:, 1:2] * params.sig_E + params.mu_E
                F_bc_phys = pred_bc[:, 0:1] * params.sig_F + params.mu_F
                loss = loss + w_bc * (
                    mse(E_bc_phys, torch.zeros_like(E_bc_phys))
                    + mse(F_bc_phys, torch.zeros_like(F_bc_phys))
                )
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
            r2_l = r2_safe(y_val[:, 0], Fv)
            r2_e = r2_safe(y_val[:, 1], Ev)
            val_score = _val_checkpoint_score(r2_l, r2_e, approach="soft")
            logger.info(
                f"    [soft] ep={ep:4d}  train_loss={loss_sum/nb:.4f}  "
                f"phys_loss={phys_sum/nb:.4f}  "
                f"val_R2_load={r2_l:.4f}  val_R2_energy={r2_e:.4f}"
            )
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

    if best_state:
        model.load_state_dict(best_state)
    return model, history, best_r2, {"training_time": time.time() - t0, "n_params": model.count_parameters(), "w_phys": cfg["w_phys"]}


def train_hard(train_df: pd.DataFrame, val_df: pd.DataFrame, scaler_disp: StandardScaler,
               scaler_out: StandardScaler, enc: OneHotEncoder, params: ScalingParams,
               seed: int, protocol: str, logger: logging.Logger) -> Tuple:
    """Train Hard-PINN model.

    Architecture: HardEnergyNet outputs normalised energy ``Ê_n``; the
    force ``F̂ = ∂Ê/∂d`` is computed by autograd at both training and
    inference, so the work-energy identity is enforced by construction.
    The boundary conditions E(0)=0 and F(0)=0 are encouraged through the
    three auxiliary soft regularisers (monotonicity, angle smoothness,
    curvature) which collectively shape F and E near d=0.  An optional
    architectural BC correction (``HardEnergyNet.configure_zero_bc``) is
    preserved in the codebase for ablation studies.

    Loss: ``w_load · SmoothL1(F̂, F) + w_energy · SmoothL1(Ê, E)`` plus
    the three auxiliary soft regularisers.  The third return value is
    the best epoch's load R², used for checkpointing including the
    final SWA snapshot.

    Training schedule: warmup + cosine LR + SWA over the final ``swa_pct``
    of epochs.
    """
    set_seed(seed)
    cfg = get_model_config("hard", protocol)
    t0 = time.time()
    rng = np.random.default_rng(seed + 300)
    
    Xtr = to_tensor(build_features(train_df, scaler_disp, enc))
    ytr = to_tensor(build_targets(train_df, scaler_out))
    Xv = to_tensor(build_features(val_df, scaler_disp, enc))
    y_val = val_df[["load_kN", "energy_J"]].values
    
    # Architecture selected via CFG.hard_architecture / cfg["architecture"];
    # default "mlp" reproduces the published production model exactly.  For
    # the baseline the optional slope-subtraction BC stays disabled (soft
    # regularisers handle BCs); the monotone/separable variants enforce
    # E(0)=0 and F>=0 architecturally (see make_hard_energy_net).
    model = make_hard_energy_net(Xtr.shape[1], cfg, params).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]) if cfg.get("optimizer", "adamw").lower() == "adam" else optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    
    # Scheduler selection: stabilized (unseen) vs reactive (random)
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
    w_curv = cfg.get("w_curvature", 0.0)
    smooth_delta = cfg.get("smooth_delta_deg", 1.5)
    colloc_ratio = cfg.get("colloc_ratio", 0.0)

    # Create collocation sampler if any physics regularization is active
    colloc_sampler = None
    if w_mono > 0 or w_smooth > 0 or w_curv > 0:
        extrapolate = cfg.get("extrapolate_angles", False)
        colloc_sampler = create_collocation_sampler(train_df, scaler_disp, enc, extrapolate_angles=extrapolate)

    rng_b = np.random.default_rng(seed)
    idx = bootstrap_indices(train_df, rng_b) if CFG.bootstrap else np.arange(Xtr.shape[0])
    loader = DataLoader(
        TensorDataset(Xtr[idx], ytr[idx]), batch_size=cfg["batch_size"], shuffle=True, **_data_loader_kwargs(seed=seed)
    )

    history = {"epoch": [], "train_loss": [], "val_load_r2": [], "val_energy_r2": [],
               "phys_residual_rms": []}
    best_state, best_r2 = None, -1e9
    # Disable early stopping for stabilized training: SWA needs the full
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
                Xc = colloc_sampler(n_colloc, rng)

                if w_mono > 0:
                    loss_mono = monotonicity_loss_hard(Xc, model, params)
                    loss = loss + w_mono * loss_mono
                
                if w_smooth > 0:
                    n_smooth = max(1, n_colloc // 2)
                    loss_smooth = angle_smoothness_loss_hard(model, colloc_sampler, n_smooth, rng, params, smooth_delta)
                    loss = loss + w_smooth * loss_smooth

                if w_curv > 0:
                    loss_curv = curvature_regularization_hard(Xc, model, params)
                    loss = loss + w_curv * loss_curv

            opt.zero_grad()
            loss.backward()
            if cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["grad_clip"])
            opt.step()
            loss_sum += loss.item()
            nb += 1
        
        # Scheduler step: warmup+cosine every epoch, ReduceLROnPlateau on eval
        if use_stabilized:
            sched.step()
            if ep >= swa_start:
                swa_active = True
                swa_model.update_parameters(model)
        
        if ep % cfg["eval_every"] == 0:
            # Evaluate SWA model if active, otherwise base model
            eval_model = swa_model.module if (use_stabilized and swa_active) else model
            eval_model.eval()
            Fv, Ev = hard_pinn_predict_load_energy(eval_model, Xv, params)
            r2_l = r2_safe(y_val[:, 0], Fv)
            r2_e = r2_safe(y_val[:, 1], Ev)
            val_score = _val_checkpoint_score(r2_l, r2_e, approach="hard")
            _swa_tag = "swa" if (use_stabilized and swa_active) else "base"
            logger.info(
                f"    [hard] ep={ep:4d}  train_loss={loss_sum/nb:.4f}  "
                f"val_R2_load={r2_l:.4f}  val_R2_energy={r2_e:.4f}  "
                f"({_swa_tag})"
            )
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

    # Final SWA evaluation
    if use_stabilized and swa_active:
        swa_model.module.eval()
        Fv, Ev = hard_pinn_predict_load_energy(swa_model.module, Xv, params)
        r2_swa = _val_checkpoint_score(
            float(r2_safe(y_val[:, 0], Fv)),
            float(r2_safe(y_val[:, 1], Ev)),
            approach="hard",
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
    failed_members = []  # (m_index, exception) for transparency
    import gc as _gc

    for m in range(CFG.n_ensemble):
        seed = CFG.seed_base + m * 1000
        try:
            # Per-member disk cache (resume-friendly): an interrupted ensemble
            # keeps every finished member; re-running trains only the missing ones.
            # Cache key includes a fingerprint of the training data and the
            # per-approach hyperparameter dict: editing either must invalidate
            # cached members, or a rerun silently reuses stale weights.
            import hashlib as _hl
            _cfg_now = get_model_config(approach, protocol)
            _fp_src = (pd.util.hash_pandas_object(train_df, index=False).values.tobytes()
                       + repr(sorted(_cfg_now.items())).encode())
            _fp = _hl.sha256(_fp_src).hexdigest()[:10]
            _mkey = (f"member_{approach}_{protocol}_m{m}_ep{getattr(CFG, 'max_epochs', 0)}"
                     f"_seed{seed}_fp{_fp}")
            model, hist, r2, meta = _explicit_disk_cache(
                _mkey,
                lambda: train_fn(train_df, val_df, scaler_disp, scaler_out, enc,
                                 params, seed, protocol, logger),
            )
            metrics = evaluate_model(model, approach, val_df, scaler_disp, scaler_out, enc, params)
            models.append(model)
            histories.append(hist)
            metas.append(meta)
            member_metrics.append(metrics)
            logger.info(f"    M{m+1}: R²_load={metrics['load_r2']:.4f}, time={meta['training_time']:.1f}s")
        except (RuntimeError, ValueError) as ex:
            # Catch OOM, NaN-grad explosions, etc. Continue with remaining members so a
            # 36-h SLURM job is not lost to a single failure on member k of M.
            failed_members.append((m, repr(ex)))
            logger.error(f"    M{m+1}: TRAINING FAILED ({type(ex).__name__}): {ex}. Skipping member.")
        finally:
            # Aggressive cleanup: prevents VRAM fragmentation on long ensembles
            # and keeps `nvidia-smi` honest. Important on >M=10 with autograd graphs.
            _gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if failed_members:
        logger.warning(
            f"  Ensemble training: {len(failed_members)}/{CFG.n_ensemble} members failed; "
            f"continuing with M_actual={len(models)}."
        )
    if not models:
        raise RuntimeError(
            f"All {CFG.n_ensemble} members of the {approach} ensemble failed to train. "
            f"Check logs for {len(failed_members)} captured exceptions."
        )
    
    # --- Convergence filter (Tukey fence on training-set R²) ----------------
    # Evaluate every member on the FULL training set (not its bootstrap
    # sample) to get a comparable quality metric.  Then apply the standard
    # Tukey fence: discard members with train R² < Q1 - k*IQR where k is
    # CFG.convergence_filter_iqr (default 1.5).  This adapts to the task
    # difficulty and only removes genuine outliers.
    k_iqr = CFG.convergence_filter_iqr
    M_total = len(models)
    # Snapshot pre-fence member metrics so survivor-only statistics can be
    # reported ALONGSIDE all-member statistics (avoids silent survivorship
    # bias in the published ensemble means/CIs).
    member_metrics_all = list(member_metrics)
    discarded_indices: List[int] = []

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
            logger.info(f"      (all-member metrics retained for survivorship-bias reporting)")
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
    
    # Safety: enforce a minimum surviving ensemble size.  Every downstream
    # uncertainty quantity (EA_std, conformal factors, GP-BO posteriors)
    # silently degrades to noise with a tiny ensemble, so in production this
    # FAILS rather than warns.  Dry runs keep the old advisory behaviour.
    min_members = int(getattr(CFG, "min_ensemble_members", 3))
    if len(models) < min_members:
        msg = (f"Only {len(models)} members survived filtering "
               f"(minimum {min_members}); increase n_ensemble or relax k_iqr.")
        if CFG.dry_run or CFG.n_ensemble < min_members:
            logger.warning(f"    WARNING: {msg}")
        else:
            raise RuntimeError(msg)
    
    ens_metrics = evaluate_ensemble(models, approach, val_df, scaler_disp, scaler_out, enc, params)
    return {"models": models, "histories": histories, "member_metrics": member_metrics,
            "metrics": ens_metrics, "avg_training_time": np.mean([m["training_time"] for m in metas]),
            "n_params": metas[0]["n_params"],
            "M_total": M_total, "M_eff": len(models),
            "train_r2_scores": train_r2_scores,
            "convergence_fence": fence,
            # Pre-fence metrics + which members were dropped: enables the
            # with/without-filter comparison in Table 1 and records WHICH
            # members the fence excluded (previously unrecorded).
            "member_metrics_all": member_metrics_all,
            "discarded_member_indices": discarded_indices}


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
    
    return {"load_r2": float(r2_safe(y_val[:, 0], Fv)), "energy_r2": float(r2_safe(y_val[:, 1], Ev)),
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
    
    return {"load_r2": float(r2_safe(y_val[:, 0], Fm)), "energy_r2": float(r2_safe(y_val[:, 1], Em)),
            "load_rmse": float(np.sqrt(mean_squared_error(y_val[:, 0], Fm))),
            "energy_rmse": float(np.sqrt(mean_squared_error(y_val[:, 1], Em))),
            "load_mae": float(mean_absolute_error(y_val[:, 0], Fm)),
            "energy_mae": float(mean_absolute_error(y_val[:, 1], Em)),
            "load_errors": Fm - y_val[:, 0], "energy_errors": Em - y_val[:, 1],
            "predictions": {"load": Fm, "energy": Em, "load_std": Fs, "energy_std": Es},
            "true_values": {"load": y_val[:, 0], "energy": y_val[:, 1]}}


def compute_statistical_tests(dual_results: Dict, logger: logging.Logger) -> Dict:
    """Compare ensemble member load-R² across approaches with three statistics.

    For each (protocol, approach-pair) the following are reported:

    1. **Welch's t-test** (``equal_var=False``) — descriptive effect-size summary.
       Note: ensemble members share the data split, so the iid assumption is
       violated. The p-value is reported with ``descriptive_only=True`` so
       downstream tables can flag it as exploratory rather than confirmatory.
    2. **Cohen's d** (pooled-σ form, see Cohen 1988).
    3. **Bootstrap 95% CI of the mean R² difference** (``n_boot=10_000``,
       resampling members with replacement). This is the *non-parametric*
       complement to (1): it does not assume iid, gives a magnitude estimate
       in R² units, and is what the manuscript can cite as a robust comparison.

    Bonferroni adjustment is applied **within each protocol** (random vs
    unseen), over the family of pairwise approach comparisons reported.
    """
    if not HAS_SCIPY:
        logger.warning("SciPy not available, skipping statistical tests")
        return {}
    rng = np.random.default_rng(CFG.seed)
    n_boot = 10_000

    def _bootstrap_mean_diff_ci(arr1: np.ndarray, arr2: np.ndarray) -> Dict[str, float]:
        """Percentile bootstrap of mean(arr1) - mean(arr2)."""
        if arr1.size == 0 or arr2.size == 0:
            return {"mean_diff": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
        idx1 = rng.integers(0, arr1.size, size=(n_boot, arr1.size))
        idx2 = rng.integers(0, arr2.size, size=(n_boot, arr2.size))
        diffs = arr1[idx1].mean(axis=1) - arr2[idx2].mean(axis=1)
        return {
            "mean_diff": float(np.mean(arr1) - np.mean(arr2)),
            "ci_lo": float(np.percentile(diffs, 2.5)),
            "ci_hi": float(np.percentile(diffs, 97.5)),
        }

    results = {}
    for protocol in ["random", "unseen"]:
        if protocol not in dual_results:
            continue
        results[protocol] = {}
        r2_vals = {a: [m["load_r2"] for m in dual_results[protocol][a]["member_metrics"]]
                   for a in ["ddns", "soft", "hard"] if a in dual_results[protocol]}
        for a1, a2 in [("ddns", "soft"), ("ddns", "hard"), ("soft", "hard")]:
            if a1 in r2_vals and a2 in r2_vals:
                arr1 = np.array(r2_vals[a1], dtype=np.float64)
                arr2 = np.array(r2_vals[a2], dtype=np.float64)
                # Drop NaNs (flat-target slices) so neither test crashes.
                arr1 = arr1[np.isfinite(arr1)]
                arr2 = arr2[np.isfinite(arr2)]
                if arr1.size == 0 or arr2.size == 0:
                    continue
                t_stat, t_pvalue = stats.ttest_ind(arr1, arr2, equal_var=False)
                n1, n2 = len(arr1), len(arr2)
                s_pooled = np.sqrt(
                    ((n1 - 1) * np.var(arr1, ddof=1) + (n2 - 1) * np.var(arr2, ddof=1))
                    / max(n1 + n2 - 2, 1)
                )
                cohens_d = (np.mean(arr1) - np.mean(arr2)) / (s_pooled + 1e-12)
                bs = _bootstrap_mean_diff_ci(arr1, arr2)
                results[protocol][f"{a1}_vs_{a2}"] = {
                    "t_statistic": float(t_stat),
                    "t_pvalue": float(t_pvalue),
                    "cohens_d": float(cohens_d),
                    "n1": n1, "n2": n2,
                    "bootstrap_mean_diff": bs["mean_diff"],
                    "bootstrap_ci_lo_95": bs["ci_lo"],
                    "bootstrap_ci_hi_95": bs["ci_hi"],
                    "bootstrap_excludes_zero": (bs["ci_lo"] > 0.0) or (bs["ci_hi"] < 0.0),
                    "descriptive_only": True,
                    "iid_violation_note": "Ensemble members share data; t-test/p-value descriptive, bootstrap CI is robust.",
                }
                logger.info(
                    f"  {protocol} {a1} vs {a2}: p={t_pvalue:.4f} (descriptive), "
                    f"d={cohens_d:.3f}, ΔR²={bs['mean_diff']:.4f} "
                    f"95% CI=[{bs['ci_lo']:.4f}, {bs['ci_hi']:.4f}] (n1={n1}, n2={n2})"
                )

    # Bonferroni: correct within each protocol's comparison family only
    for protocol in results:
        n_family = max(len(results[protocol]), 1)
        for comp in results[protocol]:
            raw_p = results[protocol][comp]["t_pvalue"]
            m = max(n_family, 1)
            results[protocol][comp]["p_bonferroni"] = min(raw_p * m, 1.0)
            results[protocol][comp]["n_comparisons_family"] = m
            # Bonferroni-significant flag retained for legacy tables but treat as descriptive
            results[protocol][comp]["significant_bonferroni_descriptive"] = results[protocol][comp]["p_bonferroni"] < 0.05
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


def compute_ipf_robust(disps: np.ndarray, loads: np.ndarray, min_disp: float = 0.5,
                       prom_frac: float = 0.05, return_info: bool = False):
    """Compute Initial Peak Force robustly.

    Fallback: if no prominent peaks found after min_disp, use the maximum
    force in the first 25% of the displacement range (not the global max,
    which for progressive crushing could occur deep in the stroke).

    When ``return_info=True`` a third element is returned:
    ``{"fallback": bool}`` — True when the value came from the
    early-window-max fallback rather than a detected prominent peak.  The
    fallback silently switches the estimand (window max vs. first prominent
    peak), so consumers that publish IPF values should record and report the
    fallback rate instead of hiding it.
    """
    # Manuscript Section 2: peak qualifies on prominence AND width.
    # Width >= 2 samples filters out 1-sample noise spikes; loads are densely
    # resampled (~650-1050 pts per curve) so 2 samples is a small fraction of stroke.
    load_range = float(loads.max() - loads.min())
    if load_range < 1e-9:
        # Flat curve: no meaningful peak; return the constant value at first valid disp.
        valid_idx = int(np.argmax(disps > min_disp)) if np.any(disps > min_disp) else 0
        out = (float(loads[valid_idx]), float(disps[valid_idx]))
        return (*out, {"fallback": True}) if return_info else out
    peaks, _ = find_peaks(
        loads,
        prominence=prom_frac * load_range,
        width=2,
    )
    valid_peaks = [p for p in peaks if disps[p] > min_disp]
    fallback = False
    if valid_peaks:
        idx = valid_peaks[0]
    else:
        fallback = True
        # Restrict fallback to first 25% of displacement range
        d_max = disps[-1] if len(disps) > 0 else 1.0
        early_mask = disps <= max(d_max * 0.25, min_disp * 2)
        if np.any(early_mask):
            idx = int(np.argmax(loads[:np.sum(early_mask)]))
        else:
            idx = np.argmax(loads)
    out = (float(loads[idx]), float(disps[idx]))
    return (*out, {"fallback": fallback}) if return_info else out


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
    ipf_results = [compute_ipf_robust(disps, all_F[i], return_info=True) for i in range(len(models))]
    ipf_per_member = np.array([r[0] for r in ipf_results])
    ipf_fallback_per_member = np.array([bool(r[2]["fallback"]) for r in ipf_results])

    ea_mean = float(np.mean(ea_per_member))
    ipf_mean = float(np.mean(ipf_per_member))
    f_mean = ea_mean / d_metric if d_metric > 0 else 0.0
    cfe = f_mean / ipf_mean if ipf_mean > 1e-12 else 0.0
    _dd_m = 1 if len(models) > 1 else 0

    # ---- Physical-plausibility diagnostics (inference-time) ----------------
    # Physics constraints (F >= 0, monotone E, EA > 0) are only SOFT training
    # penalties; nothing guarantees them at deployment.  These per-design
    # violation fractions let every consumer (design-space sweep, inverse
    # objective, Pareto filter) detect — rather than silently absorb — an
    # unphysical prediction.  Tolerances: 1 N on force; 0.1% of the energy
    # range on monotonicity (numerical-noise floor).
    _f_tol = 1e-3   # kN
    neg_force_frac = float(np.mean([np.any(all_F[i] < -_f_tol) for i in range(len(models))]))
    _e_ranges = np.maximum(np.ptp(all_E, axis=1), 1e-12)
    nonmono_E_frac = float(np.mean([
        np.any(np.diff(all_E[i]) < -1e-3 * _e_ranges[i]) for i in range(len(models))]))
    neg_ea_frac = float(np.mean(ea_per_member <= 0.0))
    plausibility = {
        "neg_force_frac": neg_force_frac,
        "nonmono_energy_frac": nonmono_E_frac,
        "neg_ea_frac": neg_ea_frac,
        "ipf_fallback_frac": float(np.mean(ipf_fallback_per_member)),
        "all_plausible": bool(neg_force_frac == 0.0 and nonmono_E_frac == 0.0
                              and neg_ea_frac == 0.0),
    }

    return {"EA": ea_mean, "IPF": ipf_mean,
            "EA_std": float(np.std(ea_per_member, ddof=_dd_m)),
            "IPF_std": float(np.std(ipf_per_member, ddof=_dd_m)),
            "F_mean": f_mean, "CFE": cfe,
            "disps": disps, "loads": Fm, "loads_std": Fs,
            "energies": Em_corrected, "energies_std": Es,
            "disp_end": d_end, "d_eval": d_metric,
            "ea_per_member": ea_per_member, "ipf_per_member": ipf_per_member,
            "plausibility": plausibility}


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
    # Sample-size-aware calibration policy: identical to ``auto_tune_lambda``.
    # With ~12-30 paired samples (n_lc0 = n_lc1 ≈ 6-15) the Platt sigmoid
    # calibrator is high-variance; in that regime calibration is deliberately
    # *skipped* and the soft-voting ensemble is returned directly.  Above 30
    # samples, sigmoid calibration with up to 5-fold inner CV is applied.
    import warnings as _w
    base_ens = _make_base()
    if n_samples < 20:
        _calib_mode_outer = "uncalibrated"
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            base_ens.fit(X_scaled, y)
        cal_ens = base_ens
    else:
        _cv_cap_outer = 3 if n_samples < 30 else 5
        n_cv = max(2, min(_cv_cap_outer, min(n_lc0, n_lc1)))
        _calib_mode_outer = f"sigmoid_cv{n_cv}"
        cal_ens = CalibratedClassifierCV(base_ens, method="sigmoid", cv=n_cv)
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            cal_ens.fit(X_scaled, y)
    logger.info(f"    Outer classifier calibration: {_calib_mode_outer} (n={n_samples})")

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
            # Same sample-size policy as the outer fit: skip calibration when
            # the train fold is too small to support a stable Platt sigmoid.
            _base_loo = _make_base()
            if n_samples < 20:
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    _base_loo.fit(_X_tr, y[train_idx])
                _ens = _base_loo
            else:
                _ens = CalibratedClassifierCV(
                    _base_loo, method="sigmoid",
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
        # Record whether Platt calibration was actually applied so downstream
        # artifacts and the manuscript describe the classifier accurately:
        # with n < 20 the ensemble is deliberately UNCALIBRATED (raw soft
        # voting), and calling it a "calibrated classifier" would overstate it.
        "calibration_mode": _calib_mode_outer,
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel (a): P(LC2 | indicators) heatmap (cividis is colourblind- and B/W-safe)
    ax = axes[0]
    im = ax.contourf(EA_grid, IPF_grid, P_lc2, levels=20, cmap="cividis")
    ax.contour(EA_grid, IPF_grid, P_lc2, levels=[0.5], colors="white",
               linewidths=1.6, linestyles="--")
    for lc_val, marker, color, label in [("LC1", MARKERS["LC1"], COLORS["LC1"], "LC1"),
                                          ("LC2", MARKERS["LC2"], COLORS["LC2"], "LC2")]:
        sub = df_metrics[df_metrics["LC"] == lc_val]
        ax.scatter(sub[ea_col], sub["IPF"], c=color, marker=marker,
                   edgecolors="black", s=70, zorder=5, label=label)
    ax.set_xlabel("Energy absorption to {:.0f} mm (J)".format(D_COMMON))
    ax.set_ylabel("Initial Peak Force IPF (kN)")
    ang_note = f" (slice at θ={angle_ref:.1f}°)" if n_in >= 3 else ""
    ax.set_title(f"(a) P(LC2 | features){ang_note}")
    ax.legend(loc="best")
    plt.colorbar(im, ax=ax, label="P(LC2)")

    # Panel (b): Penalty landscape (sequential, B/W-safe)
    ax2 = axes[1]
    Phi_lc1 = -np.log(np.maximum(1.0 - P_lc2, 1e-6))  # penalty if LC=LC1
    Phi_lc2 = -np.log(np.maximum(P_lc2, 1e-6))          # penalty if LC=LC2
    Phi_min = np.minimum(Phi_lc1, Phi_lc2)  # best-case penalty
    im2 = ax2.contourf(EA_grid, IPF_grid, Phi_min, levels=20, cmap="viridis")
    ax2.contour(EA_grid, IPF_grid, P_lc2, levels=[0.5], colors="white",
                linewidths=1.6, linestyles="--")
    for lc_val, marker, color, label in [("LC1", MARKERS["LC1"], COLORS["LC1"], "LC1"),
                                          ("LC2", MARKERS["LC2"], COLORS["LC2"], "LC2")]:
        sub = df_metrics[df_metrics["LC"] == lc_val]
        ax2.scatter(sub[ea_col], sub["IPF"], c=color, marker=marker,
                    edgecolors="black", s=70, zorder=5, label=label)
    ax2.set_xlabel("Energy absorption to {:.0f} mm (J)".format(D_COMMON))
    ax2.set_ylabel("Initial Peak Force IPF (kN)")
    ax2.set_title(f"(b) min-LC Penalty $\\Phi${ang_note}")
    ax2.legend(loc="best")
    plt.colorbar(im2, ax=ax2, label="$\\Phi$ (lower = more plausible)")

    fig.suptitle("Ensemble Classifier: Loading-Condition Plausibility")
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

    # Sample-size policy for the LOO calibrator inside the LOO probability loop:
    # ─ if n < 20: data is too small for stable Platt sigmoid calibration on n-1
    #   training-fold rows (cv would drop to 2-3 and the calibrator's fits become
    #   high-variance). Use the **uncalibrated** soft-voting ensemble; the LOO
    #   probabilities are then raw learner-averaged probabilities. λ tuned to
    #   that signal is honest about the data regime.
    # ─ if 20 ≤ n < 30: keep CalibratedClassifierCV with sigmoid but cap cv=3.
    # ─ if n ≥ 30: original behaviour (sigmoid, cv up to 5).
    if n < 20:
        _calib_mode = "uncalibrated"
        _cv_cap = None
    elif n < 30:
        _calib_mode = "sigmoid_small"
        _cv_cap = 3
    else:
        _calib_mode = "sigmoid_full"
        _cv_cap = 5
    logger.info(f"    LOO calibration mode: {_calib_mode} (n={n})")

    # LOO cross-validated probabilities (scaler refit on train fold only; matches training CV)
    from sklearn.model_selection import LeaveOneOut
    loo_probs = np.zeros(n)
    for train_idx, test_idx in LeaveOneOut().split(X_raw):
        _scaler = StandardScaler().fit(X_raw[train_idx])
        X_tr = _scaler.transform(X_raw[train_idx])
        X_te = _scaler.transform(X_raw[test_idx])
        _base = _make_lc_voting_classifier()
        if _calib_mode == "uncalibrated":
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                _base.fit(X_tr, y[train_idx])
            loo_probs[test_idx[0]] = _base.predict_proba(X_te)[0, 1]
        else:
            min_class = int(min(np.sum(y[train_idx] == 0), np.sum(y[train_idx] == 1)))
            _cv = max(2, min(_cv_cap, min_class))
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
        "calibration_mode": _calib_mode,
        "n_samples": n,
    }
    return lambda_opt, tuning_diag


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
    
    Uses scikit-optimize's gp_minimize with a joint search space:
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
    
    # Stagnation-based early-stop callback.  skopt invokes the callback
    # after every evaluation; returning truthy halts optimization with the
    # results accumulated so far.  We track the best-so-far value and stop
    # once it has not improved by more than ``stagnation_min_delta`` for
    # ``stagnation_patience`` consecutive evaluations beyond ``n_init``.
    patience = int(getattr(bo_cfg, "stagnation_patience", 6))
    min_delta = float(getattr(bo_cfg, "stagnation_min_delta", 1.0e-5))
    n_init_eff = int(bo_cfg.n_init)
    _state = {"best": float("inf"), "stale_count": 0, "stopped_at": None}

    def _early_stop_cb(skopt_res):
        n_done = len(skopt_res.func_vals)
        cur_best = float(np.min(skopt_res.func_vals))
        if n_done <= n_init_eff:
            _state["best"] = min(_state["best"], cur_best)
            return False
        if cur_best < _state["best"] - min_delta:
            _state["best"] = cur_best
            _state["stale_count"] = 0
        else:
            _state["stale_count"] += 1
        if _state["stale_count"] >= patience:
            _state["stopped_at"] = n_done
            return True
        return False

    # Run skopt GP-BO (stores models at each iteration).
    res = skopt_gp_minimize(
        joint_objective, space,
        n_calls=bo_cfg.n_calls_total,
        n_initial_points=bo_cfg.n_init,
        random_state=bo_cfg.seed,
        acq_func="EI",
        xi=bo_cfg.xi,
        callback=[_early_stop_cb],
    )
    if _state["stopped_at"] is not None:
        logger.info(
            f"    GP-BO early-stop: best-so-far stagnated for "
            f"{patience} consecutive evals; halted at eval "
            f"{_state['stopped_at']}/{bo_cfg.n_calls_total}."
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
                    cal_ens=None, feat_scaler=None, prob_weight=0.0, d_eval=None):
    """Factory function for objective closure with optional LC plausibility penalty.

    Objective:
        J = w_ea*(EA@d_eval - EA_target)^2 + w_ipf*(IPF - IPF_target)^2
            + prob_weight*(-log p_LC)
    """
    def objective(angle: float) -> float:
        m = compute_ea_ipf_ensemble(models, approach, float(angle), lc, scaler_disp, enc, params,
                                     d_eval=d_eval)
        # Physically invalid candidate (non-positive predicted EA): return a
        # large finite objective so the optimizer steers away instead of
        # potentially "matching" a target through an unphysical prediction.
        # For well-trained production surrogates this never triggers; it
        # protects the search when a member extrapolates badly.
        if m["EA"] <= 0.0:
            return 1e6 + abs(float(m["EA"]))
        fit_error = float(w_ea * (m["EA"] - target_ea)**2 + w_ipf * (m["IPF"] - target_ipf)**2)

        if cal_ens is not None and prob_weight > 0:
            penalty, _ = compute_lc_penalty(
                cal_ens, feat_scaler, m["EA"], m["IPF"], lc, prob_weight,
                angle_deg=float(angle),
            )
            return fit_error + penalty
        return fit_error
    return objective



def run_inverse_design(models: List[nn.Module], approach: str, target_ea: float, target_ipf: float,
                       scaler_disp: StandardScaler, enc: OneHotEncoder, params: ScalingParams,
                       bo_cfg: BOConfig, logger: logging.Logger,
                       cal_ens=None, feat_scaler=None,
                       prob_weight_override: Optional[float] = None) -> Dict:
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
    prob_weight_override : optional
        When set, override the auto-tuned classifier penalty weight (ablation).
    """
    lc_categories = [str(x) for x in enc.categories_[0].tolist()]
    bounds = (CFG.angle_opt_min, CFG.angle_opt_max)
    w_ea = 1.0 / (target_ea**2 + 1e-12)
    w_ipf = 1.0 / (target_ipf**2 + 1e-12)
    prob_weight = bo_cfg.prob_weight if cal_ens is not None else 0.0
    if prob_weight_override is not None:
        prob_weight = float(prob_weight_override) if cal_ens is not None else 0.0
    results = {"gpbo": {}, "gpbo_joint": None,
               "target_ea": target_ea, "target_ipf": target_ipf,
               "prob_weight": prob_weight}
    # Only GP-BO is used for inverse design.
    
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
    
    # Create objective functions for each LC (evaluated at D_COMMON)
    objective_funcs = {}
    for lc in lc_categories:
        objective_funcs[lc] = _make_objective(
            models, approach, lc, scaler_disp, enc, params,
            target_ea, target_ipf, w_ea, w_ipf,
            cal_ens=cal_ens, feat_scaler=feat_scaler, prob_weight=prob_weight,
            d_eval=D_COMMON,
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

    return results


# =============================================================================
# ILL-POSEDNESS ANALYSIS: MULTI-START BO, LANDSCAPE, SENSITIVITY, POSTERIOR
# =============================================================================
def gp_bo_minimize_joint_multistart(objective_funcs, bounds, bo_cfg, logger, n_restarts=5):
    """Multi-start GP-BO: run ``gp_bo_minimize_joint`` ``n_restarts`` times with
    different seeds, keep the lowest-y restart, and report cross-restart spread.

    Per the manuscript Section 5.8, robustness is assessed across **5 independent
    random seeds per target**. Each restart is a *fully independent* GP-BO run
    of ``bo_cfg.n_calls_total`` evaluations (default 30). The total surrogate
    cost per inverse-design call is therefore ``n_restarts × n_calls_total``
    (= 150 with defaults). The per-restart budget controls the convergence
    claim; the multi-start factor provides a robustness safety net
    ("multi-seed robustness testing ... converges to identical solutions
    across seeds").
    """
    if n_restarts <= 1:
        return gp_bo_minimize_joint(objective_funcs, bounds, bo_cfg, logger)
    n_calls_per_restart = int(getattr(bo_cfg, "n_calls_total", 30))
    total_calls_per_target = n_restarts * n_calls_per_restart
    logger.info(
        f"    Multi-start GP-BO: {n_restarts} restarts × {n_calls_per_restart} calls each "
        f"= {total_calls_per_target} surrogate evaluations per target."
    )

    all_restarts = []
    original_seed = bo_cfg.seed
    # Hash-stride seeds: avoids stride collisions with ensemble training (which
    # uses stride 1000) and produces non-adjacent integer seeds that diverge
    # under skopt's RNG initialization in just a couple of TPE iterations.
    SEED_PRIME = 1009
    for i in range(n_restarts):
        seed_i = (original_seed * 37 + i * SEED_PRIME) % (2 ** 31 - 1)
        logger.info(f"    Multi-start BO restart {i+1}/{n_restarts} (seed={seed_i})")
        bo_cfg.seed = int(seed_i)
        try:
            res = gp_bo_minimize_joint(objective_funcs, bounds, bo_cfg, logger)
            res["restart_id"] = i
            res["restart_seed"] = int(seed_i)
            all_restarts.append(res)
        except Exception as e:
            logger.warning(f"    Restart {i+1} failed: {e}")
    bo_cfg.seed = original_seed
    if not all_restarts:
        raise RuntimeError("All multi-start BO restarts failed")
    best = min(all_restarts, key=lambda r: r["y_best"])

    # Cross-restart spread (theta and y at the per-restart optimum).  This
    # is the m5 metric: it surfaces non-convergence cases that the
    # "best-only" reporting would otherwise hide.
    theta_bests = np.array([r["x_best"] for r in all_restarts], dtype=np.float64)
    y_bests = np.array([r["y_best"] for r in all_restarts], dtype=np.float64)
    lc_bests = [r.get("best_lc") for r in all_restarts]

    # LC unanimity diagnostic: count how many restarts agreed on the
    # winning LC.  A 5/5 unanimous result is strong evidence the optimum
    # is uniquely identifiable; a split (e.g. 3/5 LC1, 2/5 LC2) flags a
    # multimodal target whose recovered LC depends on the optimizer's
    # random seed.  Both cases are publication-worthy diagnostics.
    lc_counts: Dict[str, int] = {}
    for lc in lc_bests:
        if lc is None:
            continue
        lc_counts[lc] = lc_counts.get(lc, 0) + 1
    n_voters = sum(lc_counts.values())
    if n_voters > 0:
        top_lc, top_n = max(lc_counts.items(), key=lambda kv: kv[1])
        lc_unanimity_fraction = float(top_n) / float(n_voters)
        unanimous = (top_n == n_voters)
        unanimity_summary = "/".join(
            f"{n}_{lc}" for lc, n in sorted(lc_counts.items(), key=lambda kv: -kv[1])
        )
    else:
        top_lc = None
        lc_unanimity_fraction = float("nan")
        unanimous = False
        unanimity_summary = "n/a"

    best["all_restarts"] = all_restarts
    best["n_restarts_completed"] = len(all_restarts)
    best["restart_summary"] = {
        "n_restarts_requested": int(n_restarts),
        "n_restarts_completed": int(len(all_restarts)),
        "n_calls_per_restart": int(n_calls_per_restart),
        "total_surrogate_calls": int(total_calls_per_target),
        "theta_best_mean": float(np.mean(theta_bests)),
        "theta_best_std": float(np.std(theta_bests, ddof=1)) if len(theta_bests) > 1 else 0.0,
        "theta_best_min": float(np.min(theta_bests)),
        "theta_best_max": float(np.max(theta_bests)),
        "y_best_mean": float(np.mean(y_bests)),
        "y_best_std": float(np.std(y_bests, ddof=1)) if len(y_bests) > 1 else 0.0,
        "y_best_min": float(np.min(y_bests)),
        "y_best_max": float(np.max(y_bests)),
        # LC unanimity across restarts (paper-quality robustness indicator).
        "lc_unanimous":            bool(unanimous),
        "lc_unanimity_fraction":   lc_unanimity_fraction,
        "lc_majority":             top_lc,
        "lc_vote_summary":         unanimity_summary,
        "lc_counts":               dict(lc_counts),
    }
    logger.info(
        f"    Multi-start BO: best y={best['y_best']:.6f} from restart {best['restart_id']+1}; "
        f"cross-restart θ spread = ±{best['restart_summary']['theta_best_std']:.3f}°, "
        f"y spread = ±{best['restart_summary']['y_best_std']:.6f}"
    )
    logger.info(
        f"    Multi-start LC unanimity: {unanimity_summary} "
        f"({'UNANIMOUS' if unanimous else 'SPLIT'}; "
        f"fraction agreeing on top LC = {lc_unanimity_fraction:.2f})"
    )
    return best


def compute_solution_landscape(objective_funcs, bounds, best_objective, logger, n_grid=201,
                               rel_tol=0.02):
    """Evaluate J(theta) on a dense grid for both LCs; count local minima -> multiplicity index.

    Multiplicity threshold is stated in PHYSICAL units: because the fit term of
    J uses weights w = 1/target^2, an increment ``J - J_best`` equals the added
    squared combined *relative* (EA, IPF) error beyond the optimum.  A local
    minimum therefore counts as an alternative solution only when
    ``J <= J_best + rel_tol**2`` (default ``rel_tol=0.02`` -> within 2% combined
    relative error of the best solution).  This replaces the earlier ad hoc
    absolute floor (+0.01, i.e. 10% combined error), which dominated for
    well-converged optima and over-counted 'solutions'.
    """
    theta_grid = np.linspace(bounds[0], bounds[1], n_grid)
    result = {"theta_grid": theta_grid}
    total_minima = 0
    threshold = best_objective + float(rel_tol) ** 2
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
    result["multiplicity_rel_tol"] = float(rel_tol)
    logger.info(
        f"    Solution landscape: {total_minima} local minima within {rel_tol*100:.0f}% "
        f"combined relative error of the optimum (J <= {threshold:.6f})")
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
    # Grid (<=3 per row) instead of a single 1xn strip: a 1x5 row at ~5"/panel
    # is ~25" wide and shrinks to unreadable text when placed at column width.
    ncols = n if n <= 3 else 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.7 * nrows), squeeze=False)
    axes_flat = list(axes.flat)
    lc_color = {"LC1": COLORS["LC1"], "LC2": COLORS["LC2"]}
    lc_ls = {"LC1": LINESTYLES["LC1"], "LC2": LINESTYLES["LC2"]}
    for i, res in enumerate(targets_with_landscape):
        ax = axes_flat[i]
        sl = res["solution_landscape"]
        tid = res.get("target_info", {}).get("id", f"T{i+1}")
        for lc in sorted(k.replace("J_", "") for k in sl if k.startswith("J_")):
            J = sl[f"J_{lc}"]
            ax.plot(sl["theta_grid"], J, label=lc, linewidth=1.0,
                    color=lc_color.get(lc, COLORS["data"]),
                    linestyle=lc_ls.get(lc, "-"))
            for m_pt in sl.get(f"local_minima_{lc}", []):
                ax.plot(m_pt["theta"], m_pt["J"], "v", markersize=5, color=COLORS["gpbo"])
        ax.set_title(f"{tid} (SMI={sl['multiplicity_index']})")
        ax.set_xlabel(r"Angle ($^\circ$)")
        if i % ncols == 0:
            ax.set_ylabel(r"Objective $J(\theta)$")
        ax.legend()
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
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
    fig, axes = plt.subplots(2, len(lcs), figsize=(3.5 * len(lcs), 7.0), squeeze=False, sharex=True)
    bif_color = COLORS["gpbo"]
    for j, lc in enumerate(lcs):
        dea = jacobian_results[f"dEA_dtheta_{lc}"]
        dipf = jacobian_results[f"dIPF_dtheta_{lc}"]
        axes[0, j].plot(theta, dea, linewidth=1.0, color=COLORS["soft"])
        axes[0, j].axhline(0, color="0.4", linewidth=0.6, linestyle="--")
        for bf in jacobian_results.get(f"ea_bifurcations_{lc}", []):
            axes[0, j].axvline(bf, color=bif_color, linewidth=0.7, linestyle=":", alpha=0.85)
        axes[0, j].set_title(f"dEA/d$\\theta$ - {lc}")
        axes[0, j].set_ylabel("J/deg" if j == 0 else "")
        axes[1, j].plot(theta, dipf, linewidth=1.0, color=COLORS["ddns"])
        axes[1, j].axhline(0, color="0.4", linewidth=0.6, linestyle="--")
        for bf in jacobian_results.get(f"ipf_bifurcations_{lc}", []):
            axes[1, j].axvline(bf, color=bif_color, linewidth=0.7, linestyle=":", alpha=0.85)
        axes[1, j].set_title(f"dIPF/d$\\theta$ - {lc}")
        axes[1, j].set_xlabel(r"Angle ($^\circ$)")
        axes[1, j].set_ylabel("kN/deg" if j == 0 else "")
    # Panel labels (a)-(d) across the 2 x len(lcs) grid.
    for lab, ax in zip("abcdefgh", axes.flatten()):
        add_subplot_label(ax, lab)
    # Explain the reference lines: the dashed zero-slope line (a stationary point
    # of the forward map) and the detected bifurcations (sign changes that flag
    # local non-invertibility).  Only advertise bifurcations if any were found.
    has_bif = any(len(jacobian_results.get(f"{p}_bifurcations_{lc}", [])) > 0
                  for p in ("ea", "ipf") for lc in lcs)
    handles = [Line2D([0], [0], color="0.4", lw=0.8, ls="--", label="zero slope")]
    if has_bif:
        handles.append(Line2D([0], [0], color=bif_color, lw=0.9, ls=":", label="bifurcation"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 1.02), frameon=True)
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
    # Grid (<=3 per row) instead of a 1xn strip (see fig_solution_landscape).
    ncols = n if n <= 3 else 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.7 * nrows), squeeze=False)
    axes_flat = list(axes.flat)
    for i, res in enumerate(targets_with_post):
        ax = axes_flat[i]
        post = res["inverse_posterior"]
        tid = res.get("target_info", {}).get("id", f"T{i+1}")
        ax.plot(post["theta_grid"], post["posterior"], linewidth=1.0, color=COLORS["soft"])
        ax.axvline(post["mean"], color=COLORS["ddns"], linewidth=1.0, linestyle="--",
                   label=f"mean={post['mean']:.1f}$^\\circ$")
        ax.axvspan(post["ci_95_lower"], post["ci_95_upper"], alpha=0.18,
                   color=COLORS["soft"], label="95% CI")
        bo_best = res.get("gpbo_best", res.get("gpbo_joint", {}))
        if bo_best and "x_best" in bo_best:
            ax.axvline(bo_best["x_best"], color=COLORS["gpbo"], linewidth=1.0, linestyle=":",
                       label=f"BO opt={bo_best['x_best']:.1f}$^\\circ$")
        ax.set_title(tid)
        ax.set_xlabel(r"Angle ($^\circ$)")
        if i % ncols == 0:
            ax.set_ylabel(r"$P(\theta \mid$ target$)$")
        ax.legend()
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    path = os.path.join(output_dir, "Fig_inverse_posterior.png")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: Fig_inverse_posterior.png")


# =============================================================================
# INVERSE / ILL-POSEDNESS: PUBLICATION TABLES & EXTENDED POSTERIORS
# =============================================================================
def write_table_inverse_illposedness(all_inverse: List[Dict], output_dir: str, logger: logging.Logger) -> None:
    """One row per inverse target: multiplicity, sensitivity, posterior, multi-start spread.

    The multi-start columns surface the cross-restart variance of the
    GP-BO optimum (see ``gp_bo_minimize_joint_multistart``). Tight spread
    (``theta_best_std`` near 0) means GP-BO converges to the same answer
    from independent seeds; large spread flags a multimodal landscape.
    """
    if not all_inverse:
        return
    # Explicit, deterministic column order.
    COLS = [
        "Target_ID",
        f"Target_EA@{EA_COMMON_MM_TAG}_J", "Target_IPF_kN",
        "multiplicity_index",
        "posterior_mean_theta", "posterior_std_theta",
        "posterior_ci95_theta_lo", "posterior_ci95_theta_hi",
        "dJ_dtheta_at_opt", "d2J_dtheta2_at_opt",
        "n_bo_restarts_completed", "best_bo_restart_id",
        "n_calls_per_restart", "total_surrogate_calls",
        "theta_best_mean_across_restarts", "theta_best_std_across_restarts",
        "y_best_mean_across_restarts", "y_best_std_across_restarts",
        "prob_weight",
        "gpbo_y_best", "gpbo_theta_best", "gpbo_lc_best",
    ]
    rows = []
    for res in all_inverse:
        tid = res.get("target_info", {}).get("id", "?")
        sl = res.get("solution_landscape") or {}
        post = res.get("inverse_posterior") or {}
        loc = res.get("local_sensitivity") or {}
        best = res.get("gpbo_best") or {}
        gj = res.get("gpbo_joint") or {}
        rs = (gj.get("restart_summary") or {}) if gj else {}
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
            "n_calls_per_restart": rs.get("n_calls_per_restart", ""),
            "total_surrogate_calls": rs.get("total_surrogate_calls", ""),
            "theta_best_mean_across_restarts": rs.get("theta_best_mean", ""),
            "theta_best_std_across_restarts": rs.get("theta_best_std", ""),
            "y_best_mean_across_restarts": rs.get("y_best_mean", ""),
            "y_best_std_across_restarts": rs.get("y_best_std", ""),
            "prob_weight": res.get("prob_weight", ""),
            "gpbo_y_best": best.get("y_best", ""),
            "gpbo_theta_best": best.get("x_best", ""),
            "gpbo_lc_best": best.get("lc", best.get("best_lc", "")),
        })
    path = os.path.join(output_dir, "Table3_inverse_illposedness.csv")
    pd.DataFrame(rows, columns=COLS).to_csv(path, index=False)
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
    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 8), squeeze=False, sharex="col")
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
        axes[0, i].plot(th, pj, color=COLORS["soft"], linewidth=1.0, label="exp(-J/T)")
        axes[0, i].set_title(f"{tid}: J-posterior")
        if i == 0:
            axes[0, i].set_ylabel(r"$P(\theta)$")
        axes[1, i].plot(th, post_l["posterior"], color=COLORS["gpbo"], linewidth=1.0, label="Gaussian lik.")
        axes[1, i].axvline(post_l["mean"], color=COLORS["ddns"], linestyle="--", linewidth=1.0)
        axes[1, i].set_title(f"Likelihood ({post_l.get('lc', '')})")
        axes[1, i].set_xlabel(r"Angle ($^\circ$)")
        if i == 0:
            axes[1, i].set_ylabel(r"$P(\theta)$")
        bo_best = res.get("gpbo_best", {})
        if bo_best and "x_best" in bo_best:
            for axr in (axes[0, i], axes[1, i]):
                axr.axvline(bo_best["x_best"], color=COLORS["hard"], linewidth=0.9, linestyle=":", alpha=0.85)
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
    holdout_genuine = not cand.empty
    if cand.empty:
        # Under the row-level stratified random split every (Angle, LC) pair
        # appears in BOTH train and val, so this branch is the NORM, not the
        # exception: the targets below are NOT unseen configurations.  Warn
        # loudly so the resulting table cannot be mistaken for a held-out
        # stress test — it is a seen-configuration consistency check.
        logger.warning(
            "  Inverse stress targets: no (Angle, LC) pair is absent from the "
            "training split (row-level split ⇒ all configurations seen). "
            "Falling back to SEEN validation configurations — treat "
            "Table_inverse_stress_protocol.csv as a seen-configuration "
            "consistency check, not a held-out stress test.")
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
            # Same ground-truth-recovery convention as the T targets.
            "source_angle": key[0],
            "source_lc": key[1],
            "holdout_genuine": bool(holdout_genuine),
            "rationale": ("Held-out (Angle, LC) stress inverse" if holdout_genuine
                          else "SEEN-configuration consistency check (row-level split: no unseen pairs)"),
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
        rec_theta = best.get("x_best", "")
        src_theta = t.get("val_angle_deg", "")
        rec_lc = best.get("lc", best.get("best_lc", ""))
        rows.append({
            "Stress_ID": t["id"],
            "source_val_angle_deg": src_theta,
            "source_val_LC": t.get("val_LC", ""),
            # Honesty flag: under the row-level random split every (Angle, LC)
            # appears in training, so these are usually SEEN configurations.
            "config_unseen_in_train": "Yes" if t.get("holdout_genuine") else "No",
            "target_EA_J": t["EA"],
            "target_IPF_kN": t["IPF"],
            "recovered_theta_deg": rec_theta,
            "recovered_LC": rec_lc,
            "delta_theta_deg": (f"{abs(float(rec_theta) - float(src_theta)):.2f}"
                                if rec_theta != "" and src_theta != "" else ""),
            "LC_match": ("Yes" if str(rec_lc) == str(t.get("val_LC", "")) else "No") if rec_lc else "",
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
    """Extra GP-BO runs: no classifier penalty (ablation of the only regularizer in J)."""
    if not targets or CFG.dry_run or not HAS_SKOPT:
        return
    baseline_by_id = {
        r["target_info"]["id"]: r for r in baseline_results
        if r.get("target_info") and "id" in r["target_info"]
    }
    rows = []
    variants = [
        ("no_classifier_penalty", {"prob_weight_override": 0.0}),
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
) -> None:
    """Emit Q1-oriented inverse CSVs and likelihood figure."""
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
        getattr(CFG, "run_robustness_analyses", True)
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


def compute_hypervolume_2d(
    pareto_front_df: pd.DataFrame,
    ref_ea: float, ref_ipf: float,
    logger: logging.Logger,
    label: str = "Dominance"
) -> float:
    """Compute 2D hypervolume of a Pareto front relative to a reference (nadir) point.

    Objectives: maximize EA, minimize IPF.  Returns hypervolume in J*kN units.
    """
    if pareto_front_df.empty:
        return 0.0

    pts = pareto_front_df[["EA", "IPF"]].values.copy()
    mask = (pts[:, 0] > ref_ea) & (pts[:, 1] < ref_ipf)
    pts = pts[mask]
    if len(pts) == 0:
        return 0.0

    pts = pts[pts[:, 0].argsort()]
    hv = 0.0
    prev_ea = ref_ea
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


def compute_design_variance_decomposition(landscape_df: pd.DataFrame,
                                          logger: logging.Logger,
                                          output_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Global interpretability: exact variance (Sobol/ANOVA) attribution of the
    surrogate's design-level outputs to the design factors.

    The dense landscape grid is a balanced full factorial (every θ × every LC),
    so the two-way ANOVA decomposition is EXACT on the grid — no sampling, no
    surrogate-of-the-surrogate:

        Var(Y) = V_θ + V_LC + V_int
        S_θ  = V_θ  / Var(Y)   (main effect of the interior angle)
        S_LC = V_LC / Var(Y)   (main effect of the loading case)
        S_int = interaction (the design factors are not additive)

    Interpretation convention: θ ~ uniform over the swept range, LC ~ uniform
    over {LC1, LC2}.  This answers, quantitatively and faithfully to the
    trained model, the first question an engineer asks of the forward
    surrogate: *which design factor controls EA, and which controls IPF?*
    """
    if landscape_df is None or not len(landscape_df):
        return None
    rows = []
    for col in ("EA", "IPF", "EA_full"):
        if col not in landscape_df.columns:
            continue
        pivot = landscape_df.pivot_table(index="angle", columns="lc", values=col)
        if pivot.isna().any().any() or pivot.shape[1] < 2:
            continue  # unbalanced grid — decomposition would not be exact
        Y = pivot.values.astype(float)
        mu = Y.mean()
        a = Y.mean(axis=1) - mu                       # θ main effect
        b = Y.mean(axis=0) - mu                       # LC main effect
        resid = Y - mu - a[:, None] - b[None, :]      # interaction
        v_theta = float(np.mean(a ** 2))
        v_lc = float(np.mean(b ** 2))
        v_int = float(np.mean(resid ** 2))
        v_tot = v_theta + v_lc + v_int
        if v_tot <= 1e-18:
            continue
        rows.append({
            "Quantity": (f"EA@{EA_COMMON_MM_TAG}" if col == "EA" else col),
            "S_theta": round(v_theta / v_tot, 4),
            "S_LC": round(v_lc / v_tot, 4),
            "S_interaction": round(v_int / v_tot, 4),
            "total_variance": f"{v_tot:.4g}",
            "grid": f"{Y.shape[0]} angles x {Y.shape[1]} LCs",
        })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    for _, r in df.iterrows():
        logger.info(f"  Variance attribution [{r['Quantity']}]: "
                    f"θ={r['S_theta']:.1%}, LC={r['S_LC']:.1%}, "
                    f"interaction={r['S_interaction']:.1%}")
    if output_dir is not None:
        path = os.path.join(output_dir, "Table_design_variance_decomposition.csv")
        df.to_csv(path, index=False)
        logger.info(f"  Saved: {os.path.basename(path)}")
    return df


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
            plaus = m.get("plausibility", {})
            landscape.append({
                "angle": ang, "lc": lc,
                "EA": m["EA"], "IPF": m["IPF"],
                "EA_full": ea_full,
                "EA_norm": ea_norm, "IPF_norm": ipf_norm,
                "EA_std": m.get("EA_std", 0), "IPF_std": m.get("IPF_std", 0),
                # Inference-time plausibility diagnostics per sweep point.
                "neg_force_frac": plaus.get("neg_force_frac", 0.0),
                "nonmono_energy_frac": plaus.get("nonmono_energy_frac", 0.0),
                "neg_ea_frac": plaus.get("neg_ea_frac", 0.0),
                "ipf_fallback_frac": plaus.get("ipf_fallback_frac", 0.0),
            })

    landscape_df = pd.DataFrame(landscape)

    # Publish the deployment-time physics-constraint audit: what fraction of
    # sweep points (and ensemble members) violate F>=0 / monotone E / EA>0,
    # and how often IPF came from the peak-detection fallback.  Reviewers can
    # then see the soft physics constraints HOLD at deployment, not just that
    # they were penalized during training.
    if output_dir is not None and len(landscape_df):
        try:
            plaus_cols = ["neg_force_frac", "nonmono_energy_frac", "neg_ea_frac", "ipf_fallback_frac"]
            summ_rows = []
            for lc in lc_categories:
                sub = landscape_df[landscape_df["lc"] == lc]
                row = {"LC": lc, "n_sweep_points": len(sub)}
                for c in plaus_cols:
                    row[f"mean_{c}"] = f"{float(sub[c].mean()):.4f}"
                    row[f"pts_any_{c}"] = int((sub[c] > 0).sum())
                summ_rows.append(row)
            pd.DataFrame(summ_rows).to_csv(
                os.path.join(output_dir, "Table_physical_plausibility_audit.csv"), index=False)
            logger.info("  Saved: Table_physical_plausibility_audit.csv")
            n_bad = int((landscape_df[["neg_force_frac", "nonmono_energy_frac", "neg_ea_frac"]].sum(axis=1) > 0).sum())
            if n_bad:
                logger.warning(f"  Plausibility audit: {n_bad}/{len(landscape_df)} sweep points have "
                               f"at least one member violating a physics constraint — inspect "
                               f"Table_physical_plausibility_audit.csv")
            else:
                logger.info(f"  Plausibility audit: all {len(landscape_df)} sweep points physically "
                            f"plausible across every ensemble member")
        except Exception as exc:
            logger.warning(f"  plausibility audit table skipped: {exc}")
        # Global interpretability: which design factor drives EA / IPF (exact
        # ANOVA on the balanced sweep grid).
        try:
            compute_design_variance_decomposition(landscape_df, logger, output_dir)
        except Exception as exc:
            logger.warning(f"  variance decomposition skipped: {exc}")
    
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
                        "EA_std": c.get("EA_std", 0.0), "IPF_std": c.get("IPF_std", 0.0),
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
    
    # ----- LC selection diagnostic on the weighted-sum Pareto front -----
    # Two complementary views: count of α-sweep solutions per LC, and count
    # of Pareto-dominance front entries per LC.  Both are reported so the
    # paper can cite the right one depending on whether the claim is about
    # the convex hull (weighted-sum) or the full non-dominated set.
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
    pareto_df.attrs["lc_counts_alpha_sweep"] = dict(lc_counts)

    if not pareto_front_df.empty:
        front_lc_counts = pareto_front_df["lc"].value_counts().to_dict()
        logger.info(
            f"  Pareto-dominance front composition: {front_lc_counts} "
            f"({sum(front_lc_counts.values())} non-dominated points total)"
        )
        pareto_df.attrs["lc_counts_dominance_front"] = front_lc_counts

    # ----- Per-LC hypervolume contribution -----
    # How much of the front's hypervolume would be lost if each LC were
    # excluded? Computed by re-running compute_hypervolume_2d on the
    # LC-restricted sub-front and reporting the deficit relative to the full HV.
    try:
        full_hv_for_diag = compute_hypervolume_2d(
            pareto_front_df, landscape_df["EA"].min(), landscape_df["IPF"].max(), logger,
        )
    except Exception:
        full_hv_for_diag = float("nan")
    hv_by_lc: Dict[str, float] = {}
    if not pareto_front_df.empty and np.isfinite(full_hv_for_diag):
        for lc in sorted(pareto_front_df["lc"].unique()):
            sub = pareto_front_df[pareto_front_df["lc"] == lc]
            try:
                hv_lc = compute_hypervolume_2d(
                    sub, landscape_df["EA"].min(), landscape_df["IPF"].max(),
                    logger, label=f"weighted-sum:{lc}-only",
                )
            except Exception:
                hv_lc = float("nan")
            hv_by_lc[lc] = float(hv_lc)
        logger.info(f"  Per-LC hypervolume contribution: {hv_by_lc} "
                    f"(full = {full_hv_for_diag:.4f})")
        pareto_df.attrs["hypervolume_by_lc_weighted_sum"] = hv_by_lc

    # ----- Explicit LC dominance audit -----
    # For every Pareto-dominance-front point in the majority LC, find the
    # best counterfactual in the OTHER LC at the same-or-better IPF and at
    # the same-or-better EA.  This makes the LC-dominance claim
    # quantitative: instead of "LC2 dominates" (an observation), each row
    # cites "LC1's best counter-EA at IPF <= X is Y vs LC2's Z (gap of dEA
    # = Z-Y J)".  Written to Table_pareto_lc_dominance_audit.csv.
    audit_rows: List[Dict] = []
    if (
        not pareto_front_df.empty
        and "lc" in pareto_front_df.columns
        and len(lc_categories) > 1
    ):
        for _, row in pareto_front_df.iterrows():
            this_lc = row["lc"]
            other_lcs = [lc for lc in lc_categories if lc != this_lc]
            for other_lc in other_lcs:
                other_pool = landscape_df[landscape_df["lc"] == other_lc]
                if other_pool.empty:
                    audit_rows.append({
                        "pareto_lc":          this_lc,
                        "pareto_angle":       float(row["angle"]),
                        "pareto_EA":          float(row["EA"]),
                        "pareto_IPF":         float(row["IPF"]),
                        "counter_lc":         other_lc,
                        "counter_best_EA":    None,
                        "counter_best_IPF":   None,
                        "counter_best_angle": None,
                        "EA_gap_vs_pareto":   None,
                        "IPF_gap_vs_pareto":  None,
                        "dominates_pareto":   False,
                    })
                    continue
                # Best counter at IPF ≤ Pareto-IPF: maximize counter EA.
                feasible_at_ipf = other_pool[other_pool["IPF"] <= float(row["IPF"]) + 1e-12]
                if not feasible_at_ipf.empty:
                    best_at_ipf = feasible_at_ipf.loc[feasible_at_ipf["EA"].idxmax()]
                else:
                    # No counter has IPF ≤ Pareto-IPF; report the absolute
                    # minimum-IPF counter for context.
                    best_at_ipf = other_pool.loc[other_pool["IPF"].idxmin()]
                # Best counter at EA ≥ Pareto-EA: minimize counter IPF.
                feasible_at_ea = other_pool[other_pool["EA"] >= float(row["EA"]) - 1e-12]
                if not feasible_at_ea.empty:
                    best_at_ea = feasible_at_ea.loc[feasible_at_ea["IPF"].idxmin()]
                else:
                    best_at_ea = other_pool.loc[other_pool["EA"].idxmax()]
                dominates = bool(
                    (float(best_at_ipf["EA"]) > float(row["EA"]) + 1e-12)
                    and (float(best_at_ipf["IPF"]) < float(row["IPF"]) - 1e-12)
                )
                audit_rows.append({
                    "pareto_lc":          this_lc,
                    "pareto_angle":       float(row["angle"]),
                    "pareto_EA":          float(row["EA"]),
                    "pareto_IPF":         float(row["IPF"]),
                    "counter_lc":         other_lc,
                    "counter_best_EA":    float(best_at_ipf["EA"]),
                    "counter_best_IPF":   float(best_at_ipf["IPF"]),
                    "counter_best_angle": float(best_at_ipf["angle"]),
                    "EA_gap_vs_pareto":   float(best_at_ipf["EA"]) - float(row["EA"]),
                    "IPF_gap_vs_pareto":  float(best_at_ipf["IPF"]) - float(row["IPF"]),
                    "dominates_pareto":   dominates,
                })
    if audit_rows:
        audit_df = pd.DataFrame(audit_rows)
        audit_path = os.path.join(output_dir, "Table_pareto_lc_dominance_audit.csv")
        audit_df.to_csv(audit_path, index=False)
        logger.info(f"  Saved: Table_pareto_lc_dominance_audit.csv "
                    f"({len(audit_rows)} (Pareto point, counter-LC) pairs)")
        pareto_df.attrs["lc_dominance_audit"] = audit_df
        n_dominated_by_counter = int(sum(r["dominates_pareto"] for r in audit_rows))
        if n_dominated_by_counter == 0:
            logger.info(
                "  LC dominance audit: every Pareto point is *not* "
                "dominated by any counterfactual in the other LC.  "
                f"{dominant_lc} dominance is rigorous, not an artifact of "
                "coarse sampling."
            )
        else:
            logger.warning(
                f"  LC dominance audit: {n_dominated_by_counter} Pareto "
                f"point(s) ARE dominated by a counter-LC alternative.  "
                "Review Table_pareto_lc_dominance_audit.csv — the "
                "convex-hull weighted-sum sweep missed Pareto-optimal "
                "designs in the other LC."
            )

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
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    
    # Retrieve calibration for the ±2σ bands.  PRIORITY: the design-level
    # EA/IPF *scalar* factors (calibrated on |EA_pred−EA_exp|/σ_EA over the
    # measured designs) — the statistically matched estimand for these bands.
    # The earlier fallback (pointwise-energy conformal factor applied to the
    # EA scalar) mixes estimands: it is retained only as a legacy fallback
    # and is logged as heuristic when used.
    cf_ea = 1.0
    cf_ipf = 1.0
    scal = (calibration or {}).get("ea_ipf_scalar") if isinstance(calibration, dict) else None
    if scal:
        cf_ea = float(scal.get("ea_cf_2sigma", 1.0))
        cf_ipf = float(scal.get("ipf_cf_2sigma", 1.0))
        logger.info(f"  Design-space ±2σ bands using design-level scalar calibration: "
                    f"EA cf={cf_ea:.3f}, IPF cf={cf_ipf:.3f}")
    elif calibration is not None:
        for proto in ["random", "unseen"]:
            if proto in calibration and "hard" in calibration[proto]:
                cal_h = calibration[proto]["hard"]
                cf_ea = cal_h.get(
                    "energy_conformal_factor_2sigma",
                    cal_h.get("conformal_factor_2sigma",
                    cal_h.get("conformal_factor", 1.0)),
                )
                cf_ipf = cal_h.get(
                    "conformal_factor_2sigma",
                    cal_h.get("conformal_factor", 1.0),
                )
                logger.warning(f"  Design-space ±2σ bands: no design-level scalar calibration "
                               f"available; falling back to {proto} POINTWISE factors "
                               f"(EA cf={cf_ea:.3f}, IPF cf={cf_ipf:.3f}) — heuristic, "
                               f"estimand-mismatched")
                break
    
    # Prepare data for heatmaps
    angles = landscape_df["angle"].unique()
    lc_list = sorted(landscape_df["lc"].unique())
    
    # Panel (a): EA vs Angle for both LCs
    ax1 = fig.add_subplot(gs[0, 0])
    for lc in lc_list:
        lc_data = landscape_df[landscape_df["lc"] == lc].sort_values("angle")
        color = COLORS["LC1"] if lc == "LC1" else COLORS["LC2"]
        linestyle = "-" if lc == "LC1" else "--"
        marker = "o" if lc == "LC1" else "s"
        label = f"{lc} (Stable)" if lc == "LC1" else f"{lc} (Progressive)"
        ax1.plot(lc_data["angle"], lc_data["EA"], color=color, linestyle=linestyle, 
                marker=marker, markersize=4, markevery=50, linewidth=2, label=label)
        # Add conformal-calibrated uncertainty band
        if "EA_std" in lc_data.columns and lc_data["EA_std"].max() > 0:
            # cf_ea is the 2σ-coverage conformal factor; band is ±cf_ea·σ.
            ax1.fill_between(lc_data["angle"],
                            np.maximum(0.0, lc_data["EA"] - cf_ea*lc_data["EA_std"]),
                            lc_data["EA"] + cf_ea*lc_data["EA_std"],
                            color=color, alpha=0.15)
    ax1.set_xlabel("Interior Angle θ (°)")
    ax1.set_ylabel(f"Energy Absorption EA (J) @ $d$={D_COMMON:.0f} mm")
    ax1.set_title("(a) Energy Absorption Landscape")
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Panel (b): IPF vs Angle for both LCs
    ax2 = fig.add_subplot(gs[0, 1])
    for lc in lc_list:
        lc_data = landscape_df[landscape_df["lc"] == lc].sort_values("angle")
        color = COLORS["LC1"] if lc == "LC1" else COLORS["LC2"]
        linestyle = "-" if lc == "LC1" else "--"
        marker = "o" if lc == "LC1" else "s"
        label = f"{lc} (Stable)" if lc == "LC1" else f"{lc} (Progressive)"
        ax2.plot(lc_data["angle"], lc_data["IPF"], color=color, linestyle=linestyle,
                marker=marker, markersize=4, markevery=50, linewidth=2, label=label)
        if "IPF_std" in lc_data.columns and lc_data["IPF_std"].max() > 0:
            # cf_ipf is the 2σ-coverage conformal factor; band is ±cf_ipf·σ.
            ax2.fill_between(lc_data["angle"],
                            np.maximum(0.0, lc_data["IPF"] - cf_ipf*lc_data["IPF_std"]),
                            lc_data["IPF"] + cf_ipf*lc_data["IPF_std"],
                            color=color, alpha=0.15)
    ax2.set_xlabel("Interior Angle θ (°)")
    ax2.set_ylabel("Initial Peak Force IPF (kN)")
    ax2.set_title("(b) Peak Force Landscape")
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
    
    im3 = ax3.imshow(Z1, aspect='auto', cmap='cividis', origin='lower',
                     extent=[angles_lc1.min(), angles_lc1.max(), 0, 1])
    ax3.set_xlabel("Interior Angle θ (°)")
    ax3.set_ylabel("EA Weight α")
    ax3.set_title("(c) Trade-off Surface: LC1")
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
    
    im4 = ax4.imshow(Z2, aspect='auto', cmap='cividis', origin='lower',
                     extent=[angles_lc2.min(), angles_lc2.max(), 0, 1])
    ax4.set_xlabel("Interior Angle θ (°)")
    ax4.set_ylabel("EA Weight α")
    ax4.set_title("(d) Trade-off Surface: LC2")
    cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.72, pad=0.02)
    cbar4.set_label("Objective J (lower=better)")
    
    # Panel (e): Pareto front in EA-IPF space
    ax5 = fig.add_subplot(gs[1, 1])
    # Background: all design points
    for lc in lc_list:
        lc_data = landscape_df[landscape_df["lc"] == lc]
        color = COLORS["LC1"] if lc == "LC1" else COLORS["LC2"]
        ax5.scatter(lc_data["EA"], lc_data["IPF"], c=color, s=5, alpha=0.15, label=f"{lc} designs")
    
    # Pareto points
    pareto_lc1 = pareto_df[pareto_df["lc"] == "LC1"]
    pareto_lc2 = pareto_df[pareto_df["lc"] == "LC2"]
    
    if len(pareto_lc1) > 0:
        ax5.scatter(pareto_lc1["EA"], pareto_lc1["IPF"], c=COLORS["LC1"], s=150, marker='*',
                   edgecolors='black', linewidths=1.5, zorder=10, label="Pareto (LC1)")
    if len(pareto_lc2) > 0:
        ax5.scatter(pareto_lc2["EA"], pareto_lc2["IPF"], c=COLORS["LC2"], s=150, marker='D',
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
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.6))

        # High alpha (EA priority)
        high_alpha = pareto_df[pareto_df["alpha"] == pareto_df["alpha"].max()].iloc[0]
        ax5.annotate("α=1 (Max EA)", (high_alpha["EA"], high_alpha["IPF"]),
                    xytext=(36, -28), textcoords='offset points',
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.6))
    
    ax5.set_xlabel(f"Energy Absorption EA (J) @ $d$={D_COMMON:.0f} mm")
    ax5.set_ylabel("Initial Peak Force IPF (kN)")
    ax5.set_title("(e) Pareto Front: EA vs IPF Trade-off")
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
    colors = [COLORS["LC1"] if lc == "LC1" else COLORS["LC2"] for lc in pareto_df["lc"]]
    ax6_twin.bar(pareto_df["alpha"], lc_numeric, width=0.08, alpha=0.4, color=colors)
    ax6_twin.set_ylabel("Loading (LC)", color='gray')
    ax6_twin.set_ylim(0.5, 2.5)
    ax6_twin.set_yticks([1, 2])
    ax6_twin.set_yticklabels(["LC1", "LC2"])
    ax6_twin.tick_params(axis='y', labelcolor='gray')
    
    ax6.set_title("(f) Optimal Design vs Priority Weight")
    ax6.grid(True, alpha=0.3, linestyle='--', axis='x')
    
    # Add annotation boxes
    ax6.axvspan(0, 0.3, alpha=0.1, color='blue', label='IPF priority zone')
    ax6.axvspan(0.7, 1.0, alpha=0.1, color='red', label='EA priority zone')
    
    fig.suptitle("Multi-Objective Crashworthiness Optimization: EA vs IPF Trade-off")
    
    fig.savefig(os.path.join(output_dir, "Fig_multiobjective_heatmaps.png"), 
               dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("  Saved: Fig_multiobjective_heatmaps.png")


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def add_subplot_label(ax, label, dx=-7.0, dy=9.0):
    """Add a bold panel label "(a)" just outside the axes' top-left corner.

    Anchored to the corner (0, 1) in axes-fraction coordinates with a *points*
    offset and right/bottom alignment, so the label grows up-and-left into the
    empty margin.  This clears the y-tick labels (which sit below the top spine,
    regardless of their width) and any in-axes upper-left legend, fixing the
    text-overlap seen with the previous in-margin placement.  Clipping is
    disabled and ``savefig(bbox_inches='tight')`` expands to include it.
    """
    ax.annotate(f"({label})", xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(dx, dy), textcoords="offset points",
                fontweight='bold', va='bottom', ha='right',
                clip_on=False, annotation_clip=False)


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
                ax.hist(errors, bins=30, color=COLORS[approach], edgecolor='black',
                        alpha=0.75, linewidth=0.8, hatch=HATCHES.get(approach, ""))
                ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
                ax.set_xlabel(f"{xlabel} ({unit})")
                ax.set_ylabel("Frequency")
                ax.set_title(f"{MODEL_LABELS[approach]}: μ={np.mean(errors):.3f}, σ={np.std(errors):.3f}")
                add_subplot_label(ax, labels[label_idx])
                label_idx += 1
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
        fig.suptitle(f"Residual Distributions ({protocol_label(protocol)})")
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
            ax.set_title(f"{protocol_label(protocol)}")
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            add_subplot_label(ax, chr(ord('a') + row * 2 + col))
    
    fig.suptitle("Ensemble Performance Distribution")
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
                # Contrasting colour + high zorder so the 1:1 line stays visible
                # against the Hard-PINN markers (which are near-black).
                ax.plot(lims, lims, color='#d62728', linestyle='--', lw=1.5, zorder=5, label='Perfect fit')
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
        fig.suptitle(f"Parity Plots ({protocol_label(protocol)})")
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
                       mpatches.Patch(facecolor=COLORS["hard"], edgecolor='black', label='Hard-PINN')]
    # Only advertise protocols that were actually plotted; otherwise a legend
    # entry (e.g. "Random Split") appears with no matching bars in the chart.
    if "random" in dual_results:
        legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='black', hatch='', label='Random Split'))
    if "unseen" in dual_results:
        legend_elements.append(mpatches.Patch(facecolor='white', edgecolor='black', hatch='//', label='Unseen Angle'))
    # No suptitle: the previous one was drawn behind the top-anchored legend and
    # bled through it as a grey ghost.  The figure caption carries the title.
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements),
               bbox_to_anchor=(0.5, 1.02), frameon=True, columnspacing=0.8, handletextpad=0.35)
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
    
    # Retrieve conformal factors (load and energy) per approach.
    # Use the **2σ-coverage** factor (95.4-percentile of |residual|/σ) so the
    # band drawn as ``mean ± cf * σ`` achieves the nominal 95% coverage.
    # Using cf_2sigma directly avoids the under-coverage that arises from
    # linearly scaling a 1σ factor by 2 under heavy-tailed residuals.
    conformal_factors = {}
    for approach in ["ddns", "soft", "hard"]:
        cf_load = 1.0
        cf_energy = 1.0
        if (calibration is not None and "unseen" in calibration
                and approach in calibration["unseen"]):
            cal = calibration["unseen"][approach]
            cf_load = cal.get("conformal_factor_2sigma", cal.get("conformal_factor", 1.0))
            cf_energy = cal.get(
                "energy_conformal_factor_2sigma",
                cal.get("energy_conformal_factor", 1.0),
            )
            logger.info(f"    {MODEL_LABELS.get(approach, approach)} cf_2sigma: "
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
                # cf is the 2σ-coverage conformal factor; band is ±cf·σ (NOT ±2·cf·σ).
                ax.fill_between(disps, np.maximum(0.0, Fm - cf*Fs), Fm + cf*Fs,
                                color=COLORS[approach], alpha=0.15, linewidth=0)
        ax.set_xlabel("Displacement (mm)")
        ax.set_ylabel("Load (kN)")
        ax.set_title(f"{lc}, $\\theta$ = 60° (Unseen Angle)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, framealpha=0.95)
        ax.set_xlim(0, disp_end)
        add_subplot_label(ax, chr(ord('a') + idx))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    fig.suptitle("Load Predictions for Unseen Angle (Ensemble Mean, Conformal ±2σ)")
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
                # cf is the 2σ-coverage conformal factor; band is ±cf·σ.
                ax.fill_between(disps, np.maximum(0.0, Em - cf*Es), Em + cf*Es,
                                color=COLORS[approach], alpha=0.15, linewidth=0)
        ax.set_xlabel("Displacement (mm)")
        ax.set_ylabel("Energy (J)")
        ax.set_title(f"{lc}, $\\theta$ = 60° (Unseen Angle)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, framealpha=0.95)
        ax.set_xlim(0, disp_end)
        add_subplot_label(ax, chr(ord('a') + idx))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    fig.suptitle("Energy Predictions for Unseen Angle (Ensemble Mean, Conformal ±2σ)")
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
    fig, axes = plt.subplots(len(lcs), len(angles), figsize=(3.0 * len(angles), 3.0 * len(lcs) + 0.6))
    
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
            ax.set_title(f"{lc}, {ang}°", pad=2)
            if j == 0:
                ax.set_ylabel("Load (kN)")
            if i == len(lcs) - 1:
                ax.set_xlabel("Disp. (mm)")
            ax.set_xlim(0, disp_end)  # [CHANGE A] LC-specific
    
    # Create legend elements
    legend_elements = [Line2D([0], [0], color='k', linewidth=1.2, label='Experiment')]
    legend_elements += [Line2D([0], [0], color=COLORS[a], linestyle=LINESTYLES[a], linewidth=1.0, label=MODEL_LABELS[a]) for a in ["ddns", "soft", "hard"]]
    
    # Shared legend hung below the panels — a top legend overlapped the top-row
    # subplot titles.  bbox_inches='tight' expands the canvas to include it.
    fig.tight_layout()
    fig.legend(handles=legend_elements, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, -0.005), frameon=True, framealpha=0.95)
    
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
    ax.text(0.06, 0.92, summary_text, transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='black', pad=0.35))
    add_subplot_label(ax, 'd')
    fig.suptitle("Ablation Study: Effect of Physics Weight (Unseen θ=60°)")
    fig.savefig(os.path.join(output_dir, "Fig_ablation_study.png"), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("  Saved: Fig_ablation_study.png")


def fig_bo_convergence(opt_results: Dict, output_dir: str, logger: logging.Logger, tag: str = ""):
    """Generate GP-BO convergence figure (2x2 grid per inverse target).

    Panels:
      (a) Best objective vs iteration (running min).
      (b) Sampled angle θ vs iteration, coloured by LC.
      (c) Per-iteration objective + running-min trace.
      (d) Sampled (θ, J) scatter with the optimum highlighted.
    """
    gpbo = opt_results.get("gpbo_best")
    if not gpbo:
        return

    x_history = gpbo.get("x_history", [])
    y_history = gpbo.get("y_history", [])
    best_y_history = gpbo.get("best_y_history", [])

    if len(x_history) > 0:
        if isinstance(x_history[0], (tuple, list)):
            theta_history = [float(x[0]) for x in x_history]
            lc_history = [int(x[1]) for x in x_history]
        else:
            theta_history = [float(x) for x in x_history]
            lc_history = None
    else:
        theta_history = []
        lc_history = None

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # (a) Best objective over iterations
    ax = axes[0, 0]
    ax.plot(range(1, len(best_y_history) + 1), best_y_history,
            color=COLORS["gpbo"], marker='o', markersize=4, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Objective")
    add_subplot_label(ax, 'a')

    # (b) Theta over iterations
    ax = axes[0, 1]
    lc_colors = {0: COLORS["LC1"], 1: COLORS["LC2"]}
    if lc_history is not None:
        for i, (th, lc) in enumerate(zip(theta_history, lc_history)):
            ax.scatter(i + 1, th, c=lc_colors.get(lc, COLORS["soft"]),
                       s=30, edgecolors='black', linewidths=0.3, zorder=5)
        ax.plot(range(1, len(theta_history) + 1), theta_history,
                color='gray', linewidth=0.5, alpha=0.5, zorder=1)
    else:
        ax.plot(range(1, len(theta_history) + 1), theta_history,
                color=COLORS["soft"], marker='D', markersize=4, linewidth=1.0)

    best_theta = gpbo.get("x_best", theta_history[-1] if theta_history else 0)
    if isinstance(best_theta, (tuple, list)):
        best_theta = float(best_theta[0])
    ax.axhline(best_theta, color=COLORS["hard"], linestyle='--', linewidth=1.5,
               label=fr"$\theta^*$ = {best_theta:.1f}°")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Angle $\theta$ (°)")
    ax.legend(loc='best')
    add_subplot_label(ax, 'b')

    # (c) All evaluations and best so far
    ax = axes[1, 0]
    ax.scatter(range(1, len(y_history) + 1), y_history, c=COLORS["ddns"], s=30,
               alpha=0.7, edgecolors='black', linewidths=0.5)
    ax.plot(range(1, len(best_y_history) + 1), best_y_history,
            color=COLORS["hard"], linewidth=2.0, label='Best so far')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective Value")
    ax.legend(loc='best')
    add_subplot_label(ax, 'c')

    # (d) Theta vs Objective landscape
    ax = axes[1, 1]
    if lc_history is not None:
        for i, (th, y, lc) in enumerate(zip(theta_history, y_history, lc_history)):
            marker = '*' if lc == 0 else 'D'
            ax.scatter(th, y, c=lc_colors.get(lc, COLORS["soft"]),
                       s=40, marker=marker, alpha=0.7,
                       edgecolors='black', linewidths=0.3, zorder=5)
    else:
        ax.scatter(theta_history, y_history, c=COLORS["soft"], s=30,
                   alpha=0.7, edgecolors='black', linewidths=0.5,
                   label='Evaluations')

    best_y = gpbo.get("y_best", None)
    if best_y is None:
        if isinstance(y_history, np.ndarray):
            best_y = float(np.min(y_history)) if y_history.size > 0 else 0.0
        else:
            best_y = float(min(y_history)) if len(y_history) > 0 else 0.0
    else:
        best_y = float(best_y)
    ax.scatter([best_theta], [best_y], c='#DC143C', marker='*', s=250,
               zorder=10, edgecolors='black', linewidths=0.8, label='Optimum')
    ax.set_xlabel(r"Angle $\theta$ (°)")
    ax.set_ylabel("Objective Value")
    ax.legend(loc='best')
    add_subplot_label(ax, 'd')

    t_ea = opt_results.get('target_ea', float('nan'))
    t_ipf = opt_results.get('target_ipf', float('nan'))
    fig.suptitle(
        f"GP-BO Optimization: Target EA@{D_COMMON:.0f}mm = {t_ea:.1f} J, "
        f"IPF = {t_ipf:.2f} kN",
    )
    suffix = f"_{tag}" if tag else ""
    fig.savefig(os.path.join(output_dir, f"Fig_bo_convergence{suffix}.png"),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"  Saved: Fig_bo_convergence{suffix}.png")


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
    lc_colors = {"LC1": COLORS["LC1"], "LC2": COLORS["LC2"]}
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
            # Fewer than 8 snapshots: show each once.  (Padding by repeating the
            # last index rendered the final iteration two or three times as
            # visually identical panels.)
            snapshot_indices = list(range(n_iters))

        # Clamp indices to valid range
        snapshot_indices = [min(idx, n_iters - 1) for idx in snapshot_indices]
        
        # Create 2x4 figure (extra height for multi-line subplot titles + bottom legend)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for ax_idx, snap_idx in enumerate(snapshot_indices[:8]):
            ax = axes[ax_idx]
            snap = snapshots[snap_idx]
            theta_grid = snap["theta_grid"]
            
            # Plot each LC from the SAME snapshot (same GP model)
            for lc in lc_list:
                color = lc_colors.get(lc, COLORS["LC1"])
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
            ax.set_title(f"Iter {iter_num}/{total_evals}\n({obs_str})")
            ax.set_xlim(45, 70)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        # Hide any unused panels (when fewer than 8 snapshots are available) so
        # the grid does not show blank or repeated axes.
        for extra_ax in axes[len(snapshot_indices):]:
            extra_ax.set_visible(False)

        # Shared legend hung just below the grid: loc='upper center' with a y
        # slightly under 0 keeps it clear of the bottom row's x-axis tick labels,
        # and bbox_inches='tight' expands the canvas to include it.
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3,
                   bbox_to_anchor=(0.5, -0.01), frameon=True, framealpha=0.95, columnspacing=0.6)
        
        title_suffix = f" ({tag})" if tag else ""
        fig.suptitle(f"GP-BO Posterior Evaluation{title_suffix}")
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
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
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
            
            color = lc_colors.get(lc, COLORS["LC1"])
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
        ax.set_title(f"Iter {snap_idx + 1}/{total_iters}")
        ax.set_xlim(45, 70)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, 0.02), frameon=True, framealpha=0.95, columnspacing=0.6)
    
    title_suffix = f" ({tag})" if tag else ""
    fig.suptitle(f"GP-BO Posterior Evaluation{title_suffix}")
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
    """GP-BO inverse-design summary across all targets.

    A compact 1x2 panel: (a) best-so-far objective convergence overlaid for
    every target, and (b) the converged (final) objective per target.  This
    replaces the previous one-column-per-target grid, whose per-panel
    single-bar charts (only GP-BO is run) read as uninformative filled boxes;
    per-target convergence detail lives in ``Fig_inverse_convergence_T*``.
    """
    if not all_inverse_results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), squeeze=False)
    ax_conv, ax_bar = axes[0, 0], axes[0, 1]
    cmap = plt.get_cmap("viridis", max(2, len(all_inverse_results)))

    tids, finals, fcolors = [], [], []
    conv_lines = 0
    for i, result in enumerate(all_inverse_results):
        tid = result.get("target_info", {}).get("id", f"T{i + 1}")
        best = result.get("gpbo_best", {})
        color = cmap(i)
        hist = best.get("best_y_history", [])
        if len(hist):
            ax_conv.plot(range(1, len(hist) + 1), hist, color=color, linewidth=1.8, label=tid)
            conv_lines += 1
        if "y_best" in best:
            tids.append(tid)
            finals.append(float(best["y_best"]))
            fcolors.append(color)

    # (a) convergence overlay
    ax_conv.set_xlabel("Function evaluations")
    ax_conv.set_ylabel("Best objective (lower is better)")
    ax_conv.set_title("GP-BO convergence (all targets)")
    ax_conv.xaxis.set_minor_locator(AutoMinorLocator())
    ax_conv.yaxis.set_minor_locator(AutoMinorLocator())
    # Only draw the target legend when convergence traces were actually plotted
    # (a best-so-far history may be absent), so no empty legend box appears.
    if 0 < conv_lines <= 8:
        ax_conv.legend(title="Target", fontsize=8, loc="upper right", framealpha=0.92)
    elif conv_lines == 0:
        ax_conv.text(0.5, 0.5, "convergence history unavailable",
                     ha="center", va="center", transform=ax_conv.transAxes,
                     fontsize=9, color="0.5")
    add_subplot_label(ax_conv, "a")

    # (b) converged objective per target
    if tids:
        x = np.arange(len(tids))
        bars = ax_bar.bar(x, finals, color=fcolors, edgecolor="black", linewidth=0.7, width=0.7)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(tids)
        ax_bar.set_xlabel("Target")
        ax_bar.set_ylabel("Final objective")
        ax_bar.set_title("Converged objective per target")
        for b, val in zip(bars, finals):
            ax_bar.annotate(f"{val:.1e}", (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=8, fontweight="bold",
                            xytext=(0, 2), textcoords="offset points")
        ax_bar.margins(y=0.18)
    add_subplot_label(ax_bar, "b")

    fig.suptitle("GP-BO inverse design summary")
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
        # Best-so-far is a monotone staircase; draw it as a step (not a diagonal
        # interpolation) and mark each evaluation, so a trace that converges early
        # reads as "optimum found at eval k" rather than a featureless flat line.
        ax.step(x, yb, where="post", color=color, linestyle=ls, linewidth=1.8, label=label)
        ax.scatter(x, yb, color=color, s=22, zorder=3, edgecolor="black", linewidth=0.4)
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
        ax.annotate(t["id"], (t["EA"], t["IPF"]), textcoords="offset points", xytext=(4, 4))

    if ea_col == "EA_common":
        ax.set_xlabel(f"Energy absorbed to {D_COMMON:.0f} mm (J)")
    else:
        ax.set_xlabel(f"Energy Absorption, EA (J) @ $d$={D_COMMON:.0f} mm")
    ax.set_ylabel("Initial Peak Force, IPF (kN)")
    ax.legend(loc='best', framealpha=0.95)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.set_title("Feasibility of Inverse Design Targets in Empirical EA-IPF Space")

    filepath = os.path.join(output_dir, "Fig_inverse_target_feasibility.png")
    fig.savefig(filepath, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("  Saved: Fig_inverse_target_feasibility.png")


def fig_design_space(models: List[nn.Module], approach: str, scaler_disp: StandardScaler, enc: OneHotEncoder, params: ScalingParams, output_dir: str, logger: logging.Logger,
                     df_metrics: Optional[pd.DataFrame] = None):
    """Generate design space prediction figure (EA at ``D_COMMON`` for LC-fair comparison; IPF unchanged).

    When ``df_metrics`` is provided, the experimental (EA@D_COMMON, IPF) points
    are overlaid on the surrogate sweep — the single most direct visual
    validation of the forward-design claim: the model's design-space curves
    must pass through (or near) the measured designs.
    """
    angles = np.linspace(CFG.angle_opt_min, CFG.angle_opt_max, 26)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for lc, marker, ls, color in [("LC1", "o", "-", COLORS["soft"]), ("LC2", "s", "--", COLORS["ddns"])]:
        EA_vals, IPF_vals, EA_std, IPF_std = [], [], [], []
        for ang in angles:
            m = compute_ea_ipf_ensemble(
                models, approach, ang, lc, scaler_disp, enc, params, d_eval=D_COMMON)
            EA_vals.append(m["EA"])
            IPF_vals.append(m["IPF"])
            EA_std.append(float(m.get("EA_std", 0.0)))
            IPF_std.append(float(m.get("IPF_std", 0.0)))
        EA_vals = np.asarray(EA_vals); IPF_vals = np.asarray(IPF_vals)
        EA_std = np.asarray(EA_std);   IPF_std = np.asarray(IPF_std)
        axes[0].plot(angles, EA_vals, linestyle=ls, marker=marker, markersize=5, linewidth=1.5, label=lc, color=color, markerfacecolor='white', markeredgecolor=color)
        # ±1σ ensemble band (epistemic spread across the M members).
        axes[0].fill_between(angles, EA_vals - EA_std, EA_vals + EA_std, color=color, alpha=0.16, linewidth=0)
        axes[1].plot(angles, IPF_vals, linestyle=ls, marker=marker, markersize=5, linewidth=1.5, label=lc, color=color, markerfacecolor='white', markeredgecolor=color)
        axes[1].fill_between(angles, IPF_vals - IPF_std, IPF_vals + IPF_std, color=color, alpha=0.16, linewidth=0)
        # Experimental design points for this LC (EA@D_COMMON + IPF).
        if df_metrics is not None and len(df_metrics):
            sub = df_metrics[df_metrics["LC"].astype(str) == lc]
            ea_col = "EA_common" if "EA_common" in sub.columns else None
            if len(sub):
                if ea_col is not None:
                    axes[0].scatter(sub["Angle"].astype(float), sub[ea_col].astype(float),
                                    s=70, marker="*", color=color, edgecolor="black",
                                    linewidth=0.7, zorder=5,
                                    label=f"{lc} experiment")
                if "IPF" in sub.columns:
                    axes[1].scatter(sub["Angle"].astype(float), sub["IPF"].astype(float),
                                    s=70, marker="*", color=color, edgecolor="black",
                                    linewidth=0.7, zorder=5,
                                    label=f"{lc} experiment")
    axes[0].set_xlabel("Angle $\\theta$ (°)")
    axes[0].set_ylabel(f"Energy absorption EA (J) @ $d$={D_COMMON:.0f} mm")
    axes[0].legend(loc='best', fontsize=9)
    add_subplot_label(axes[0], 'a')
    axes[1].set_xlabel("Angle $\\theta$ (°)")
    axes[1].set_ylabel("Initial Peak Force (kN)")
    axes[1].legend(loc='best', fontsize=9)
    add_subplot_label(axes[1], 'b')
    fig.suptitle("Design Space Predictions (Hard-PINN; shaded ±1σ ensemble)")
    fig.savefig(os.path.join(output_dir, "Fig_design_space.png"), dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("  Saved: Fig_design_space.png")


def table_forward_design_errors(models: List[nn.Module], approach: str,
                                scaler_disp: StandardScaler, enc: OneHotEncoder,
                                params: ScalingParams, df_metrics: pd.DataFrame,
                                output_dir: str, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Design-level forward validation: predicted vs experimental EA/IPF per design.

    This is the quantitative companion to the Fig_design_space overlay — for
    each measured (θ, LC) design it reports the surrogate's EA@D_COMMON and
    IPF against the experimental values with percent errors, plus per-LC and
    overall MAPE.  Pointwise load/energy R² (Table 1) says the curves fit;
    THIS table says the *design-level engineering quantities* that forward and
    inverse design actually operate on are predicted correctly.
    """
    if df_metrics is None or not len(df_metrics):
        return None
    dm = df_metrics.copy()
    if "EA_common" not in dm.columns:
        logger.warning("  forward design-error table skipped: df_metrics lacks EA_common")
        return None
    rows = []
    for _, r in dm.iterrows():
        ang, lc = float(r["Angle"]), str(r["LC"])
        m = compute_ea_ipf_ensemble(models, approach, ang, lc, scaler_disp, enc, params,
                                    d_eval=D_COMMON)
        ea_exp, ipf_exp = float(r["EA_common"]), float(r["IPF"])
        ea_pred, ipf_pred = float(m["EA"]), float(m["IPF"])
        rows.append({
            "Angle_deg": f"{ang:.0f}",
            "LC": lc,
            f"EA@{EA_COMMON_MM_TAG}_exp_J": f"{ea_exp:.2f}",
            f"EA@{EA_COMMON_MM_TAG}_pred_J": f"{ea_pred:.2f}",
            "EA_pred_std_J": f"{float(m.get('EA_std', 0.0)):.2f}",
            "EA_error_pct": f"{abs(ea_pred - ea_exp) / max(abs(ea_exp), 1e-12) * 100:.2f}",
            "IPF_exp_kN": f"{ipf_exp:.3f}",
            "IPF_pred_kN": f"{ipf_pred:.3f}",
            "IPF_pred_std_kN": f"{float(m.get('IPF_std', 0.0)):.3f}",
            "IPF_error_pct": f"{abs(ipf_pred - ipf_exp) / max(abs(ipf_exp), 1e-12) * 100:.2f}",
        })
    df = pd.DataFrame(rows)
    # Per-LC + overall MAPE summary rows.
    for scope, sub in [("LC1", df[df["LC"] == "LC1"]), ("LC2", df[df["LC"] == "LC2"]), ("ALL", df)]:
        if len(sub):
            rows.append({
                "Angle_deg": f"MAPE_{scope}",
                "LC": scope,
                "EA_error_pct": f"{sub['EA_error_pct'].astype(float).mean():.2f}",
                "IPF_error_pct": f"{sub['IPF_error_pct'].astype(float).mean():.2f}",
            })
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Table_forward_design_errors.csv")
    out.to_csv(path, index=False)
    logger.info(f"  Saved: {os.path.basename(path)} (design-level EA/IPF validation, "
                f"{len(dm)} designs)")
    return out


def compute_ea_ipf_scalar_calibration(models: List[nn.Module], approach: str,
                                      scaler_disp: StandardScaler, enc: OneHotEncoder,
                                      params: ScalingParams, df_metrics: pd.DataFrame,
                                      output_dir: Optional[str], logger: logging.Logger) -> Optional[Dict]:
    """Calibrate DESIGN-LEVEL (EA, IPF) uncertainty against the measured designs.

    The pointwise-energy conformal factor is calibrated on per-displacement
    energy residuals and is statistically unrelated to the error distribution
    of the EA *scalar* (an integral) or the IPF *scalar* (a peak) — applying
    it to EA_std/IPF_std, as the heatmaps previously did, mixes estimands.
    This computes the factor at the right level: z_i = |pred_i − exp_i| / σ_i
    over the measured designs, and returns the 95th-percentile z as the
    multiplier that makes ±cf·σ a nominal-95% design-level band.

    Caveat (stated for reviewers): with only ~12 designs the factor is coarse,
    and the same designs are in the full-data surrogate's training set, so
    this calibrates *in-sample* design-level spread — bands at unmeasured θ
    remain heuristic.  That is still strictly better than reusing a
    pointwise-energy factor for a scalar quantity.
    """
    if df_metrics is None or not len(df_metrics) or "EA_common" not in df_metrics.columns:
        return None
    z_ea, z_ipf, rows = [], [], []
    for _, r in df_metrics.iterrows():
        ang, lc = float(r["Angle"]), str(r["LC"])
        m = compute_ea_ipf_ensemble(models, approach, ang, lc, scaler_disp, enc, params,
                                    d_eval=D_COMMON)
        ea_exp, ipf_exp = float(r["EA_common"]), float(r["IPF"])
        ea_sig = max(float(m.get("EA_std", 0.0)), 1e-9)
        ipf_sig = max(float(m.get("IPF_std", 0.0)), 1e-9)
        ze = abs(float(m["EA"]) - ea_exp) / ea_sig
        zi = abs(float(m["IPF"]) - ipf_exp) / ipf_sig
        z_ea.append(ze); z_ipf.append(zi)
        rows.append({"Angle_deg": f"{ang:.0f}", "LC": lc,
                     "z_EA": f"{ze:.3f}", "z_IPF": f"{zi:.3f}"})
    cal = {
        "ea_cf_2sigma": float(np.percentile(z_ea, 95.4)),
        "ipf_cf_2sigma": float(np.percentile(z_ipf, 95.4)),
        "ea_cf_1sigma": float(np.percentile(z_ea, 68.3)),
        "ipf_cf_1sigma": float(np.percentile(z_ipf, 68.3)),
        "n_designs": len(rows),
    }
    logger.info(f"  EA/IPF scalar calibration over {len(rows)} designs: "
                f"cf_2σ(EA)={cal['ea_cf_2sigma']:.2f}, cf_2σ(IPF)={cal['ipf_cf_2sigma']:.2f}")
    if output_dir is not None:
        df = pd.DataFrame(rows)
        for k, v in cal.items():
            df.attrs[k] = v
        summary = pd.DataFrame([{"Angle_deg": "cf_2sigma", "LC": "",
                                 "z_EA": f"{cal['ea_cf_2sigma']:.3f}",
                                 "z_IPF": f"{cal['ipf_cf_2sigma']:.3f}"}])
        pd.concat([df, summary], ignore_index=True).to_csv(
            os.path.join(output_dir, "Table_ea_ipf_scalar_calibration.csv"), index=False)
        logger.info("  Saved: Table_ea_ipf_scalar_calibration.csv")
    return cal


def fig_separable_interpretability(models: List[nn.Module],
                                   enc: OneHotEncoder, params: ScalingParams,
                                   output_dir: str, logger: logging.Logger,
                                   tag: str = "") -> Optional[str]:
    """Faithful-by-construction interpretability readout of the separable Hard-PINN.

    The separable variant IS its own explanation: E(d, θ, LC) =
    Σ_k φ_k(θ, LC)·(B_k(d) − B_k(0)).  This figure plots exactly those learned
    components — no post-hoc approximation:

      (a) displacement basis functions B_k(d) − B_k(0): the k crush "modes"
          the model composes (units: normalized energy × σ_E gives J);
      (b) design coefficients φ_k(θ) per LC: how each mode's contribution
          varies across the design space — by construction first-order
          Fourier in θ, so the printed trend is the model's entire
          θ-dependence, not a slice of a black box.

    Plots the first separable member solid with remaining members faint
    (basis indices are not aligned across members — each member's
    decomposition is individually faithful).
    """
    sep_models = [m for m in models if isinstance(m, SeparableHardEnergyNet)]
    if not sep_models:
        return None
    set_publication_style()
    device = next(sep_models[0].parameters()).device
    K = sep_models[0].n_basis
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, K))

    d_raw = np.linspace(0.0, max(disp_end_mm("LC1"), disp_end_mm("LC2")), 200)
    d_s = torch.tensor(((d_raw - params.mu_d) / max(1e-12, params.sig_d)).reshape(-1, 1),
                       dtype=torch.float32, device=device)
    d0_s = torch.full((1, 1), -params.mu_d / max(1e-12, params.sig_d),
                      dtype=torch.float32, device=device)
    thetas = np.linspace(CFG.angle_opt_min, CFG.angle_opt_max, 101)
    lc_cats = [str(c) for c in enc.categories_[0].tolist()]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for mi, net in enumerate(sep_models):
        net.eval()
        alpha = 1.0 if mi == 0 else 0.25
        with torch.no_grad():
            B = (net.basis(d_s) - net.basis(d0_s)).cpu().numpy()      # (200, K)
            for k in range(K):
                axes[0].plot(d_raw, B[:, k] * params.sig_E, color=colors[k],
                             lw=1.8 if mi == 0 else 1.0, alpha=alpha,
                             label=f"$B_{{{k + 1}}}$" if mi == 0 else None)
            for lc_i, (lc, ls) in enumerate(zip(lc_cats, ("-", "--"))):
                onehot = np.zeros((len(thetas), len(lc_cats)), dtype=np.float32)
                onehot[:, lc_i] = 1.0
                z = torch.tensor(np.column_stack([
                    np.sin(np.deg2rad(thetas)), np.cos(np.deg2rad(thetas)), onehot,
                ]).astype(np.float32), device=device)
                phi = net.phi(z)
                if net.monotone:
                    phi = nn.functional.softplus(phi)
                phi = phi.cpu().numpy()
                for k in range(K):
                    axes[1].plot(thetas, phi[:, k], ls, color=colors[k],
                                 lw=1.8 if mi == 0 else 1.0, alpha=alpha,
                                 label=(f"$\\varphi_{{{k + 1}}}$ ({lc})"
                                        if (mi == 0 and k < 2) else None))
    axes[0].set_xlabel("Displacement $d$ (mm)")
    axes[0].set_ylabel("Basis energy $B_k(d)-B_k(0)$ ($\\times\\sigma_E$, J)")
    axes[0].set_title("(a) Learned crush-mode basis functions")
    axes[0].legend(fontsize=9, ncol=2)
    axes[0].grid(True, alpha=0.3, linestyle="--")
    axes[1].set_xlabel("Interior angle $\\theta$ (°)")
    axes[1].set_ylabel("Design coefficient $\\varphi_k(\\theta,\\mathrm{LC})$")
    axes[1].set_title("(b) Mode activation across the design space")
    axes[1].legend(fontsize=8, ncol=2, title="solid=LC1, dashed=LC2")
    axes[1].grid(True, alpha=0.3, linestyle="--")
    fig.suptitle("Separable Hard-PINN decomposition: $E=\\sum_k \\varphi_k(\\theta,\\mathrm{LC})\\,[B_k(d)-B_k(0)]$")
    suffix = f"_{tag}" if tag else ""
    out = os.path.join(output_dir, f"Fig_separable_interpretability{suffix}.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"  Saved: {os.path.basename(out)}")
    return out


def table_null_baseline_design_level(df_all: pd.DataFrame, theta_star: float,
                                     output_dir: str, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Design-level null model: 1-D interpolation of experimental EA(θ) / IPF(θ).

    The natural competitor for design-level forward prediction with a single
    geometric variable is simply interpolating the measured EA and IPF over θ
    (per LC) — no learning at all.  This table holds out θ* (the unseen-angle
    protocol's angle), linearly interpolates EA@D_COMMON(θ) and IPF(θ) through
    the remaining angles, and reports the null model's error at θ*.  The
    surrogate's design-level errors (Table_forward_design_errors.csv and the
    unseen-protocol results) must be judged against THIS baseline — pointwise
    R² against tree/GP baselines does not answer the design-level question.
    """
    try:
        dm = compute_design_space_metrics(df_all, logger)
        enrich_df_metrics_ea_common(dm, df_all, logger=logger)
    except Exception as exc:
        logger.warning(f"  null-baseline table skipped: {exc}")
        return None
    if "EA_common" not in dm.columns or not len(dm):
        return None
    rows = []
    for lc in sorted(dm["LC"].astype(str).unique()):
        sub = dm[dm["LC"].astype(str) == lc].sort_values("Angle")
        angs = sub["Angle"].astype(float).values
        if theta_star not in angs or len(angs) < 3:
            continue
        train_mask = angs != theta_star
        for col, unit in [("EA_common", "J"), ("IPF", "kN")]:
            y = sub[col].astype(float).values
            y_true = float(y[~train_mask][0])
            y_pred = float(np.interp(theta_star, angs[train_mask], y[train_mask]))
            rows.append({
                "LC": lc,
                "Quantity": f"EA@{EA_COMMON_MM_TAG} ({unit})" if col == "EA_common" else f"{col} ({unit})",
                "theta_star_deg": f"{theta_star:.0f}",
                "Experimental": f"{y_true:.3f}",
                "Null_interp_pred": f"{y_pred:.3f}",
                "Null_error_pct": f"{abs(y_pred - y_true) / max(abs(y_true), 1e-12) * 100:.2f}",
                "Note": "linear interpolation of experimental values over θ (no model)",
            })
    if not rows:
        logger.warning("  null-baseline table skipped: θ* not in measured angle grid")
        return None
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Table_null_baseline_design_level.csv")
    df.to_csv(path, index=False)
    logger.info(f"  Saved: {os.path.basename(path)} — the design-level accuracy floor the "
                f"surrogates must beat at θ*={theta_star:.0f}°")
    return df


def fig_pareto_tradeoff(df_pareto: pd.DataFrame, output_dir: str, logger: logging.Logger):
    """Generate Pareto trade-off figure with per-LC conditional fronts.
    
    Uses dark, high-contrast colours for all lines, labels, and axes.
    Avoids overlapping markers via offset and distinct styles.
    """
    if df_pareto.empty:
        return
    
    pareto_by_lc = df_pareto.attrs.get("pareto_by_lc", {})
    has_per_lc = len(pareto_by_lc) > 0
    
    # ---- colour palette (Wong-derived, B/W-safe via lightness contrast) ----
    C_GLOBAL = COLORS["hard"]   # black, global Pareto front anchor
    C_LC1    = COLORS["LC1"]    # deep blue
    C_LC2    = COLORS["LC2"]    # vermillion
    C_EA     = COLORS["gpbo"]   # bluish-green
    C_IPF    = "#CC79A7"        # reddish-purple, Wong-palette auxiliary distinct from EA
    
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
    # ±1σ ensemble error bars on each Pareto-optimal design (EA and IPF).
    if "EA_std" in pareto_sorted.columns and "IPF_std" in pareto_sorted.columns:
        ax.errorbar(pareto_sorted["EA"], pareto_sorted["IPF"],
                    xerr=pareto_sorted["EA_std"], yerr=pareto_sorted["IPF_std"],
                    fmt="none", ecolor=C_GLOBAL, elinewidth=0.8, capsize=2, alpha=0.5, zorder=4)
    # Recommended "knee": front point nearest the utopia (max EA, min IPF) in
    # normalised objective space — an actionable single-design takeaway.
    _ea = pareto_sorted["EA"].to_numpy(dtype=float)
    _ipf = pareto_sorted["IPF"].to_numpy(dtype=float)
    if _ea.size >= 3 and np.ptp(_ea) > 0 and np.ptp(_ipf) > 0:
        nE = (_ea - _ea.min()) / np.ptp(_ea)
        nI = (_ipf - _ipf.min()) / np.ptp(_ipf)
        k = int(np.argmin(np.hypot(1.0 - nE, nI)))
        ax.scatter([_ea[k]], [_ipf[k]], s=280, marker="*", facecolor="#F0E442",
                   edgecolor="black", linewidth=1.4, zorder=7, label="Recommended (knee)")
        ax.annotate("recommended", (_ea[k], _ipf[k]), textcoords="offset points",
                    xytext=(6, 8), fontsize=8, fontweight="bold", zorder=8)
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
    
    fig.suptitle("Multi-Objective EA vs IPF Trade-off")
    fig.savefig(os.path.join(output_dir, "Fig_pareto_tradeoff.png"),
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info("  Saved: Fig_pareto_tradeoff.png")


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
    fig.suptitle("Training Convergence")
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
    ax.margins(y=0.12)
    add_subplot_label(ax, 'a')
    # Offset-points annotation (scale-independent): an absolute data-unit offset
    # placed the label far outside the axes on the small-valued time panel.
    for bar, val in zip(bars, n_params):
        ax.annotate(f'{val:,}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', xytext=(0, 3), textcoords='offset points')
    ax = axes[1]
    bars = ax.bar(x, train_times, color=colors_bar, edgecolor='black', linewidth=1.2, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[a] for a in approaches])
    ax.set_ylabel("Training Time (s)")
    ax.margins(y=0.12)
    add_subplot_label(ax, 'b')
    for bar, val in zip(bars, train_times):
        ax.annotate(f'{val:.1f}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', xytext=(0, 3), textcoords='offset points')
    fig.suptitle("Model Complexity Comparison")
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
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
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
    ax.set_title("Physics Residual Distribution")
    ax.legend(loc='upper right')
    ax.set_yscale('log')
    # FIX: Set x-axis limits with small padding to prevent overlap with y-axis
    ax.set_xlim(-max_residual * 0.02, max_residual * 1.05)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    add_subplot_label(ax, 'a')
    
    # Panel (b): dE/dd vs F_actual scatter for each model
    ax = axes[1]
    
    # Plot in reverse order so Hard-PINN (most important) is on top.
    # Local RNG to avoid polluting the global numpy seed state mid-pipeline.
    _subsample_rng = np.random.default_rng(42)
    for approach in ["ddns", "soft", "hard"]:
        if approach in residual_data:
            dEdd = residual_data[approach]["dE_dd"]
            F = residual_data[approach]["F_pred"]
            # Subsample for visibility
            idx = _subsample_rng.choice(len(dEdd), min(500, len(dEdd)), replace=False)
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
    ax.set_title("Constraint Satisfaction: dE/dd vs F$_{pred}$")
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
    ax.set_title("Physics Violation Magnitude")
    ax.set_yscale("log")
    
    # FIX: Set y-limits with MORE headroom to prevent annotation overlap with border
    ymin = 1e-13  # Small but visible on log scale
    ymax = max(means_plot.max(), 1) * 20  # Much more headroom for annotations
    ax.set_ylim(ymin, ymax)
    
    # FIX: Add value labels with better positioning
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
            fontweight=fontweight,
            color=text_color
        )
    
    add_subplot_label(ax, "c")
    
    fig.suptitle("Thermodynamic Consistency Verification: F$_{pred}$ = dE/dd")
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
            "load_r2": r2_safe(y_val_load, pred_load),
            "energy_r2": r2_safe(y_val_energy, pred_energy),
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
            "load_r2": r2_safe(y_val_load, pred_load),
            "energy_r2": r2_safe(y_val_energy, pred_energy),
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
            "load_r2": r2_safe(y_val_load, pred_load),
            "energy_r2": r2_safe(y_val_energy, pred_energy),
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
                idx = np.random.default_rng(CFG.seed).choice(len(X_train), max_gp_samples, replace=False)
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
                "load_r2": r2_safe(y_val_load, pred_load),
                "energy_r2": r2_safe(y_val_energy, pred_energy),
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
            # Ensemble member spread -> error bars on the PINN bars (baselines
            # are single models, so they carry no ensemble std).
            _mem = dual_results[protocol][approach].get("member_metrics", [])
            _lr = [mm["load_r2"] for mm in _mem]
            _er = [mm["energy_r2"] for mm in _mem]
            _dd = 1 if len(_lr) > 1 else 0
            all_results[MODEL_LABELS[approach]] = {
                "load_r2": m["load_r2"],
                "energy_r2": m["energy_r2"],
                "load_r2_std": float(np.std(_lr, ddof=_dd)) if _lr else 0.0,
                "energy_r2_std": float(np.std(_er, ddof=_dd)) if _er else 0.0,
                "load_rmse": m["load_rmse"],
                "energy_rmse": m["energy_rmse"],
                "load_mae": m.get("load_mae", 0),
                "energy_mae": m.get("energy_mae", 0),
                "train_time": dual_results[protocol][approach]["avg_training_time"],
                "n_params": dual_results[protocol][approach]["n_params"]
            }
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Baseline ML models use Wong-palette auxiliaries (B/W-safe via lightness).
    # PINN families keep their canonical COLORS assignments.
    _BASELINE_COLORS = {
        "Linear":          "#56B4E9",   # sky blue
        "RandomForest":    "#009E73",   # bluish green
        "XGBoost":         "#E69F00",   # orange
        "GaussianProcess": "#CC79A7",   # reddish purple
        "DDNS":            COLORS["ddns"],
        "Soft-PINN":       COLORS["soft"],
        "Hard-PINN":       COLORS["hard"],
    }
    models = list(all_results.keys())
    colors_list = [_BASELINE_COLORS.get(m, COLORS["data"]) for m in models]
    
    x = np.arange(len(models))
    
    # Panel (a): Load R²  (PINN bars carry ±1σ ensemble error bars)
    ax = axes[0]
    load_r2 = [all_results[m]["load_r2"] for m in models]
    load_r2_err = [all_results[m].get("load_r2_std", 0.0) for m in models]
    bars = ax.bar(x, load_r2, yerr=load_r2_err, color=colors_list, edgecolor='black',
                  linewidth=1, capsize=3, error_kw={'linewidth': 1.0})
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=40, ha='right')
    ax.set_ylabel("Load R²")
    ax.set_ylim(0, 1.05)
    add_subplot_label(ax, 'a')

    # Panel (b): Energy R²
    ax = axes[1]
    energy_r2 = [all_results[m]["energy_r2"] for m in models]
    energy_r2_err = [all_results[m].get("energy_r2_std", 0.0) for m in models]
    bars = ax.bar(x, energy_r2, yerr=energy_r2_err, color=colors_list, edgecolor='black',
                  linewidth=1, capsize=3, error_kw={'linewidth': 1.0})
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
    
    fig.suptitle(f"Baseline Model Comparison ({ptitle})")
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
                # Soft-PINN uses soft (penalty) BC, not architectural — match train_soft.
                model.configure_zero_bc(params, enabled=False)
                opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
                mse = nn.MSELoss()
                sl1 = nn.SmoothL1Loss(beta=cfg["smoothl1_beta"])
                colloc_sampler = create_collocation_sampler(train_df, scaler_disp, enc)
                rng = np.random.default_rng(CFG.seed)
                
                loader = DataLoader(
                    TensorDataset(Xtr, ytr), batch_size=cfg["batch_size"], shuffle=True,
                    **_data_loader_kwargs(seed=CFG.seed),
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
                
                load_r2 = r2_safe(y_val[:, 0], Fv)
                energy_r2 = r2_safe(y_val[:, 1], Ev)
                
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
        
        im = ax.imshow(matrix, cmap='viridis', aspect='auto', vmin=0.5, vmax=1.0)
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
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)
        
        plt.colorbar(im, ax=ax, shrink=0.8)
        add_subplot_label(ax, chr(ord('a') + idx))
    
    ptitle = f" ({tag})" if tag else ""
    fig.suptitle(f"Hyperparameter Sensitivity Analysis (Soft-PINN{ptitle})")
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
    """Run inverse design with multiple outer seeds for robustness analysis.

    Two layers of randomization combine:
      - **Inner multi-start** (``gp_bo_minimize_joint_multistart``, n_bo_restarts):
        within one ``run_inverse_design`` call, GP-BO is restarted from
        different acquisition seeds; the best is kept.
      - **Outer multi-seed** (this function, n_seeds): the entire inner
        multi-start GP-BO is repeated with ``n_seeds`` different ``BOConfig.seed``
        values. The cross-seed spread is the published robustness statistic
        ("σ_θ across seeds" in Section 5.8).

    Output schema mirrors ``gp_bo_minimize_joint_multistart.restart_summary``
    so downstream consumers and tables can treat both paths uniformly.
    ``target_ea`` is EA at ``D_COMMON`` (displacement-fair vs. full-stroke EA).
    """
    logger.info(f"    Running multi-seed inverse design (n_seeds={n_seeds})...")

    results_by_method = {"gpbo": []}

    for seed_offset in range(n_seeds):
        seed = CFG.seed + seed_offset * 100

        # Preserve full BOConfig (multi-start count, etc.); only vary seed.
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
        "n_seeds": n_seeds,
    }

    for method in ["gpbo"]:
        if results_by_method[method]:
            x_vals = np.array([r["x_best"] for r in results_by_method[method]], dtype=np.float64)
            y_vals = np.array([r["y_best"] for r in results_by_method[method]], dtype=np.float64)
            ea_errs = np.array([r["ea_error_pct"] for r in results_by_method[method]], dtype=np.float64)
            ipf_errs = np.array([r["ipf_error_pct"] for r in results_by_method[method]], dtype=np.float64)

            # Legacy keys (existing consumers).
            aggregated[f"{method}_x_mean"] = float(np.mean(x_vals))
            aggregated[f"{method}_x_std"] = float(np.std(x_vals, ddof=1)) if x_vals.size > 1 else 0.0
            aggregated[f"{method}_y_mean"] = float(np.mean(y_vals))
            aggregated[f"{method}_y_std"] = float(np.std(y_vals, ddof=1)) if y_vals.size > 1 else 0.0
            aggregated[f"{method}_ea_err_mean"] = float(np.mean(ea_errs))
            aggregated[f"{method}_ea_err_std"] = float(np.std(ea_errs, ddof=1)) if ea_errs.size > 1 else 0.0
            aggregated[f"{method}_ipf_err_mean"] = float(np.mean(ipf_errs))
            aggregated[f"{method}_ipf_err_std"] = float(np.std(ipf_errs, ddof=1)) if ipf_errs.size > 1 else 0.0

            # Unified restart_summary schema (matches gp_bo_minimize_joint_multistart).
            n_calls_per_restart = int(getattr(bo_cfg, "n_calls_total", 30))
            n_inner = int(getattr(bo_cfg, "n_bo_restarts", 5))
            aggregated[f"{method}_restart_summary"] = {
                "scope": "outer_multi_seed",
                "n_outer_seeds": int(n_seeds),
                "n_inner_restarts_per_seed": n_inner,
                "n_calls_per_restart": n_calls_per_restart,
                "total_surrogate_calls": int(n_seeds * n_inner * n_calls_per_restart),
                "theta_best_mean": float(np.mean(x_vals)),
                "theta_best_std":  float(np.std(x_vals, ddof=1)) if x_vals.size > 1 else 0.0,
                "theta_best_min":  float(np.min(x_vals)),
                "theta_best_max":  float(np.max(x_vals)),
                "y_best_mean":     float(np.mean(y_vals)),
                "y_best_std":      float(np.std(y_vals, ddof=1)) if y_vals.size > 1 else 0.0,
                "y_best_min":      float(np.min(y_vals)),
                "y_best_max":      float(np.max(y_vals)),
            }

    return aggregated


def generate_inverse_robustness_table(robust_results: List[Dict], output_dir: str, logger: logging.Logger):
    """Generate inverse design robustness table.

    Columns mirror ``Table3_inverse_illposedness.csv`` so the two tables can
    be read interchangeably: same nomenclature for cross-seed spread, with
    explicit cost accounting (``total_surrogate_calls``).
    """
    if not robust_results:
        return

    COLS = [
        f"Target_EA@{EA_COMMON_MM_TAG}", "Target_IPF",
        "Method", "N_outer_seeds", "N_inner_restarts_per_seed",
        "N_calls_per_restart", "Total_surrogate_calls",
        "Angle_mean_std", "Theta_best_min", "Theta_best_max",
        "Objective_mean_std", "Y_best_min", "Y_best_max",
        "EA_err_mean_std", "IPF_err_mean_std",
    ]
    rows = []
    for res in robust_results:
        for method in ["gpbo"]:
            if f"{method}_x_mean" in res:
                rs = res.get(f"{method}_restart_summary", {})
                rows.append({
                    f"Target_EA@{EA_COMMON_MM_TAG}": f"{res.get('target_ea', 0):.3f}",
                    "Target_IPF": f"{res['target_ipf']:.3f}",
                    "Method": method.upper(),
                    "N_outer_seeds": rs.get("n_outer_seeds", res.get("n_seeds", "")),
                    "N_inner_restarts_per_seed": rs.get("n_inner_restarts_per_seed", ""),
                    "N_calls_per_restart": rs.get("n_calls_per_restart", ""),
                    "Total_surrogate_calls": rs.get("total_surrogate_calls", ""),
                    "Angle_mean_std": f"{res[f'{method}_x_mean']:.2f} ± {res[f'{method}_x_std']:.2f}",
                    "Theta_best_min": f"{rs.get('theta_best_min', float('nan')):.2f}",
                    "Theta_best_max": f"{rs.get('theta_best_max', float('nan')):.2f}",
                    "Objective_mean_std": f"{res[f'{method}_y_mean']:.4f} ± {res[f'{method}_y_std']:.4f}",
                    "Y_best_min": f"{rs.get('y_best_min', float('nan')):.6f}",
                    "Y_best_max": f"{rs.get('y_best_max', float('nan')):.6f}",
                    "EA_err_mean_std": f"{res[f'{method}_ea_err_mean']:.2f} ± {res[f'{method}_ea_err_std']:.2f}%",
                    "IPF_err_mean_std": f"{res[f'{method}_ipf_err_mean']:.2f} ± {res[f'{method}_ipf_err_std']:.2f}%",
                })

    df = pd.DataFrame(rows, columns=COLS)
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

            # ---- SPLIT-CONFORMAL calibration (curve-level split) -----------
            # The factors c1/c2 (68.3 / 95.4 percentiles of |residual|/sigma)
            # must be FIT on residuals disjoint from those used to REPORT
            # coverage, otherwise "corrected coverage" is true by construction.
            # The exchangeable unit here is the curve (residuals within one
            # load-displacement curve are strongly dependent), so validation
            # curves are split into a calibration half and a report half.
            # For the unseen protocol (2 curves: LC1/LC2 at θ*) this means
            # calibrate-on-one-curve / report-on-the-other.  When curve
            # identity is unavailable, fall back to in-sample calibration and
            # say so loudly.
            val_df_p = dual_results[protocol].get("val_df")
            calib_mask = None
            if val_df_p is not None and len(val_df_p) == len(residuals):
                curve_keys = list(zip(val_df_p["Angle"].astype(float),
                                      val_df_p["LC"].astype(str)))
                uniq = sorted(set(curve_keys))
                if len(uniq) >= 2:
                    calib_curves = set(uniq[0::2])   # deterministic alternation
                    calib_mask = np.array([k in calib_curves for k in curve_keys])
            def _fit_and_report(res, sig):
                """Fit factors on the calib half, report coverage on the rest."""
                if calib_mask is not None and 0 < calib_mask.sum() < len(res):
                    fit_res, fit_sig = res[calib_mask], sig[calib_mask]
                    rep_res, rep_sig = res[~calib_mask], sig[~calib_mask]
                    split_ok = True
                else:
                    fit_res, fit_sig = res, sig
                    rep_res, rep_sig = res, sig
                    split_ok = False
                nr = fit_res / fit_sig
                c1 = float(np.percentile(nr, 68.3))
                c2 = float(np.percentile(nr, 95.4))
                cov1 = np.array([float(np.mean(rep_res < s * c1 * rep_sig)) for s in sigma_levels])
                cov2 = np.array([float(np.mean(rep_res < s * c2 * rep_sig)) for s in sigma_levels])
                # In-sample (legacy) factors for reference/back-compat analyses.
                nr_all = res / sig
                c1_ins = float(np.percentile(nr_all, 68.3))
                c2_ins = float(np.percentile(nr_all, 95.4))
                return c1, c2, cov1, cov2, c1_ins, c2_ins, split_ok
            (conformal_factor, conformal_factor_2sigma,
             corrected_coverage, corrected_coverage_2sigma,
             cf1_insample, cf2_insample, split_ok) = _fit_and_report(residuals, sigma_safe)
            if not split_ok:
                logger.warning(
                    f"  {protocol} {approach}: curve-level split unavailable — conformal "
                    f"factors are IN-SAMPLE (legacy mode); corrected coverage is "
                    f"tautological and must not be reported as evidence")

            # Also compute energy calibration if available.
            energy_cal = {}
            if "energy_std" in preds:
                e_residuals = np.abs(true_vals["energy"] - preds["energy"])
                e_sigma = np.maximum(preds["energy_std"], 1e-12)
                e_observed = np.array([float(np.mean(e_residuals < s * e_sigma)) for s in sigma_levels])
                (e_conformal, e_conformal_2, e_corrected, _e_cov2,
                 e_cf1_ins, e_cf2_ins, _e_split) = _fit_and_report(e_residuals, e_sigma)
                energy_cal = {
                    "energy_observed_coverage": e_observed,
                    "energy_conformal_factor": e_conformal,
                    "energy_conformal_factor_2sigma": e_conformal_2,
                    "energy_corrected_coverage": e_corrected,
                    "energy_conformal_factor_insample": e_cf1_ins,
                    "energy_conformal_factor_2sigma_insample": e_cf2_ins,
                }

            calibration[protocol][approach] = {
                "sigma_levels": sigma_levels,
                "expected_coverage": expected_coverage,
                "observed_coverage": observed_coverage,
                "conformal_factor": conformal_factor,
                "conformal_factor_2sigma": conformal_factor_2sigma,
                "corrected_coverage": corrected_coverage,
                "corrected_coverage_2sigma": corrected_coverage_2sigma,
                # Provenance: True => factors fit on a curve-level calibration
                # half and coverage reported on held-out curves (honest
                # split-conformal); False => legacy in-sample (tautological).
                "split_conformal": bool(split_ok),
                "conformal_factor_insample": cf1_insample,
                "conformal_factor_2sigma_insample": cf2_insample,
                # Legacy keys for backward compatibility
                "load_within_1sigma": float(observed_coverage[1]),  # index 1 = 1.0 sigma
                "load_within_2sigma": float(observed_coverage[3]),  # index 3 = 2.0 sigma
                "expected_1sigma": 0.683,
                "expected_2sigma": 0.954,
                **energy_cal,
            }

            logger.info(f"  {protocol} {approach}: {observed_coverage[1]:.1%} within +/-1sigma "
                       f"(expected: 68.3%), {observed_coverage[3]:.1%} within +/-2sigma (expected: 95.4%)")
            logger.info(f"    Conformal factor ({'split, curve-level' if split_ok else 'IN-SAMPLE'}): "
                       f"{conformal_factor:.3f} (1.0=perfectly calibrated, >1=overconfident)")
            logger.info(f"    Held-out corrected coverage: {corrected_coverage[1]:.1%} within +/-1sigma, "
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
        ax.set_title(f"{protocol_label(protocol)}")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_aspect('equal')
        ax.legend(loc='lower right', framealpha=0.95, ncol=1,
                  borderpad=0.35, labelspacing=0.35, handletextpad=0.45)
        ax.grid(True, alpha=0.3)
        add_subplot_label(ax, chr(ord('a') + ax_idx))
        ax_idx += 1
    
    fig.suptitle("Uncertainty Calibration: Reliability Diagram")
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
            # Soft-PINN: soft BC via loss penalty, NOT architectural correction.
            model.configure_zero_bc(params, enabled=False)
        else:
            model = HardEnergyNet(Xtr.shape[1], architecture, cfg.get("dropout", 0.0), cfg["softplus_beta"]).to(DEVICE)
            # Hard-PINN: bare MLP outputting E; F = dE/dd via autograd.
            # Architectural BC disabled to match the production forward
            # training architecture (which these surrogates mirror).
            model.configure_zero_bc(params, enabled=False)
        
        n_params = model.count_parameters()
        
        opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        mse = nn.MSELoss()
        sl1 = nn.SmoothL1Loss(beta=cfg["smoothl1_beta"])
        
        if approach == "soft":
            colloc_sampler = create_collocation_sampler(train_df, scaler_disp, enc)
            rng = np.random.default_rng(CFG.seed)

        loader = DataLoader(
            TensorDataset(Xtr, ytr), batch_size=cfg["batch_size"], shuffle=True,
            **_data_loader_kwargs(seed=CFG.seed),
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
        
        load_r2 = r2_safe(y_val[:, 0], Fv)
        energy_r2 = r2_safe(y_val[:, 1], Ev)
        
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
               f'{m["load_r2"]:.3f}', ha='center', va=va, fontweight='bold')
    
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
               f'{m["energy_r2"]:.3f}', ha='center', va=va, fontweight='bold')
    
    fig.suptitle("Same-Capacity Experiment: Hard-PINN vs Soft-PINN with Identical Architecture")
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
            # Soft-PINN: soft BC penalty, not architectural — match train_soft.
            model.configure_zero_bc(params, enabled=False)
            opt = optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
            mse = nn.MSELoss()
            sl1 = nn.SmoothL1Loss(beta=cfg["smoothl1_beta"])

            colloc_sampler = create_collocation_sampler(train_df, scaler_disp, enc) if cfg.get("colloc_ratio", 0) > 0 else None
            rng = np.random.default_rng(CFG.seed)

            loader = DataLoader(
                TensorDataset(Xtr, ytr), batch_size=cfg["batch_size"], shuffle=True,
                **_data_loader_kwargs(seed=CFG.seed),
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
            
            load_r2 = r2_safe(y_val[:, 0], Fv)
            energy_r2 = r2_safe(y_val[:, 1], Ev)
            
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
    fig.suptitle(f"Extended Ablation Study: Component Contributions{ptitle}")
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
            member_load = [x["load_r2"] for x in mm]
            member_energy = [x["energy_r2"] for x in mm]
            ci_load = compute_confidence_intervals(member_load)
            ci_energy = compute_confidence_intervals(member_energy)
            # All-members (pre-convergence-fence) mean, when recorded: lets
            # readers verify the fence is direction-conservative rather than
            # quietly inflating the survivor-only statistics.
            mm_all = dual_results[protocol][approach].get("member_metrics_all") or mm
            # Load_R2 / Energy_R2 are the ENSEMBLE-aggregated metrics (the headline).
            # The 95% CI is a bootstrap over the per-member R² distribution, so it
            # brackets the mean-member R² (reported alongside), not the ensemble
            # aggregate — which normally exceeds the member mean.  Reporting both
            # keeps the CI consistent with the statistic it actually describes.
            row = {"Protocol": protocol_label(protocol), "Model": MODEL_LABELS[approach],
                        "M_total": dual_results[protocol][approach].get("M_total", len(mm)),
                        "M_eff": dual_results[protocol][approach].get("M_eff", len(mm)),
                        "Load_R2": f"{m['load_r2']:.4f}",
                        "Mean_Member_Load_R2": f"{np.mean(member_load):.4f}",
                        "Mean_Member_Load_R2_all": f"{np.mean([x['load_r2'] for x in mm_all]):.4f}",
                        "Member_Load_R2_95CI": f"[{ci_load['ci_lower']:.4f}, {ci_load['ci_upper']:.4f}]",
                        "Energy_R2": f"{m['energy_r2']:.4f}",
                        "Mean_Member_Energy_R2": f"{np.mean(member_energy):.4f}",
                        "Member_Energy_R2_95CI": f"[{ci_energy['ci_lower']:.4f}, {ci_energy['ci_upper']:.4f}]",
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
    
    # Table 2: Statistical tests.
    # Welch t-test + Cohen's d are reported for completeness but flagged as
    # *descriptive* (ensemble members violate the iid assumption). The
    # bootstrap 95% CI of the mean R² difference (n_boot=10000, percentile
    # method) is the primary inferential statistic — it is non-parametric and
    # robust to dependence among members from the same data split.
    if stat_tests:
        # Explicit column order so the CSV layout is deterministic across runs.
        STAT_COLS = [
            "Protocol", "Comparison",
            "DeltaR2_load", "DeltaR2_load_CI95_lo", "DeltaR2_load_CI95_hi",
            "Bootstrap_excludes_zero",
            "Cohens_d",
            "t_statistic", "p_value_descriptive", "p_bonferroni_descriptive",
            "Significant_descriptive", "n1", "n2",
            "Note",
        ]
        rows = []
        for protocol in ["random", "unseen"]:
            if protocol not in stat_tests:
                continue
            for comp, vals in stat_tests[protocol].items():
                p_bonf = vals.get("p_bonferroni", vals["t_pvalue"])
                rows.append({
                    "Protocol": protocol_label(protocol),
                    "Comparison": comp.replace("_vs_", " vs ").upper(),
                    "DeltaR2_load": f"{vals.get('bootstrap_mean_diff', float('nan')):.4f}",
                    "DeltaR2_load_CI95_lo": f"{vals.get('bootstrap_ci_lo_95', float('nan')):.4f}",
                    "DeltaR2_load_CI95_hi": f"{vals.get('bootstrap_ci_hi_95', float('nan')):.4f}",
                    "Bootstrap_excludes_zero": "Yes" if vals.get("bootstrap_excludes_zero") else "No",
                    "Cohens_d": f"{vals['cohens_d']:.3f}",
                    "t_statistic": f"{vals['t_statistic']:.4f}",
                    "p_value_descriptive": f"{vals['t_pvalue']:.4f}",
                    "p_bonferroni_descriptive": f"{p_bonf:.4f}",
                    "Significant_descriptive": "Yes" if p_bonf < 0.05 else "No",
                    "n1": int(vals["n1"]),
                    "n2": int(vals["n2"]),
                    "Note": "Bootstrap CI is the inferential statistic; t-test is descriptive only (iid violated).",
                })
        pd.DataFrame(rows, columns=STAT_COLS).to_csv(
            os.path.join(output_dir, "Table2_statistical_tests.csv"), index=False)
        logger.info("  Saved: Table2_statistical_tests.csv")
    
    # Table 3: Inverse design results
    if all_inverse:
        rows = []
        for res in all_inverse:
            tinfo = res.get("target_info") or {}
            row = {"Target": tinfo.get("id", ""),
                   f"Target_EA@{EA_COMMON_MM_TAG}_J": f"{res.get('target_ea', 0):.3f}",
                   "Target_IPF_kN": f"{res['target_ipf']:.3f}"}
            # Ground-truth provenance: targets are drawn from observed
            # experimental rows, so the source design is known.  Reporting
            # θ_true / LC_true and the recovery error Δθ turns the table
            # from optimizer self-consistency into a verifiable round-trip.
            src_angle = tinfo.get("source_angle")
            src_lc = tinfo.get("source_lc")
            row["True_theta_deg"] = f"{src_angle:.1f}" if src_angle is not None else ""
            row["True_LC"] = src_lc or ""
            for method in ["gpbo"]:
                key = f"{method}_best"
                if key in res:
                    best = res[key]
                    prefix = method.upper()
                    rec_angle = float(best["x_best"])
                    rec_lc = best.get("lc", "")
                    row[f"{prefix}_Angle"] = f"{rec_angle:.1f}"
                    row[f"{prefix}_LC"] = rec_lc
                    if src_angle is not None:
                        row["Delta_theta_deg"] = f"{abs(rec_angle - float(src_angle)):.2f}"
                    if src_lc:
                        row["LC_match"] = "Yes" if str(rec_lc) == str(src_lc) else "No"
                    # Bound-active flag: the recovered angle sits on the search
                    # boundary, so the unconstrained optimum may lie outside
                    # the tested angle range — reviewers need to see this.
                    tol = 1e-6 + 0.05 * (CFG.angle_opt_max - CFG.angle_opt_min) / 100.0
                    at_bound = (abs(rec_angle - CFG.angle_opt_min) <= tol
                                or abs(rec_angle - CFG.angle_opt_max) <= tol)
                    row["Bound_active"] = "Yes" if at_bound else "No"
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

    # Table 5: Per-LC performance breakdown (disaggregates the aggregated R²)
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


# =============================================================================
# PIPELINE STATE SAVE / REPLOT
# =============================================================================
# After a full run, the trained models, scalers, and analysis results are
# pickled to ``pipeline_state.pkl`` in the output directory.  Reload via
# ``load_pipeline_state`` and pass to ``replot_figures_from_state`` (or run
# the CLI with ``--replot_from <path>``) to re-render every figure with a
# different style or layout, without retraining anything.
_PIPELINE_STATE_FILENAME = "pipeline_state.pkl"


def save_pipeline_state(state: Dict, output_dir: str, logger: logging.Logger) -> str:
    """Pickle figure-relevant state to ``pipeline_state.pkl`` (atomic write).

    Uses ``torch.save`` so tensors are device-aware on load: a state pickled
    from a CUDA run can be reloaded on CPU via ``map_location='cpu'``.  Trained
    ensembles are pickled in full; this works because every model class is
    defined in this module and the file is loaded against the same module
    version.
    """
    path = os.path.join(output_dir, _PIPELINE_STATE_FILENAME)
    tmp = path + ".tmp"
    torch.save(state, tmp, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    size_mb = os.path.getsize(path) / (1024.0 * 1024.0)
    logger.info(f"  Saved pipeline state ({size_mb:.1f} MB): {path}")
    return path


def load_pipeline_state(path: str, logger: Optional[logging.Logger] = None) -> Dict:
    """Inverse of :func:`save_pipeline_state` — returns the saved state dict.

    Loads tensors onto CPU regardless of the original device, so figure
    regeneration is portable between machines.  Pass the result to
    :func:`replot_figures_from_state`.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    if logger is not None:
        size_mb = os.path.getsize(path) / (1024.0 * 1024.0)
        logger.info(f"  Loaded pipeline state ({size_mb:.1f} MB): {path}")
    return state


def replot_figures_from_state(state: Dict, output_dir: str, logger: logging.Logger) -> None:
    """Re-run every figure routine from a saved pipeline state.

    Mirrors the figure sequence in :func:`run_pipeline` and skips all training
    and heavy compute.  Figures whose inputs are absent from ``state`` (e.g.
    baseline_results when --no_robustness was used in the original run) are
    skipped silently.  Apply :func:`set_publication_style` so any figure-style
    edits in this module take effect.
    """
    os.makedirs(output_dir, exist_ok=True)
    set_publication_style()

    # Conceptual schematics (data-independent) — always rendered.
    fig_framework_schematic(output_dir, logger)
    fig_architecture_schematic(output_dir, logger)

    dual_results        = state.get("dual_results")
    df_all              = state.get("df_all")
    df_metrics          = state.get("df_metrics")
    df_ablation         = state.get("df_ablation")
    calibration         = state.get("calibration", {}) or {}
    val_df_u            = state.get("val_df_u")
    scaler_disp_u       = state.get("scaler_disp_u")
    enc_u               = state.get("enc_u")
    params_u            = state.get("params_u")
    inv_models          = state.get("inv_models")
    inv_scaler_disp     = state.get("inv_scaler_disp")
    inv_enc             = state.get("inv_enc")
    inv_params          = state.get("inv_params")
    cal_ens             = state.get("cal_ens")
    clf_feat_scaler     = state.get("clf_feat_scaler")
    clf_diag            = state.get("clf_diag")
    all_inverse_results = state.get("all_inverse_results", []) or []
    jacobian_results    = state.get("jacobian_results")
    pareto_df           = state.get("pareto_df")
    landscape_df        = state.get("landscape_df")
    baseline_results_u  = state.get("baseline_results_u")
    sensitivity_df_u    = state.get("sensitivity_df_u")

    # ---- Forward model figures ----
    if dual_results is not None:
        logger.info("\n[REPLOT] Forward-model figures")
        fig_parity_plots(dual_results, output_dir, logger)
        fig_residual_histograms(dual_results, output_dir, logger)
        fig_boxplot_comparison(dual_results, output_dir, logger)
        fig_training_curves(dual_results, output_dir, logger)
        fig_cross_protocol_comparison(dual_results, output_dir, logger)
        if df_all is not None:
            fig_unseen_curves(dual_results, df_all, output_dir, logger, calibration=calibration)
            fig_random_grid_curves(dual_results, df_all, output_dir, logger)
        fig_validation_error_maps(dual_results, output_dir, logger)
        fig_qq_load_residuals(dual_results, output_dir, logger)

    # ---- Robustness / extended-analysis figures ----
    if (dual_results is not None and val_df_u is not None
            and scaler_disp_u is not None and enc_u is not None and params_u is not None):
        logger.info("[REPLOT] Physics verification")
        try:
            fig_physics_verification(dual_results, val_df_u, scaler_disp_u, enc_u, params_u, output_dir, logger)
        except Exception as e:
            logger.warning(f"  fig_physics_verification skipped: {e}")
    if baseline_results_u is not None and dual_results is not None:
        fig_baseline_comparison(baseline_results_u, dual_results, output_dir, logger, protocol="unseen")
    if sensitivity_df_u is not None:
        fig_hyperparam_sensitivity(sensitivity_df_u, output_dir, logger, tag="unseen")
    if sensitivity_df_u is not None and len(sensitivity_df_u):
        fig_penalty_weight_sensitivity(sensitivity_df_u, dual_results, output_dir, logger)
    if calibration:
        fig_reliability_diagram(calibration, output_dir, logger)

    # ---- Inverse-design figures ----
    if clf_diag:
        logger.info("[REPLOT] Classifier diagnostics")
        fig_lc_classifier_diagnostics(clf_diag, output_dir, logger)
        if cal_ens is not None and clf_feat_scaler is not None and df_metrics is not None:
            fig_classifier_decision_boundary(cal_ens, clf_feat_scaler, df_metrics, output_dir, logger)
    _clf_ab = state.get("classifier_ablation_diag")
    if _clf_ab is not None and len(_clf_ab):
        fig_classifier_effect(_clf_ab, output_dir, logger)
    _lam_df = state.get("lambda_sweep_df")
    if _lam_df is not None and len(_lam_df):
        fig_lambda_sensitivity(_lam_df, output_dir, logger, lambda_opt=state.get("lambda_opt"))

    if all_inverse_results:
        logger.info("[REPLOT] Inverse-design figures")
        for res in all_inverse_results:
            tid = res.get("target_info", {}).get("id", "")
            fig_inverse_optimizer_convergence(res, output_dir, logger, tag=tid)
            fig_bo_posterior_evaluation(res, output_dir, logger, tag=tid)
        fig_inverse_parity_uncertainty(all_inverse_results, output_dir, logger)
        if df_all is not None and inv_models is not None:
            fig_inverse_vs_nearest_experimental_curve(
                df_all, all_inverse_results, inv_models,
                inv_scaler_disp, inv_enc, inv_params, output_dir, logger,
            )
        fig_solution_landscape(all_inverse_results, output_dir, logger)
        fig_inverse_posterior(all_inverse_results, output_dir, logger)
        fig_inverse_posterior_likelihood(all_inverse_results, output_dir, logger)

    if inv_models is not None:
        fig_design_space(inv_models, "hard", inv_scaler_disp, inv_enc, inv_params, output_dir, logger,
                         df_metrics=df_metrics)
        if df_metrics is not None:
            try:
                table_forward_design_errors(inv_models, "hard", inv_scaler_disp, inv_enc,
                                            inv_params, df_metrics, output_dir, logger)
            except Exception as e:
                logger.warning(f"  forward design-error table skipped: {e}")

    if jacobian_results is not None:
        fig_forward_map_jacobian(jacobian_results, output_dir, logger)

    # ---- Multi-objective figures ----
    if pareto_df is not None and not pareto_df.empty:
        logger.info("[REPLOT] Multi-objective figures")
        fig_pareto_tradeoff(pareto_df, output_dir, logger)
        if landscape_df is not None and not landscape_df.empty:
            fig_multiobjective_heatmaps(pareto_df, landscape_df, output_dir, logger, calibration=calibration)
    if landscape_df is not None and not landscape_df.empty:
        fig_landscape_ensemble_disagreement(landscape_df, output_dir, logger)
        if inv_models is not None:
            fig_d_common_sensitivity_ea(
                inv_models, "hard", inv_scaler_disp, inv_enc, inv_params,
                landscape_df, output_dir, logger,
            )

    logger.info("\n[REPLOT] Figure regeneration complete.")


def table_inverse_design_explanation(all_inverse: List[Dict], cal_ens, feat_scaler,
                                     output_dir: str, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Inverse interpretability: decompose each recovered design's objective.

    For every target the selected design (θ*, LC*) minimized

        J = w_EA·(EA − EA_t)² + w_IPF·(IPF − IPF_t)² + λ·(−log p_LC)

    This table reports the three terms AT the recovered optimum, the LC
    plausibility p_LC, each fit term re-expressed as a relative error
    (with the 1/target² weighting, sqrt(term) IS the relative error), and
    which term dominated — i.e. *why the optimizer stopped where it did*:
    was the design EA-limited, IPF-limited, or steered by the plausibility
    penalty?  This turns the black-box optimizer outcome into an auditable
    decision record per target.
    """
    if not all_inverse:
        return None
    rows = []
    for res in all_inverse:
        best = res.get("gpbo_best") or {}
        if not best or "x_best" not in best:
            continue
        tinfo = res.get("target_info") or {}
        t_ea = float(res.get("target_ea", 0.0))
        t_ipf = float(res.get("target_ipf", 0.0))
        w_ea = 1.0 / (t_ea ** 2 + 1e-12)
        w_ipf = 1.0 / (t_ipf ** 2 + 1e-12)
        pred_ea = float(best.get("pred_ea", float("nan")))
        pred_ipf = float(best.get("pred_ipf", float("nan")))
        fit_ea = w_ea * (pred_ea - t_ea) ** 2
        fit_ipf = w_ipf * (pred_ipf - t_ipf) ** 2
        prob_weight = float(res.get("prob_weight", 0.0) or 0.0)
        rec_theta = float(best["x_best"])
        rec_lc = str(best.get("lc", best.get("best_lc", "")))
        penalty, p_lc = 0.0, float("nan")
        if cal_ens is not None and feat_scaler is not None and rec_lc:
            try:
                penalty, p_lc = compute_lc_penalty(
                    cal_ens, feat_scaler, pred_ea, pred_ipf, rec_lc,
                    prob_weight=prob_weight, angle_deg=rec_theta)
            except Exception:
                pass
        terms = {"EA_fit": fit_ea, "IPF_fit": fit_ipf, "classifier_penalty": float(penalty)}
        dominant = max(terms, key=terms.get)
        rows.append({
            "Target": tinfo.get("id", ""),
            "recovered_theta_deg": f"{rec_theta:.2f}",
            "recovered_LC": rec_lc,
            "J_at_optimum": f"{float(best.get('y_best', float('nan'))):.6f}",
            "EA_fit_term": f"{fit_ea:.6f}",
            "IPF_fit_term": f"{fit_ipf:.6f}",
            "classifier_penalty_term": f"{float(penalty):.6f}",
            "EA_rel_error_pct": f"{np.sqrt(fit_ea) * 100:.2f}",
            "IPF_rel_error_pct": f"{np.sqrt(fit_ipf) * 100:.2f}",
            "p_LC_at_optimum": (f"{p_lc:.4f}" if np.isfinite(p_lc) else ""),
            "lambda_prob_weight": f"{prob_weight:.4g}",
            "dominant_term": dominant,
        })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "Table_inverse_design_explanation.csv")
    df.to_csv(path, index=False)
    logger.info(f"  Saved: {os.path.basename(path)} (per-target objective decomposition)")
    return df


def generate_optimizer_comparison_table(all_inverse_results: List[Dict], output_dir: str, logger: logging.Logger):
    """Summarize GP-BO inverse runs per target, anchored to a dense-grid baseline.

    Each row reports the GP-BO recovery alongside the exhaustive dense-grid
    reference (from the already-computed solution landscape: 201 θ × both LCs),
    the BO-vs-grid angle discrepancy, the total multi-start evaluation cost,
    and a Bound_active flag when the recovered angle sits on the search-box
    boundary.  The grid columns give reviewers the honest context for GP-BO in
    this low-dimensional design space: BO reaches the grid optimum with far
    fewer objective evaluations.

    Winner labels (populated only when >1 optimizer competes):
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
        # Dense-grid reference baseline from the already-computed solution
        # landscape (201 θ points × both LCs = 402 objective evaluations).
        # This anchors the GP-BO result against an exhaustive equal-fidelity
        # search: BO should reach the grid optimum with far fewer evaluations,
        # which is the honest argument for using BO in this low-dimensional
        # design space.
        grid_theta, grid_lc, grid_J, grid_n = "", "", "", ""
        landscape = result.get("solution_landscape") or {}
        tg = landscape.get("theta_grid")
        if tg is not None:
            g_best_J, g_best_theta, g_best_lc, g_count = float("inf"), None, None, 0
            for k, v in landscape.items():
                if k.startswith("J_"):
                    J_vals = np.asarray(v, dtype=float)
                    g_count += len(J_vals)
                    j = int(np.nanargmin(J_vals))
                    if J_vals[j] < g_best_J:
                        g_best_J = float(J_vals[j])
                        g_best_theta = float(np.asarray(tg)[j])
                        g_best_lc = k[2:]
            if g_best_theta is not None:
                grid_theta = f"{g_best_theta:.2f}"
                grid_lc = g_best_lc
                grid_J = f"{g_best_J:.6f}"
                grid_n = g_count
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
                rec_angle = float(best["x_best"])
                # Total surrogate calls across the multi-start (the honest
                # search cost, not just the winning restart's budget).
                rs = (result.get("gpbo_joint") or {}).get("restart_summary", {})
                total_calls = rs.get("total_surrogate_calls", best.get("n_evals", ""))
                tol = 1e-6 + 0.05 * (CFG.angle_opt_max - CFG.angle_opt_min) / 100.0
                at_bound = (abs(rec_angle - CFG.angle_opt_min) <= tol
                            or abs(rec_angle - CFG.angle_opt_max) <= tol)
                row = {"Target": target_id,
                            "Optimizer": display,
                            "Best_Angle": f"{rec_angle:.2f}",
                            "Selected_LC": best.get("lc", ""),
                            "Bound_active": "Yes" if at_bound else "No",
                            "EA_Error_%": f"{best.get('ea_error_pct', float('nan')):.2f}",
                            "IPF_Error_%": f"{best['ipf_error_pct']:.2f}",
                            f"Delivered_EA@{EA_COMMON_MM_TAG}_J": f"{best.get('pred_ea', 0):.2f}",
                            "Delivered_EA_full_J": f"{best.get('pred_ea_full', best.get('pred_ea', 0)):.2f}",
                            "Delivered_IPF_kN": f"{best.get('pred_ipf', 0):.3f}",
                            "Objective": f"{best['y_best']:.6f}",
                            "N_Evals": best.get("n_evals", ""),
                            "Total_Multistart_Evals": total_calls,
                            "Iters_to_99pct": conv_iter,
                            "Wall_Time_s": f"{best.get('wall_time', 0):.2f}",
                            "Grid_Best_Angle": grid_theta,
                            "Grid_Best_LC": grid_lc,
                            "Grid_Best_Objective": grid_J,
                            "Grid_N_Evals": grid_n,
                            "Winner": badge}
                if grid_theta:
                    row["BO_vs_Grid_dTheta_deg"] = f"{abs(rec_angle - float(grid_theta)):.2f}"
                rows.append(row)
    if rows:
        df_comp = pd.DataFrame(rows)
        # The Winner badge only carries information when more than one optimizer
        # competes.  With the single-optimizer GP-BO configuration it is always
        # empty and misreads as missing data, so drop it in that case.
        if "Winner" in df_comp.columns and df_comp["Winner"].astype(str).str.strip().eq("").all():
            df_comp = df_comp.drop(columns=["Winner"])
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
def log_runtime_environment(output_dir: str, logger: logging.Logger,
                              tag: str = "") -> Dict[str, Any]:
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
        "scikit_optimize": _pkg_ver("scikit-optimize"),
        "openpyxl": _pkg_ver("openpyxl"),
        "git_sha": _git_sha_short(),
        "data_fingerprint": getattr(CFG, "data_fingerprint", ""),
        "HAS_SCIPY": bool(HAS_SCIPY),
        "HAS_SKOPT": bool(HAS_SKOPT),
        "HAS_SKLEARN_GP": bool(HAS_SKLEARN_GP),
        "CFG_seed": CFG.seed,
        "CFG_n_ensemble": CFG.n_ensemble,
        "D_COMMON_mm": D_COMMON,
    }
    fname = f"runtime_environment_{tag}.json" if tag else "runtime_environment.json"
    path = os.path.join(output_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)
    logger.info(f"  Saved: {fname}")
    logger.info("  Runtime stack (see JSON for full strings):")
    logger.info(f"    Python {info['python_short']} | numpy {info['numpy']} | torch {info['torch']}")
    logger.info(f"    sklearn {info.get('sklearn')} | skopt={HAS_SKOPT}")
    return info


def apply_dry_run_settings(logger: logging.Logger) -> None:
    """Tighten global budgets for ``CFG.dry_run`` (CI smoke; not for publication numbers)."""
    if not getattr(CFG, "dry_run", False):
        return
    CFG.n_ensemble = min(int(CFG.n_ensemble), 2)
    CFG.run_robustness_analyses = False
    CFG.run_ablation = False
    CFG.run_gpbo = False
    BO_CFG.lambda_sweep = False
    BO_CFG.run_classifier_ablation = False
    CFG.run_inverse_ablation = False
    CFG.run_inverse_stress_validation = False
    logger.info(
        "DRY RUN MODE: M<=2, short epochs, coarser MO landscape, "
        "no robustness / ablation / GP-BO; inverse uses coarse angle grid."
    )


def check_publication_dependencies(logger: logging.Logger) -> None:
    """Fail fast when ``CFG.strict_paper_deps`` and a required stack piece is missing."""
    if not getattr(CFG, "strict_paper_deps", False):
        return
    missing: List[str] = []
    if CFG.run_gpbo and not HAS_SKOPT:
        missing.append("scikit-optimize (skopt) required for GP-BO inverse design")
    if missing:
        for m in missing:
            logger.error(f"  strict_paper_deps: {m}")
        raise RuntimeError(
            "strict_paper_deps is enabled but optional dependencies are missing: "
            + "; ".join(missing)
            + ". Install requirements.txt (skopt). If inverse GP-BO is not required, "
            "set CFG.run_gpbo = False before run."
        )


def write_statistical_testing_policy(output_dir: str, logger: logging.Logger) -> None:
    """Document multiplicity and primary vs exploratory tests."""
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
- Inverse design, multi-objective sweeps, classifier ablations, and
  hyperparameter sensitivity analyses are **descriptive** unless explicitly
  framed as confirmatory with a pre-specified alpha budget.

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
    # Hardware row: wall-time claims are uninterpretable without the device.
    try:
        hw = (torch.cuda.get_device_name(0) if torch.cuda.is_available()
              else f"CPU ({__import__('platform').processor() or 'unknown'})")
    except Exception:
        hw = "unknown"
    rows.append({
        "stage": f"hardware: {hw} | torch {torch.__version__} | git {_git_sha_short() or 'n/a'}",
        "ensemble_M_total": "", "ensemble_M_effective": "", "n_parameters": "",
        "avg_train_time_s_per_member": "",
    })
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

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
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
                ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(4, 4))
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.28, linestyle="--")
        ax.legend(loc="upper left", framealpha=0.95)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    add_subplot_label(axes[0], "a")
    add_subplot_label(axes[1], "b")
    fig.suptitle("Inverse design validation: parity with ensemble uncertainty")
    fig.savefig(os.path.join(output_dir, "Fig_inverse_parity_uncertainty.png"),
                dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("  Saved: Fig_inverse_parity_uncertainty.png")

    # Second small figure: error vs recovered angle (interpolation stress)
    fig2, ax2 = plt.subplots(figsize=(7.5, 5.0))
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
                         textcoords="offset points", xytext=(3, 3))
    ax2.set_xlabel(r"Recovered angle $\theta$ (°)")
    ax2.set_ylabel("Relative EA error (%)")
    ax2.set_title("Inverse error vs optimised angle")
    ax2.grid(True, alpha=0.28, linestyle="--")
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    # Single-panel figure: no "(a)" label.
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
    fig, axes = plt.subplots(
        2, len(protocols),
        figsize=(3.5 * len(protocols), 7.0),
        squeeze=False,
    )
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
        # When every validation row shares one angle (the held-out θ* of the
        # unseen protocol), a 2-D angle×displacement hexbin collapses to a
        # single vertical stripe in a mostly empty panel.  Show the error
        # profile vs displacement at that angle instead.
        single_angle = float(np.ptp(ang)) < 2.0
        for i, (errs, title) in enumerate([(abs_le, "|load error| (kN)"), (abs_ee, "|energy error| (J)")]):
            ax = axes[i, j]
            if single_angle:
                order = np.argsort(disp)
                sc = ax.scatter(disp[order], errs[order], c=errs[order], cmap="magma",
                                s=46, edgecolors="black", linewidths=0.3)
                plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, label=title)
                ax.set_xlabel("Displacement (mm)")
                ax.set_ylabel(title)
                ax.set_title(protocol_label(protocol))
            else:
                hb = ax.hexbin(
                    ang, disp, C=errs, reduce_C_function=np.mean,
                    gridsize=(28, 24), mincnt=1, cmap="magma", linewidths=0,
                )
                plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.02, label=title)
                ax.set_xlabel(r"Angle $\theta$ (°)")
                ax.set_ylabel("Displacement (mm)")
                # Keep titles short (model name goes in the suptitle) so they do
                # not run into the panel label in these colorbar-narrowed panels.
                ax.set_title(protocol_label(protocol))
            ax.grid(True, alpha=0.22, linestyle="--")
            add_subplot_label(ax, chr(ord("a") + i * len(protocols) + j))
    fig.suptitle(f"Validation error maps: {MODEL_LABELS.get(approach, approach)}")
    # constrained_layout already handles spacing — tight_layout would conflict.
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
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
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
        f"{MODEL_LABELS.get(approach, approach)}"
    )
    # probplot sets generic axis labels; make them specific and unit-bearing.
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Ordered load residuals (kN)")
    ax.grid(True, alpha=0.28, linestyle="--")
    # Single-panel figure: no "(a)" panel label needed.
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
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax00, ax01, ax10, ax11 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    cm = confusion_matrix(y, pred, labels=[0, 1])
    im = ax00.imshow(cm, cmap="Blues")
    ax00.set_xticks([0, 1])
    ax00.set_yticks([0, 1])
    ax00.set_xticklabels(["Pred LC1", "Pred LC2"])
    ax00.set_yticklabels(["True LC1", "True LC2"])
    for (i, j), v in np.ndenumerate(cm):
        ax00.text(j, i, int(v), ha="center", va="center", color="black")
    plt.colorbar(im, ax=ax00, fraction=0.046)
    ax00.set_title("CV confusion")
    fpr, tpr, _ = roc_curve(y, pr)
    ax01.plot(fpr, tpr, color=COLORS.get("hard", "#0072B2"), lw=2.0, label=f"AUC = {auc(fpr, tpr):.3f}")
    ax01.plot([0, 1], [0, 1], "k--", alpha=0.45)
    ax01.set_xlabel("FPR")
    ax01.set_ylabel("TPR")
    ax01.set_title("ROC (score = P(LC2))")
    ax01.legend(loc="lower right")
    ax01.grid(True, alpha=0.25)
    prec, rec, _ = precision_recall_curve(y, pr)
    ap = average_precision_score(y, pr)
    ax10.plot(rec, prec, color=COLORS.get("soft", "#4DBBD5"), lw=2.0, label=f"AP = {ap:.3f}")
    ax10.set_xlabel("Recall (LC2)")
    ax10.set_ylabel("Precision (LC2)")
    ax10.set_title("Precision–recall (CV)")
    ax10.set_xlim(0, 1.02)
    ax10.set_ylim(0, 1.02)
    ax10.legend(loc="upper right")
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
        f"LC plausibility classifier — {clf_diag.get('cv_method', 'CV')} on design metrics"
    )
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
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
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
        ax.legend(title="LC")
        ax.grid(True, alpha=0.26, linestyle="--")
    add_subplot_label(axes[0], "a")
    add_subplot_label(axes[1], "b")
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
    fig, axes = plt.subplots(1, len(lc_list), figsize=(5.0 * len(lc_list), 4.5), squeeze=False)
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
        ax.set_ylabel(r"EA up to $d$ (J)")
        ax.set_title(f"{lc} (d_end={d_end:.0f} mm)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.26, linestyle="--")
    for ax, lab in zip(axes, "ab"):
        add_subplot_label(ax, lab)
    fig.suptitle(
        "Sensitivity of EA metric to displacement endpoint (ensemble mean; same trained models)"
    )
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows + 0.6), squeeze=False)
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
        ax.legend()
        ax.grid(True, alpha=0.26, linestyle="--")
        add_subplot_label(ax, chr(ord("a") + idx))
    for j in range(len(panels), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle(
        "Inverse optimum vs nearest experimental load–displacement (same LC)"
    )
    fig.savefig(
        os.path.join(output_dir, "Fig_inverse_vs_nearest_experimental_curve.png"),
        dpi=600, bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)
    logger.info("  Saved: Fig_inverse_vs_nearest_experimental_curve.png")


# Core artifacts a full production run (--mode all, robustness on) MUST emit.
# The manifest compares against this list and ends the run with a prominent
# failure summary when any is missing — previously a swallowed exception could
# silently drop a required paper artifact and the run still "succeeded".
REQUIRED_PAPER_ARTIFACTS = [
    "Table1_forward_results.csv",
    "Table2_statistical_tests.csv",
    "Table3_inverse_design.csv",
    "Table4_model_complexity.csv",
    "Table5_per_LC_breakdown.csv",
    "Table_optimizer_comparison.csv",
    "Table_forward_design_errors.csv",
    "Table_null_baseline_design_level.csv",
    "Table_physical_plausibility_audit.csv",
    "Table_design_variance_decomposition.csv",
    "Table_inverse_design_explanation.csv",
    "Fig_dataset_overview.png",
    "Fig_parity_unseen.png",
    "Fig_residuals_unseen.png",
    "Fig_physics_verification.png",
    "Fig_reliability_diagram.png",
    "Fig_design_space.png",
    "Fig_pareto_tradeoff.png",
    "Fig_multiobjective_heatmaps.png",
    "Fig_optimizer_comparison.png",
    "Fig_inverse_parity_uncertainty.png",
]


def generate_output_manifest(output_dir: str, logger: logging.Logger,
                             check_required: bool = False) -> None:
    """List generated figures, tables, and key sidecar files (no extra experiments).

    With ``check_required=True`` (full-pipeline runs) the manifest is compared
    against :data:`REQUIRED_PAPER_ARTIFACTS` and missing entries produce a
    prominent failure summary in the log.
    """
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
    if check_required:
        present = {r["filename"] for r in rows}
        missing = [a for a in REQUIRED_PAPER_ARTIFACTS if a not in present]
        if missing:
            logger.warning("=" * 80)
            logger.warning(f"MISSING REQUIRED PAPER ARTIFACTS ({len(missing)}):")
            for a in missing:
                logger.warning(f"  MISSING: {a}")
            logger.warning("Search the log above for 'skipped'/'WARNING' to find the cause.")
            logger.warning("=" * 80)
        else:
            logger.info(f"  Required-artifact check: all {len(REQUIRED_PAPER_ARTIFACTS)} present")


# =============================================================================
# PIPELINE ORCHESTRATION, FIGURE LAYER, AND CLI
# =============================================================================
# Below: multi-panel manuscript figures, the three-bundle pipeline state
# save/load helpers, and the ``main()`` entry point with ``--mode all/forward/
# inverse/replot``.  The state bundles let forward training (slow) and
# downstream analysis / plotting (fast) iterate independently — replot mode
# regenerates every figure from whichever subset of bundles is present.
# =============================================================================

# Bundle filenames
_FORWARD_BUNDLE  = "forward_models.pt"
_INVERSE_BUNDLE  = "inverse_models.pt"
_ANALYSIS_BUNDLE = "analysis_results.pt"


def _git_sha_short() -> str:
    """Current git commit (short), or '' when unavailable (e.g. tarball run)."""
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10,
                             cwd=os.path.dirname(os.path.abspath(__file__)))
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _bundle_meta() -> Dict:
    """Provenance stamp: ties a bundle to the data, config, and code that made it."""
    try:
        cfg_sha = __import__("hashlib").sha256(
            repr(sorted(vars(CFG).items())).encode()).hexdigest()[:12]
    except Exception:
        cfg_sha = ""
    return {
        "git_sha": _git_sha_short(),
        "data_fingerprint": getattr(CFG, "data_fingerprint", ""),
        "cfg_sha": cfg_sha,
        "seed": int(getattr(CFG, "seed_base", 0)),
    }


def _save_bundle(state: Dict, output_dir: str, filename: str, logger: logging.Logger) -> str:
    """Atomic ``torch.save`` (tmp + os.replace) — SLURM-preempt safe.

    Every bundle carries a ``_meta`` provenance stamp (git commit, data
    fingerprint, CFG hash, seed) so forward/inverse/analysis bundles produced
    by different jobs can be checked for consistency before being combined —
    previously nothing prevented silently mixing bundles from different data
    or configs.
    """
    state = dict(state)
    state.setdefault("_meta", _bundle_meta())
    path = os.path.join(output_dir, filename)
    tmp = path + ".tmp"
    torch.save(state, tmp, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    size_mb = os.path.getsize(path) / (1024.0 * 1024.0)
    m = state["_meta"]
    logger.info(f"  Saved {filename} ({size_mb:.1f} MB) "
                f"[git={m.get('git_sha') or 'n/a'} data={m.get('data_fingerprint') or 'n/a'} "
                f"cfg={m.get('cfg_sha') or 'n/a'}]")
    return path


def save_forward_bundle(state: Dict, output_dir: str, logger: logging.Logger) -> str:
    """Trained forward ensembles + scalers + calibration."""
    return _save_bundle(state, output_dir, _FORWARD_BUNDLE, logger)


def save_inverse_bundle(state: Dict, output_dir: str, logger: logging.Logger) -> str:
    """Full-data Hard-PINN + classifier ensemble + diagnostics."""
    return _save_bundle(state, output_dir, _INVERSE_BUNDLE, logger)


def save_analysis_bundle(state: Dict, output_dir: str, logger: logging.Logger) -> str:
    """All non-NN analysis outputs (BO, Pareto, robustness, sensitivity)."""
    return _save_bundle(state, output_dir, _ANALYSIS_BUNDLE, logger)


def _load_bundle(path: str, logger: Optional[logging.Logger] = None) -> Optional[Dict]:
    """``torch.load`` with ``map_location='cpu'`` so CUDA-trained pickles
    reload on CPU-only machines for figure regeneration.

    Returns None if the bundle is missing OR fails to deserialize (rare but
    possible if a writer was killed mid-write before ``os.replace`` completed,
    leaving a corrupt ``.tmp`` that somehow ended up at the final path).
    Replot mode degrades gracefully — corrupt bundles produce placeholders,
    not stack traces.
    """
    if not os.path.exists(path):
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        if logger is not None:
            logger.warning(f"  Bundle load failed: {os.path.basename(path)} → {type(e).__name__}: {e}")
        return None
    if logger is not None:
        size_mb = os.path.getsize(path) / (1024.0 * 1024.0)
        logger.info(f"  Loaded {os.path.basename(path)} ({size_mb:.1f} MB)")
    return state


def load_forward_bundle(output_dir: str, logger: Optional[logging.Logger] = None) -> Optional[Dict]:
    return _load_bundle(os.path.join(output_dir, _FORWARD_BUNDLE), logger)


def load_inverse_bundle(output_dir: str, logger: Optional[logging.Logger] = None) -> Optional[Dict]:
    return _load_bundle(os.path.join(output_dir, _INVERSE_BUNDLE), logger)


def load_analysis_bundle(output_dir: str, logger: Optional[logging.Logger] = None) -> Optional[Dict]:
    return _load_bundle(os.path.join(output_dir, _ANALYSIS_BUNDLE), logger)


# -----------------------------------------------------------------------------
# Figure helpers
# -----------------------------------------------------------------------------
def _savefig(fig, output_dir: str, name: str, logger: logging.Logger, *, dpi: int = 600) -> str:
    """Save and close ``fig``; log and return the absolute path."""
    path = os.path.join(output_dir, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"  Saved: {name}")
    return path


def _panel_label(ax, label: str, *, x: float = -0.16, y: float = 1.06) -> None:
    """Place a bold panel label '(a)' at axes-fraction coordinates.

    Default ``x = -0.16`` keeps the label clear of the y-tick labels;
    override per axes when needed (e.g. for axes without a y-axis label).
    """
    ax.text(x, y, f"({label})", transform=ax.transAxes,
            fontweight='bold', va='top', ha='left', clip_on=False)


# =============================================================================
# Conceptual schematics (data-independent): framework + architectures
# =============================================================================
def _schematic_box(ax, cx, cy, w, h, text, *, fc="white", ec="black",
                   fontsize=9, fontcolor="black", lw=1.3):
    """Rounded box centred at (cx, cy) with wrapped, centred text (axes coords)."""
    ax.add_patch(mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=lw, edgecolor=ec, facecolor=fc, mutation_aspect=0.6, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=fontcolor, zorder=3)


def _schematic_arrow(ax, x1, y1, x2, y2, *, color="black", style="-|>",
                     lw=1.6, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                linestyle=ls, shrinkA=2, shrinkB=2), zorder=1)


def fig_architecture_schematic(output_dir: str, logger: logging.Logger) -> str:
    """Schematic contrasting the three surrogate architectures and how each
    treats the work–energy identity F = dE/dd (data-independent)."""
    set_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.8))
    in_txt = r"inputs:  $d,\ \sin\theta,\ \cos\theta,\ \mathrm{LC}$"
    for ax, kind, title in [
        (axes[0], "ddns", "DDNS — data-driven"),
        (axes[1], "soft", "Soft-PINN — penalty"),
        (axes[2], "hard", "Hard-PINN — exact"),
    ]:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.set_title(title, fontweight="bold", color=COLORS[kind])
        col = COLORS[kind]
        _schematic_box(ax, 0.5, 0.90, 0.94, 0.13, in_txt, fc="#EEEEEE", fontsize=8.5)
        _schematic_box(ax, 0.5, 0.635, 0.6, 0.15, "MLP\n(Softplus)", fc="#F6F6F6")
        _schematic_arrow(ax, 0.5, 0.825, 0.5, 0.715)
        if kind == "ddns":
            _schematic_box(ax, 0.27, 0.34, 0.34, 0.13, r"$\hat F$", fc="#FBE6D9", ec=col)
            _schematic_box(ax, 0.73, 0.34, 0.34, 0.13, r"$\hat E$", fc="#FBE6D9", ec=col)
            _schematic_arrow(ax, 0.42, 0.56, 0.30, 0.41)
            _schematic_arrow(ax, 0.58, 0.56, 0.70, 0.41)
            ax.text(0.5, 0.12, "two independent heads\n(no physics coupling)",
                    ha="center", va="center", fontsize=8, style="italic")
        elif kind == "soft":
            _schematic_box(ax, 0.27, 0.40, 0.34, 0.13, r"$\hat F$", fc="#DCE9F5", ec=col)
            _schematic_box(ax, 0.73, 0.40, 0.34, 0.13, r"$\hat E$", fc="#DCE9F5", ec=col)
            _schematic_arrow(ax, 0.42, 0.56, 0.30, 0.47)
            _schematic_arrow(ax, 0.58, 0.56, 0.70, 0.47)
            _schematic_arrow(ax, 0.44, 0.40, 0.56, 0.40, color=col, style="<|-|>", ls="--", lw=1.4)
            ax.text(0.5, 0.15,
                    r"penalty $w_\phi\,\|\hat F-\partial\hat E/\partial d\|^2$",
                    ha="center", va="center", fontsize=8.5, color=col, fontweight="bold")
            ax.text(0.5, 0.06, "(soft; weight-tuned)", ha="center", va="center",
                    fontsize=8, style="italic")
        else:  # hard
            _schematic_box(ax, 0.5, 0.40, 0.5, 0.13, r"$\hat E$  (single output)",
                           fc="#E6E6E6", ec=col)
            _schematic_arrow(ax, 0.5, 0.56, 0.5, 0.47)
            _schematic_arrow(ax, 0.5, 0.335, 0.5, 0.245, color=col, lw=1.8)
            _schematic_box(ax, 0.5, 0.16, 0.74, 0.135,
                           r"$\hat F=\partial\hat E/\partial d$  (autograd)",
                           fc="#1A1A1A", ec="#1A1A1A", fontcolor="white", fontsize=8.5)
            ax.text(0.5, 0.035, "exact by construction", ha="center", va="center",
                    fontsize=8, style="italic")
    fig.suptitle("Surrogate architectures: physics-enforcement strategies",
                 fontweight="bold")
    return _savefig(fig, output_dir, "Fig_architecture_schematic.png", logger)


def fig_framework_schematic(output_dir: str, logger: logging.Logger) -> str:
    """End-to-end IPINN framework pipeline (data-independent graphical abstract)."""
    set_publication_style()
    fig, ax = plt.subplots(1, 1, figsize=(13.0, 3.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    stages = [
        ("Experimental\ncrush data\n(LC1, LC2;\n$\\theta\\in[45,70]^\\circ$)", "#EEEEEE"),
        ("Three surrogates\nDDNS · Soft-PINN ·\nHard-PINN\n($M{=}20$ ensemble)", "#FBE6D9"),
        ("Dual-protocol\nvalidation\n(random + unseen $\\theta$)\n+ conformal UQ", "#DCE9F5"),
        ("Full-data Hard-PINN\n+ GP-BO inverse\n+ LC plausibility\nclassifier", "#E2EFE8"),
        ("Outcomes:\ntarget matching,\nPareto EA–IPF,\nill-posedness", "#F3E6F0"),
    ]
    n = len(stages)
    cxs = np.linspace(0.10, 0.90, n)
    w, h, cy = 0.155, 0.66, 0.55
    for i, ((text, fc), cx) in enumerate(zip(stages, cxs)):
        ec = COLORS["hard"] if i == 1 else "black"
        _schematic_box(ax, cx, cy, w, h, text, fc=fc, ec=ec, fontsize=8.4,
                       lw=1.6 if i == 1 else 1.3)
        if i > 0:
            _schematic_arrow(ax, cxs[i - 1] + w / 2, cy, cx - w / 2, cy, lw=2.0)
    fig.suptitle("IPINN framework: forward surrogates → inverse design → trade-off",
                 fontweight="bold", y=1.02)
    return _savefig(fig, output_dir, "Fig_framework_schematic.png", logger)


# =============================================================================
# Methodology-justification figures (classifier, sensitivity sweeps)
# =============================================================================
def _as_floats(seq):
    out = []
    for v in seq:
        try:
            out.append(float(str(v).replace("%", "")))
        except (TypeError, ValueError):
            out.append(np.nan)
    return out


def fig_classifier_effect(clf_ablation_df, output_dir: str, logger: logging.Logger):
    """Effect of the LC-plausibility penalty on inverse design: the loading-mode
    plausibility p(LC) at the recovered optimum, with vs without the classifier
    penalty, per target.  Justifies the plausibility-gate contribution."""
    if clf_ablation_df is None or len(clf_ablation_df) == 0:
        return None
    set_publication_style()
    tids = [str(t) for t in clf_ablation_df["Target"]]
    no_p = _as_floats(clf_ablation_df["No_Penalty_p_LC"])
    with_p = _as_floats(clf_ablation_df["With_Penalty_p_LC"])
    x = np.arange(len(tids)); w = 0.38
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.3))
    ax.bar(x - w / 2, no_p, w, label="without penalty", color="#BBBBBB",
           edgecolor="black", linewidth=0.8)
    ax.bar(x + w / 2, with_p, w, label="with penalty", color=COLORS["gpbo"],
           edgecolor="black", linewidth=0.8)
    ax.axhline(0.5, color="0.4", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(tids)
    ax.set_xlabel("Inverse-design target")
    ax.set_ylabel(r"$p(\mathrm{LC})$ at recovered optimum")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_title("Plausibility penalty steers GP-BO toward LC-consistent designs")
    return _savefig(fig, output_dir, "Fig_classifier_effect.png", logger)


def fig_lambda_sensitivity(lambda_df, output_dir: str, logger: logging.Logger,
                           lambda_opt=None):
    """Sensitivity to the classifier-penalty weight λ: target-matching error and
    LC plausibility vs λ (λ values shown on an even axis; auto-tuned λ marked)."""
    if lambda_df is None or len(lambda_df) == 0:
        return None
    set_publication_style()
    lam = _as_floats(lambda_df["lambda"])
    order = np.argsort(lam)
    lam = [lam[i] for i in order]
    ea = [_as_floats(lambda_df["EA_err%"])[i] for i in order]
    ipf = [_as_floats(lambda_df["IPF_err%"])[i] for i in order]
    plc = [_as_floats(lambda_df["p_LC"])[i] for i in order]
    xi = np.arange(len(lam))
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.4))
    l1 = ax.plot(xi, ea, "o-", color=COLORS["gpbo"], label="EA error (%)")
    l2 = ax.plot(xi, ipf, "s--", color="#CC79A7", label="IPF error (%)")
    ax.set_xticks(xi); ax.set_xticklabels([f"{v:g}" for v in lam])
    ax.set_xlabel(r"Classifier penalty weight $\lambda$")
    ax.set_ylabel("Target-matching error (%)")
    ax2 = ax.twinx()
    l3 = ax2.plot(xi, plc, "^:", color=COLORS["hard"], label=r"$p(\mathrm{LC})$ at optimum")
    ax2.set_ylabel(r"$p(\mathrm{LC})$ at optimum"); ax2.set_ylim(0, 1.05)
    lines = l1 + l2 + l3
    if lambda_opt is not None:
        try:
            j = int(np.argmin([abs(v - float(lambda_opt)) for v in lam]))
            ax.axvline(xi[j], color="0.3", lw=1.3, ls="-.")
            ax.text(xi[j], ax.get_ylim()[1] * 0.96, f" auto-tuned λ={float(lambda_opt):.3g}",
                    fontsize=8, color="0.2", ha="left", va="top")
        except Exception:
            pass
    ax.legend(lines, [ln.get_label() for ln in lines], loc="best", fontsize=8, framealpha=0.95)
    ax.set_title(r"Sensitivity to the classifier penalty weight $\lambda$")
    return _savefig(fig, output_dir, "Fig_lambda_sensitivity.png", logger)


def fig_penalty_weight_sensitivity(sensitivity_df, dual_results, output_dir: str,
                                   logger: logging.Logger):
    """Soft-PINN accuracy vs physics-penalty weight w_phi, with Hard-PINN as a
    weight-free reference (Hard enforces the constraint by construction, so it
    has no such knob).  Justifies the 'eliminates penalty-weight sensitivity' claim.

    Derived from the hyperparameter-sensitivity sweep (``sensitivity_df`` with
    columns w_phys, lr, load_r2, energy_r2): for each w_phys we take the best R²
    over the learning-rate axis, i.e. the accuracy attainable at that weight.
    """
    if sensitivity_df is None or len(sensitivity_df) == 0:
        return None
    set_publication_style()
    df = sensitivity_df.copy()
    df["w_phys"] = _as_floats(df["w_phys"])
    df["load_r2"] = _as_floats(df["load_r2"])
    df["energy_r2"] = _as_floats(df["energy_r2"])
    g = (df.groupby("w_phys", as_index=False)
           .agg(load_r2=("load_r2", "max"), energy_r2=("energy_r2", "max"))
           .sort_values("w_phys"))
    w = list(g["w_phys"]); lr = list(g["load_r2"]); er = list(g["energy_r2"])
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.4))
    ax.plot(w, lr, "o-", color=COLORS["soft"], label=r"Soft-PINN load $R^2$")
    ax.plot(w, er, "s--", color="#56B4E9", label=r"Soft-PINN energy $R^2$")
    try:
        hm = dual_results["unseen"]["hard"]["metrics"]
        ax.axhline(hm["load_r2"], color=COLORS["hard"], lw=1.6, ls="-",
                   label=r"Hard-PINN load $R^2$ (no $w_\phi$ knob)")
    except (TypeError, KeyError):
        pass
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlim(-0.05, (max(w) * 1.6) if w else 100)  # crop empty negative decades
    ax.set_xlabel(r"Physics-penalty weight $w_\phi$ (Soft-PINN)")
    ax.set_ylabel(r"Validation $R^2$ (unseen $\theta$)")
    ax.legend(loc="best", fontsize=8, framealpha=0.95)
    ax.set_title(r"Soft-PINN depends on $w_\phi$; Hard-PINN is weight-free")
    return _savefig(fig, output_dir, "Fig_penalty_weight_sensitivity.png", logger)


# =============================================================================
# Dataset overview — single focused figure (1x3 grid)
# =============================================================================
def fig_dataset_overview(df_all: pd.DataFrame, output_dir: str,
                          logger: logging.Logger) -> str:
    """Dataset overview: representative load–displacement curves, EA and IPF
    distributions per (angle, LC).  Clean 1x3 layout suitable for journal
    full-column width without text overlap."""
    set_publication_style()
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, i]) for i in range(3))

    angles = sorted(df_all["Angle"].unique())
    lcs    = sorted(df_all["LC"].unique())
    cmap   = plt.get_cmap("viridis", max(1, len(angles)))

    # (a) sample load–displacement curves coloured by angle
    seen_lc = set()
    for i, ang in enumerate(angles):
        for lc in lcs:
            sub = df_all[(df_all["Angle"] == ang) & (df_all["LC"] == lc)].sort_values("disp_mm")
            if sub.empty:
                continue
            label = lc if lc not in seen_lc else None
            seen_lc.add(lc)
            ax_a.plot(sub["disp_mm"], sub["load_kN"], color=cmap(i),
                      linestyle=LINESTYLES.get(lc, "-"), lw=1.4, alpha=0.85,
                      label=label)
    sm = matplotlib.cm.ScalarMappable(
        norm=Normalize(vmin=min(angles), vmax=max(angles)), cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_a, fraction=0.05, pad=0.03)
    cbar.set_label("Angle θ (°)")
    ax_a.set_xlabel("Displacement d (mm)")
    ax_a.set_ylabel("Load F (kN)")
    ax_a.set_title("Experimental curves")
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(loc="upper left", title="LC")
    _panel_label(ax_a, "a")

    # (b, c) EA / IPF per (angle, LC) as grouped boxplots
    df_metrics = compute_design_space_metrics(df_all, logger)
    if "Angle" not in df_metrics.columns and df_metrics.index.name == "Angle":
        df_metrics = df_metrics.reset_index()

    _LC_BOX_COLORS = {"LC1": "#a8d5f7", "LC2": "#ffc999"}

    def _grouped_box(ax, value_col: str, ylabel: str, title: str) -> None:
        n_lc = max(1, len(lcs))
        bw = 0.8 / n_lc
        any_data = False
        for i, ang in enumerate(angles):
            for j, lc in enumerate(lcs):
                sub = df_metrics[(df_metrics["Angle"] == ang) & (df_metrics["LC"] == lc)]
                if sub.empty or value_col not in sub.columns:
                    continue
                data = sub[value_col].dropna().values
                if data.size == 0:
                    continue
                pos = i + (j - (n_lc - 1) / 2.0) * bw
                # Few replicates per (angle, LC): a boxplot collapses to a bare
                # median line (reads as a stray red dash and hides the LC fill
                # colour), so plot the points themselves in the LC colour so they
                # match the LC1/LC2 legend swatches.
                fc = _LC_BOX_COLORS.get(lc, "#cccccc")
                ax.scatter(np.full(data.shape, pos), data, s=34, color=fc,
                           edgecolor="black", linewidth=0.6, zorder=3)
                if data.size > 1:
                    ax.plot([pos, pos], [float(np.min(data)), float(np.max(data))],
                            color=fc, lw=1.3, zorder=2)
                any_data = True
        if not any_data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            return
        ax.set_xticks(np.arange(len(angles)),
                      [f"{a:.0f}°" for a in angles])
        ax.set_xlim(-0.5, len(angles) - 0.5)
        ax.set_xlabel(r"Angle $\theta$ (°)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        handles = [mpatches.Patch(facecolor=_LC_BOX_COLORS[lc],
                                   edgecolor="black", label=lc)
                   for lc in lcs if lc in _LC_BOX_COLORS]
        if handles:
            ax.legend(handles=handles, title="LC", loc="upper left")

    _grouped_box(ax_b, "EA",  "Energy absorbed EA (J)",     "EA distribution")
    _grouped_box(ax_c, "IPF", "Initial peak force IPF (kN)", "IPF distribution")
    _panel_label(ax_b, "b")
    _panel_label(ax_c, "c")

    fig.suptitle("Dataset overview")
    return _savefig(fig, output_dir, "Fig_dataset_overview.png", logger)


# =============================================================================
# PIPELINE ORCHESTRATION
# =============================================================================
# =============================================================================
# TRAINING HELPERS — splittable for parallel HPC submission
# =============================================================================
def _train_forward_only(data_dir: str, output_dir: str,
                        logger: logging.Logger,
                        df_all: Optional[pd.DataFrame] = None) -> Dict:
    """Train all forward ensembles (DDNS, Soft, Hard × random + unseen) and
    save ``forward_models.pt``.  Returns the forward-state dict.

    Used both by ``run_pipeline`` (mode='all') and directly by
    ``main`` when invoked with ``--mode forward`` for parallel HPC
    submission.  ``df_all`` is loaded from disk if not provided.
    """
    if df_all is None:
        df_all = load_data(data_dir, logger)
    dual_results: Dict = {}

    logger.info("\n[forward 1/3] Random 80/20 split — DDNS, Soft, Hard")
    train_df_r, val_df_r = split_random_80_20(df_all, CFG.split_seed, logger)
    sd_r, so_r, en_r, p_r = create_preprocessors(train_df_r, logger)
    save_reproducibility_artifacts(output_dir, "random",
                                   train_df_r, sd_r, so_r, en_r, p_r, logger)
    dual_results["random"] = {a: train_ensemble(a, train_df_r, val_df_r,
                                                sd_r, so_r, en_r, p_r,
                                                "random", logger)
                              for a in ["ddns", "soft", "hard"]}
    dual_results["random"].update({"train_df": train_df_r, "val_df": val_df_r,
                                   "scaler_disp": sd_r, "scaler_out": so_r,
                                   "enc": en_r, "params": p_r})

    logger.info(f"\n[forward 2/3] Unseen θ={CFG.theta_star}° — DDNS, Soft, Hard")
    train_df_u, val_df_u = split_unseen_angle(df_all, CFG.theta_star, logger)
    sd_u, so_u, en_u, p_u = create_preprocessors(train_df_u, logger)
    save_reproducibility_artifacts(output_dir, "unseen",
                                   train_df_u, sd_u, so_u, en_u, p_u, logger)
    dual_results["unseen"] = {a: train_ensemble(a, train_df_u, val_df_u,
                                                sd_u, so_u, en_u, p_u,
                                                "unseen", logger)
                              for a in ["ddns", "soft", "hard"]}
    dual_results["unseen"].update({"train_df": train_df_u, "val_df": val_df_u,
                                   "scaler_disp": sd_u, "scaler_out": so_u,
                                   "enc": en_u, "params": p_u})

    logger.info("\n[forward 3/3] Calibration + statistical tests")
    calibration = compute_uncertainty_calibration(dual_results, logger)
    stat_tests  = compute_statistical_tests(dual_results, logger)

    baseline_results_u = None
    sensitivity_df_u   = None
    if CFG.run_robustness_analyses:
        try:
            baseline_results_u = train_baseline_models(train_df_u, val_df_u,
                                                      sd_u, en_u, p_u, logger)
        except Exception as e:
            logger.warning(f"  baseline_results_u skipped: {e}")
        try:
            sensitivity_df_u = run_hyperparam_sensitivity(
                train_df_u, val_df_u, sd_u, so_u, en_u, p_u, "unseen", logger)
        except Exception as e:
            logger.warning(f"  sensitivity_df_u skipped: {e}")

    forward_state = {
        "dual_results": dual_results, "df_all": df_all,
        "calibration": calibration, "stat_tests": stat_tests,
        "val_df_u": val_df_u, "scaler_disp_u": sd_u, "scaler_out_u": so_u,
        "enc_u": en_u, "params_u": p_u,
        "baseline_results_u": baseline_results_u,
        "sensitivity_df_u": sensitivity_df_u,
    }
    save_forward_bundle(forward_state, output_dir, logger)
    return forward_state


def _train_inverse_and_analyze(data_dir: str, output_dir: str,
                               logger: logging.Logger,
                               df_all: Optional[pd.DataFrame] = None,
                               pretrained_inverse: Optional[Dict] = None,
                               ) -> Tuple[Dict, Dict]:
    """Train the full-data Hard-PINN inverse ensemble + LC plausibility
    classifier, then run every inverse-design analysis (GP-BO target
    matching, multi-seed robustness, forward Jacobian, multi-objective
    Pareto sweep, Pareto-target recovery, λ-sensitivity, classifier
    ablation, D_COMMON sensitivity).  Saves ``inverse_models.pt`` and
    ``analysis_results.pt``.  Returns ``(inverse_state, analysis_state)``.

    Used by ``run_pipeline`` (mode='all') and directly by ``main``
    when invoked with ``--mode inverse`` for parallel HPC submission.

    ``pretrained_inverse`` (optional): a bundle produced by
    ``hpo.inverse_merge``.  When given, this function skips the
    expensive ``train_full_data_hard_pinn`` call and reconstructs the
    surrogate from the bundle's state_dicts + preprocessors.  Used by the
    SLURM array path that parallelises the M=20 member training across GPUs.
    """
    if df_all is None:
        df_all = load_data(data_dir, logger)

    if pretrained_inverse is not None:
        logger.info("\n[inverse 1/4] Loading pretrained full-data Hard-PINN surrogate "
                    f"(M={pretrained_inverse.get('M_eff', '?')} members)")
        cfg_pre = pretrained_inverse["cfg"]
        in_d = int(pretrained_inverse["in_d"])
        inv_sd = pretrained_inverse["scaler_disp"]
        inv_so = pretrained_inverse["scaler_out"]
        inv_en = pretrained_inverse["enc"]
        inv_p  = pretrained_inverse["params"]
        # Validate the bundle contains at least one member state_dict before
        # iterating.  An empty ``inv_models_state`` would silently produce
        # an empty ``inv_models`` list and downstream calls to
        # ``compute_ea_ipf_ensemble`` / ``evaluate_ensemble`` would fail
        # obliquely (e.g. ``np.mean`` over an empty list).
        states = pretrained_inverse.get("inv_models_state") or []
        if len(states) == 0:
            raise RuntimeError(
                "Pretrained inverse bundle contains zero member "
                "state_dicts (inv_models_state is empty).  This usually "
                "means every per-member SLURM task failed or no members "
                "passed the convergence filter.  Inspect the inverse-"
                "merge log and rerun the failed members before invoking "
                "--use_pretrained_inverse.",
            )
        if len(states) < 3:
            logger.warning(
                f"  Pretrained inverse bundle has only {len(states)} "
                f"surviving member(s); GP-BO posterior widths will be "
                f"noisy.  Consider rerunning failed members or relaxing "
                f"the convergence filter.",
            )
        inv_models = []
        for m_state in states:
            model = HardEnergyNet(in_d, cfg_pre["hidden_layers"],
                                  cfg_pre["dropout"], cfg_pre["softplus_beta"]).to(DEVICE)
            # BC disabled to mirror the forward-training architecture.
            model.configure_zero_bc(inv_p, enabled=False)
            model.load_state_dict({k: v.to(DEVICE) for k, v in m_state["state_dict"].items()})
            model.eval()
            inv_models.append(model)
        logger.info(f"  Reconstructed {len(inv_models)} inverse-design models on {DEVICE}.")
    else:
        logger.info("\n[inverse 1/4] Full-data Hard-PINN + LC plausibility classifier")
        inv_models, inv_sd, inv_so, inv_en, inv_p = train_full_data_hard_pinn(df_all, logger)
    # Persist the full-data preprocessor state on disk so the inverse model
    # is fully reproducible from the artifacts directory (parity with the
    # ``random`` and ``unseen`` artifacts saved in ``_train_forward_only``).
    save_reproducibility_artifacts(output_dir, "full_data",
                                   df_all, inv_sd, inv_so, inv_en, inv_p, logger)
    df_metrics = compute_design_space_metrics(df_all, logger)
    enrich_df_metrics_ea_common(df_metrics, df_all, logger)
    cal_ens, clf_feat_scaler, clf_diag = train_lc_plausibility_classifier(df_metrics, logger)
    lambda_opt, lambda_diag = auto_tune_lambda(cal_ens, clf_feat_scaler,
                                               df_metrics, logger)
    BO_CFG.prob_weight = lambda_opt

    # Design-level (EA, IPF) scalar uncertainty calibration for the deployed
    # surrogate — feeds the heatmap/design-space ±2σ bands with a factor whose
    # estimand actually matches the plotted quantity.
    ea_ipf_scalar_cal = None
    try:
        ea_ipf_scalar_cal = compute_ea_ipf_scalar_calibration(
            inv_models, "hard", inv_sd, inv_en, inv_p, df_metrics, output_dir, logger)
    except Exception as e:
        logger.warning(f"  EA/IPF scalar calibration skipped: {e}")

    inverse_state = {
        "inv_models": inv_models,
        "inv_scaler_disp": inv_sd, "inv_scaler_out": inv_so,
        "inv_enc": inv_en, "inv_params": inv_p,
        "cal_ens": cal_ens, "clf_feat_scaler": clf_feat_scaler,
        "clf_diag": clf_diag, "lambda_diag": lambda_diag,
        "df_metrics": df_metrics,
        "ea_ipf_scalar_cal": ea_ipf_scalar_cal,
    }
    save_inverse_bundle(inverse_state, output_dir, logger)

    logger.info("\n[inverse 2/4] GP-BO target matching (5 targets)")
    inverse_targets = generate_feasible_targets(df_metrics, logger, df_all=df_all)
    all_inverse_results: List[Dict] = []

    # Pull the forward-design R² of the inverse surrogate (mean over
    # surviving members) from the pretrained bundle if available, so each
    # per-target output records the surrogate quality alongside its
    # posterior CI.  ``train_r2_per_member`` lives on the pretrained bundle
    # via ``inverse_merge.py``; if absent (legacy bundles) this becomes
    # ``None`` and the field is omitted.
    if pretrained_inverse is not None and "member_train_r2" in pretrained_inverse:
        try:
            forward_R2_used = float(np.mean(pretrained_inverse["member_train_r2"]))
        except Exception:
            forward_R2_used = None
    else:
        forward_R2_used = None

    # Per-target BO history will be written to disk for inspection.
    bo_history_dir = os.path.join(output_dir, "bo_history")
    os.makedirs(bo_history_dir, exist_ok=True)

    for i, t in enumerate(inverse_targets):
        res = run_inverse_design(inv_models, "hard", t["EA"], t["IPF"],
                                 inv_sd, inv_en, inv_p, BO_CFG, logger,
                                 cal_ens=cal_ens, feat_scaler=clf_feat_scaler)
        res["target_info"] = t
        if forward_R2_used is not None:
            res["forward_R2_used"] = forward_R2_used

        # Persist the per-target BO trajectory (every evaluated point) so
        # the convergence claim ("the optimum is found in ~7 / 20 evals")
        # can be verified independently of the run log.  One CSV per target.
        tid = t.get("id", f"T{i+1}")
        try:
            gj = res.get("gpbo_best") or res.get("gpbo_joint") or {}
            x_hist = gj.get("x_history")
            y_hist = gj.get("y_history")
            if x_hist is not None and y_hist is not None and len(y_hist) > 0:
                # x_history is a list of (theta, lc_idx) tuples.
                hist_rows = []
                running_best = float("inf")
                for eval_num, (xh, yh) in enumerate(zip(x_hist, y_hist), start=1):
                    if isinstance(xh, (list, tuple)) and len(xh) >= 2:
                        theta_h = float(xh[0])
                        lc_idx_h = int(xh[1])
                        # Translate the encoder index to the actual LC label
                        # ('LC1'/'LC2') so the published convergence-trace CSVs
                        # are self-describing rather than encoder-order-dependent.
                        try:
                            lc_cats = [str(c) for c in inv_en.categories_[0].tolist()]
                            lc_h = lc_cats[lc_idx_h] if 0 <= lc_idx_h < len(lc_cats) else str(lc_idx_h)
                        except Exception:
                            lc_h = str(lc_idx_h)
                    else:
                        theta_h = float(xh)
                        lc_h = ""
                    yh_f = float(yh)
                    running_best = min(running_best, yh_f)
                    hist_rows.append({
                        "target_id":  tid,
                        "eval_num":   eval_num,
                        "theta":      theta_h,
                        "lc":         lc_h,
                        "y":          yh_f,
                        "best_so_far": running_best,
                    })
                bo_hist_df = pd.DataFrame(hist_rows)
                bo_hist_path = os.path.join(
                    bo_history_dir, f"bo_history_{tid}.csv",
                )
                bo_hist_df.to_csv(bo_hist_path, index=False)
                logger.info(
                    f"  Saved per-target BO history: bo_history/{os.path.basename(bo_hist_path)} "
                    f"({len(hist_rows)} evaluations)"
                )
        except Exception as exc:
            logger.warning(f"  per-target BO history save skipped for {tid}: {exc}")

        all_inverse_results.append(res)

    robust_inverse_results = None
    if CFG.run_robustness_analyses:
        robust_inverse_results = []
        # Sweep ALL targets (previously only the first 3): T4/T5 include the
        # bound-active and LC2 recoveries, which are exactly the cases whose
        # seed-robustness a reviewer will question.
        for t in inverse_targets:
            r = run_inverse_design_robust(inv_models, "hard", t["EA"], t["IPF"],
                                          inv_sd, inv_en, inv_p, BO_CFG, logger,
                                          n_seeds=5, cal_ens=cal_ens,
                                          feat_scaler=clf_feat_scaler)
            r["target_info"] = t
            robust_inverse_results.append(r)

    # Verification targets: off-grid round-trips (true θ known by construction)
    # + a deliberately infeasible probe.  These make the inverse demonstration
    # non-circular: T1-T5 recover observed rows, V1-V2 recover angles that
    # exist in no data row, and V3 shows non-attainability is flagged.
    if CFG.run_robustness_analyses and not CFG.dry_run:
        try:
            run_verification_inverse_and_save(
                inv_models, "hard", inv_sd, inv_en, inv_p, df_metrics,
                BO_CFG, cal_ens, clf_feat_scaler, output_dir, logger)
        except Exception as e:
            logger.warning(f"  inverse verification targets skipped: {e}")

    # Design-level forward validation of the deployed (full-data) surrogate:
    # predicted vs experimental EA@D_COMMON / IPF at the 12 measured designs.
    try:
        table_forward_design_errors(inv_models, "hard", inv_sd, inv_en, inv_p,
                                    df_metrics, output_dir, logger)
    except Exception as e:
        logger.warning(f"  forward design-error table skipped: {e}")

    # Inverse interpretability: per-target objective decomposition at the
    # recovered optimum (EA-fit vs IPF-fit vs plausibility penalty).
    try:
        table_inverse_design_explanation(all_inverse_results, cal_ens,
                                         clf_feat_scaler, output_dir, logger)
    except Exception as e:
        logger.warning(f"  inverse explanation table skipped: {e}")

    logger.info("\n[inverse 3/4] Forward Jacobian + multi-objective Pareto sweep")
    jacobian_results = None
    try:
        jacobian_results = compute_forward_map_jacobian(
            inv_models, "hard", inv_sd, inv_en, inv_p,
            (CFG.angle_opt_min, CFG.angle_opt_max), logger)
    except Exception as e:
        logger.warning(f"  jacobian skipped: {e}")

    pareto_df, landscape_df = run_multiobjective_sweep(
        inv_models, "hard", inv_sd, inv_en, inv_p, df_metrics, logger,
        output_dir=output_dir, df_all=df_all)

    pareto_targets, pareto_inverse_results = None, None
    if CFG.run_robustness_analyses and not pareto_df.empty:
        pdom = pareto_df.attrs.get("pareto_dominance", pd.DataFrame())
        if not pdom.empty and len(pdom) >= 5:
            pareto_targets = generate_pareto_targets(pdom, logger, n_targets=5)
            if pareto_targets:
                pareto_inverse_results = []
                for t in pareto_targets:
                    res = run_inverse_design(inv_models, "hard", t["EA"], t["IPF"],
                                             inv_sd, inv_en, inv_p, BO_CFG, logger,
                                             cal_ens=cal_ens, feat_scaler=clf_feat_scaler)
                    res["target_info"] = t
                    pareto_inverse_results.append(res)

    logger.info("\n[inverse 4/4] Sensitivity sweeps (λ, D_COMMON, classifier ablation)")
    lambda_sweep_df = None
    if BO_CFG.lambda_sweep:
        try:
            run_lambda_sensitivity(inv_models, "hard", inverse_targets,
                                   inv_sd, inv_en, inv_p, BO_CFG,
                                   cal_ens, clf_feat_scaler, output_dir, logger)
            p = os.path.join(output_dir, "Table_lambda_sensitivity.csv")
            if os.path.exists(p):
                lambda_sweep_df = pd.read_csv(p)
        except Exception as e:
            logger.warning(f"  λ sweep skipped: {e}")

    classifier_ablation_diag = None
    if BO_CFG.run_classifier_ablation:
        all_no = []
        for t in inverse_targets:
            r = run_inverse_design(inv_models, "hard", t["EA"], t["IPF"],
                                   inv_sd, inv_en, inv_p, BO_CFG, logger,
                                   cal_ens=None, feat_scaler=None)
            r["target_info"] = t
            all_no.append(r)
        rows = []
        for i, t in enumerate(inverse_targets):
            tid = t["id"]; w = all_inverse_results[i]; n = all_no[i]
            wb, nb = w.get("gpbo_best", {}), n.get("gpbo_best", {})
            def _plc(best):
                if not best: return float("nan")
                ang, lc = best.get("x_best"), best.get("lc", best.get("best_lc", ""))
                if ang is None or not lc: return float("nan")
                m = compute_ea_ipf_ensemble(inv_models, "hard", ang, lc,
                                             inv_sd, inv_en, inv_p, d_eval=D_COMMON)
                _, p_lc = compute_lc_penalty(cal_ens, clf_feat_scaler,
                                             m["EA"], m["IPF"], lc,
                                             prob_weight=0.0, angle_deg=float(ang))
                return p_lc
            # Record the DECISION effect, not just p_LC: whether the penalty
            # changed the recovered angle or flipped the selected LC.  A row of
            # near-identical p_LC values with zero decision changes shows the
            # penalty was inactive for feasible targets (expected — it exists
            # to guard implausible regions, cf. the V3 infeasibility probe).
            w_ang = wb.get("x_best"); n_ang = nb.get("x_best")
            w_lc = wb.get("lc", wb.get("best_lc", "")); n_lc = nb.get("lc", nb.get("best_lc", ""))
            rows.append({"Target": tid,
                         "With_Penalty_p_LC": f"{_plc(wb):.4f}",
                         "No_Penalty_p_LC":   f"{_plc(nb):.4f}",
                         "With_Penalty_theta": f"{float(w_ang):.2f}" if w_ang is not None else "",
                         "No_Penalty_theta":   f"{float(n_ang):.2f}" if n_ang is not None else "",
                         "Delta_theta_deg": (f"{abs(float(w_ang) - float(n_ang)):.2f}"
                                             if (w_ang is not None and n_ang is not None) else ""),
                         "With_Penalty_LC": w_lc,
                         "No_Penalty_LC":   n_lc,
                         "LC_flipped": ("Yes" if (w_lc and n_lc and str(w_lc) != str(n_lc)) else "No")})
        classifier_ablation_diag = pd.DataFrame(rows)
        classifier_ablation_diag.to_csv(
            os.path.join(output_dir, "Table_classifier_ablation.csv"), index=False)

    dcommon_diag = None
    try:
        # Symmetric d_common sweep: capped at min(LC1_stroke, LC2_stroke) so
        # BOTH LCs are evaluable at every d_star.  This avoids silent
        # LC-asymmetry from d_star values that fall beyond LC1's natural
        # 80 mm stroke (which would conflate physical limits with reporting
        # gaps).
        lcs_observed = sorted(df_all["LC"].unique())
        min_stroke = min(disp_end_mm(lc) for lc in lcs_observed) if lcs_observed else 80.0
        # Sweep capped at the lowest-stroke LC; step 10 mm starting at 40 mm.
        symmetric_d_grid = [d for d in [40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
                            if d <= min_stroke]
        if not symmetric_d_grid:
            symmetric_d_grid = [int(min_stroke)]
        rows = []
        for d_star in symmetric_d_grid:
            for lc in lcs_observed:
                # Defensive: every (d_star, lc) here is by construction
                # within the LC's natural stroke; double-checked.
                if d_star > disp_end_mm(lc):
                    continue
                ea_list = []
                ipf_list = []
                for ang in np.arange(CFG.angle_opt_min, CFG.angle_opt_max + 1, 5):
                    try:
                        m = compute_ea_ipf_ensemble(inv_models, "hard",
                                                    float(ang), lc,
                                                    inv_sd, inv_en, inv_p,
                                                    d_eval=float(d_star))
                        ea_list.append(m["EA"])
                        if "IPF" in m:
                            ipf_list.append(m["IPF"])
                    except Exception:
                        continue
                if ea_list:
                    row = {"d_common": d_star, "lc": lc,
                           "EA_mean": float(np.mean(ea_list)),
                           "EA_std":  float(np.std(ea_list))}
                    if ipf_list:
                        row["IPF_mean"] = float(np.mean(ipf_list))
                        row["IPF_std"]  = float(np.std(ipf_list))
                    rows.append(row)
        if rows:
            dcommon_diag = pd.DataFrame(rows)
            dcommon_diag.to_csv(os.path.join(output_dir,
                                             "Table_d_common_sensitivity.csv"),
                                index=False)
            n_lc1 = int((dcommon_diag["lc"] == "LC1").sum())
            n_lc2 = int((dcommon_diag["lc"] == "LC2").sum())
            logger.info(
                f"  D_common sweep: {len(rows)} rows over d ∈ {symmetric_d_grid} mm "
                f"({n_lc1} LC1 + {n_lc2} LC2 entries; cap = min-stroke "
                f"{min_stroke:.0f} mm so both LCs are reported at every d_common)"
            )
    except Exception as e:
        logger.warning(f"  D_common sweep skipped: {e}")

    analysis_state = {
        "all_inverse_results": all_inverse_results,
        "robust_inverse_results": robust_inverse_results,
        "jacobian_results": jacobian_results,
        "pareto_df": pareto_df, "landscape_df": landscape_df,
        "pareto_targets": pareto_targets,
        "pareto_inverse_results": pareto_inverse_results,
        "lambda_diag": lambda_diag,
        "lambda_sweep_df": lambda_sweep_df,
        "classifier_ablation_diag": classifier_ablation_diag,
        "dcommon_diag": dcommon_diag,
        "inverse_targets": inverse_targets,
    }
    save_analysis_bundle(analysis_state, output_dir, logger)
    return inverse_state, analysis_state


def _render_all_tables(forward_state: Dict, inverse_state: Dict,
                           analysis_state: Dict, output_dir: str,
                           logger: logging.Logger) -> None:
    """Render every paper table.  Most tables need both forward and inverse
    bundles; missing inputs produce a warning rather than crash."""
    F = forward_state or {}
    I = inverse_state or {}
    A = analysis_state or {}
    dual_results = F.get("dual_results")
    df_metrics   = I.get("df_metrics", pd.DataFrame())
    calibration  = F.get("calibration", {}) or {}
    stat_tests   = F.get("stat_tests")
    all_inv      = A.get("all_inverse_results") or []
    robust_inv   = A.get("robust_inverse_results")

    if dual_results is not None:
        try:
            generate_summary_tables(dual_results, df_metrics, all_inv,
                                    stat_tests or {}, output_dir, logger,
                                    calibration=calibration)
        except Exception as e:
            logger.warning(f"  generate_summary_tables: {e}")
    if all_inv:
        try: generate_optimizer_comparison_table(all_inv, output_dir, logger)
        except Exception as e: logger.warning(f"  optimizer_comparison: {e}")
    if robust_inv:
        try: generate_inverse_robustness_table(robust_inv, output_dir, logger)
        except Exception as e: logger.warning(f"  inverse_robustness: {e}")
    try: write_statistical_testing_policy(output_dir, logger)
    except Exception as e: logger.warning(f"  stat_policy: {e}")
    if dual_results is not None and all_inv:
        try: generate_compute_budget_summary(dual_results, all_inv, output_dir, logger)
        except Exception as e: logger.warning(f"  compute_budget: {e}")
    # Design-level null baseline (experimental EA/IPF interpolation over θ):
    # the accuracy floor the surrogates must beat at θ*.
    df_all_f = F.get("df_all")
    if df_all_f is not None:
        try: table_null_baseline_design_level(df_all_f, CFG.theta_star, output_dir, logger)
        except Exception as e: logger.warning(f"  null_baseline: {e}")
    # Interpretability tables (global variance attribution + per-target
    # inverse objective decomposition).
    landscape_df_a = A.get("landscape_df")
    if landscape_df_a is not None and len(landscape_df_a):
        try: compute_design_variance_decomposition(landscape_df_a, logger, output_dir)
        except Exception as e: logger.warning(f"  variance_decomposition: {e}")
    if all_inv:
        try: table_inverse_design_explanation(all_inv, I.get("cal_ens"),
                                              I.get("clf_feat_scaler"), output_dir, logger)
        except Exception as e: logger.warning(f"  inverse_explanation: {e}")


# =============================================================================
# PIPELINE ORCHESTRATION
# =============================================================================
def run_pipeline(data_dir: str, output_dir: str,
                     logger: Optional[logging.Logger] = None) -> Dict:
    """Inverse-design + multi-objective focused pipeline (mode='all').

    Trains forward ensembles, the full-data Hard-PINN inverse model, the LC
    plausibility classifier, runs every analysis, saves three torch.save
    bundles in ``output_dir``, and renders 17 figures + 6+ tables.

    For HPC parallel submission, run forward and inverse training as two
    separate jobs with ``--mode forward`` / ``--mode inverse`` (each saves
    its bundles into the same ``output_dir``), then run ``--mode replot``
    to load every bundle and produce figures + tables.

    If ``logger`` is provided (the typical path from ``main``), the
    function reuses it instead of calling ``setup_logging`` again — which
    would truncate the existing ``run_log.txt`` and overwrite the banner
    lines already written by ``main``.
    """
    refresh_device()
    os.makedirs(output_dir, exist_ok=True)
    owned_logger = logger is None
    if owned_logger:
        logger = setup_logging(output_dir)
        log_runtime_environment(output_dir, logger)
    apply_dry_run_settings(logger)
    set_publication_style()
    check_publication_dependencies(logger)

    logger.info("=" * 80)
    logger.info("IPINN CRASHWORTHINESS FRAMEWORK — (mode=all)")
    logger.info("=" * 80)
    logger.info(f"Data: {data_dir}  Out: {output_dir}  M={CFG.n_ensemble}  seed={CFG.seed_base}")

    df_all = load_data(data_dir, logger)
    forward_state = _train_forward_only(data_dir, output_dir, logger, df_all=df_all)
    inverse_state, analysis_state = _train_inverse_and_analyze(
        data_dir, output_dir, logger, df_all=df_all)

    logger.info("\n[tables]")
    _render_all_tables(forward_state, inverse_state, analysis_state,
                           output_dir, logger)
    logger.info("\n[figures]")
    _render_all_figures(forward_state, inverse_state, analysis_state,
                            output_dir, logger)

    # Inventory every figure/table/sidecar written above into
    # MANIFEST_outputs.csv.  Must run after all writes so the listing is
    # complete.  Full-pipeline runs also verify the required-artifact list
    # (dry runs skip inverse stages, so the check would false-positive).
    generate_output_manifest(output_dir, logger,
                             check_required=not CFG.dry_run)

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"All results in: {output_dir}")
    logger.info("=" * 80)
    return {"forward": forward_state, "inverse": inverse_state,
            "analysis": analysis_state}



# =============================================================================
# Render all paper figures from already-prepared bundle dicts
# =============================================================================
def _render_all_figures(forward_state: Dict, inverse_state: Dict,
                            analysis_state: Dict, output_dir: str,
                            logger: logging.Logger) -> None:
    """Render the full focused figure suite from already-prepared bundles.

    Each figure handles ONE analysis (parity, residuals, calibration, etc.)
    and renders cleanly in a 3x2 grid (or smaller).  Every call is guarded
    by data availability so missing bundles silently skip their figures
    rather than crashing the replot.

    The figure inventory roughly matches the v16 / v17 focused-figure suite,
    plus a few project-specific additions (validation error maps,
    classifier decision boundary, jacobian) that were already in this
    module.
    """
    set_publication_style()
    F = forward_state or {}
    I = inverse_state or {}
    A = analysis_state or {}

    df_all       = F.get("df_all")
    dual_results = F.get("dual_results")
    calibration  = F.get("calibration", {}) or {}
    # Merge the design-level EA/IPF scalar calibration (inverse bundle) into
    # the calibration dict so the heatmap bands use the matched-estimand
    # factor instead of the pointwise-energy fallback.
    if I.get("ea_ipf_scalar_cal"):
        calibration = {**calibration, "ea_ipf_scalar": I["ea_ipf_scalar_cal"]}
    val_df_u     = F.get("val_df_u")
    scaler_disp_u= F.get("scaler_disp_u")
    enc_u        = F.get("enc_u")
    params_u     = F.get("params_u")
    baseline_u   = F.get("baseline_results_u")
    sensitivity_u= F.get("sensitivity_df_u")

    inv_models   = I.get("inv_models")
    inv_sd       = I.get("inv_scaler_disp")
    inv_en       = I.get("inv_enc")
    inv_p        = I.get("inv_params")
    clf_diag     = I.get("clf_diag")
    cal_ens      = I.get("cal_ens")
    clf_feat_sc  = I.get("clf_feat_scaler")
    df_metrics_i = I.get("df_metrics")

    all_inv      = A.get("all_inverse_results")
    robust_inv   = A.get("robust_inverse_results")
    jacobian     = A.get("jacobian_results")
    pareto_df    = A.get("pareto_df")
    landscape_df = A.get("landscape_df")
    pareto_tgts  = A.get("pareto_targets")
    pareto_inv   = A.get("pareto_inverse_results")
    lambda_diag  = A.get("lambda_sweep_df")
    dcommon_diag = A.get("dcommon_diag")
    clf_ab_diag  = A.get("classifier_ablation_diag")
    inv_targets  = A.get("inverse_targets")

    def _try(name: str, fn, *args, **kwargs):
        """Call ``fn`` and log/swallow any exception."""
        try:
            fn(*args, **kwargs)
        except Exception as e:
            logger.warning(f"  {name} skipped: {e}")

    # ---- Conceptual schematics (data-independent) ----
    _try("Fig_framework_schematic", fig_framework_schematic, output_dir, logger)
    _try("Fig_architecture_schematic", fig_architecture_schematic, output_dir, logger)

    # ---- Dataset overview ----
    if df_all is not None:
        _try("Fig_dataset_overview", fig_dataset_overview,
             df_all, output_dir, logger)

    # ---- Forward-model accuracy figures ----
    if dual_results is not None:
        _try("Fig_parity_*", fig_parity_plots,
             dual_results, output_dir, logger)
        _try("Fig_residuals_*", fig_residual_histograms,
             dual_results, output_dir, logger)
        _try("Fig_boxplot_comparison", fig_boxplot_comparison,
             dual_results, output_dir, logger)
        _try("Fig_training_curves", fig_training_curves,
             dual_results, output_dir, logger)
        _try("Fig_model_complexity", fig_model_complexity,
             dual_results, output_dir, logger)
        _try("Fig_validation_error_maps", fig_validation_error_maps,
             dual_results, output_dir, logger)
        if {"random", "unseen"} & set(dual_results.keys()):
            _try("Fig_cross_protocol", fig_cross_protocol_comparison,
                 dual_results, output_dir, logger)
        if df_all is not None:
            _try("Fig_unseen_curves", fig_unseen_curves,
                 dual_results, df_all, output_dir, logger,
                 calibration=calibration)
            _try("Fig_random_grid_curves", fig_random_grid_curves,
                 dual_results, df_all, output_dir, logger)

    # ---- Physics consistency + calibration ----
    if (dual_results is not None and val_df_u is not None
            and scaler_disp_u is not None and enc_u is not None
            and params_u is not None):
        _try("Fig_physics_verification", fig_physics_verification,
             dual_results, val_df_u, scaler_disp_u, enc_u, params_u,
             output_dir, logger)
        _try("Fig_qq_load_residuals_unseen", fig_qq_load_residuals,
             dual_results, output_dir, logger)
    if calibration:
        _try("Fig_reliability_diagram", fig_reliability_diagram,
             calibration, output_dir, logger)

    # ---- Baselines and HP sensitivity (optional) ----
    if baseline_u is not None and dual_results is not None:
        _try("Fig_baseline_comparison_unseen", fig_baseline_comparison,
             baseline_u, dual_results, output_dir, logger, protocol="unseen")
    if sensitivity_u is not None:
        _try("Fig_hyperparam_sensitivity_unseen", fig_hyperparam_sensitivity,
             sensitivity_u, output_dir, logger, tag="unseen")
        _try("Fig_penalty_weight_sensitivity", fig_penalty_weight_sensitivity,
             sensitivity_u, dual_results, output_dir, logger)

    # ---- Inverse problem ill-posedness diagnostics ----
    if jacobian is not None:
        _try("Fig_forward_map_jacobian", fig_forward_map_jacobian,
             jacobian, output_dir, logger)
    if all_inv:
        _try("Fig_solution_landscape", fig_solution_landscape,
             all_inv, output_dir, logger)
        _try("Fig_inverse_posterior", fig_inverse_posterior,
             all_inv, output_dir, logger)
        _try("Fig_inverse_posterior_likelihood",
             fig_inverse_posterior_likelihood,
             all_inv, output_dir, logger)
    if landscape_df is not None and not landscape_df.empty:
        _try("Fig_landscape_ensemble_disagreement",
             fig_landscape_ensemble_disagreement,
             landscape_df, output_dir, logger)

    # ---- LC plausibility classifier ----
    if clf_diag is not None:
        _try("Fig_lc_classifier_cv_diagnostics",
             fig_lc_classifier_diagnostics,
             clf_diag, output_dir, logger)
    if (cal_ens is not None and clf_feat_sc is not None
            and df_metrics_i is not None):
        _try("Fig_classifier_decision_boundary",
             fig_classifier_decision_boundary,
             cal_ens, clf_feat_sc, df_metrics_i, output_dir, logger)
    if clf_ab_diag is not None and len(clf_ab_diag):
        _try("Fig_classifier_effect", fig_classifier_effect,
             clf_ab_diag, output_dir, logger)
    if lambda_diag is not None and len(lambda_diag):
        _try("Fig_lambda_sensitivity", fig_lambda_sensitivity,
             lambda_diag, output_dir, logger, lambda_opt=A.get("lambda_opt"))

    # ---- Inverse design (GP-BO target matching) ----
    if all_inv:
        for res in all_inv:
            tid = (res.get("target_info") or {}).get("id", "T?")
            _try(f"Fig_bo_convergence_{tid}", fig_bo_convergence,
                 res, output_dir, logger, tag=tid)
            _try(f"Fig_gpbo_posterior_evaluation_{tid}",
                 fig_bo_posterior_evaluation,
                 res, output_dir, logger, tag=tid)
            _try(f"Fig_inverse_convergence_{tid}",
                 fig_inverse_optimizer_convergence,
                 res, output_dir, logger, tag=tid)
        _try("Fig_optimizer_comparison", fig_optimizer_comparison,
             all_inv, output_dir, logger)
        _try("Fig_inverse_parity_uncertainty",
             fig_inverse_parity_uncertainty,
             all_inv, output_dir, logger)
    if (all_inv and inv_models is not None and df_all is not None
            and inv_sd is not None and inv_en is not None
            and inv_p is not None):
        _try("Fig_inverse_vs_nearest_experimental_curve",
             fig_inverse_vs_nearest_experimental_curve,
             df_all, all_inv, inv_models, inv_sd, inv_en, inv_p,
             output_dir, logger)
    if df_metrics_i is not None and inv_targets:
        _try("Fig_inverse_target_feasibility", fig_target_feasibility,
             df_metrics_i, inv_targets, output_dir, logger)

    # ---- Design space sweep ----
    if (inv_models is not None and inv_sd is not None
            and inv_en is not None and inv_p is not None):
        _try("Fig_design_space", fig_design_space,
             inv_models, "hard", inv_sd, inv_en, inv_p, output_dir, logger,
             df_metrics=df_metrics_i)
        if df_metrics_i is not None:
            _try("Table_forward_design_errors", table_forward_design_errors,
                 inv_models, "hard", inv_sd, inv_en, inv_p, df_metrics_i,
                 output_dir, logger)

    # ---- Multi-objective Pareto ----
    if pareto_df is not None and not pareto_df.empty:
        _try("Fig_pareto_tradeoff", fig_pareto_tradeoff,
             pareto_df, output_dir, logger)
        if landscape_df is not None and not landscape_df.empty:
            _try("Fig_multiobjective_heatmaps", fig_multiobjective_heatmaps,
                 pareto_df, landscape_df, output_dir, logger,
                 calibration=calibration)

    # ---- Robustness / sensitivity sweeps ----
    if (landscape_df is not None and not landscape_df.empty
            and inv_models is not None and inv_sd is not None
            and inv_en is not None and inv_p is not None):
        _try("Fig_d_common_sensitivity_EA_vs_disp_endpoint",
             fig_d_common_sensitivity_ea,
             inv_models, "hard", inv_sd, inv_en, inv_p,
             landscape_df, output_dir, logger)


# =============================================================================
# Replot mode: load whatever bundles are present and re-render figures
# =============================================================================
def replot_from_bundles(source_dir: str, output_dir: str, logger: logging.Logger) -> None:
    """Reload bundle(s) from ``source_dir`` and regenerate every figure +
    table whose inputs are available.  Missing bundles are silently skipped.

    Used after parallel ``--mode forward`` / ``--mode inverse`` jobs land
    their bundles into a common directory: invoking ``--mode replot
    --replot_from <dir>`` unifies them into the final figure + table set.
    """
    os.makedirs(output_dir, exist_ok=True)
    F = load_forward_bundle(source_dir, logger) or {}
    I = load_inverse_bundle(source_dir, logger) or {}
    A = load_analysis_bundle(source_dir, logger) or {}
    if not (F or I or A):
        logger.warning(f"  No bundles found in: {source_dir}")
        return
    # Provenance cross-check: refuse to silently combine bundles produced from
    # different data or configs (they may individually look fine but the
    # combined figures/tables would be inconsistent).
    metas = {name: b.get("_meta") for name, b in (("forward", F), ("inverse", I), ("analysis", A))
             if b and b.get("_meta")}
    for key in ("data_fingerprint", "cfg_sha", "git_sha"):
        vals = {name: m.get(key) for name, m in metas.items() if m.get(key)}
        if len(set(vals.values())) > 1:
            logger.warning(f"  BUNDLE PROVENANCE MISMATCH on '{key}': {vals} — "
                           f"the loaded bundles were produced by different runs; "
                           f"combined outputs may be inconsistent")
    _render_all_tables(F, I, A, output_dir, logger)
    _render_all_figures(F, I, A, output_dir, logger)
    # Refresh the output manifest so MANIFEST_outputs.csv reflects the
    # regenerated figures and tables.
    generate_output_manifest(output_dir, logger)


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="IPINN Crashworthiness Framework.  Run modes "
                    "let you split forward and inverse training across two "
                    "parallel HPC jobs and merge the outputs in a third quick "
                    "replot step.",
    )
    parser.add_argument("--data_dir",   type=str, default=".",
                        help="Directory containing data files (LC1.xlsx, LC2.xlsx).")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for figures, tables, and bundles.")
    parser.add_argument("--n_ensemble", type=int, default=20,
                        help="Forward ensemble size (default: 20).")
    parser.add_argument("--max_epochs", type=int, default=0,
                        help="Cap every training config's epochs at this value (>0) for "
                             "feasible reduced/local runs; 0 = full tuned budgets (default).")
    parser.add_argument("--seed",       type=int, default=2026,
                        help="Random seed base.")
    parser.add_argument("--no_robustness", action="store_true",
                        help="Skip baseline + sensitivity + robustness extras.")
    parser.add_argument("--force_cpu",  action="store_true",
                        help="Use CPU even if CUDA is available.")
    parser.add_argument("--strict_paper", action="store_true",
                        help="Abort if optional deps (skopt) are missing.")
    parser.add_argument("--dry_run",    action="store_true",
                        help="CI/smoke: tiny ensemble, short epochs.")
    parser.add_argument(
        "--mode", choices=["all", "forward", "inverse", "replot"],
        default="all",
        help=("'all' (default): full pipeline — train everything, then tables + "
              "figures.  'forward': train only the forward ensembles and write "
              "forward_models.pt; no figures or tables.  'inverse': train only "
              "the full-data Hard-PINN + classifier and run all inverse-design "
              "analyses (BO, Pareto, robustness, sensitivity); writes "
              "inverse_models.pt + analysis_results.pt; no figures or tables.  "
              "'replot': read whatever bundles are present in --replot_from and "
              "regenerate figures + tables from them.  Forward and inverse "
              "modes are independent — submit them as two parallel HPC jobs "
              "into a shared output_dir, then run replot to merge."))
    parser.add_argument(
        "--replot_from", type=str, default=None,
        help="Source directory for replot mode.  Defaults to --output_dir.")
    parser.add_argument(
        "--use_pretrained_inverse", type=str, default=None,
        help=("Path to a pretrained inverse-design surrogate bundle produced "
              "by hpo.inverse_merge.  When set with --mode inverse, "
              "the full-data Hard-PINN training is skipped — the M=20 ensemble "
              "is reconstructed from the bundle's state_dicts and the pipeline "
              "proceeds directly to classifier training, GP-BO target matching, "
              "Jacobian, Pareto, and the rest.  Use this when the per-member "
              "SLURM array (hpo/inverse_member.py) has produced the bundle."))
    args = parser.parse_args()

    CFG.dry_run                = bool(args.dry_run)
    CFG.n_ensemble             = args.n_ensemble
    CFG.max_epochs             = int(args.max_epochs)
    CFG.seed                   = args.seed
    CFG.seed_base              = args.seed
    CFG.split_seed             = args.seed
    BO_CFG.seed                = args.seed
    CFG.run_robustness_analyses = not args.no_robustness
    CFG.strict_paper_deps       = bool(args.strict_paper)
    # Replot mode never needs GPU: pin to CPU so a CUDA-trained bundle can
    # be re-rendered on a CPU-only box.  (Forward and inverse training modes
    # honour --force_cpu but otherwise prefer the GPU.)
    CFG.force_cpu = bool(args.force_cpu) or args.mode == "replot"
    refresh_device()

    os.makedirs(args.output_dir, exist_ok=True)

    # Mode-tagged log + runtime-env filenames so parallel forward + inverse
    # SLURM jobs sharing one output_dir don't truncate each other's run_log
    # or runtime_environment.  --mode all and --mode replot use untagged
    # names (single-process; nothing to collide with).
    log_tag = args.mode if args.mode in {"forward", "inverse"} else ""
    logger = setup_logging(args.output_dir, tag=log_tag)
    logger.info("=" * 80)
    logger.info(f"IPINN (mode={args.mode!r})")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 80)

    # Apply dry-run + dependency guards BEFORE the mode dispatch so every
    # mode (not just ``all``) honours ``--dry_run`` and ``--strict_paper``.
    # Both are idempotent — ``run_pipeline`` may call them again for
    # ``--mode all`` with no effect.
    apply_dry_run_settings(logger)
    set_publication_style()
    check_publication_dependencies(logger)

    if args.mode == "replot":
        src = args.replot_from or args.output_dir
        logger.info(f"Replot source: {src}")
        replot_from_bundles(src, args.output_dir, logger)
        return

    if args.mode == "forward":
        log_runtime_environment(args.output_dir, logger, tag=log_tag)
        _train_forward_only(args.data_dir, args.output_dir, logger)
        logger.info("\n[forward-only mode COMPLETE] forward_models.pt saved; "
                    "run --mode replot to render figures + tables once "
                    "inverse_models.pt and analysis_results.pt are also present.")
        return

    if args.mode == "inverse":
        log_runtime_environment(args.output_dir, logger, tag=log_tag)
        pretrained = None
        if args.use_pretrained_inverse:
            if not os.path.isfile(args.use_pretrained_inverse):
                raise FileNotFoundError(
                    f"--use_pretrained_inverse path does not exist: "
                    f"{args.use_pretrained_inverse}"
                )
            logger.info(
                f"  Loading pretrained inverse surrogate from "
                f"{args.use_pretrained_inverse}"
            )
            try:
                pretrained = torch.load(
                    args.use_pretrained_inverse, map_location="cpu", weights_only=False,
                )
            except TypeError:
                pretrained = torch.load(args.use_pretrained_inverse, map_location="cpu")
            if pretrained.get("kind") != "inverse_full_data_pretrained":
                logger.warning(
                    f"  Pretrained bundle's 'kind' is "
                    f"{pretrained.get('kind')!r}; expected "
                    f"'inverse_full_data_pretrained'.  Proceeding anyway."
                )
        _train_inverse_and_analyze(args.data_dir, args.output_dir, logger,
                                   pretrained_inverse=pretrained)
        logger.info("\n[inverse-only mode COMPLETE] inverse_models.pt + "
                    "analysis_results.pt saved; run --mode replot to render "
                    "figures + tables once forward_models.pt is also present.")
        return

    # mode == "all"
    log_runtime_environment(args.output_dir, logger, tag=log_tag)
    run_pipeline(args.data_dir, args.output_dir, logger=logger)


if __name__ == "__main__":
    main()