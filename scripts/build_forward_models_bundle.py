"""Build a combined ``forward_models.pt`` from the per-approach
``forward_<approach>_bundle.pt`` files produced by ``hpo/forward_merge.py``.

The replot mode of ``composite_design.py`` expects a single combined
bundle with the structure assembled by ``_train_and_save_forward``
(see composite_design.py line ~10226).  Per-approach bundles written by
the parallel-SLURM workflow only contain the unseen-protocol ensemble;
this script wraps them into the expected ``dual_results['unseen'][approach]``
layout and computes the calibration block from the loaded models.

Usage:
    python scripts/build_forward_models_bundle.py \\
        --bundle_dir ./results_paper \\
        --data_dir   ./data

Writes ``./results_paper/forward_models.pt``.  After this, the replot
stage can render Fig01, Fig03, Fig04, Fig07 (Fig02 — which requires a
random-80/20 ensemble — will be skipped or use placeholders since the
random protocol was not trained in this workflow).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from typing import Dict, List, Optional

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import composite_design as cd  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning, module=r"matplotlib")


def _make_logger() -> logging.Logger:
    log = logging.getLogger("build_forward_models")
    log.setLevel(logging.INFO)
    log.handlers = []
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    log.addHandler(ch)
    return log


def _reconstruct_models(
    approach: str, bundle: Dict, params: cd.ScalingParams, device: torch.device,
) -> List[torch.nn.Module]:
    """Rebuild M models from saved state_dicts."""
    cfg = bundle["cfg"]
    # Input dim: take the first 2-D weight (layer-0 .weight) and read its
    # second axis.  Robust to whatever ordering torch.save chose for the
    # state_dict keys.
    sd0 = bundle["models_state"][0]["state_dict"]
    first_weight = next(v for k, v in sd0.items() if k.endswith(".weight") and v.ndim == 2)
    in_d = int(first_weight.shape[1])

    models: List[torch.nn.Module] = []
    for entry in bundle["models_state"]:
        if approach in ("ddns", "soft"):
            m = cd.SoftPINNNet(
                in_d, cfg["hidden_layers"], cfg["dropout"], cfg["softplus_beta"],
            ).to(device)
        else:
            m = cd.HardEnergyNet(
                in_d, cfg["hidden_layers"], cfg["dropout"], cfg["softplus_beta"],
            ).to(device)
        # All three approaches: bare-MLP form for inference (BC handled by soft penalties).
        m.configure_zero_bc(params, enabled=False)
        m.load_state_dict({k: v.to(device) for k, v in entry["state_dict"].items()})
        m.eval()
        models.append(m)
    return models


def _build_protocol_dict(
    approach: str, bundle: Dict, params: cd.ScalingParams, device: torch.device,
    logger: logging.Logger,
) -> Dict:
    """Mirror what ``train_ensemble`` would return for the unseen protocol."""
    models = _reconstruct_models(approach, bundle, params, device)
    val_df = bundle["val_df"]
    scaler_disp = bundle["scaler_disp"]
    scaler_out = bundle["scaler_out"]
    enc = bundle["enc"]
    ens_metrics = cd.evaluate_ensemble(
        models, approach, val_df, scaler_disp, scaler_out, enc, params,
    )
    member_metrics = bundle.get("member_metrics", [])
    n_params = sum(p.numel() for p in models[0].parameters())
    logger.info(
        f"  {approach:5s}: M={len(models):2d}  "
        f"ens_load_R²={ens_metrics.get('load_r2', float('nan')):.4f}  "
        f"ens_energy_R²={ens_metrics.get('energy_r2', float('nan')):.4f}  "
        f"n_params={n_params}"
    )
    return {
        "models":             models,
        "histories":          [],  # not preserved per-member; replot fig functions tolerate empty
        "member_metrics":     member_metrics,
        "metrics":            ens_metrics,
        "avg_training_time":  0.0,  # not needed for figures
        "n_params":           n_params,
        "M_total":            len(models),
        "M_eff":              len(models),
        "train_r2_scores":    [],
        "convergence_fence":  float("-inf"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle_dir", default="./results_paper",
                   help="Directory holding forward_<approach>_bundle.pt files.")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--force_cpu", action="store_true")
    args = p.parse_args()

    logger = _make_logger()
    cd.CFG.force_cpu = bool(args.force_cpu)
    cd.refresh_device()
    device = cd.DEVICE

    # Load the three per-approach bundles.
    bundles: Dict[str, Dict] = {}
    for approach in ("ddns", "soft", "hard"):
        path = os.path.join(args.bundle_dir, f"forward_{approach}_bundle.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing bundle: {path}")
        logger.info(f"Loading {path}")
        try:
            bundles[approach] = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            bundles[approach] = torch.load(path, map_location="cpu")

    # The three bundles share scalers / params / val_df / train_df (same unseen split).
    params = bundles["hard"]["params"]
    val_df_u = bundles["hard"]["val_df"]
    train_df_u = bundles["hard"]["train_df"]
    scaler_disp_u = bundles["hard"]["scaler_disp"]
    scaler_out_u = bundles["hard"]["scaler_out"]
    enc_u = bundles["hard"]["enc"]

    # Build dual_results["unseen"] with the three approaches.
    logger.info("\nReconstructing dual_results['unseen']:")
    unseen_dict: Dict = {}
    for approach in ("ddns", "soft", "hard"):
        unseen_dict[approach] = _build_protocol_dict(
            approach, bundles[approach], params, device, logger,
        )
    unseen_dict.update({
        "train_df":     train_df_u,
        "val_df":       val_df_u,
        "scaler_disp":  scaler_disp_u,
        "scaler_out":   scaler_out_u,
        "enc":          enc_u,
        "params":       params,
    })

    # Load original full df for fig_01 etc.
    df_all = cd.load_data(args.data_dir, logger)

    # Compute calibration (needs models from dual_results, so do it now).
    dual_results: Dict = {"unseen": unseen_dict}
    logger.info("\nComputing calibration block ...")
    try:
        calibration = cd.compute_uncertainty_calibration(dual_results, logger)
    except Exception as exc:
        logger.warning(f"  calibration skipped: {type(exc).__name__}: {exc}")
        calibration = {}

    # Statistical tests need both protocols; with only unseen they degrade gracefully.
    logger.info("\nComputing statistical tests ...")
    try:
        stat_tests = cd.compute_statistical_tests(dual_results, logger)
    except Exception as exc:
        logger.warning(f"  stat_tests skipped: {type(exc).__name__}: {exc}")
        stat_tests = {}

    # Assemble forward_state matching what _train_and_save_forward writes.
    forward_state = {
        "dual_results":       dual_results,
        "df_all":             df_all,
        "calibration":        calibration,
        "stat_tests":         stat_tests,
        "val_df_u":           val_df_u,
        "scaler_disp_u":      scaler_disp_u,
        "scaler_out_u":       scaler_out_u,
        "enc_u":              enc_u,
        "params_u":           params,
        "baseline_results_u": None,  # not trained in this workflow
        "sensitivity_df_u":   None,  # not trained in this workflow
    }

    out_path = os.path.join(args.bundle_dir, "forward_models.pt")
    cd.save_forward_bundle(forward_state, args.bundle_dir, logger)
    logger.info(f"\nDone.  Wrote: {out_path}")
    logger.info(
        "Re-run `composite_design.py --mode replot --output_dir <bundle_dir> "
        "--replot_from <bundle_dir>` to generate figures + tables.",
    )


if __name__ == "__main__":
    main()
