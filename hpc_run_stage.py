#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HPC Stage Dispatcher for IPINN Crashworthiness Pipeline
========================================================
Splits the monolithic run_pipeline() into independent SLURM stages that
communicate via pickle files in {output_dir}/.staging/.

Usage:
    python hpc_run_stage.py --stage prep --data_dir . --output_dir ./results_hpc
    python hpc_run_stage.py --stage train_random_hard --output_dir ./results_hpc
    ...

The existing `python composite_design_v19.py --strict_paper` still works
unchanged for single-machine sequential runs.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import logging
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Import the main pipeline module
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import composite_design_v19 as cd  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _staging(output_dir: str) -> str:
    d = os.path.join(output_dir, ".staging")
    os.makedirs(d, exist_ok=True)
    return d


def _save(obj: Any, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _load(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _logger(output_dir: str, tag: str) -> logging.Logger:
    return cd.setup_logging(output_dir)


# ---------------------------------------------------------------------------
# Model serialisation (state_dict based — portable across import paths)
# ---------------------------------------------------------------------------

def _serialize_model(model, approach: str, protocol: str, scaler_disp) -> Dict:
    cfg = cd.get_model_config(approach, protocol)
    info = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "class": type(model).__name__,
        "init_args": {
            "in_d": 5,  # always: 1 disp + 2 angle_trig + 2 LC_onehot
            "hidden_layers": cfg["hidden_layers"],
            "dropout": cfg["dropout"],
            "softplus_beta": cfg["softplus_beta"],
        },
    }
    return info


def _deserialize_model(info: Dict, device: torch.device):
    cls = cd.HardEnergyNet if info["class"] == "HardEnergyNet" else cd.SoftPINNNet
    model = cls(**info["init_args"]).to(device)
    model.load_state_dict(info["state_dict"])
    model.eval()
    return model


def _serialize_ensemble(result: Dict, approach: str, protocol: str, scaler_disp) -> Dict:
    """Deep-copy result dict, replace live nn.Module list with state_dict list."""
    out = {}
    for k, v in result.items():
        if k == "models":
            out[k] = [_serialize_model(m, approach, protocol, scaler_disp) for m in v]
        else:
            out[k] = v
    return out


def _deserialize_ensemble(result: Dict, device: torch.device) -> Dict:
    out = dict(result)
    if "models" in out and out["models"] and isinstance(out["models"][0], dict):
        out["models"] = [_deserialize_model(m, device) for m in out["models"]]
    return out


# ---------------------------------------------------------------------------
# Config application (mirrors main() CLI → CFG mapping)
# ---------------------------------------------------------------------------

def _apply_cfg(args) -> None:
    cd.CFG.dry_run = bool(getattr(args, "dry_run", False))
    cd.CFG.n_ensemble = int(getattr(args, "n_ensemble", 20))
    cd.CFG.seed = int(getattr(args, "seed", 2026))
    cd.CFG.seed_base = cd.CFG.seed
    cd.CFG.split_seed = cd.CFG.seed
    cd.BO_CFG.seed = cd.CFG.seed
    cd.CFG.force_cpu = bool(getattr(args, "force_cpu", False))
    cd.CFG.strict_paper_deps = bool(getattr(args, "strict_paper", False))
    cd.CFG.show_plots = False
    cd.CFG.run_reviewer_proof = True
    cd.CFG.run_ablation = True
    cd.CFG.run_mobo_qnehvi = True
    cd.CFG._skip_mobo = False
    cd.CFG.run_loao_cv = True
    cd.CFG.run_rar = True
    cd.refresh_device()
    cd.set_publication_style()
    if cd.CFG.dry_run:
        cd.apply_dry_run_settings(logging.getLogger("hpc_setup"))


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def run_prep(args):
    out = args.output_dir
    stg = _staging(out)
    logger = _logger(out, "prep")
    cd.log_runtime_environment(out, logger)
    cd.check_publication_dependencies(logger)

    df_all = cd.load_data(args.data_dir, logger)
    _save(df_all, os.path.join(stg, "df_all.pkl"))

    # Random split
    train_r, val_r = cd.split_random_80_20(df_all, cd.CFG.split_seed, logger)
    sd_r, so_r, enc_r, par_r = cd.create_preprocessors(train_r, logger)
    cd.save_reproducibility_artifacts(out, "random", train_r, sd_r, so_r, enc_r, par_r, logger)
    _save({"train_df": train_r, "val_df": val_r, "scaler_disp": sd_r,
           "scaler_out": so_r, "enc": enc_r, "params": par_r},
          os.path.join(stg, "random_split.pkl"))

    # Unseen split
    train_u, val_u = cd.split_unseen_angle(df_all, cd.CFG.theta_star, logger)
    sd_u, so_u, enc_u, par_u = cd.create_preprocessors(train_u, logger)
    cd.save_reproducibility_artifacts(out, "unseen", train_u, sd_u, so_u, enc_u, par_u, logger)
    _save({"train_df": train_u, "val_df": val_u, "scaler_disp": sd_u,
           "scaler_out": so_u, "enc": enc_u, "params": par_u},
          os.path.join(stg, "unseen_split.pkl"))

    # Design-space metrics
    df_metrics = cd.compute_design_space_metrics(df_all, logger)
    cd.enrich_df_metrics_ea_common(df_metrics, df_all, logger)
    _save(df_metrics, os.path.join(stg, "df_metrics.pkl"))

    # Save config snapshot
    _save({
        "CFG": dataclasses.asdict(cd.CFG),
        "BO_CFG": dataclasses.asdict(cd.BO_CFG),
        "MOBO_CFG": dataclasses.asdict(cd.MOBO_QNEHVI_CFG),
    }, os.path.join(stg, "cfg.pkl"))

    logger.info("PREP COMPLETE")


def run_train_ensemble(args):
    """Generic training stage — parses protocol/approach from stage name."""
    stage = args.stage  # e.g. "train_random_ddns"
    parts = stage.replace("train_", "").split("_", 1)
    protocol, approach = parts[0], parts[1]
    out = args.output_dir
    stg = _staging(out)
    logger = _logger(out, stage)
    logger.info(f"TRAINING: protocol={protocol}, approach={approach}")

    split = _load(os.path.join(stg, f"{protocol}_split.pkl"))
    train_df = split["train_df"]
    val_df = split["val_df"]
    sd, so, enc, par = split["scaler_disp"], split["scaler_out"], split["enc"], split["params"]

    result = cd.train_ensemble(approach, train_df, val_df, sd, so, enc, par, protocol, logger)
    serialised = _serialize_ensemble(result, approach, protocol, sd)
    _save(serialised, os.path.join(stg, f"train_{protocol}_{approach}.pkl"))
    logger.info(f"TRAINING COMPLETE: {stage}")


def run_train_inverse(args):
    out = args.output_dir
    stg = _staging(out)
    logger = _logger(out, "train_inverse")

    df_all = _load(os.path.join(stg, "df_all.pkl"))
    inv_models, inv_sd, inv_so, inv_enc, inv_par = cd.train_full_data_hard_pinn(df_all, logger)

    serialised_models = [_serialize_model(m, "hard", "unseen", inv_sd) for m in inv_models]
    _save({
        "models": serialised_models,
        "scaler_disp": inv_sd, "scaler_out": inv_so,
        "enc": inv_enc, "params": inv_par,
    }, os.path.join(stg, "train_inverse.pkl"))
    logger.info("INVERSE TRAINING COMPLETE")


def run_loao_cv(args):
    out = args.output_dir
    stg = _staging(out)
    logger = _logger(out, "loao_cv")

    df_all = _load(os.path.join(stg, "df_all.pkl"))
    loao = cd.run_leave_one_angle_out_cv(df_all, logger)
    _save(loao, os.path.join(stg, "loao_cv.pkl"))

    import pandas as pd
    cd.fig_loao_cv_results(loao, out, logger)
    pd.DataFrame(loao["per_fold_r2"]).to_csv(os.path.join(out, "Table_loao_cv.csv"), index=False)
    logger.info("LOAO-CV COMPLETE")


def run_forward_analysis(args):
    import pandas as pd
    out = args.output_dir
    stg = _staging(out)
    logger = _logger(out, "forward_analysis")

    df_all = _load(os.path.join(stg, "df_all.pkl"))
    df_metrics = _load(os.path.join(stg, "df_metrics.pkl"))
    r_split = _load(os.path.join(stg, "random_split.pkl"))
    u_split = _load(os.path.join(stg, "unseen_split.pkl"))
    device = cd.DEVICE

    # Reassemble dual_results
    dual_results = {}
    for protocol, split in [("random", r_split), ("unseen", u_split)]:
        dual_results[protocol] = {
            "train_df": split["train_df"], "val_df": split["val_df"],
            "scaler_disp": split["scaler_disp"], "scaler_out": split["scaler_out"],
            "enc": split["enc"], "params": split["params"],
        }
        for approach in ["ddns", "soft", "hard"]:
            pkl_path = os.path.join(stg, f"train_{protocol}_{approach}.pkl")
            if os.path.exists(pkl_path):
                ens = _load(pkl_path)
                ens = _deserialize_ensemble(ens, device)
                dual_results[protocol][approach] = ens

    # Statistical tests
    stat_tests = cd.compute_statistical_tests(dual_results, logger)
    _save(stat_tests, os.path.join(stg, "stat_tests.pkl"))

    # Ablation
    df_ablation = pd.DataFrame()
    if cd.CFG.run_ablation:
        s = u_split
        df_ablation = cd.run_ablation_study(
            s["train_df"], s["val_df"], s["scaler_disp"], s["scaler_out"],
            s["enc"], s["params"], "unseen", logger)
        df_ablation.to_csv(os.path.join(out, "Table5_ablation.csv"), index=False)

    # Calibration
    calibration = cd.compute_uncertainty_calibration(dual_results, logger)
    _save(calibration, os.path.join(stg, "calibration.pkl"))

    # Forward figures
    cd.fig_parity_plots(dual_results, out, logger)
    cd.fig_residual_histograms(dual_results, out, logger)
    cd.fig_boxplot_comparison(dual_results, out, logger)
    cd.fig_training_curves(dual_results, out, logger)
    cd.fig_cross_protocol_comparison(dual_results, out, logger)
    cd.fig_unseen_curves(dual_results, df_all, out, logger, calibration=calibration)
    cd.fig_random_grid_curves(dual_results, df_all, out, logger)
    cd.fig_validation_error_maps(dual_results, out, logger)
    cd.table_validation_errors_by_angle_bin(dual_results, out, logger)
    cd.fig_qq_load_residuals(dual_results, out, logger)

    # Reviewer-proof
    if cd.CFG.run_reviewer_proof:
        s = u_split
        cd.fig_physics_verification(
            dual_results, s["val_df"], s["scaler_disp"], s["enc"], s["params"], out, logger)
        baseline_r = cd.train_baseline_models(
            s["train_df"], s["val_df"], s["scaler_disp"], s["enc"], s["params"], logger)
        cd.fig_baseline_comparison(baseline_r, dual_results, out, logger, protocol="unseen")
        sens_df = cd.run_hyperparam_sensitivity(
            s["train_df"], s["val_df"], s["scaler_disp"], s["scaler_out"],
            s["enc"], s["params"], "unseen", logger)
        cd.fig_hyperparam_sensitivity(sens_df, out, logger, tag="unseen")
        cd.fig_reliability_diagram(calibration, out, logger)
        try:
            cal_vs_M = cd.compute_calibration_vs_M(dual_results, logger)
            cd.fig_calibration_vs_M(cal_vs_M, out, logger)
        except Exception as e:
            logger.warning(f"Calibration vs M failed: {e}")

    # Save dual_results for downstream (serialise models)
    dr_ser = {}
    for protocol in dual_results:
        dr_ser[protocol] = {}
        for k, v in dual_results[protocol].items():
            if k in ["ddns", "soft", "hard"]:
                dr_ser[protocol][k] = _serialize_ensemble(v, k, protocol, dual_results[protocol]["scaler_disp"])
            else:
                dr_ser[protocol][k] = v
    _save(dr_ser, os.path.join(stg, "dual_results.pkl"))
    logger.info("FORWARD ANALYSIS COMPLETE")


def run_inverse_analysis(args):
    import pandas as pd
    out = args.output_dir
    stg = _staging(out)
    logger = _logger(out, "inverse_analysis")
    device = cd.DEVICE

    df_all = _load(os.path.join(stg, "df_all.pkl"))
    df_metrics = _load(os.path.join(stg, "df_metrics.pkl"))
    calibration = _load(os.path.join(stg, "calibration.pkl"))

    # Load and deserialise dual_results (needed for publication artifacts)
    dr_ser = _load(os.path.join(stg, "dual_results.pkl"))
    dual_results = {}
    for protocol in dr_ser:
        dual_results[protocol] = {}
        for k, v in dr_ser[protocol].items():
            if k in ["ddns", "soft", "hard"] and isinstance(v, dict) and "models" in v:
                dual_results[protocol][k] = _deserialize_ensemble(v, device)
            else:
                dual_results[protocol][k] = v

    # Load inverse models
    inv_data = _load(os.path.join(stg, "train_inverse.pkl"))
    inv_models = [_deserialize_model(m, device) for m in inv_data["models"]]
    inv_sd = inv_data["scaler_disp"]
    inv_so = inv_data["scaler_out"]
    inv_enc = inv_data["enc"]
    inv_par = inv_data["params"]

    # Targets
    inverse_targets = cd.generate_feasible_targets(df_metrics, logger, df_all=df_all)

    # Classifier
    cal_ens, clf_fs, clf_diag = cd.train_lc_plausibility_classifier(df_metrics, logger)
    cd.fig_lc_classifier_diagnostics(clf_diag, out, logger)
    cd.generate_classifier_diagnostics_table(cal_ens, clf_fs, clf_diag, out, logger)
    cd.fig_classifier_decision_boundary(cal_ens, clf_fs, df_metrics, out, logger)

    lambda_opt, _ = cd.auto_tune_lambda(cal_ens, clf_fs, df_metrics, logger)
    cd.BO_CFG.prob_weight = lambda_opt
    beta_robust = cd.auto_tune_beta(inv_models, "hard", df_metrics, inv_sd, inv_enc, inv_par, logger)
    cd.BO_CFG.beta_robust = beta_robust

    # Inverse design loop
    all_inverse_results = []
    for target in inverse_targets:
        logger.info(f"\n  Target {target['id']}: EA={target['EA']:.2f}J, IPF={target['IPF']:.3f}kN")
        res = cd.run_inverse_design(
            inv_models, "hard", target["EA"], target["IPF"],
            inv_sd, inv_enc, inv_par, cd.BO_CFG, logger,
            cal_ens=cal_ens, feat_scaler=clf_fs)
        res["target_info"] = target
        all_inverse_results.append(res)
        cd.fig_inverse_optimizer_convergence(res, out, logger, tag=target['id'])
        cd.fig_bo_posterior_evaluation(res, out, logger, tag=target['id'])

    # Inverse figures
    if all_inverse_results:
        cd.generate_optimizer_comparison_table(all_inverse_results, out, logger)
    cd.fig_design_space(inv_models, "hard", inv_sd, inv_enc, inv_par, out, logger)
    if all_inverse_results:
        cd.fig_inverse_parity_uncertainty(all_inverse_results, out, logger)
        cd.fig_inverse_vs_nearest_experimental_curve(
            df_all, all_inverse_results, inv_models, inv_sd, inv_enc, inv_par, out, logger)
        cd.fig_solution_landscape(all_inverse_results, out, logger)
        cd.fig_inverse_posterior(all_inverse_results, out, logger)

    # Jacobian
    jacobian_results = None
    try:
        jacobian_results = cd.compute_forward_map_jacobian(
            inv_models, "hard", inv_sd, inv_enc, inv_par,
            (cd.CFG.angle_opt_min, cd.CFG.angle_opt_max), logger)
        cd.fig_forward_map_jacobian(jacobian_results, out, logger)
    except Exception as e:
        logger.warning(f"Jacobian analysis failed: {e}")

    # Lambda sensitivity
    if cd.BO_CFG.lambda_sweep:
        cd.run_lambda_sensitivity(
            inv_models, "hard", inverse_targets, inv_sd, inv_enc, inv_par,
            cd.BO_CFG, cal_ens, clf_fs, out, logger)

    # Classifier ablation
    if cd.BO_CFG.run_classifier_ablation:
        all_inv_no_clf = []
        for target in inverse_targets:
            res_no = cd.run_inverse_design(
                inv_models, "hard", target["EA"], target["IPF"],
                inv_sd, inv_enc, inv_par, cd.BO_CFG, logger,
                cal_ens=None, feat_scaler=None)
            res_no["target_info"] = target
            all_inv_no_clf.append(res_no)
        # Build comparison table
        rows = []
        for t, r_with, r_without in zip(inverse_targets, all_inverse_results, all_inv_no_clf):
            bw = r_with.get("gpbo_best", {})
            bwo = r_without.get("gpbo_best", {})
            rows.append({
                "target_id": t["id"],
                "with_clf_theta": bw.get("x_best"), "with_clf_lc": bw.get("lc"),
                "with_clf_J": bw.get("y_best"),
                "without_clf_theta": bwo.get("x_best"), "without_clf_lc": bwo.get("lc"),
                "without_clf_J": bwo.get("y_best"),
            })
        pd.DataFrame(rows).to_csv(os.path.join(out, "Table_classifier_ablation.csv"), index=False)

    # Multi-seed robustness
    if cd.CFG.run_reviewer_proof:
        robust_results = []
        for target in inverse_targets[:3]:
            rr = cd.run_inverse_design_robust(
                inv_models, "hard", target["EA"], target["IPF"],
                inv_sd, inv_enc, inv_par, cd.BO_CFG, logger, n_seeds=5,
                cal_ens=cal_ens, feat_scaler=clf_fs)
            robust_results.append(rr)
        cd.generate_inverse_robustness_table(robust_results, out, logger)

    # Multiobjective sweep
    pareto_df, landscape_df = cd.run_multiobjective_sweep(
        inv_models, "hard", inv_sd, inv_enc, inv_par,
        df_metrics, logger, output_dir=out, df_all=df_all)
    cd.fig_pareto_tradeoff(pareto_df, out, logger)
    cd.fig_multiobjective_heatmaps(pareto_df, landscape_df, out, logger, calibration=calibration)

    # MOBO
    mobo_result = None
    if cd.HAS_BOTORCH and not getattr(cd.CFG, '_skip_mobo', False):
        mobo_result = cd.run_multiobjective_mobo_qnehvi(
            inv_models, "hard", inv_sd, inv_enc, inv_par,
            landscape_df, logger, output_dir=out, cfg=cd.MOBO_QNEHVI_CFG)
    cd.fig_moo_objective_space_validation(
        pareto_df, landscape_df, out, logger, mobo_result=mobo_result, df_metrics=df_metrics)
    cd.fig_mobo_qnehvi_diagnostics(mobo_result, out, logger)

    # Publication artifacts
    cd.generate_inverse_publication_artifacts(
        out, logger, all_inverse_results,
        calibration=calibration, jacobian_results=jacobian_results,
        inv_models=inv_models, inv_scaler_disp=inv_sd, inv_enc=inv_enc,
        inv_params=inv_par, dual_results=dual_results, df_metrics=df_metrics,
        df_all=df_all, bo_cfg=cd.BO_CFG, cal_ens=cal_ens, clf_feat_scaler=clf_fs,
        mobo_result=mobo_result, landscape_df=landscape_df)

    # Save for aggregation
    _save(all_inverse_results, os.path.join(stg, "all_inverse_results.pkl"))
    _save({"pareto_df": pareto_df, "landscape_df": landscape_df}, os.path.join(stg, "landscape.pkl"))
    _save(mobo_result, os.path.join(stg, "mobo_result.pkl"))
    _save(jacobian_results, os.path.join(stg, "jacobian_results.pkl"))
    logger.info("INVERSE ANALYSIS COMPLETE")


def run_aggregate(args):
    out = args.output_dir
    stg = _staging(out)
    logger = _logger(out, "aggregate")
    device = cd.DEVICE

    df_metrics = _load(os.path.join(stg, "df_metrics.pkl"))
    stat_tests = _load(os.path.join(stg, "stat_tests.pkl"))
    calibration = _load(os.path.join(stg, "calibration.pkl"))
    all_inverse = _load(os.path.join(stg, "all_inverse_results.pkl"))

    # Reconstruct dual_results
    dr_ser = _load(os.path.join(stg, "dual_results.pkl"))
    dual_results = {}
    for protocol in dr_ser:
        dual_results[protocol] = {}
        for k, v in dr_ser[protocol].items():
            if k in ["ddns", "soft", "hard"] and isinstance(v, dict) and "models" in v:
                dual_results[protocol][k] = _deserialize_ensemble(v, device)
            else:
                dual_results[protocol][k] = v

    cd.generate_summary_tables(dual_results, df_metrics, all_inverse, stat_tests, out, logger,
                               calibration=calibration)
    cd.write_statistical_testing_policy(out, logger)
    cd.generate_compute_budget_summary(dual_results, all_inverse, out, logger)
    cd.generate_output_manifest(out, logger)
    logger.info("AGGREGATION COMPLETE — all results in: " + out)


# ---------------------------------------------------------------------------
# Stage dispatcher
# ---------------------------------------------------------------------------
STAGES = {
    "prep":                run_prep,
    "train_random_ddns":   run_train_ensemble,
    "train_random_soft":   run_train_ensemble,
    "train_random_hard":   run_train_ensemble,
    "train_unseen_ddns":   run_train_ensemble,
    "train_unseen_soft":   run_train_ensemble,
    "train_unseen_hard":   run_train_ensemble,
    "train_inverse":       run_train_inverse,
    "loao_cv":             run_loao_cv,
    "forward_analysis":    run_forward_analysis,
    "inverse_analysis":    run_inverse_analysis,
    "aggregate":           run_aggregate,
}


def main():
    parser = argparse.ArgumentParser(description="IPINN HPC Stage Dispatcher")
    parser.add_argument("--stage", required=True, choices=list(STAGES.keys()))
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="./results_hpc")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n_ensemble", type=int, default=20)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--strict_paper", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    _apply_cfg(args)

    t0 = time.time()
    print(f"=== Stage: {args.stage} | output: {args.output_dir} | "
          f"device: {cd.DEVICE} | start: {time.strftime('%Y-%m-%d %H:%M:%S')} ===")

    STAGES[args.stage](args)

    elapsed = time.time() - t0
    print(f"=== Stage {args.stage} finished in {elapsed/3600:.2f} h ===")


if __name__ == "__main__":
    main()
