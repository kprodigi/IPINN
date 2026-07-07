# Paper tables

CSV tables that back every numerical result in the manuscript.

## Main tables (manuscript)

| File | Manuscript table | Contents |
|---|---|---|
| `Table1_forward_results.csv` | Table 1 | Forward-prediction metrics: Load R², Energy R², RMSE, MAE, conformal factors per approach |
| `Table2_statistical_tests.csv` | Table 2 | Welch t-tests between approaches with Bonferroni correction |
| `Table3_inverse_design.csv` | Table 3 | Inverse-design recovery: target vs recovered θ, residual error, multiplicity, posterior credible intervals |
| `Table4_model_complexity.csv` | Table 4 | Parameter counts and training time per approach |
| `Table5_per_LC_breakdown.csv` | Table 5 | LC1 vs LC2 split of forward metrics |
| `Table_optimizer_comparison.csv` | Table 6 | GP-BO inverse-design outcomes across the 5 strategic targets: best angle, selected LC, EA/IPF errors, delivered crashworthiness metrics, evaluation count, iterations-to-99%, and wall time |

## Verification and validation tables (manuscript Sections 5.4–5.7)

| File | Contents |
|---|---|
| `Table_inverse_verification.csv` | Off-grid round-trip targets (V1, V2) and the infeasibility probe (V3) with verdicts |
| `Table_forward_design_errors.csv` | Design-level EA/IPF accuracy of the deployed surrogate over all 12 experimental configurations |
| `Table_baseline_comparison.csv` | Conventional-ML baselines (ridge, random forest, XGBoost, subsampled Matérn GP) on the held-out-angle split |
| `Table_null_baseline_design_level.csv` | Model-free interpolation floor at the held-out angle |
| `Table_uncertainty_calibration.csv` | Curve-level split-conformal calibration: raw/corrected coverage and factors |
| `Table_ea_ipf_scalar_calibration.csv` | Design-level scalar calibration (z-scores of realised design errors vs ensemble σ) |
| `Table_physics_verification.csv` | Thermodynamic-consistency residuals per approach |
| `Table_physical_plausibility_audit.csv` | Plausibility audit of all 1,002 design-sweep evaluations (zero violations) |
| `Table_design_variance_decomposition.csv` | Exact ANOVA attribution of design-metric variance to θ, LC, and interaction |
| `Table_inverse_design_explanation.csv` | Per-target decomposition of the inverse objective at the optimum |

## Supplementary tables

| File | Contents |
|---|---|
| `Table_inverse_robustness.csv` | Multi-seed robustness sweep on the inverse search (5 seeds × 5 restarts per target) |
| `Table_compute_reproducibility_budget.csv` | Per-stage compute budget, training times, evaluation counts |
| `Table_pareto_chebyshev.csv` | Pareto sweep results under the Chebyshev scalarisation |
| `Table_pareto_dominance.csv` | Non-dominated subset of the dense surrogate landscape |
| `Table_pareto_lc_dominance_audit.csv` | Pareto dominance check broken down by loading-case |
| `Table_lambda_sensitivity.csv` | Sensitivity of inverse design to the classifier-penalty weight λ |
| `Table_d_common_sensitivity.csv` | Sensitivity to the common-displacement (d_common) cap |
| `Table_d_common_sensitivity_EA_grid.csv` | EA(d) grid behind the d_common sensitivity sweep |
| `Table_classifier_ablation.csv` | LC plausibility-penalty ablation: p(LC) at each inverse target with vs without the classifier penalty |
| `Table_lc_classifier_cv_predictions.csv` | Leave-one-out classifier predictions per design cell |
| `STATISTICAL_TESTING_POLICY.txt` | Statement of the inferential policy (bootstrap CIs primary; t-tests descriptive) |

## Mechanics-analysis tables (data-only, `scripts/mechanics_analysis.py`)

| File | Contents |
|---|---|
| `Table_crush_mode_signatures.csv` | Per-specimen crush signatures: stiffness, plateau force, densification onset, CFE, oscillation |
| `Table_densification_kinematics.csv` | Candidate kinematic H(θ) fits per quantity and LC (LC2 plateau: sinθ·cosθ, R² = 0.947) |
| `Table_master_curve_collapse.csv` | Pairwise curve dispersion under raw vs mechanics normalisations |

## Per-approach forward results

| File | Approach |
|---|---|
| `forward_ddns_results.json` | DDNS — per-member R², ensemble metrics, cfg |
| `forward_soft_results.json` | Soft-PINN — same |
| `forward_hard_results.json` | Hard-PINN — same |

These match the headline figures reported in `Table1_forward_results.csv`
and provide the per-member breakdown for reviewers verifying the
ensemble statistics.
