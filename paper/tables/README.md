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
| `Table_optimizer_comparison.csv` | Table 6 | Optimizer comparison (Adam vs AdamW vs SGD) on the reference Hard-PINN |

## Supplementary tables

| File | Contents |
|---|---|
| `Table_inverse_robustness.csv` | Multi-seed robustness sweep on the inverse search |
| `Table_compute_reproducibility_budget.csv` | Per-stage compute budget, training times, evaluation counts |
| `Table_pareto_chebyshev.csv` | Pareto sweep results under the Chebyshev scalarisation |
| `Table_pareto_dominance.csv` | Non-dominated subset of the dense surrogate landscape |
| `Table_pareto_lc_dominance_audit.csv` | Pareto dominance check broken down by loading-case |
| `Table_lambda_sensitivity.csv` | Sensitivity of inverse design to the classifier-penalty weight λ |
| `Table_d_common_sensitivity.csv` | Sensitivity to the common-displacement (d_common) cap |
| `Table_classifier_ablation.csv` | LC plausibility classifier — ablation across the four learner backends |

## Per-approach forward results

| File | Approach |
|---|---|
| `forward_ddns_results.json` | DDNS — per-member R², ensemble metrics, cfg |
| `forward_soft_results.json` | Soft-PINN — same |
| `forward_hard_results.json` | Hard-PINN — same |

These match the headline figures reported in `Table1_forward_results.csv`
and provide the per-member breakdown for reviewers verifying the
ensemble statistics.
