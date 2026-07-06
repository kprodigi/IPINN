# Paper figures

All figures are PNG at 600 DPI, rendered with Arial bold typography
sized for insertion at *Composite Structures* full-column width
(7.48 inches / 190 mm).

This is the focused single-purpose figure suite: each figure handles ONE
analysis (parity, residuals, calibration, etc.) and renders cleanly in a
3×2 grid (or smaller).  No subplot-title overflow, no x-label / row-2
collisions, no legend obscuring data.

**This snapshot ships 48 figures** (2 schematics + 28 analysis figures +
15 per-target GP-BO traces + 3 mechanics-analysis figures), regenerated
2026-07-06 from the final production bundles with all layout fixes
applied and every figure visually verified.  The pipeline can
additionally emit 7 *optional* figures that depend on analysis data not
included in this snapshot — they are listed at the end under
[Optional figures](#optional-figures-not-included-in-this-snapshot).

## Recommended placement: main text vs supplementary

Every figure named below is present on disk in this snapshot.

**Main text:** `Fig_framework_schematic`, `Fig_architecture_schematic`,
`Fig_dataset_overview`, `Fig_parity_unseen`, `Fig_residuals_unseen`,
`Fig_cross_protocol`, `Fig_physics_verification`, `Fig_reliability_diagram`,
`Fig_classifier_decision_boundary`, `Fig_design_space`,
`Fig_pareto_tradeoff`, `Fig_multiobjective_heatmaps`,
`Fig_solution_landscape`, `Fig_inverse_parity_uncertainty`,
`Fig_optimizer_comparison`.

**Supplementary:** `Fig_boxplot_comparison`, `Fig_unseen_load_curves`,
`Fig_unseen_energy_curves`, `Fig_validation_error_maps_angle_disp`,
`Fig_qq_load_residuals_unseen`, `Fig_forward_map_jacobian`,
`Fig_inverse_posterior`, `Fig_landscape_ensemble_disagreement`,
`Fig_lc_classifier_cv_diagnostics`,
`Fig_inverse_vs_nearest_experimental_curve`,
`Fig_inverse_target_feasibility`, `Fig_inverse_error_vs_angle`,
`Fig_d_common_sensitivity_EA_vs_disp_endpoint`, plus the 15 per-target
GP-BO traces (`Fig_bo_convergence_T*`, `Fig_gpbo_posterior_evaluation_T*`,
`Fig_inverse_convergence_T*` — keep one representative in the main text and
`Fig_optimizer_comparison` for the cross-target summary).

## Methodology / framework (conceptual schematics)

| File | Caption summary |
|---|---|
| `Fig_framework_schematic.png` | End-to-end pipeline: data → three surrogates → dual-protocol validation → full-data Hard-PINN + GP-BO + LC classifier → Pareto trade-off (graphical abstract) |
| `Fig_architecture_schematic.png` | DDNS (two independent heads), Soft-PINN (heads + soft penalty), Hard-PINN (single energy output → autograd F = ∂E/∂d) |

## Forward model accuracy

| File | Caption summary |
|---|---|
| `Fig_dataset_overview.png` | Experimental load–displacement curves, EA distribution, IPF distribution |
| `Fig_parity_unseen.png` | Predicted vs actual (load + energy) parity for DDNS, Soft-PINN, Hard-PINN at θ\*=60° |
| `Fig_residuals_unseen.png` | Residual histograms (load + energy) with μ/σ in each panel title |
| `Fig_boxplot_comparison.png` | Ensemble R² distributions per approach × protocol |
| `Fig_cross_protocol.png` | Random vs unseen protocol bar chart (R² for load + energy) |
| `Fig_unseen_load_curves.png` | Load curves at θ\*=60° per LC with conformal ±2σ bands |
| `Fig_unseen_energy_curves.png` | Energy curves at θ\*=60° per LC with conformal ±2σ bands |
| `Fig_validation_error_maps_angle_disp.png` | Pointwise \|load\| / \|energy\| errors vs displacement at the held-out angle |

## Physics consistency + uncertainty calibration

| File | Caption summary |
|---|---|
| `Fig_physics_verification.png` | \|∂E/∂d − F\| residual histogram per approach |
| `Fig_qq_load_residuals_unseen.png` | Normal Q–Q plot of Hard-PINN validation load residuals |
| `Fig_reliability_diagram.png` | Observed vs nominal coverage (raw and conformal-corrected) |

## Inverse problem diagnostics

| File | Caption summary |
|---|---|
| `Fig_forward_map_jacobian.png` | Forward-map sensitivity ∂{EA, IPF}/∂θ per LC |
| `Fig_solution_landscape.png` | GP-BO objective landscape J(θ, LC) showing local minima |
| `Fig_inverse_posterior.png` | GP-BO posterior solutions per target |
| `Fig_landscape_ensemble_disagreement.png` | Ensemble disagreement σ(EA), σ(IPF) heat-map |

## LC plausibility classifier

| File | Caption summary |
|---|---|
| `Fig_lc_classifier_cv_diagnostics.png` | Confusion matrix, ROC, PR, calibration (LOO CV) |
| `Fig_classifier_decision_boundary.png` | P(LC2) probability landscape in EA-IPF space |
| `Fig_classifier_effect.png` | p(LC) at the recovered optimum, with vs without the plausibility penalty, per target |
| `Fig_lambda_sensitivity.png` | Target-matching error and p(LC) vs the classifier penalty weight λ (auto-tuned λ marked) |

## Inverse design (GP-BO target matching)

| File | Caption summary |
|---|---|
| `Fig_bo_convergence_T*.png` | Per-target GP-BO convergence trace (best-objective + sampled θ) — T1–T5 |
| `Fig_gpbo_posterior_evaluation_T*.png` | Per-target GP posterior snapshots — T1–T5 |
| `Fig_inverse_convergence_T*.png` | Per-target best-so-far objective vs evaluations — T1–T5 |
| `Fig_optimizer_comparison.png` | GP-BO outcomes across all 5 strategic targets |
| `Fig_inverse_parity_uncertainty.png` | Recovered vs target (EA, IPF) with ensemble ±2σ |
| `Fig_inverse_vs_nearest_experimental_curve.png` | Recovered curve vs nearest experimental curve |
| `Fig_inverse_target_feasibility.png` | Inverse-design target reachability on the empirical EA-IPF space |
| `Fig_inverse_error_vs_angle.png` | Recovery error vs held-out angle |

## Design space sweep

| File | Caption summary |
|---|---|
| `Fig_design_space.png` | Surrogate EA / IPF vs θ at d\*=D_COMMON (LC-fair sweep) |

## Multi-objective Pareto

| File | Caption summary |
|---|---|
| `Fig_pareto_tradeoff.png` | Pareto front in (EA, IPF) space coloured by α |
| `Fig_multiobjective_heatmaps.png` | 2D objective heat-maps over (θ, α) |

## Robustness / sensitivity sweeps

| File | Caption summary |
|---|---|
| `Fig_d_common_sensitivity_EA_vs_disp_endpoint.png` | EA(d) sensitivity vs displacement endpoint (LC1 / LC2) |

## Mechanics analysis (data-only, `scripts/mechanics_analysis.py`)

| File | Caption summary |
|---|---|
| `Fig_mode_signatures.png` | Crush-mode signatures per specimen (CFE vs densification onset) + plateau/IPF design trends |
| `Fig_densification_kinematics.png` | Candidate kinematic H(θ) fits: LC2 plateau force ~ sinθ·cosθ (R² = 0.95) |
| `Fig_master_curve_collapse.png` | Master-curve collapse test: raw vs mechanics scalings per LC |

## Optional figures (not included in this snapshot)

The pipeline emits these figures only when the corresponding analysis data
is produced (baseline sweep, HP sensitivity, penalty-weight sweep,
random-grid curves, per-member training curves, and the
posterior-likelihood diagnostic).  They are **gated behind optional
stages** and are **not part of this 48-figure snapshot**.  Their
generating functions exist in `composite_design.py`:

| File | Caption summary |
|---|---|
| `Fig_baseline_comparison_unseen.png` | DDNS/Soft/Hard vs ML baselines (requires `baseline_results_u`) |
| `Fig_hyperparam_sensitivity_unseen.png` | HP sensitivity sweep (requires `sensitivity_df_u`) |
| `Fig_penalty_weight_sensitivity.png` | Soft-PINN R² vs physics-penalty weight w_φ, with Hard-PINN as a weight-free reference line |
| `Fig_random_grid_curves.png` | Per-(angle, LC) grid of predicted curves (random 80/20 protocol) |
| `Fig_model_complexity.png` | Parameter count + training time per approach |
| `Fig_training_curves.png` | Per-approach training loss curves (M-member ensemble) |
| `Fig_inverse_posterior_likelihood.png` | Posterior-likelihood diagnostic |

To regenerate from saved model bundles see the
[project README](../../README.md#reproducing-the-figures-without-retraining).
