# Paper figures

All figures are PNG at 600 DPI, rendered with Arial bold typography
sized for insertion at *Composite Structures* full-column width
(7.48 inches / 190 mm).

This is the focused single-purpose figure suite: each figure handles ONE
analysis (parity, residuals, calibration, etc.) and renders cleanly in a
3×2 grid (or smaller).  No subplot-title overflow, no x-label / row-2
collisions, no legend obscuring data.

## Forward model accuracy

| File | Caption summary |
|---|---|
| `Fig_dataset_overview.png` | Experimental load–displacement curves, EA distribution, IPF distribution |
| `Fig_parity_unseen.png` | Predicted vs actual (load + energy) parity for DDNS, Soft-PINN, Hard-PINN at θ\*=60° |
| `Fig_residuals_unseen.png` | Residual histograms (load + energy) with μ/σ in each panel title |
| `Fig_boxplot_comparison.png` | Ensemble R² distributions per approach × protocol |
| `Fig_cross_protocol.png` | Random vs unseen protocol bar chart (R² for load + energy) |
| `Fig_training_curves.png` | Per-approach training loss curves (M-member ensemble) |
| `Fig_model_complexity.png` | Parameter count + training time per approach |
| `Fig_unseen_load_curves.png` | Load curves at θ\*=60° per LC with conformal ±2σ bands |
| `Fig_unseen_energy_curves.png` | Energy curves at θ\*=60° per LC with conformal ±2σ bands |
| `Fig_random_grid_curves.png` | Per-(angle, LC) grid of predicted curves (random 80/20 protocol) |
| `Fig_validation_error_maps_angle_disp.png` | Hexbin maps of pointwise |load| / |energy| errors vs (angle, displacement) |

## Physics consistency + uncertainty calibration

| File | Caption summary |
|---|---|
| `Fig_physics_verification.png` | |∂E/∂d − F| residual histogram per approach |
| `Fig_qq_load_residuals_unseen.png` | Normal Q–Q plot of Hard-PINN validation load residuals |
| `Fig_reliability_diagram.png` | Observed vs nominal coverage (raw and conformal-corrected) |

## Baselines and HP sensitivity (optional)

| File | Caption summary |
|---|---|
| `Fig_baseline_comparison_unseen.png` | DDNS/Soft/Hard vs ML baselines (when baseline_results_u is present) |
| `Fig_hyperparam_sensitivity_unseen.png` | HP sensitivity sweep (when sensitivity_df_u is present) |

## Inverse problem diagnostics

| File | Caption summary |
|---|---|
| `Fig_forward_map_jacobian.png` | Forward-map sensitivity ∂{EA, IPF}/∂θ per LC |
| `Fig_solution_landscape.png` | GP-BO objective landscape J(θ, LC) showing local minima |
| `Fig_inverse_posterior.png` | GP-BO posterior solutions per target |
| `Fig_inverse_posterior_likelihood.png` | Posterior-likelihood diagnostic |
| `Fig_landscape_ensemble_disagreement.png` | Ensemble disagreement σ(EA), σ(IPF) heat-map |

## LC plausibility classifier

| File | Caption summary |
|---|---|
| `Fig_lc_classifier_cv_diagnostics.png` | Confusion matrix, ROC, PR, calibration (LOO CV) |
| `Fig_classifier_decision_boundary.png` | P(LC2) probability landscape in EA-IPF space |

## Inverse design (GP-BO target matching)

| File | Caption summary |
|---|---|
| `Fig_bo_convergence_T*.png` | Per-target GP-BO convergence trace (best-objective + sampled θ) |
| `Fig_gpbo_posterior_evaluation_T*.png` | Per-target GP posterior snapshots (8 iterations) |
| `Fig_inverse_convergence_T*.png` | Per-target best-so-far objective vs evaluations |
| `Fig_optimizer_comparison.png` | GP-BO outcomes across all 5 strategic targets |
| `Fig_inverse_parity_uncertainty.png` | Recovered vs target (EA, IPF) with ensemble ±2σ |
| `Fig_inverse_vs_nearest_experimental_curve.png` | Recovered curve vs nearest experimental curve |
| `Fig_inverse_target_feasibility.png` | Inverse-design target reachability on the empirical EA-IPF space |

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

To regenerate from saved model bundles see the
[project README](../../README.md#reproducing-the-figures-without-retraining).
</content>
</invoke>