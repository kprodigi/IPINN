# IPINN crashworthiness (composite ring)

Physics-informed neural networks (DDNS / Soft-PINN / Hard-PINN) for quasi-static crushing: forward prediction, dual validation protocols, inverse design (multi-start GP-BO with Tikhonov regularisation), loading-condition classifier, ill-posedness analysis, multi-objective landscape, and **qLogNEHVI** MOBO (BoTorch, primary method).

## Reproducibility (Q1 / Zenodo bundle)

1. **Environment**  
   - Pip: `pip install -r requirements.txt`  
   - Conda: `conda env create -f environment.yml` then `conda activate ipinn-crash`

2. **Strict submission run** (fails if `skopt` or `botorch` missing):  
   ```bash
   python composite_design_v19.py --data_dir . --output_dir ./results_paper --strict_paper
   ```

3. **Artifacts** (under `--output_dir`):  
   - `runtime_environment.json` — Python / torch / numpy / optional flags  
   - `STATISTICAL_TESTING_POLICY.txt` — multiplicity / exploratory vs confirmatory framing  
   - `Table_compute_reproducibility_budget.csv` — training budgets and inverse / MOBO call counts  
   - Figures (`Fig_*.png`), tables (`Table*.csv`), `run_log.txt`  
   - `MANIFEST_outputs.csv` — auto-generated inventory of every `Fig_*` / `Table*` file

4. **Tests**  
   ```bash
   pytest
   ```

## CLI options

| Flag | Effect |
|------|--------|
| `--data_dir DIR` | Input CSV / XLSX location |
| `--output_dir DIR` | All outputs |
| `--seed N` | Global seed base |
| `--n_ensemble M` | Ensemble size (default 20) |
| `--strict_paper` | Require skopt (GP-BO) **and** botorch (MOBO); abort if either missing |
| `--no_mobo_qnehvi` | Explicitly skip MOBO after landscape sweep (disallowed with `--strict_paper`) |
| `--no_reviewer_proof` | Skip extended reviewer analyses (baselines, sensitivity, calibration vs M, stress inverses) |
| `--force_cpu` | CPU even if CUDA is available |
| `--show_plots` | Display plots on screen (default: off) |
| `--no_ablation` | Skip forward-model physics-weight ablation study |
| `--no_loao_cv` | Skip leave-one-angle-out cross-validation |
| `--no_rar` | Skip residual-based adaptive refinement for collocation sampling |
| `--inverse_ablation` | Run extra GP-BO inverse ablations (no Tikhonov / no classifier / no robustness) on first targets; expensive |
| `--no_inverse_stress` | Skip validation-row inverse stress targets |
| `--no_inverse_member_spread` | Skip per-member spread table at the inverse optimum |
| `--dry_run` | CI/smoke: short training, `M<=2`, no MOBO / reviewer-proof / ablation / GP-BO / LOAO-CV / RAR; inverse uses a coarse angle grid. Use with `tests/fixtures/tiny_crush.csv` (`--data_dir tests/fixtures`). |

## Figure and table map (typical run)

Outputs vary with flags (`--no_reviewer_proof`, `--no_mobo_qnehvi`, `--dry_run`, `--inverse_ablation`). The table links **primary artifacts** to the claims they support.

### Forward prediction

| File (prefix) | Supports |
|---------------|----------|
| `Fig_parity_*`, `Fig_residual_*`, `Fig_boxplot_*`, `Fig_training_*` | Forward accuracy and spread across DDNS / Soft / Hard and protocols. |
| `Fig_cross_protocol_*`, `Fig_unseen_*`, `Fig_random_grid_*` | Generalization: random split vs unseen-angle holdout. |
| `Fig_validation_error_maps_angle_disp.png`, `Table_validation_errors_by_angle_bin.csv` | Error distribution in (theta, displacement) and by angle bin. |
| `Fig_qq_load_residuals_unseen.png` | Tail behaviour of load residuals (normality check). |
| `Fig_loao_cv.png`, `Table_loao_cv.csv` | Leave-one-angle-out cross-validation: per-angle R-squared for Hard-PINN. |
| `Fig_reliability_diagram.png`, `Table_uncertainty_calibration.csv` | Ensemble interval calibration vs nominal coverage. |
| `Fig_calibration_vs_M.png` | Conformal factor and coverage as a function of ensemble size M (reviewer-proof). |

### Inverse design and ill-posedness

| File (prefix) | Supports |
|---------------|----------|
| `Fig_lc_classifier_cv_diagnostics.png`, `Table_lc_classifier_cv_predictions.csv` | LC plausibility model: CV confusion, ROC, PR, calibration. |
| `Fig_inverse_convergence_<T>.png`, `Fig_gpbo_posterior_evaluation_<T>.png` | Per-target inverse: best-objective vs evaluations; GP-BO posterior evaluation (2x4 snapshots). |
| `Table3_inverse_illposedness.csv` | **Ill-posedness summary**: multiplicity index, posterior stats, dJ/dtheta, d2J/dtheta2, BO restarts, Tikhonov gamma, classifier/robustness weights, best (theta, LC, J) per target. |
| `Table_inverse_local_minima.csv` | Long-form grid local minima per LC for each target. |
| `Table_inverse_topk_basins.csv` | Ranked basins from landscape minima + multi-start BO terminals. |
| `Fig_solution_landscape.png` | J(theta) for both LCs per target with local minima marked; solution multiplicity index (SMI). |
| `Fig_inverse_posterior_likelihood.png`, `Table_inverse_posterior_likelihood.csv` | Gaussian likelihood on (EA, IPF) vs theta, best LC. |
| `Fig_inverse_posterior.png` | Approximate inverse posterior P(theta | target) with 95% credible interval. |
| `Fig_forward_map_jacobian.png`, `Table_forward_jacobian_summary.csv` | dEA/dtheta and dIPF/dtheta for both LCs; bifurcation point detection for local invertibility analysis. |
| `Table_inverse_vs_calibration.csv` | Inverse errors cross-referenced with random-protocol Hard-PINN conformal factors. |
| `Table_inverse_theta_member_spread.csv` | Per-member EA/IPF min/max/std at the reported (theta*, LC*). |
| `Table_inverse_stress_protocol.csv` | Stress-test inverses on (Angle, LC) pairs absent from training (requires `--no_reviewer_proof` off). |
| `Table_inverse_ablation.csv` | Inverse ablation: no Tikhonov / no classifier / no robustness vs main run (requires `--inverse_ablation`). |

### Multi-objective analysis

| File (prefix) | Supports |
|---------------|----------|
| `Table6_pareto_sweep.csv`, `Table_design_landscape.csv`, `Fig_multiobjective_*`, `Fig_pareto_*` | MO trade-off and dense surrogate landscape (EA@`D_COMMON`, IPF). |
| `Fig_landscape_ensemble_disagreement.png` | Epistemic spread of EA/IPF along the landscape. |
| `Fig_d_common_sensitivity_EA_vs_disp_endpoint.png`, `Table_d_common_sensitivity_EA_grid.csv` | Sensitivity of EA metric to displacement endpoint (same trained ensemble). |
| `Table_mobo_qnehvi_*`, `mobo_qnehvi_reference_point.json`, `Fig_mobo_*`, `Fig_moo_*` | qLogNEHVI MOBO (primary method) and objective-space checks. |
| `Table_mobo_vs_landscape_pareto_distance.csv` | ND hypervolumes + normalised mean distance MOBO ND to landscape ND. |

### Reproducibility

| File (prefix) | Supports |
|---------------|----------|
| `MANIFEST_outputs.csv` | Complete inventory of that run's deliverables (auto-generated). |
| `runtime_environment.json`, `STATISTICAL_TESTING_POLICY.txt`, `Table_compute_reproducibility_budget.csv` | Reproducibility and compute narrative. |

## Methodology notes (short)

- **Hard-PINN boundary enforcement**: E = g(d) * NN(x) where g(d) = d_scaled - d_zero_scaled, guaranteeing E(d=0) = 0 exactly by construction (matching F = dE/dd structural enforcement).
- **Residual-based adaptive refinement (RAR)**: collocation points are re-weighted every 50 epochs proportional to local physics residual magnitude, concentrating constraint enforcement where it is most needed (Lu et al., 2021).
- **Inverse ill-posedness**: addressed via (1) Tikhonov regularisation gamma*(theta - theta_center)^2, (2) multi-start GP-BO (N=5 restarts), (3) solution landscape mapping with multiplicity index, (4) forward-map Jacobian dEA/dtheta and dIPF/dtheta with bifurcation detection, (5) approximate inverse posterior P(theta | target).
- **MOBO** is the primary multi-objective method (qLogNEHVI / BoTorch). `--strict_paper` requires botorch.
- **Validation**: dual protocols (random 80-20 + unseen angle), leave-one-angle-out CV, calibration vs ensemble size, conformal post-hoc correction.
- **Statistics**: see `STATISTICAL_TESTING_POLICY.txt` in each run folder.  
- **MOBO reference point**: documented in `mobo_qnehvi_reference_point.json`.

## Citation

Set author/year/journal per your submission; cite BoTorch / skopt / PyTorch as appropriate in the paper methods.
