# Architecture and code map

This document maps the paper's sections and major design decisions to
specific line ranges in `composite_design.py` (≈11,094 lines, single
file). Use it as a reviewer's table of contents — every numbered section
points at a self-contained block.

The file is organized in nine top-level blocks, in source order:

1. Configuration and styling (lines 1–949)
2. Data loading and preprocessing (lines 950–1388)
3. Network architectures (lines 1389–1529)
4. Physics losses (lines 1530–1796)
5. Training (lines 1797–2290)
6. Evaluation and uncertainty (lines 2291–2679)
7. LC plausibility classifier (lines 2680–3271)
8. Inverse design and optimization (lines 3272–5174)
9. Plotting, summary tables, pipeline, CLI (lines 5175–11094)

## Block 1 — Configuration and styling (1 – 949)

| Lines | Content |
|-------|---------|
| 109–179 | Logging configuration (`setup_logging`, mode-tagged log filenames) |
| 180–425 | Publication style (Arial bold, font size scaling, gridspec helpers) |
| 426–475 | Color and style constants (Wong-equivalent palette, marker map) |
| 476–576 | Reproducibility (`set_global_seed`, deterministic torch flags) |
| 577–698 | `Config`/`Params` dataclasses, default knob values |
| 699–948 | `get_model_config(approach, protocol, w_phys_override)` — per-approach hyperparameter dictionary; the HPO output is patched in here |

## Block 2 — Data loading and preprocessing (950 – 1388)

| Lines | Content |
|-------|---------|
| 950–1040 | `IndentationDataset`, scaling utilities |
| 1041–1247 | `train_full_data_hard_pinn` — full-data Hard-PINN trained for inverse design (Section 3.6) |
| 1248–1300 | `validate_input_data` — schema + monotonicity checks |
| 1301–1388 | `load_data` — multi-file Excel/CSV ingestion, column auto-rename |

## Block 3 — Network architectures (1389 – 1529) ← paper Section 3.4–3.5

| Lines | Content |
|-------|---------|
| 1391–1462 | `SoftPINNNet` — two-headed MLP for force and energy (DDNS and Soft-PINN both use this) |
| 1428–1462 | `SoftPINNNet.configure_zero_bc` — optional architectural `E(0)=0`/`F(0)=0` correction (production cfgs use the soft `w_bc` penalty instead) |
| 1463–1529 | `HardEnergyNet` — single-output energy network; force is `dE/dd` via `torch.autograd.grad` |
| 1501–1529 | `HardEnergyNet.configure_zero_bc` — optional architectural BC correction (preserved for ablation studies; the production Hard-PINN uses the bare MLP form together with the auxiliary soft penalties) |

The Hard-PINN production architecture is the standard bare-MLP
energy network with `F = ∂E/∂d` computed by autograd at both
training and inference.  The boundary conditions `E(0) = 0` and
`F(0) = 0` are encouraged through the auxiliary soft penalties
(monotonicity, angle smoothness, curvature) which collectively
shape the force–displacement curve near the origin (Section 3.5.2).
The Soft-PINN adds an explicit paired BC penalty
`w_bc · (E(0)² + F(0)²)` to the work-energy residual loss.

Two flavours:

- **SoftPINNNet** (two-headed `[F, E]`) keeps the standard
  two-output forward pass; the BC is enforced by the paired soft-MSE
  loss term `w_bc · (E(0)² + F(0)²)` added to the work-energy
  residual loss.
- **HardEnergyNet** (single output `E`, with `F = dE/dd`) enforces the
  work-energy identity by construction.  An optional architectural
  BC correction (`configure_zero_bc`) is available for ablation
  studies but is not used in the production cfgs reported in the
  main results.

## Block 4 — Physics losses (1530 – 1796) ← paper Section 3.5.2

| Lines | Content |
|-------|---------|
| 1530–1726 | Loss components: data fit, force–energy consistency, monotonicity, angle smoothness, curvature regularisers |
| 1726–1796 | `_val_checkpoint_score(r2_load, r2_energy, approach)` — load-only checkpoint metric (`r2_load if finite else -inf`); same rule for every approach |

## Block 5 — Training (1797 – 2290) ← paper Section 3.5.3 / 3.6

| Lines | Content |
|-------|---------|
| 1797–1882 | `train_ddns` |
| 1883–2009 | `train_soft` |
| 2010–2163 | `train_hard` (warmup + SWA over the last `swa_pct` of epochs) |
| 2164–2290 | `train_ensemble` — Tukey-fence convergence filter; per-approach ensemble training |

## Block 6 — Evaluation and uncertainty (2291 – 2679) ← paper Section 3.7.1, 4.1

| Lines | Content |
|-------|---------|
| 2291–2467 | Per-trajectory metrics, R² / RMSE / MAE |
| 2468–2678 | Conformal calibration — split-conformal ±2σ factor estimation, reliability binning |

## Block 7 — LC plausibility classifier (2680 – 3271) ← paper Section 3.7.2

| Lines | Content |
|-------|---------|
| 2685 | `CLASSIFIER_FEATURES = ["EA_common", "IPF", "Angle"]` — feature schema |
| 2696–2714 | `_make_lc_voting_classifier` — VotingClassifier(GaussianNB + SVC + RF + MLP) |
| 2715–3249 | `train_lc_plausibility_classifier` — calibration, OOF probability, λ-scan |
| 3252–3271 | `compute_lc_plausibility(...)` — penalty added to the inverse objective |

## Block 8 — Inverse design and optimization (3272 – 5174) ← paper Section 3.8

| Lines | Content |
|-------|---------|
| 3273–3620 | Surrogate evaluation, target sweep utilities |
| 3621–3852 | `run_inverse_design(...)` — multi-start GP-BO with classifier penalty `J = fit_error + λ · LC_penalty` |
| 3853–4043 | `compute_solution_landscape` — landscape mapping and basin enumeration |
| 4046–5173 | Forward-map Jacobian, multiplicity index, Pareto sweep (weighted-sum + Chebyshev) |
| 6907–6999 | `run_inverse_design_robust` — multi-start robustness sweep |

## Block 9 — Plotting, summary tables, pipeline, CLI (5175 – 11094)

### Robustness, ablation, calibration figures (5175 – 7637)

These produce the supplementary diagnostics: optimizer comparison,
target feasibility, design-space scatter, training curves, model-complexity
plot, physics residuals, baseline comparison, hyperparameter sensitivity,
reliability diagram, same-capacity experiment, extended ablation.

### Summary tables (7639 – 8704)

| Lines | Content |
|-------|---------|
| 7641–7793 | `generate_summary_tables` — Table 1 (forward), Table 2 (Welch t-tests), Table 3 (inverse ill-posedness), Table 4–6 |
| 7794–7816 | `save_reproducibility_artifacts` — `runtime_environment.json`, `STATISTICAL_TESTING_POLICY.txt` |
| 7817–8035 | Pipeline state save/load (`save_pipeline_state`, `load_pipeline_state`, `replot_figures_from_state`) |
| 8036–8230 | Compute-budget summary, statistical-testing policy text, dry-run settings |

### Manuscript figures (focused single-purpose suite)

Each figure handles ONE analysis (parity, residuals, calibration, etc.)
and renders cleanly in a 3×2 grid or smaller — no subplot-title overflow,
no x-label / row-2 collisions, no legend obscuring data.  See
`paper/figures/README.md` for the full inventory; key functions:

| Function | Output file | Purpose |
|----------|-------------|---------|
| `fig_dataset_overview` | `Fig_dataset_overview.png` | Experimental curves + EA/IPF distributions |
| `fig_parity_plots` | `Fig_parity_unseen.png` | 3×2 parity grid (approach × load/energy) |
| `fig_residual_histograms` | `Fig_residuals_unseen.png` | 3×2 residual-histogram grid with μ/σ |
| `fig_boxplot_comparison` | `Fig_boxplot_comparison.png` | Ensemble R² boxplots per approach |
| `fig_unseen_curves` | `Fig_unseen_{load,energy}_curves.png` | Load + energy curves at θ\*=60° with conformal ±2σ |
| `fig_physics_verification` | `Fig_physics_verification.png` | |∂E/∂d − F| residual histogram |
| `fig_reliability_diagram` | `Fig_reliability_diagram.png` | Calibration (raw + conformal) |
| `fig_lc_classifier_diagnostics` | `Fig_lc_classifier_cv_diagnostics.png` | Confusion matrix, ROC, PR, calibration |
| `fig_bo_convergence` | `Fig_bo_convergence_T*.png` | Per-target GP-BO convergence trace |
| `fig_inverse_parity_uncertainty` | `Fig_inverse_parity_uncertainty.png` | Recovered vs target (EA, IPF) with ±2σ |
| `fig_pareto_tradeoff` | `Fig_pareto_tradeoff.png` | EA-IPF Pareto front |
| `fig_multiobjective_heatmaps` | `Fig_multiobjective_heatmaps.png` | 2D objective heat-maps |
| `fig_training_curves` | `Fig_training_curves.png` | Per-approach training loss curves |
| `fig_model_complexity` | `Fig_model_complexity.png` | Parameter count + training time |

The `_render_all_figures` orchestrator at the bottom of the module calls
every figure whose input data is available; missing bundles silently
skip their figures.

### Bundle save/load (9447 – 9524)

| Function | Lines | Bundle file |
|----------|-------|-------------|
| `save_forward_bundle` | 9447–9451 | `forward_models.pt` (every trained PINN + scalers) |
| `save_inverse_bundle` | 9452–9456 | `inverse_models.pt` (full-data Hard-PINN + classifier) |
| `save_analysis_bundle` | 9457–9524 | `analysis_results.pt` (BO traces, landscape, Pareto frame) |

These three bundles are what `--mode replot` re-loads to regenerate every
figure without retraining. The forward-only and inverse-only HPC modes
emit the matching subset.

### Appendix / supplementary figures

The remaining single-purpose plot routines (random-grid curves, baseline
comparison, hyperparameter sensitivity, classifier decision boundary,
solution landscape, inverse-posterior likelihood, validation error maps,
Q–Q plot, etc.) are all called directly by `_render_all_figures` when
their input data is present.  There is no longer a separate
`fig_appendix_all` aggregator — every supplementary figure is treated as
a first-class citizen of the focused suite.

### Pipeline and CLI (10845 – 11094)

| Function | Lines | Purpose |
|----------|-------|---------|
| `run_pipeline` | 10845–10990 | Threaded logger, mode dispatcher (`forward`/`inverse`/`replot`/`all`) |
| `main` | 10991–11094 | CLI argparse + dry-run handling + mode dispatch |

The HPO knobs that `hpo/hpo_search.py` searches are looked up via
`get_model_config(approach, protocol, w_phys_override)` (block 1, lines
699–948) — `hpo_search.py` monkey-patches that function per trial via a
`patched_factory`.
