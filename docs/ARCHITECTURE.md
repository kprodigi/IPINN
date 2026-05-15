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
| 699–948  | `get_model_config(approach, protocol, w_phys_override)` — per-approach hyperparameter dictionary; the HPO output is patched in here |

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
| 1428–1462 | `SoftPINNNet.configure_zero_bc` — architectural `E(0)=0`/`F(0)=0` correction |
| 1463–1529 | `HardEnergyNet` — single-output energy network; force is `dE/dd` via `torch.autograd.grad` |
| 1501–1529 | `HardEnergyNet.configure_zero_bc` — slope-subtraction architectural BC (both `E(0)=0` and `F(0)=0`) |

The architectural BC is the v_20 contribution: it replaces v_19's soft `w_bc`
penalty with an exact subtraction so `w_bc` is no longer needed in the loss.
Two flavors:

- **SoftPINNNet** (two-headed `[F, E]`) corrects each output directly at
  `d=0`:
  `F_corr(x) = F_net(x) − F_net(x|d=0) + c_{0,F}`,
  `E_corr(x) = E_net(x) − E_net(x|d=0) + c_{0,E}`.
- **HardEnergyNet** (single output `E`, with `F = dE/dd`) must pin BOTH the
  value AND the d-slope at the boundary to recover `F(0)=0`. Slope
  subtraction does this:
  `E_corr(x) = E_net(x) − E_net(x|d=0) − (d_s − d_s0)·∂E_net/∂d_s|_{x|d=0}
    + c_{0,E}`.
  Cost: one extra inner `autograd.grad` per forward pass plus a second-order
  graph during training (~2–3× the value-only correction).

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

### Manuscript figures (9525 – 10510)

The 10 multi-panel paper figures, each a single function with the same
gridspec/styling discipline:

| Function | Lines | Paper figure |
|----------|-------|--------------|
| `figv20_01_dataset_overview` | 9525–9622 | Fig. 1 — dataset summary, layup-angle distribution |
| `figv20_02_forward_parity` | 9623–9681 | Fig. 2 — parity, residuals, R² boxplots across approaches |
| `figv20_03_unseen_generalization` | 9682–9758 | Fig. 3 — load/energy curves at unseen θ=60° with conformal bands |
| `figv20_04_physics_calibration` | 9759–9906 | Fig. 4 — calibration / reliability / coverage |
| `figv20_05_ill_posedness` | 9907–10007 | Fig. 5 — solution landscape, Jacobian, multiplicity |
| `figv20_06_classifier` | 10008–10092 | Fig. 6 — LC plausibility classifier diagnostics |
| `figv20_07_inverse_design` | 10093–10271 | Fig. 7 — inverse-design recovered θ vs target |
| `figv20_08_pareto` | 10272–10349 | Fig. 8 — Pareto front (EA vs IPF) |
| `figv20_09_pareto_recovery` | 10350–10419 | Fig. 9 — recovery on Pareto-optimal targets |
| `figv20_10_robustness` | 10420–10510 | Fig. 10 — robustness, ablation, λ-sensitivity |

### Bundle save/load (9447 – 9524)

| Function | Lines | Bundle file |
|----------|-------|-------------|
| `save_forward_bundle` | 9447–9451 | `forward_models.pt` (every trained PINN + scalers) |
| `save_inverse_bundle` | 9452–9456 | `inverse_models.pt` (full-data Hard-PINN + classifier) |
| `save_analysis_bundle` | 9457–9524 | `analysis_results.pt` (BO traces, landscape, Pareto frame) |

These three bundles are what `--mode replot` re-loads to regenerate every
figure without retraining. The forward-only and inverse-only HPC modes
emit the matching subset.

### Appendix figures (10511 – 10844)

`figv20_appendix_all` calls 7 single-purpose appendix wrappers (per-approach
training curves, conformal histograms, classifier confusion matrices, etc.).

### Pipeline and CLI (10845 – 11094)

| Function | Lines | Purpose |
|----------|-------|---------|
| `run_pipeline_v20` | 10845–10990 | Threaded logger, mode dispatcher (`forward`/`inverse`/`replot`/`all`) |
| `main_v20` | 10991–11094 | CLI argparse + dry-run handling + mode dispatch |

The HPO knobs that `hpo/tune_v20.py` searches are looked up via
`get_model_config(approach, protocol, w_phys_override)` (block 1, lines
699–948) — `tune_v20.py` monkey-patches that function per trial via a
`patched_factory`.
