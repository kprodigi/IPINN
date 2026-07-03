# Limitations audit and improvement program

A systematic audit (forward-design demonstration, inverse-design demonstration,
physics/methodology, dataset, validation protocols, code robustness,
reproducibility) identified ~60 limitations.  Every code-addressable one has
been fixed; items requiring the production HPC rerun or new experiments are
listed at the end.  All changes are verified: **144/144 tests pass**, the full
dry-run pipeline completes end-to-end, and the figure suite renders cleanly.

## A. Inverse-design demonstration (was: optimizer self-consistency; now: verifiable round-trip)

| Limitation | Fix |
|---|---|
| Recovered θ was never compared to the target's known source design — the most convincing check was silently discarded | Targets now store `source_angle`/`source_lc`; **Table 3** reports `True_theta_deg`, `True_LC`, `Delta_theta_deg`, `LC_match` |
| Reported 0.3–3.5% errors were surrogate-vs-itself | Ground-truth recovery columns (above) are the headline; surrogate-fit errors remain as convergence evidence |
| All 5 targets were observed rows (trivially recoverable) | New **verification targets** (`Table_inverse_verification.csv`): V1/V2 off-grid round-trips at θ=52.5°/62.5° (true θ known by construction, present in no data row) + V3 deliberately infeasible probe that must be flagged non-attainable |
| GP-BO unmotivated vs. the 402-point grid the pipeline already evaluates | Optimizer table now anchors each recovery to the dense-grid optimum (`Grid_Best_*`, `BO_vs_Grid_dTheta_deg`) and reports true multi-start cost (`Total_Multistart_Evals`) |
| Recovered angles sitting on the search bound were unflagged | `Bound_active` column in Tables 3 and 6 |
| Robustness sweep covered only T1–T3 | All 5 targets |
| "Validation stress" test was vacuous (row-level split ⇒ mask never triggers) and silently fell back to seen configurations | Loud warning + `config_unseen_in_train` honesty column + `delta_theta_deg`/`LC_match` in the stress table |
| Classifier called "calibrated" while silently uncalibrated at n=12 | `calibration_mode` recorded in diagnostics; README/tables wording fixed |
| Classifier ablation showed only (null) p_LC deltas | Ablation now records decision effects: `Delta_theta_deg`, `LC_flipped` per target |
| Multiplicity threshold ad hoc (absolute +0.01 ⇒ ~10% error "solutions") | Threshold stated physically: minima within `rel_tol=2%` combined relative error of the optimum |
| BO history CSVs stored bare encoder indices for LC | Real `LC1`/`LC2` labels |

## B. Forward-design demonstration (was: pointwise R² only; now: design-level validation)

| Limitation | Fix |
|---|---|
| Design-space predictions never compared to the 12 experimental (EA, IPF) points | Experimental stars overlaid on `Fig_design_space`; new **`Table_forward_design_errors.csv`**: predicted vs experimental EA@80mm/IPF per design + per-LC/overall MAPE — evaluates the exact (full-data) surrogate that drives inverse design |
| No plausibility guards at inference: negative load/EA, non-monotone energy flowed silently into BO/Pareto | `compute_ea_ipf_ensemble` returns per-design violation fractions; sweep-wide **`Table_physical_plausibility_audit.csv`**; the BO objective rejects EA≤0 candidates; IPF peak-detection fallback rate recorded (`ipf_fallback_frac`) |
| No design-level null baseline | **`Table_null_baseline_design_level.csv`**: linear interpolation of experimental EA/IPF over θ at θ\*=60° (errors 6–34% on the real data) — the floor the surrogates must beat |
| EA meant different things across artifacts | `EA@80mm` qualifier added to all landscape/Pareto axis labels; conventions documented in `data/README.md` |
| Pointwise-energy conformal factor misapplied to the EA scalar in heatmap bands | New design-level scalar calibration (`Table_ea_ipf_scalar_calibration.csv`, `compute_ea_ipf_scalar_calibration`) feeds the bands; pointwise fallback demoted + logged as heuristic |

## C. Methodology honesty

| Limitation | Fix |
|---|---|
| "Split-conformal" fit + evaluated on the same residuals (corrected coverage tautological) | True split-conformal at the **curve level**: factors fit on a calibration half of validation curves, coverage reported on the held-out half; `split_conformal` provenance flag; in-sample factors kept under `*_insample` |
| Random 80/20 split is row-level within-curve interpolation, mislabeled | Relabeled everywhere: **"Within-Curve Interpolation (random 80/20)"**; README explains what it does and does not measure |
| Energy channel is the trapezoidal integral of load (physics holds in data by construction) | Disclosed in README + `data/README.md`; PINN physics reframed as inductive bias / structural regularizer |
| "Hard" BC claim overstated (architectural BC disabled in production) | README states precisely: derivative consistency F=dE/dd by construction; E(0)/F(0) soft |
| HPO tuned on the same θ\*=60° reported as unseen (selection contamination) | HPO default objective moved to inner angle 55° (`--loao_folds` default, slurm default); explicit warning when 60 is requested |
| Fence-filtered (survivor-only) stats entered the paper unflagged | Pre-fence metrics retained (`member_metrics_all`, `discarded_member_indices`); Table 1 adds `Mean_Member_Load_R2_all` |
| Row-level bootstrap under-disperses the ensemble (root cause of ~3× conformal factors) | Curve-level cluster bootstrap available (`CFG.bootstrap_unit="curve"`); default unchanged, trade-off documented |
| Curvature penalty may flatten the IPF peak it competes with | Opt-in window `CFG.curvature_window_after_mm` excludes the pre-peak region; default unchanged |

## D. Data integrity

| Limitation | Fix |
|---|---|
| Corrupted out-of-order row in LC2 θ=70° trained on silently | `load_data` auto-drops non-monotone-displacement rows with logged counts (verified: exactly 1 row dropped on the real data) |
| Data QA (`validate_input_data`) was dead code; per-file load errors swallowed | QA wired into `load_data`; per-file failures now fatal; both-LCs presence asserted |
| `data/README.md` factually wrong (wrong angle set, false replicates claim) | Rewritten with verified facts: angles 45–70°, single specimen/config, 10 N load quantization, LC1 θ=50° gap, 80 mm truncation, energy-derivation disclosure |

## E. Engineering robustness

| Limitation | Fix |
|---|---|
| Required paper artifacts could silently vanish (exceptions swallowed at DEBUG) | `REQUIRED_PAPER_ARTIFACTS` list checked by the manifest with a prominent failure summary; table handlers raised to WARNING |
| Ensembles could shrink to 1–2 members with only a warning | `CFG.min_ensemble_members` enforced (production fails; dry runs warn) |
| Bundles carried no provenance; stages could mix inconsistent artifacts | Every bundle stamped with `_meta` (git SHA, dataset SHA-256 fingerprint, CFG hash, seed); `replot` cross-checks and warns on mismatch |
| Resume cache reused stale members after config/data edits | Cache key now includes SHA of training data + hyperparameter dict |
| Deterministic `.tmp` name raced across concurrent jobs | Unique pid+counter tmp names |
| GP baseline drew from the global RNG (path-dependent) | Local seeded RNG |
| Production SLURM path could only train the unseen protocol | `--protocol {unseen,random}` added to `hpo/forward_member.py` + `hpo/forward_merge.py` — both protocols now come from the same array workflow |

## F. Tests / CI / reproducibility

| Limitation | Fix |
|---|---|
| Zero test coverage of the inverse machinery | New `tests/test_inverse_design.py` (13 tests): GP-BO quadratic recovery, coarse-grid argmin, objective EA≤0 guard, physical multiplicity threshold, target provenance, LC-penalty monotonicity + honest calibration mode, plausibility diagnostics, bootstrap units, IPF fallback flag, committed-dataset ingestion (xlsx + QA + fingerprint) |
| CI asserted only file existence and never touched inverse artifacts | CI now checks Table 3 (with truth columns), design-error/null-baseline/plausibility tables, and finite content in Tables 1/3; CI deps constrained to the declared ranges |
| Runtime environment omitted skopt version + git SHA | Both recorded, plus data fingerprint; hardware row added to the compute-budget table |

## θ-generalization program: Hard-PINN architecture variants (Tier 1+2)

Follow-up to the audit's core scientific finding: **F = ∂E/∂d constrains only
the displacement direction** — it is satisfied by any smooth E(d, θ) and adds
no structure across the design variable θ, where generalization actually
fails.  The measured design trend itself is jagged (model-free leave-one-angle
interpolation of the *experimental* EA/IPF misses held-out angles by 8–41%),
so the variants below attack inductive bias, and the harness measures each
against that floor.

| Variant (`CFG.hard_architecture` / `cfg["architecture"]`) | Guarantee |
|---|---|
| `mlp` (default) | published production architecture, unchanged |
| `monotone` | **F ≥ 0, E(0) = 0, E ≥ 0 by construction** at every input incl. unseen θ (positive-weight d-path + non-decreasing activations + intrinsic value-subtraction BC) |
| `separable` | E = Σ_k φ_k(θ, LC)·B_k(d) with θ-capacity limited to ONE linear map over [sinθ, cosθ, LC] (~4 dof/basis) — matched to 5 training angles |
| `monotone_separable` | both (φ_k ≥ 0, monotone B_k) |

Implementation: `PositiveLinear`, `MonotoneHardEnergyNet`,
`SeparableHardEnergyNet`, `make_hard_energy_net` in `composite_design.py`;
wired into `train_hard` and the full-data inverse trainer; property-tested in
`tests/test_hard_architectures.py` (F ≥ 0 at θ up to 180° and d to 200 mm,
exact E(0)=0, double-backward training compatibility, capacity checks).

**Ablation harness:** `scripts/ablation_theta_generalization.py` trains a
small ensemble per (variant × held-out angle) and reports design-level
EA@80mm/IPF error at the held-out angle vs the interpolation floor, plus
pointwise R² and plausibility-violation fractions.  Full run (HPC):

```bash
python scripts/ablation_theta_generalization.py --data_dir ./data \
    --output_dir ./results_ablation --members 4
```

Note the full run at tuned budgets is Hard-PINN-expensive (4 variants × 6
angles × 4 members × ~4 h/member ≈ 380 GPU-h; parallelize or reduce
`--members`/`--angles`).  A CPU smoke mode (`--epochs 60 --batch_size 32`)
validates the machinery in minutes.

## Explainability & interpretability layer (verified gap → implemented)

Verification finding: the pipeline claimed a physics-informed forward/inverse
framework but produced **no dedicated explainability artifacts** beyond the
Jacobian sensitivity figure (no attribution, no faithful decomposition, no
per-decision explanation).  Implemented — all *exact model readouts*, not
post-hoc approximations:

| Artifact | What it explains |
|---|---|
| `Table_design_variance_decomposition.csv` (`compute_design_variance_decomposition`) | Exact Sobol/ANOVA split of EA and IPF variance into θ, LC, and interaction on the balanced dense sweep grid — *which design factor controls which objective* (unit-tested: synthetic θ-only/LC-only surfaces recover 100/0 exactly; interaction detected) |
| `Table_inverse_design_explanation.csv` (`table_inverse_design_explanation`) | Per-target objective decomposition at the recovered optimum: EA-fit vs IPF-fit vs classifier penalty, p_LC, dominant term — *why the optimizer selected each design* (terms verified to recompose J; √term = relative error by the 1/target² weighting) |
| `Fig_separable_interpretability` | Faithful decomposition of the separable variant: learned crush-mode basis Bₖ(d) and design coefficients φₖ(θ, LC) — *the figure is the model* (the θ-dependence is architecturally a printed first-order Fourier map) |

The inverse-uses-forward-surrogate chain was also verified in code (not
assumed): `train_full_data_hard_pinn` → `run_inverse_design(inv_models, …)` →
objective → `compute_ea_ipf_ensemble(models, …)` — GP-BO inverts the frozen
trained forward PINN; the diagnostics (landscape, posterior, Jacobian,
multiplicity) sit on top.  Both tables are wired into the inverse stage, the
replot path, and `REQUIRED_PAPER_ARTIFACTS`.

## Phase 1+2: reviewer-proof held-out-design validation + mechanics injection

Built in response to the "you have not shown LOAO results" risk.  Measured
context that shapes the strategy: the *experimental* design trend itself is
jagged (model-free leave-one-angle interpolation errs 8–41%), and LC2 θ=70°
sits past a deformation-regime transition (plateau force 0.37→0.79 kN,
EA +71%) that is unlearnable from the other five single-specimen angles.

**Phase 1 — protocol, metrics, and calibration (`scripts/ablation_theta_generalization.py`):**
- Two pre-declared protocols: **LOAO** (hold out both LCs of an angle;
  boundary folds 45°/70° stratified as extrapolation stress tests) and
  **LOCO** (hold out one (θ, LC) curve; cross-loading transfer, 12 folds).
- Curve metric suite with **level/shape decomposition** (`curve_error_metrics`):
  raw R² is reported but retired as the headline — a pure level offset (the
  single-specimen scatter mode) drives it negative with perfect shape
  (unit-tested); `R2_shape`, `NRMSE_range`, bias, Pearson r carry the claim.
- **Skill scores vs two floors** per fold (`skill_score`): the linear
  interpolation of the experiments and a 2-parameter mechanics trend —
  S > 0 = beats the floor, the correct notion of success under irreducible
  jaggedness.
- **Jackknife+ prediction intervals with empirical coverage**
  (`jackknife_plus_intervals`): each fold's interval never uses its own
  residual (unit-tested with an outlier fold), so boundary-fold ignorance is
  calibrated, not hidden (`Table_ablation_jackknife_coverage.csv`).
- Outputs: per-fold detail + stratified summary (interior/boundary) +
  skill figure (`Fig_ablation_skill.png`) + error-vs-floors figure.

**Phase 2 — mechanics analysis (`scripts/mechanics_analysis.py`, data-only, runs in seconds):**
- **Crush-mode signature table** per specimen (IPF, plateau force,
  densification onset, CFE, oscillation, initial stiffness) →
  `Table_crush_mode_signatures.csv` + `Fig_mode_signatures.png` — the
  regime-transition evidence that reframes boundary-fold failure as detected
  mode change.
- **Densification kinematics** (`fit_densification_kinematics`): candidate
  H(θ) forms regressed per quantity.  Measured on the real data: LC2 plateau
  force follows sinθ·cosθ with **R²=0.947** (all six angles, including the
  70° rise); LC1 initial stiffness R²=0.77; detected onsets decrease with θ
  (78→70→61 mm) — the kinematic-stroke mechanism behind the LC2-70° EA jump.
  → `Table_densification_kinematics.csv` + `Fig_densification_kinematics.png`.
- **Master-curve collapse test** (`fig_master_curve_collapse`): mechanics
  scalings collapse LC2's chaotic curves 1.9× (pairwise NRMSE 0.85→0.43);
  LC1 already shares a common shape.  → figure + table.
- **Mechanics-trend baseline** (`mechanics_trend_baseline`): candidate form
  selected on training folds only; reported per fold next to the
  interpolation floor.
- **Mechanics θ-features** (`CFG.theta_feature_map="mechanics"` /
  `--theta_features mechanics`): the networks receive kinematic coordinates
  [sinθ, (1+cosθ)/2] instead of raw Fourier — same dimensionality, opt-in.

All functions unit-tested (10 new tests; 171 total).  The candidate H(θ)
forms are stated as candidates because the mandrel parameterization is not
recorded in-repo; affine-redundant candidates are excluded by construction.

## Requires the production HPC rerun (code is ready)

1. **Retrain forward ensembles** (both protocols now supported by the SLURM path)
   and rerun `--mode inverse` + `--mode replot`: regenerates every table/figure
   with the new honest metrics (ground-truth recovery, design-level errors,
   split-conformal coverage, plausibility audit, verification targets).
2. **Re-run HPO with the decontaminated objective** (default now θ=55 inner
   holdout) if the unseen-angle claim is to be selection-clean; report pre/post.
3. Optional ablations now wired: `CFG.bootstrap_unit="curve"`,
   `CFG.curvature_window_after_mm`, penalty-off classifier ablation.

## Requires new data / author input (documented, not code-fixable)

- Replicate specimens (aleatoric scatter is unmeasurable from 1 test/config).
- Off-grid experimental validation of a recovered design (the gold-standard
  inverse check); the V1/V2 surrogate round-trips are the code-only proxy.
- Material/specimen/test-standard details in `data/README.md`.
- Public release of the trained bundle (GitHub Release / Zenodo DOI) so readers
  can regenerate figures without ~80 GPU-hours.
