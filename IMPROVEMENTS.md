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
