# IPINN — Physics-Informed Inverse Design for Crashworthiness Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Release](https://img.shields.io/badge/release-v1.0--paper--final-blue.svg)](https://github.com/kprodigi/IPINN/releases/tag/v1.0-paper-final)

Forward prediction and inverse design of hexagonal composite ring structures
under quasi-static crushing, using three physics-informed neural-network
architectures compared on the same data + protocol, with multi-start GP-BO
inverse design, ill-posedness characterization, and multi-objective Pareto
sweep.

---

## Headline results

Forward-prediction accuracy on the **unseen-angle protocol** (θ\*=60°
held out; M=20 bootstrap ensemble; conformal-calibrated):

| Approach   | Load R² | Energy R² | Load RMSE (kN) | Load MAE (kN) | M_eff/M_total |
|------------|--------:|----------:|---------------:|--------------:|:-------------:|
| DDNS       |  0.7175 |    0.9803 |          0.082 |          0.060 |     18/20     |
| Soft-PINN  |  0.7875 |    0.9911 |          0.071 |          0.050 |     20/20     |
| **Hard-PINN** | **0.8217** | **0.9888** |     **0.065** |    **0.046** | **20/20**   |

Full results in [`paper/tables/Table1_forward_results.csv`](paper/tables/Table1_forward_results.csv).
All 43 figures (main text + supplementary) rendered at 600 DPI in
[`paper/figures/`](paper/figures/); see
[`paper/figures/README.md`](paper/figures/README.md) for the main-vs-SI split.

---

## Repository layout

```
IPINN/
├── composite_design.py        # Main pipeline (forward + inverse + analysis)
├── data/                       # Experimental dataset (LC1.xlsx, LC2.xlsx)
├── hpo/                        # Optuna HPO + per-member parallel training
├── slurm/                      # HPC submission scripts
├── scripts/                    # Utility scripts (bundle assembly, etc.)
├── tests/                      # 124 unit + integration tests
├── docs/                       # ARCHITECTURE.md (file/line map for reviewers)
├── paper/                      # Publication artifacts
│   ├── figures/                # 43 figures: main text + SI (PNG, 600 DPI)
│   └── tables/                 # Numerical tables (CSV) backing the manuscript
└── CHANGELOG.md
```

---

## Reproducing the paper from scratch

```bash
git clone https://github.com/kprodigi/IPINN.git
cd IPINN
git checkout v1.0-paper-final

# Environment
conda env create -f environment.yml
conda activate ipinn

# Full pipeline (forward + inverse + figures + tables)
python composite_design.py --mode all \
    --data_dir ./data \
    --output_dir ./results_paper \
    --strict_paper

# Or on HPC, parallel SLURM submission (see slurm/README.md)
bash slurm/submit_pipeline.sh
```

End-to-end wall time on a 15-GPU SLURM allocation is approximately
14–18 hours (Hard-PINN forward training dominates).

### Reproducing the figures without retraining

If you have a previously trained set of model bundles staged at
`./results_paper/`:

```bash
python composite_design.py --mode replot \
    --output_dir ./results_paper \
    --replot_from ./results_paper \
    --force_cpu
```

Runs in approximately 5 minutes on a single CPU.

---

## Hyperparameter optimisation

The hyperparameters reported in the paper are tuned per approach by an
Optuna TPE study with a MedianPruner across trials.  Re-run on your own
hardware:

```bash
APPROACH=ddns N_TRIALS=150 N_WORKERS=6  bash slurm/submit_hpo.sh
APPROACH=soft N_TRIALS=150 N_WORKERS=6  bash slurm/submit_hpo.sh
APPROACH=hard N_TRIALS=150 N_WORKERS=15 bash slurm/submit_hpo.sh
```

See [`hpo/README.md`](hpo/README.md) for the full workflow, including
warm-start configuration and the resume-from-preemption protocol.

---

## Methodology highlights

- **Hard-PINN architecture:** single-output energy network with force
  recovered exactly by autograd as F = ∂E/∂d — the *derivative consistency*
  F = dE/dd holds by construction.  Boundary conditions E(0)=0 and F(0)=0
  are **not** architecturally enforced in production; they are shaped
  through three auxiliary soft regularisers (monotonicity, angle
  smoothness, energy curvature).
- **Soft-PINN architecture:** two-headed (F, E) network with the
  work-energy identity penalised by a soft residual loss and a paired
  E(0)/F(0) BC penalty.
- **DDNS baseline:** data-driven only — same two-headed network, no
  physics terms.
- **Honest-reporting note:** in the experimental data the energy channel is
  the trapezoidal integral of the load channel (verified to machine
  precision), so F = dE/dd holds in the data *by construction*.  The PINN
  physics terms act as an inductive bias / structural regulariser — they are
  not validation against an independently measured physical channel.
- **M=20 bootstrap ensemble** with Tukey-fence convergence filter on
  training-set R²; survivor-only and all-member statistics are both
  reported (`Mean_Member_Load_R2` vs `Mean_Member_Load_R2_all`) so the
  filter's effect is visible.
- **Split-conformal calibration (curve-level):** ±1σ/±2σ inflation factors
  are fit on a calibration half of the validation *curves* and coverage is
  reported on the held-out half — corrected coverage is a measurement, not
  a tautology.  In-sample factors are retained under `*_insample` keys for
  reference.
- **Design-level validation:** predicted vs experimental EA@80mm / IPF at
  every measured design (`Table_forward_design_errors.csv`, overlay stars in
  `Fig_design_space`), a no-model interpolation baseline
  (`Table_null_baseline_design_level.csv`), a deployment-time physics audit
  (`Table_physical_plausibility_audit.csv`), and inverse-design ground-truth
  recovery (Δθ / LC-match columns in Table 3, off-grid + infeasibility
  verification targets in `Table_inverse_verification.csv`).
- **Multi-start GP-BO inverse design:** 5 restarts × 20 calls/restart,
  joint kernel over continuous θ and the categorical loading case, with
  a calibrated VotingClassifier penalty enforcing LC plausibility.
- **Ill-posedness diagnostics:** solution-landscape mapping, multiplicity
  index, forward-map Jacobian with bifurcation detection, and an
  approximate inverse posterior with 95% credible interval.
- **Pareto sweep:** weighted-sum and Chebyshev scalarisations with
  2-D dominance filtering on a dense surrogate landscape.

---

## CLI reference

| Flag | Effect |
|------|--------|
| `--data_dir DIR` | Input data location (default: `./data`) |
| `--output_dir DIR` | Output directory (default: `./results_paper`) |
| `--mode {all,forward,inverse,replot}` | Pipeline stage |
| `--seed N` | Global seed base (default: 2026) |
| `--n_ensemble M` | Bootstrap ensemble size (default: 20) |
| `--strict_paper` | Require optional deps (skopt) — abort if missing |
| `--force_cpu` | Use CPU even when CUDA is available |
| `--no_robustness` | Skip baselines + sensitivity + ablation |
| `--use_pretrained_inverse PATH` | Reuse a pre-trained inverse ensemble bundle |
| `--dry_run` | CI/smoke: tiny budgets, M ≤ 2, no GP-BO |

Run `python composite_design.py --help` for the full list.

---

## Testing

```bash
pytest tests/ -q
```

124 unit + integration tests covering network forward passes, physics
losses, training schedules, ensemble aggregation, conformal calibration,
classifier behaviour, and inverse-design diagnostics.

---

## License

[MIT](LICENSE)
