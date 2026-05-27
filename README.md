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
All 10 paper figures rendered at 600 DPI in [`paper/figures/`](paper/figures/).

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
│   ├── figures/                # 10 main figures + appendix (PNG, 600 DPI)
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
  recovered exactly by autograd as F = ∂E/∂d, enforcing the work-energy
  identity by construction.  Boundary conditions E(0)=0 and F(0)=0 are
  shaped through three auxiliary soft regularisers (monotonicity, angle
  smoothness, energy curvature).
- **Soft-PINN architecture:** two-headed (F, E) network with the
  work-energy identity penalised by a soft residual loss and a paired
  E(0)/F(0) BC penalty.
- **DDNS baseline:** data-driven only — same two-headed network, no
  physics terms.
- **M=20 bootstrap ensemble** with Tukey-fence convergence filter on
  training-set R² to drop unconverged members before reporting.
- **Conformal calibration:** split-conformal estimation of ±1σ and ±2σ
  inflation factors so reported uncertainty bands attain nominal coverage
  on the held-out angle.
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
