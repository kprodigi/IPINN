# IPINN — Physics-Informed Neural Networks for Crashworthiness

[![CI](https://github.com/kprodigi/IPINN/actions/workflows/ci.yml/badge.svg)](https://github.com/kprodigi/IPINN/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Forward prediction and inverse design of hexagonal composite ring structures under quasi-static crushing using three PINN architectures (DDNS, Soft-PINN, Hard-PINN), dual validation protocols, regularised inverse design with ill-posedness characterisation, and multi-objective optimisation via qLogNEHVI MOBO.

## Key Features

- **Three-tier PINN hierarchy**: Data-driven (DDNS), soft physics constraint (Soft-PINN), and structurally enforced physics (Hard-PINN with E=g(d)*NN and F=dE/dd by construction)
- **Explicit boundary enforcement**: E(d=0)=0 guaranteed by network architecture, not soft penalty
- **Residual-based adaptive refinement (RAR)**: Collocation points concentrate where physics residuals are largest
- **Regularised inverse design**: Tikhonov regularisation + multi-start GP-BO + LC plausibility classifier
- **Ill-posedness analysis**: Solution landscape mapping, multiplicity index, forward-map Jacobian with bifurcation detection, approximate inverse posterior with credible intervals
- **Multi-objective optimisation**: Dense Pareto sweep + qLogNEHVI MOBO (BoTorch)
- **Dual validation**: Random 80/20 split + unseen-angle holdout + leave-one-angle-out CV
- **HPC parallel pipeline**: 12-stage SLURM workflow for ~2.2x speedup on GPU clusters

## Quick Start

### Installation

```bash
git clone https://github.com/kprodigi/IPINN.git
cd IPINN
pip install -e ".[all]"    # core + GP-BO + MOBO
```

Or using conda:

```bash
conda env create -f environment.yml
conda activate ipinn-crash
```

### Run Tests

```bash
pytest
```

### Single-Machine Run

```bash
# Full paper run (requires skopt + botorch):
python composite_design_v19.py --data_dir . --output_dir ./results_paper --strict_paper

# Quick smoke test:
python composite_design_v19.py --dry_run --data_dir tests/fixtures --output_dir ./results_test --force_cpu
```

### HPC Parallel Run (SLURM)

```bash
# Submits 12 jobs: prep -> 8 parallel GPU training -> analysis -> aggregate
DATA_DIR=. OUTPUT_DIR=./results_paper CONDA_ENV=ipinn bash submit_pipeline.sh
```

See the [dependency graph](#hpc-parallel-pipeline) below.

## CLI Options

| Flag | Effect |
|------|--------|
| `--data_dir DIR` | Input CSV / XLSX location |
| `--output_dir DIR` | All outputs |
| `--seed N` | Global seed base (default: 2026) |
| `--n_ensemble M` | Ensemble size (default: 20) |
| `--strict_paper` | Require skopt + botorch; abort if missing |
| `--force_cpu` | CPU even if CUDA is available |
| `--show_plots` | Display plots on screen |
| `--no_mobo_qnehvi` | Skip MOBO (disallowed with `--strict_paper`) |
| `--no_reviewer_proof` | Skip baselines, sensitivity, calibration vs M |
| `--no_ablation` | Skip physics-weight ablation study |
| `--no_loao_cv` | Skip leave-one-angle-out cross-validation |
| `--no_rar` | Skip residual-based adaptive refinement |
| `--inverse_ablation` | Run inverse ablations (no Tikhonov / no classifier / no robustness) |
| `--no_inverse_stress` | Skip validation-row inverse stress targets |
| `--no_inverse_member_spread` | Skip per-member spread table |
| `--dry_run` | CI/smoke: tiny budgets, M<=2, no MOBO/GP-BO/LOAO-CV/RAR |

## Output Artifacts

All outputs are written to `--output_dir`. `MANIFEST_outputs.csv` auto-inventories every file.

### Forward Prediction

| Artifact | Purpose |
|----------|---------|
| `Fig_parity_*`, `Fig_residual_*`, `Fig_boxplot_*` | Forward accuracy across approaches and protocols |
| `Fig_unseen_*`, `Fig_random_grid_*` | Load/energy curves with conformal uncertainty bands |
| `Fig_loao_cv.png`, `Table_loao_cv.csv` | Leave-one-angle-out CV per-angle R-squared |
| `Fig_reliability_diagram.png`, `Fig_calibration_vs_M.png` | Ensemble calibration and coverage vs M |
| `Table1_forward_results.csv` | Summary: R-squared, RMSE, MAE, conformal factors |
| `Table2_statistical_tests.csv` | Welch t-tests with Bonferroni correction |

### Inverse Design and Ill-Posedness

| Artifact | Purpose |
|----------|---------|
| `Table3_inverse_illposedness.csv` | Multiplicity index, posterior stats, sensitivity, penalties |
| `Fig_solution_landscape.png` | J(theta) per LC with local minima and SMI |
| `Fig_inverse_posterior.png` | Approximate posterior P(theta \| target) with 95% CI |
| `Fig_forward_map_jacobian.png` | dEA/dtheta, dIPF/dtheta with bifurcation detection |
| `Table_inverse_topk_basins.csv` | Ranked basins from landscape + multi-start BO |
| `Table_inverse_theta_member_spread.csv` | Per-member uncertainty at the optimum |

### Multi-Objective Analysis

| Artifact | Purpose |
|----------|---------|
| `Table6_pareto_sweep.csv`, `Fig_pareto_tradeoff.png` | EA vs IPF Pareto front |
| `Fig_multiobjective_heatmaps.png` | Dense landscape visualisation |
| `Table_mobo_qnehvi_*`, `Fig_mobo_*` | qLogNEHVI MOBO diagnostics |

### Reproducibility

| Artifact | Purpose |
|----------|---------|
| `runtime_environment.json` | Python, torch, numpy versions |
| `STATISTICAL_TESTING_POLICY.txt` | Confirmatory vs exploratory test framing |
| `Table_compute_reproducibility_budget.csv` | Training times, evaluation counts |
| `MANIFEST_outputs.csv` | Complete file inventory |

## HPC Parallel Pipeline

The `submit_pipeline.sh` script splits the monolithic pipeline into 12 SLURM jobs:

```
                             prep
                              |
        +-------+-------+----+----+-------+-------+-------+--------+
        |       |       |         |       |       |       |        |
      train   train   train     train   train   train   train    loao
      rand    rand    rand      uns     uns     uns     inv      cv
      ddns    soft    hard      ddns    soft    hard
        |       |       |         |       |       |       |        |
        +-------+-------+---------+-------+-------+------+--------+
                              |                           |
                    forward_analysis                      |
                              |                           |
                              +---------------------------+
                              |
                    inverse_analysis
                              |
                         aggregate
```

Critical path: ~36 hours (vs ~80 hours sequential).

## Methodology

- **Hard-PINN boundary enforcement**: `E = (d_scaled - d_zero_scaled) * NN(x)` guarantees E(d=0) = 0 exactly, matching the structural F = dE/dd enforcement
- **RAR collocation**: Sampling weights updated every 50 epochs proportional to |physics residual|, concentrating enforcement at unseen angles
- **Regularised inverse**: J = fit_error + lambda * LC_penalty + beta * uncertainty_penalty + gamma * Tikhonov
- **Multi-start GP-BO**: N=5 restarts with different seeds for global optimality
- **Ill-posedness diagnostics**: Forward-map Jacobian identifies where dEA/dtheta or dIPF/dtheta change sign (bifurcation = non-uniqueness source)
- **MOBO**: qLogNEHVI (BoTorch) as the primary multi-objective method; reference point documented in `mobo_qnehvi_reference_point.json`
- **Statistics**: Welch t-tests with within-protocol Bonferroni correction; see `STATISTICAL_TESTING_POLICY.txt`

## Project Structure

```
IPINN/
  composite_design_v19.py   # Main pipeline (~9500 lines)
  hpc_run_stage.py          # HPC stage dispatcher for SLURM
  submit_pipeline.sh        # SLURM job submission script
  requirements.txt          # Pip dependencies
  environment.yml           # Conda environment
  pyproject.toml            # Package metadata
  LC1.xlsx, LC2.xlsx        # Experimental data
  tests/
    conftest.py             # Shared test fixtures
    test_core.py            # 80+ unit tests
    test_smoke.py           # Smoke tests
    test_helpers_manifest.py # Manifest and figure tests
    fixtures/
      tiny_crush.csv        # Minimal dataset for CI
```

## Citation

If you use this software, please cite it using the metadata in [CITATION.cff](CITATION.cff).

## License

[MIT](LICENSE)
