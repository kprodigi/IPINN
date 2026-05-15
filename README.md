# IPINN — Physics-Informed Neural Networks for Crashworthiness

[![CI](https://github.com/kprodigi/IPINN/actions/workflows/ci.yml/badge.svg)](https://github.com/kprodigi/IPINN/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Forward prediction and inverse design of hexagonal composite ring structures under quasi-static crushing using three PINN architectures (DDNS, Soft-PINN, Hard-PINN), dual validation protocols, multi-start GP-BO inverse design with ill-posedness characterization, and weighted-sum multi-objective optimization.

## Key Features

- **Three-tier PINN hierarchy**: Data-driven (DDNS), soft physics constraint (Soft-PINN), and structurally enforced physics (Hard-PINN with F = dE/dd by construction)
- **Inverse design**: multi-start GP-BO with LC plausibility classifier penalty
- **Ill-posedness analysis**: Solution landscape mapping, multiplicity index, forward-map Jacobian with bifurcation detection, approximate inverse posterior with credible intervals
- **Multi-objective optimization**: Dense Pareto sweep with weighted-sum and Chebyshev scalarizations + dominance filtering
- **Dual validation**: Random 80/20 split + unseen-angle holdout
- **HPC parallel pipeline**: SLURM workflow with parallel GPU training stages

## Quick Start

### Installation

```bash
git clone https://github.com/kprodigi/IPINN.git
cd IPINN
pip install -e ".[all]"    # core + GP-BO
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
# Full paper run (requires skopt):
python composite_design.py --data_dir ./data --output_dir ./results_paper --strict_paper

# Quick smoke test:
python composite_design.py --dry_run --data_dir tests/fixtures --output_dir ./results_test --force_cpu
```

### HPC Parallel Run (SLURM)

```bash
# Run from the repo root.  Submits parallel SLURM jobs:
#   prep -> training stages -> analysis -> aggregate
CONDA_ENV=ipinn bash slurm/submit_pipeline.sh
```

See the [dependency graph](#hpc-parallel-pipeline) below.

### Hyperparameter Optimization (Optuna TPE)

Before running the full pipeline, retune the forward training hyperparameters
on your hardware.  Three independent Optuna studies (one per approach) each
write a SQLite study DB that survives SLURM preemption:

```bash
# From the repo root, on an HPC cluster with SLURM:
APPROACH=ddns N_TRIALS=80  bash slurm/submit_hpo.sh
APPROACH=soft N_TRIALS=120 bash slurm/submit_hpo.sh
APPROACH=hard N_TRIALS=100 bash slurm/submit_hpo.sh
```

See [`hpo/README.md`](hpo/README.md) for the full HPO workflow.

## CLI Options

| Flag | Effect |
|------|--------|
| `--data_dir DIR` | Input CSV / XLSX location |
| `--output_dir DIR` | All outputs |
| `--seed N` | Global seed base (default: 2026) |
| `--n_ensemble M` | Ensemble size (default: 20) |
| `--strict_paper` | Require skopt; abort if missing |
| `--force_cpu` | CPU even if CUDA is available |
| `--show_plots` | Display plots on screen |
| `--no_robustness` | Skip baselines, sensitivity, calibration |
| `--no_ablation` | Skip physics-weight ablation study |
| `--inverse_ablation` | Run inverse ablation (no classifier penalty) on first targets |
| `--no_inverse_stress` | Skip validation-row inverse stress targets |
| `--no_inverse_member_spread` | Skip per-member spread table |
| `--dry_run` | CI/smoke: tiny budgets, M<=2, no GP-BO |

## Output Artifacts

All outputs are written to `--output_dir`. `MANIFEST_outputs.csv` auto-inventories every file.

### Forward Prediction

| Artifact | Purpose |
|----------|---------|
| `Fig_parity_*`, `Fig_residual_*`, `Fig_boxplot_*` | Forward accuracy across approaches and protocols |
| `Fig_unseen_*`, `Fig_random_grid_*` | Load/energy curves with conformal uncertainty bands |
| `Fig_reliability_diagram.png` | Ensemble calibration |
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
| `Table_pareto_dominance.csv` | Non-dominated subset of the dense surrogate landscape |

### Reproducibility

| Artifact | Purpose |
|----------|---------|
| `runtime_environment.json` | Python, torch, numpy versions |
| `STATISTICAL_TESTING_POLICY.txt` | Confirmatory vs exploratory test framing |
| `Table_compute_reproducibility_budget.csv` | Training times, evaluation counts |
| `MANIFEST_outputs.csv` | Complete file inventory |

## HPC Parallel Pipeline

The `submit_pipeline.sh` script splits the monolithic pipeline into SLURM jobs:

```
                                  prep
                                   |
       +------+------+------+------+------+------+------+
       |      |      |      |      |      |      |      |
      train  train  train  train  train  train  train
      rand   rand   rand   uns    uns    uns    inv
      ddns   soft   hard   ddns   soft   hard
       |      |      |      |      |      |      |
       +------+------+------+------+------+------+ -----+
                       |                                |
                forward_analysis                        |
                       |                                |
                       +--------------------------------+
                       |
                inverse_analysis
                       |
                  aggregate
```

## Methodology

- **Hard-PINN**: single-output energy network; force is recovered exactly by autograd as F = dE/dd
- **Inverse objective**: J = fit_error + lambda * LC_penalty (matches manuscript Section 3.7.2)
- **Multi-start GP-BO**: N=5 restarts with different seeds for global optimality
- **Ill-posedness diagnostics**: Forward-map Jacobian identifies where dEA/dtheta or dIPF/dtheta change sign (bifurcation = non-uniqueness source)
- **Multi-objective sweep**: weighted-sum and Chebyshev scalarizations + 2D dominance filter on the dense landscape
- **Statistics**: Welch t-tests with within-protocol Bonferroni correction; see `STATISTICAL_TESTING_POLICY.txt`

## Project Structure

```
IPINN/
  composite_design.py     # Main pipeline (single module — forward + inverse + analysis)
  pyproject.toml              # Package metadata
  environment.yml             # Conda environment
  requirements.txt            # Pip dependencies
  README.md
  CONTRIBUTING.md
  CITATION.cff
  LICENSE

  data/                       # Experimental dataset
    LC1.xlsx, LC2.xlsx
    README.md                 # dataset description and column schema

  hpo/                        # Hyperparameter optimisation + per-member training
    hpo_search.py             # Optuna TPE entry point: --approach {ddns,soft,hard}
    forward_member.py         # per-member forward ensemble trainer (1 SLURM task / member)
    forward_merge.py          # forward-ensemble aggregator (Tukey filter + bundle)
    inverse_member.py         # per-member inverse-design surrogate trainer
    inverse_merge.py          # inverse-design pretrained-surrogate aggregator
    compare_methods.py        # cross-approach physics-correctness comparison
    README.md                 # HPO + per-member training workflow

  slurm/                      # HPC submission scripts
    submit_pipeline.sh        # full pipeline: 11-job dependency chain
    submit_hpo.sh             # Optuna HPO launcher, APPROACH={ddns,soft,hard}
    submit_forward.sh         # SLURM array launcher for forward per-member training
    submit_inverse.sh         # SLURM array launcher for inverse per-member + analysis
    hpc_run_stage.py          # per-stage CLI used by submit_pipeline.sh
    README.md                 # HPC submission guide

  docs/
    ARCHITECTURE.md           # file/line map: paper sections → code

  tests/
    conftest.py               # shared test fixtures
    test_core.py              # unit tests (training, losses, classifier, GP-BO)
    test_smoke.py             # CLI smoke tests
    test_helpers_manifest.py  # manifest and figure tests
    fixtures/
      tiny_crush.csv          # minimal dataset for CI dry runs
```

## Citation

If you use this software, please cite it using the metadata in [CITATION.cff](CITATION.cff).

## License

[MIT](LICENSE)
