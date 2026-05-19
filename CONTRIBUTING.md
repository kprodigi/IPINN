# Contributing

Thank you for your interest in IPINN.

## Development Setup

```bash
git clone https://github.com/kprodigi/IPINN.git
cd IPINN
pip install -r requirements.txt
pytest
```

## Running Tests

```bash
pytest tests/ -v
```

All tests must pass before submitting a pull request.

## Repository Layout

The project uses a flat single-module layout for the main pipeline so that
the full forward + inverse + analysis flow lives in one auditable file, with
HPO and HPC infrastructure in dedicated sibling directories:

```
.
├── composite_design.py    # main pipeline (single module)
├── data/                      # LC1.xlsx, LC2.xlsx
├── hpo/                       # Optuna HPO entry point
├── slurm/                     # SLURM submission scripts + HPC dispatcher
├── docs/                      # ARCHITECTURE.md (file/line map for reviewers)
└── tests/
```

## Code Style

- Python 3.10+ with type hints
- Single main module (`composite_design.py`) for reproducibility
- Publication figures follow Elsevier/Composite Structures formatting (600 DPI, 7.48" width)

## Reporting Issues

Open an issue on GitHub with:
1. Steps to reproduce
2. Expected vs actual behavior
3. Output of `python -c "import torch, skopt; print(torch.__version__, skopt.__version__)"`
