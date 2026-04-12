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

All 103 tests must pass before submitting a pull request.

## Code Style

- Python 3.10+ with type hints
- Single main module (`composite_design_v19.py`) for reproducibility
- Publication figures follow Elsevier/Composite Structures formatting (600 DPI, 7.48" width)

## Reporting Issues

Open an issue on GitHub with:
1. Steps to reproduce
2. Expected vs actual behavior
3. Output of `python -c "import torch; print(torch.__version__); import botorch; print(botorch.__version__)"`
