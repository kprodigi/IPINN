# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to semantic versioning.

## [1.0-paper-final] — 2026-05-27

The release accompanying the manuscript submission.

### Final headline metrics (M=20 ensemble, unseen-angle θ\*=60° protocol)

| Approach   | Load R² | Energy R² | Load RMSE (kN) | M_eff/M_total |
|------------|--------:|----------:|---------------:|:-------------:|
| DDNS       |  0.7175 |    0.9803 |          0.082 |     18/20     |
| Soft-PINN  |  0.7875 |    0.9911 |          0.071 |     20/20     |
| Hard-PINN  |  0.8217 |    0.9888 |          0.065 |     20/20     |

### Added

- `paper/figures/` — all 10 main paper figures and the bonus validation
  error-map figure (PNG, 600 DPI).
- `paper/tables/` — all paper tables (CSV) plus supplementary numerical
  outputs (Pareto dominance, λ-sensitivity, classifier ablation, etc.).
- `scripts/build_forward_models_bundle.py` — combines the
  per-approach forward bundles produced by `hpo/forward_merge.py` into
  the combined `forward_models.pt` layout expected by `--mode replot`.
- Auto-scaling figure typography: `scaled_fonts(fig_width)` now scales
  fonts and line widths up so the on-page sizes after journal insertion
  at the full-column width are always the publication targets
  (label=16 pt, tick=14 pt, legend=12 pt, all Arial bold).
- SQLite race-condition mitigation for concurrent HPO worker startup
  (worker-ID staggered DB init + retry with exponential backoff).
- `PYTHONUNBUFFERED=1` in the SLURM HPO launcher so per-worker
  `.out` logs are not block-buffered.
- MedianPruner across HPO trials with seed-1 pruning for Hard-PINN
  (which cannot use per-trial EarlyStopping while SWA is active).
- CHANGELOG.md (this file).

### Changed

- Hard-PINN production architecture is the bare-MLP form with force
  recovered as F = ∂E/∂d via autograd (work-energy identity by
  construction).  The optional architectural BC correction
  (`HardEnergyNet.configure_zero_bc`) is retained as ablation
  infrastructure but disabled in production training.
- `cfg_ddns`, `cfg_soft`, `cfg_hard` in `get_model_config` now contain
  the hyperparameters tuned by the Optuna search reported in the paper.
- `forward_merge.py` reconstructs Hard models with the BC disabled to
  match how production training instantiates them.
- HPO `n_startup_trials` default raised from 15 to 30 (~1.5× the
  search-space dimensionality) for a more diverse posterior before TPE
  engages.
- HPO `HPO_EPOCHS` default raised from 200 to 800 so trial ranking
  reflects production deployment behaviour exactly.
- Repository prose throughout the codebase audited and rewritten so
  docstrings and comments describe the methodology in publication
  language, with no development-history references.
- Manuscript sections 1–4 rewritten end-to-end to match the current
  methodology (separate repository / local manuscript file; not tracked
  in git).

### Fixed

- `forward_merge.py` previously activated the slope-subtraction
  architectural BC at merge-time evaluation, producing nonsensical
  ensemble metrics for Hard-PINN whose weights had been trained with
  the BC disabled.  Unified to load all three approaches in the
  bare-MLP form, matching production training.
- HPO search ranges re-centred so every documented optimum sits between
  11% and 73% of its range, with no parameter pinned against a
  boundary that would prevent TPE from exploring around it.

### Documentation

- `README.md` rewritten as a publication landing page with headline
  results, reproduction recipe, and links to figures and tables.
- `docs/ARCHITECTURE.md` updated to reflect the production Hard-PINN
  architecture.
- `paper/figures/README.md` and `paper/tables/README.md` provide
  one-line descriptions of every artifact.

### Tagged

`v1.0-paper-final` — fingerprints the exact code state used to
generate every number, figure, and table in the paper.
