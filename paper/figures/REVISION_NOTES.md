# Figure revision notes

All figure fixes below were applied to `composite_design.py` **and visually
verified** by rendering the full suite from the staged intermediate results
(`results_hpc_test/.staging`) with a CPU PyTorch build. That staged run is a
small 2-member test ensemble with *old numbers*, so it validates
**layout / overlap / text / legends** — not the final values.

> **IMPORTANT — regenerate the shipped PNGs from the FINAL bundle.**
> The PNGs committed in this directory pre-date these code fixes, and the
> staged test data must NOT be used to overwrite them (it would regress the
> numbers). On the machine with the final trained bundle, run:
> ```bash
> python composite_design.py --mode replot \
>     --output_dir <bundle_dir> --replot_from <bundle_dir> --force_cpu
> ```
> The layout/overlap fixes below then apply on top of the final numbers.

## Fixed and visually verified (32 figures rendered clean)

| Figure | Fix |
|---|---|
| `Fig_dataset_overview` | Panels (b)/(c) boxplots on ~1 point/group collapsed to a red median dash; now scatter the points in LC1/LC2 colours matching the legend. |
| `Fig_cross_protocol` | Removed the grey ghost suptitle bleeding through the legend; protocol legend entries added only for protocols actually plotted. |
| `Fig_forward_map_jacobian` | Units corrected (dEA/dθ → J/deg); added (a)–(d) panel labels and a legend for the zero-slope / bifurcation reference lines. |
| `Fig_parity_unseen` | 1:1 "Perfect fit" line was black and hidden by near-black Hard-PINN markers; now a contrasting red, drawn above the scatter. |
| `Fig_multiobjective_heatmaps` | Large mid-figure whitespace gap removed by switching to `constrained_layout`; ±2σ EA/IPF bands clipped at 0 (no impossible negative peak force). |
| `Fig_validation_error_maps_angle_disp` | Unseen panels collapsed to a vertical stripe (single held-out angle); recast as an error-vs-displacement profile filling the panel. |
| `Fig_model_complexity` | Bar-value labels used an absolute data-unit offset that threw the time-panel labels far outside the axes; now scale-independent offset-points annotation. |
| `Fig_random_grid_curves` | Shared legend overlapped the top-row subplot titles; moved below all panels. |
| `Fig_unseen_load_curves`, `Fig_unseen_energy_curves` | ±2σ conformal load/energy bands clipped at 0 (were dipping to negative load). |
| `Fig_qq_load_residuals_unseen` | Added unit-bearing axis labels ("Ordered load residuals (kN)"); removed the orphan panel label. |
| `Fig_optimizer_comparison` | Empty "Target" legend box replaced by a "convergence history unavailable" note when no traces exist. |
| `Fig_d_common_sensitivity...` | Fixed garbled y-label `EA (J) to d` → `EA up to d (J)`. |
| Panel labels globally | `add_subplot_label` clearance increased (6→9 pt) — verified it clears the top y-tick in `parity`, `physics-verification`, `reliability`, `inverse-parity`, `landscape-disagreement`, `inverse-vs-nearest`, etc. |

Also spot-checked clean: `residuals_unseen`, `boxplot_comparison`, `training_curves`,
`design_space`, `pareto_tradeoff`, `inverse_error_vs_angle`,
`inverse_target_feasibility`, `bo_convergence_T*`, both schematics.

## Fixed in code but not renderable from the staged data (re-check after regeneration)

These need data not present in the test staging (posterior snapshots,
best-so-far histories, classifier CV). The code fixes are in place; verify
visually once regenerated from the final bundle:

- `Fig_gpbo_posterior_evaluation_T*` — fixed the duplicate final-panel bug (padding no longer repeats the last iteration) and moved the shared legend below the grid.
- `Fig_inverse_convergence_T*` — best-so-far now drawn as a step function with evaluations marked.
- `Fig_solution_landscape`, `Fig_inverse_posterior`, `Fig_lc_classifier_cv_diagnostics`, `Fig_classifier_decision_boundary` — inherit the global panel-label clearance fix.
