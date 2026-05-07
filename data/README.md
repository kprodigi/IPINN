# Experimental Dataset

This directory holds the quasi-static crushing measurements used to train and
evaluate every PINN variant in `composite_design_v20.py`.

## Files

| File | Loading condition | Crush distance |
|------|-------------------|----------------|
| `LC1.xlsx` | Lateral compression of the hexagonal ring (axis horizontal) | `d_end = 80 mm` |
| `LC2.xlsx` | Axial compression (axis vertical) | `d_end = 130 mm` |

Both files contain repeated experiments at the same set of layup angles
(θ ∈ {15°, 30°, 45°, 60°, 75°, 90°}). Each row is one displacement sample
within one experiment.

## Required columns (case-insensitive after rename)

The loader (`composite_design_v20.py::load_data`) renames columns by keyword
match, then validates against this canonical schema:

| Canonical name | Units | Source aliases recognised |
|----------------|-------|---------------------------|
| `disp_mm` | mm | any column containing `disp`, or named `d` / `u` |
| `load_kN` | kN | any column containing `load` or `force`, or named `f` |
| `energy_J` | J | any column containing `energy`, or named `e` |
| `Angle` | degrees | any column containing `angle` or `theta` |
| `LC` | string | column named `LC`, or any column containing `loading` |

If `LC` is absent from the spreadsheet itself, the loader infers it from the
filename (`LC1.xlsx` → `LC1`, `LC2.xlsx` → `LC2`).

## Validation rules enforced by `validate_input_data`

1. All five required columns are present with no NaNs.
2. `disp_mm` is non-negative everywhere.
3. Within each `(Angle, LC)` group, `disp_mm` is monotonically non-decreasing
   in row order — quasi-static crushing cannot retract.
4. `LC` values are warned-on but not rejected if they fall outside `{LC1, LC2}`.

## Use

```bash
# from the repo root:
python composite_design_v20.py --data_dir ./data --output_dir ./results --strict_paper
```

The CLI flag `--data_dir` is what every pipeline entry point reads; it
accepts either an absolute or repo-root-relative path. The HPO scripts and
`slurm/submit_pipeline.sh` default to `./data` for this reason.

## Cross-LC EA comparison (paper note)

LC1 and LC2 are evaluated at a common displacement `d_common = 80 mm` (the
shorter of the two crush paths) for the energy-absorption-per-unit-length
metric so the comparison is not confounded by LC2's longer crush window.
This convention is enforced in `composite_design_v20.py::compute_ea_per_unit`.
