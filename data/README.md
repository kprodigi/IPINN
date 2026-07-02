# Experimental Dataset

This directory holds the quasi-static crushing measurements used to train and
evaluate every PINN variant in `composite_design.py`.

## Files

| File | Loading condition | Nominal crush distance |
|------|-------------------|------------------------|
| `LC1.xlsx` | Lateral compression of the hexagonal ring (axis horizontal) | `d_end = 80 mm` (see truncation note) |
| `LC2.xlsx` | Axial compression (axis vertical) | `d_end = 130 mm` |

## What the data actually contains (verified)

- **Interior angles:** θ ∈ {45°, 50°, 55°, 60°, 65°, 70°} — six designs per
  loading case, on a 5° grid.
- **One specimen per configuration.** There are **no replicate tests**: each
  (θ, LC) pair has exactly one crush curve.  Consequently, specimen-to-specimen
  (aleatoric) scatter is **not measurable from this dataset**, and all
  uncertainty bands produced by the pipeline are epistemic (ensemble) only.
- **Row counts:** LC1 = 4,248 rows; LC2 = 6,252 rows (≈640–1,050 displacement
  samples per curve, median step ≈ 0.12 mm).
- **Load resolution:** the load channel is quantized at 0.01 kN (10 N).
  Initial peak forces span ≈0.3–0.8 kN, so IPF values closer than ~10 N are
  indistinguishable in the ground truth — treat sub-quantization IPF "errors"
  accordingly.
- **Energy is derived, not independently measured:** `energy_J` reproduces the
  cumulative trapezoidal integral of `load_kN` over `disp_mm` to machine
  precision on 10 of 12 curves (max deviation 0.055%, explained by an excised
  segment).  The work–energy relation F = dE/dd therefore holds in the data
  **by construction**; PINN "physics consistency" against this channel is an
  inductive-bias/regularization benefit, not validation against independent
  physics.

### Known data caveats

- `LC1` θ=50°: a ~10.6 mm displacement segment (≈31.9→42.5 mm) is absent
  (≈85 rows).  Curves are treated as piecewise-sampled; EA integration
  interpolates across the gap.
- `LC2` θ=70°: the raw file contains one corrupted out-of-order logger row
  (disp 88.00 → 83.33 → 88.25 mm).  `load_data` now **auto-drops**
  non-monotone-displacement rows with a logged count, so this row never
  reaches training.
- `LC1` extends to 92 mm for some curves (481 rows > 80 mm, ≈11%): rows beyond
  `d_end = 80 mm` are used in training but excluded from metric evaluation by
  `disp_end_mm` — EA numbers are defined on [0, 80] mm for LC1.

### Provenance to be completed by the experimental authors

Material system / layup, specimen dimensions, test machine, crosshead rate,
and the test standard followed are not recorded in this repository and should
be added here (they are required for independent reproduction of the
experiments themselves).

## Required columns (case-insensitive after rename)

The loader (`composite_design.py::load_data`) renames columns by keyword
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

## Validation enforced by `load_data` (QA is active, not advisory)

1. A per-file read failure is **fatal** (a silent single-LC run would
   invalidate every cross-LC analysis).
2. Rows with missing values are dropped with a logged count.
3. Within each `(Angle, LC)` curve, rows whose displacement steps backwards
   are **auto-dropped** with a logged count (corrupted logger rows).
4. `validate_input_data` runs on the concatenated frame; issues are logged.
5. Both `LC1` and `LC2` must be present or the run aborts.

## Use

```bash
# from the repo root:
python composite_design.py --data_dir ./data --output_dir ./results --strict_paper
```

The CLI flag `--data_dir` is what every pipeline entry point reads; it
accepts either an absolute or repo-root-relative path. The HPO scripts and
`slurm/submit_pipeline.sh` default to `./data` for this reason.

## Cross-LC EA comparison (paper note)

LC1 and LC2 are evaluated at a common displacement `d_common = 80 mm` (the
shorter of the two crush paths) for the energy-absorption metric so the
comparison is not confounded by LC2's longer crush window.  Tables and
figures label this quantity `EA@80mm`; full-stroke values are labelled
`EA_full`.
