# Paper artifacts

This directory contains the publication-ready outputs that back the
manuscript: every figure (10 main + 1 appendix) and every table (6 main
+ supplementary).

| Subdirectory | Contents |
|---|---|
| [`figures/`](figures/) | PNG renderings of all manuscript figures at 600 DPI (Arial bold, 7.48-inch full-page width sized for journal insertion). |
| [`tables/`](tables/) | CSV tables backing every reported number, including Tables 1–6 of the manuscript plus supplementary analysis tables. |

The figures and tables are regenerable from the trained-model archive
(see the project [README](../README.md) under "Trained model bundles")
in approximately 5 minutes on a single CPU using
`python composite_design.py --mode replot --output_dir <bundle_dir>`.

The numeric values in [`tables/Table1_forward_results.csv`](tables/Table1_forward_results.csv)
are the headline forward-prediction metrics; see the [project README](../README.md)
for the summarised table.
