---
title: "Supplementary Material for: Physics-Informed Inverse Design and Crashworthiness Optimization of Hexagonal Composite Ring Structures"
author: "Nahid Sarker^a^, Elsadig Mahdi^b^, Monzure-Khoda Kazi^a,\\*^"
---

*^a^Karen M. Swindler Department of Chemical and Biological Engineering, South Dakota School of Mines and Technology, SD, 57701, Rapid City, USA*

*^b^Department of Mechanical and Industrial Engineering, College of Engineering, Qatar University, PO. Box-2713, Doha, Qatar*

\* Corresponding author: Monzure-Khoda Kazi. *E-mail:* Kazi.Khoda@sdsmt.edu.

This document supplements the main article. Section, table, figure, and equation numbers of the form "Section 4.3", "Table 3", or "Fig. 5" refer to the main article; numbers prefixed with "S" refer to this document. All quantities shown here regenerate deterministically from the released code, data, and trained model bundles (see the Data availability statement of the main article).

**S1. Hard-PINN training stabilization**

Because the Hard-PINN's force prediction is a derivative of its energy output with respect to an input, the predicted force is sensitive to weight-space perturbations early in training, and a full base learning rate can amplify these into catastrophic updates. Three mechanisms address this instability jointly: (i) linear learning-rate warmup from zero over the first 80 epochs; (ii) deterministic cosine annealing thereafter; and (iii) stochastic weight averaging (SWA) — averaging the network weights over the final 20% of training instead of keeping only the last values — which converges to a flatter region of the loss landscape and reduces variance across ensemble members. Gradient norms are clipped (||∇L||~2~ ≤ 0.98) to guard against isolated explosion events from the second-order autodifferentiation graph, and early stopping is disabled for the Hard-PINN because SWA requires the full training trajectory.

**S2. Hyperparameter search protocol and selected configurations**

The hyperparameters of every surrogate are tuned independently using Optuna's multivariate Tree-structured Parzen Estimator with a budget of 150 trials per approach: 30 random startup trials, one warm-start prior, a MedianPruner that terminates clearly losing trials early, and SQLite-backed storage that lets multiple workers share one study. Each trial trains three independently seeded members for the production epoch budget (800 epochs), and the trial objective is the mean validation load R² across surviving members. Table S1 lists the selected configurations.

Table S1. HPO-selected configurations of the three surrogates.

| Setting | DDNS | Soft-PINN | Hard-PINN |
|:---|:---:|:---:|:---:|
| Hidden layers | [128, 64, 32] | [256, 128] | [128, 64] |
| Trainable parameters | 11,170 | 34,690 | 9,089 |
| Dropout | 0.016 | 0.008 | 0.006 |
| Softplus β | 18.9 | 12.1 | 11.7 |
| Batch size | 64 | 64 | 8 |
| Learning rate | 4.21 × 10^−5^ | 8.07 × 10^−3^ | 9.95 × 10^−5^ |
| Weight decay | 3.16 × 10^−5^ | 6.84 × 10^−4^ | 3.75 × 10^−3^ |
| Data-loss weights (w~F~, w~E~) | 3.57, 3.45 | 2.97, 0.95 | 6.80, 8.65 |
| Physics / BC weights | — | w~phys~ 0.52, w~BC~ 0.60 | — |
| Regularizer weights | — | w~mono~ 4.10, w~smooth~ 0.019 | w~mono~ 7.72, w~smooth~ 0.016, w~curv~ 1.29 × 10^−3^ |

**S3. Master-curve collapse test**

The mechanics analysis of Section 5.1 (main article) is corroborated by a master-curve collapse test: each measured load–displacement curve is normalized by its own plateau force and densification onset, and the pairwise dispersion of the normalized curves is compared against the raw-normalized case. For LC2, mechanics-based normalization halves the pairwise curve dispersion (NRMSE 0.85 → 0.43), showing that the specimens share one underlying shape once plateau level and densification timing are factored out; LC1 curves are already nearly self-similar (0.24 → 0.22). Figure S1 shows all four cases.

![](build/figs/Fig_master_curve_collapse.png){width=16cm}

Fig. S1. Master-curve collapse test: (a, b) raw-normalized LC1 and LC2 curves; (c, d) mechanics-normalized curves (each divided by its plateau force, displacement scaled by its densification onset), with pairwise NRMSE in each panel title.

**S4. Extended forward-prediction diagnostics**

Figure S2 shows parity plots at the held-out angle θ\* = 60° for load and energy, per surrogate. The progressive tightening of the load-parity cloud from DDNS to Soft-PINN to Hard-PINN mirrors the R² ordering of Table 3 (main article); the energy parities are nearly indistinguishable, consistent with the intrinsically smooth cumulative-energy surface.

![](build/figs/Fig_parity_unseen.png){width=16cm}

Fig. S2. Parity plots at the held-out angle for load (top row) and energy (bottom row): DDNS, Soft-PINN, and Hard-PINN, with the red dashed line marking perfect prediction.

**S5. Optimizer transparency and physical-plausibility audit**

Table S2 anchors every GP-BO solution of the main article against an exhaustive dense-grid evaluation of the identical objective (402 evaluations per target) and against 25 independent optimizer replicates (5 outer seeds × 5 restarts). The sample-efficient search needed 11–20 sequential evaluations per target (≈ 47 s wall time including GP fitting) and lands on the grid optimum to ≤ 1.0° for four of five targets; the T4 discrepancy is the cross-LC degeneracy discussed in Section 5.7 of the main article, where the two branches differ by ΔJ ≈ 4 × 10^−4^. Replicate dispersion is ≤ 0.35° except for the degenerate T4. In this two-variable space the dense grid is itself affordable — which is precisely what makes the anchor possible; the sample-efficient search is the component that transfers to richer design spaces (several geometric variables, material parameters, or simulation-in-the-loop objectives) where exhaustive evaluation is not.

Table S2. Optimizer transparency: GP-BO solutions versus the dense-grid anchor (402 objective evaluations per target) and angle dispersion across 25 independent runs (5 seeds × 5 restarts).

| Target | BO (θ, LC) | BO J | Grid (θ, LC) | Grid J | Seq. evals | θ spread (°) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| T1 | 63.5, LC1 | 0.00281 | 64.5, LC1 | 0.00258 | 11 | ± 0.35 |
| T2 | 45.0, LC1 | 0.00153 | 45.0, LC1 | 0.00153 | 11 | ± 0.00 |
| T3 | 68.1, LC1 | 0.00174 | 68.1, LC1 | 0.00174 | 13 | ± 0.18 |
| T4 | 70.0, LC1 | 0.00287 | 57.0, LC2 | 0.00244 | 15 | ± 5.88 |
| T5 | 60.4, LC2 | 0.00139 | 60.4, LC2 | 0.00139 | 20 | ± 0.34 |

A physical-plausibility audit of all 1,002 design-sweep evaluations recorded zero negative-force predictions, zero energy non-monotonicities, zero negative EA values, and zero IPF-fallback activations — the deployed surrogate is physically well-behaved over the entire queried design space, not merely at the reported optima. The objective decomposition at each optimum shows the fit terms driven to O(10^−5^)–O(10^−4^) with the residual objective dominated by the (intentionally small) plausibility penalty, confirming that the classifier steered rather than distorted the solutions; consistently, the classifier ablation flips no LC decision, moves no recovered angle by more than 0.4°, and leaves target-matching accuracy flat for penalty weights λ ≤ 0.02.

Figure S3 shows the inverse objective landscapes J(θ) per loading configuration for the five targets with detected minima and the solution multiplicity index per panel; Figure S4 overlays the predicted load–displacement curve at each recovered design on the nearest experimental curve of the same loading configuration.

![](build/figs/Fig_solution_landscape.png){width=16cm}

Fig. S3. Inverse objective landscapes J(θ) per loading configuration for the five targets, with detected minima; panel titles report the solution multiplicity index.

![](build/figs/Fig_inverse_vs_nearest_experimental_curve.png){width=16cm}

Fig. S4. Recovered designs versus nearest experimental curves (same LC): predicted load–displacement response at each inverse optimum (T1–T5) against the closest measured curve.

**S6. Additional archived diagnostics**

Beyond the material shown here, the released repository archives, per inverse target, the GP-BO convergence traces and posterior-evolution snapshots; the multi-seed robustness sweep; the λ-sensitivity and classifier-ablation tables; the leave-one-out classifier diagnostics; the Pareto scalarization and dominance tables; extended uncertainty-calibration diagnostics (including the design-level scalar calibration audit); and the full mechanics-analysis outputs (crush-mode signatures and densification-kinematics fits). Every artifact regenerates deterministically from the released model bundles with a single command documented in the repository README.
