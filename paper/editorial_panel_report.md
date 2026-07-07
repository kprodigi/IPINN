# Five-lens editorial review and revision record (manuscript_v2)

An independent five-agent editorial panel reviewed the manuscript, one agent
per lens. Scores (out of 10) on first read, and the action taken.

| Lens | Score | Summary |
|---|---|---|
| Simple-language / logic | 6.5 | Hard concepts explained well, but several ML/statistics terms lacked a plain-word gloss and a few sentences were over-nested. |
| Flow / engagement | 7 | Strong section openings and hand-offs; drags from signature-phrase repetition, back-to-back parameter lists, and two rough section seams. |
| Structure | 7 | Sound macro-architecture; hurt by a table/paragraph merge bug, four uncited figures, and methods-before-data forcing forward references. |
| Claims / novelty defense | 8 | Claim discipline well above norm; a few over/underclaims and one internal Table 1/Table 7 inconsistency. |
| American English / consistency | 3 | Body was systematically British while title/abstract were American — mixed variety, a copyedit fail. |

## Actions taken (all applied to manuscript_v2)

**Structure.** Swapped the section order so Experimental methods and data
(Section 3) now precedes Methodology (Section 4), matching the
problem→experiment→model→result convention and removing forward references;
updated the roadmap and every cross-reference. Added a Section 3 chapeau and
bridges into Sections 3, 5.6, and 5.7. Moved the dataset-overview figure to
Fig. 1 and renumbered all figures; every figure (1–13) is now cited in the
body in order. Added a Methods subsection for the data-only mechanics
analysis and a multi-objective-characterisation paragraph, so every Results
subsection now mirrors a Methods counterpart. Fixed two table/paragraph
merge bugs (Tables 2 and 12 had prose glued to their last row). Converted
Limitations to a numbered list. Consolidated the three inline
hyperparameter litanies into Table 2 (HPO-selected configurations).

**Simple language.** Added plain-word glosses for conformal calibration,
the Tukey fence (dropping the MCMC analogy), Chebyshev and weighted-sum
scalarisation, GP-BO and Expected Improvement, collocation points, bootstrap
resampling, automatic differentiation, SWA, the pseudo-posterior
temperature, AUC, the GP prior, Pareto optimality, and a plain restatement
of Eq. (1). Split the four longest sentences (contribution 2, the T4
degeneracy, the Hamiltonian antecedent, the coverage caveat).

**Claims.** Fixed the abstract's twelve-curves/withheld-60° contradiction;
split the conflated "several-fold better than floors" claim and corrected
the skill-margin range to the arithmetically correct 1.4–4.6×; corrected
Highlight 4 to scope sub-degree recovery to the off-grid round trips; softened
contribution 1's "essential" to "supporting" (no ablation) and reworded the
"eliminates penalty-weight sensitivity" claim; differentiated the hard-
constraint antecedents [22, 23]; removed the circular physics-residual
corroboration; recomputed Table 1 with the paper's own IPF definition so it
agrees with Table 8 (the T3-outlier narrative now rests on consistent
numbers); added the LC2 qualifier to the abstract plateau-force law;
tempered the "calibrated" wording to "recalibrated … honestly reported";
and surfaced two underclaimed results (the rout of conventional ML baselines
and the data-consistent-yet-model-inconsistent DDNS finding) in Conclusion 1.

**Flow.** Trimmed the repeated "honest/honestly" and the twice-told protocol
definition and classifier-ablation numbers; removed the duplicated
Hard-PINN clause in the coverage caveat.

**Language.** Converted the entire body to American English (108
replacements across 32 word forms: -ize/-ization/-izer families, fiber,
behavior, modeling, labeled, neighboring, favorable, toward, etc.), leaving
reference-list titles as published. Unified terminology: held-out-angle
protocol (not held-out-design), loading configuration (not loading-case),
Matérn, EA@80 mm, fallback.
