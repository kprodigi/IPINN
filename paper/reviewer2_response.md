# Anticipated Reviewer 2 report and point-by-point responses

An adversarial pre-submission review of `manuscript_v2` (Physics-Informed
Inverse Design and Crashworthiness Optimization of Hexagonal Composite Ring
Structures), written from the perspective of a skeptical expert reviewer,
together with the action taken on each point. Comments marked **[FIXED]**
are addressed in the manuscript; **[DISCLOSED]** are acknowledged
limitations argued honestly; **[REBUTTAL]** are defended with evidence.

---

**R2.1 — "Twelve curves, one specimen per configuration, no replicates.
No statistical claim about the design space can survive this sample size.
The 'jagged trends' you describe may simply be noise."** [DISCLOSED +
REBUTTAL]

Partly correct, and the manuscript says so before the reviewer can:
Section 4.2 states that aleatoric scatter is unidentifiable and all
uncertainty is epistemic; Limitation (i) caps the achievable design-level
accuracy with measured model-free floors. Two rebuttal points: (a) curve-
level claims rest on N ≈ 10,500 points, not N = 12; the design-level claims
are the ones bounded by floors, and they are presented as such. (b) The
design trends are not pure noise: the data-only mechanics analysis recovers
a systematic kinematic law (LC2 plateau force ~ sinθ·cosθ, R² = 0.947 with
the ranking robust across five candidate forms) that random specimen
scatter would not produce.

**R2.2 — "R² = 0.82 is far below the 0.95+ routinely reported for
crashworthiness surrogates."** [FIXED]

Those 0.95+ figures come from random row splits, which for densely sampled
curves measure within-curve interpolation. The manuscript's argument was
already present; one sentence quoted a specific interpolation score
(R² > 0.97) from a preliminary run that is not part of the released
artifact set — that number is now removed and the argument made purely
structural. The floors of Section 5.5 give the honest yardstick.

**R2.3 — "θ\* = 60° is an interior angle. Calling this generalisation is
generous — it is interpolation in design space."** [FIXED]

Correct, and now stated in exactly those words in Section 3.5: the
protocol tests geometric interpolation at an unobserved design, not
extrapolation beyond the observed range; boundary angles are harder still
(Limitation ii). No claim of extrapolation is made anywhere.

**R2.4 — "Your energy channel is the integral of your load channel. The
work–energy 'constraint' is a tautology, and the Hard-PINN's zero residual
is trivial."** [FIXED — argument added]

The zero residual is indeed architectural, and the data-level identity is
disclosed in Section 4.2. The new sentence in Section 5.3 makes the
non-trivial part explicit: the DDNS is trained on the *same* identity-
satisfying data and still violates the identity by ≈19% of the mean
plateau load — model-level consistency does not follow from data-level
consistency, which is precisely what architectural enforcement adds.

**R2.5 — "Hyperparameters were tuned on the fold you report. The
comparison is contaminated."** [DISCLOSED — strengthened]

Disclosed in Section 3.7 and Limitation (iii). Strengthened: identical
treatment does not guarantee identical benefit, so a residual effect on
gap sizes cannot be excluded; the qualitative ranking is corroborated by
tuning-independent diagnostics (physics residual, calibration factors),
and the released code defaults to an inner tuning fold for future work.

**R2.6 — "Where is the comparison against standard ML surrogates (random
forest, XGBoost, Gaussian processes)? Neural networks may be unnecessary."**
[FIXED — new analysis]

Regenerated from the released data on the exact held-out-angle split, with
the pipeline's own baseline harness (`train_baseline_models`), and added as
Table 4: all four conventional baselines collapse at the held-out angle
(details in the table); tree ensembles partition the feature space and
cannot produce new angle behaviour, and the subsampled GP reverts toward
prior behaviour off its support. Every neural surrogate — including the
unconstrained DDNS — beats every conventional baseline by a wide margin,
and the Hard-PINN adds its physics margin on top.

**R2.7 — "Your calibrated bands still under-cover (74–86% at nominal
95%). How can downstream decisions rely on them?"** [DISCLOSED —
strengthened]

The honest coverage numbers are the point: a row-level split would have
reported near-nominal coverage by construction. Added caveat in Section
5.4: coverage estimated from few held-out curves carries sampling
uncertainty and is read as a ranking (Hard-PINN needs the smallest
corrections) rather than a guarantee; design-level decisions use the
separately calibrated (≈4.4×) design-level factors.

**R2.8 — "The classifier is a coin flip (LOO AUC 0.53). Remove it."**
[REBUTTAL — justification added]

If it changed answers, with AUC 0.53, *that* would be alarming. The
ablation shows it flips nothing and moves angles ≤ 0.4°; it is retained as
a cost-free guardrail against implausible (EA, IPF) proposals in sparsely
observed corners, and it strengthens automatically as data accumulate
(Section 3.8).

**R2.9 — "T3 misses the true angle by 8.1° — 32% of the design range.
The posterior spanning [45.9°, 69.7°] means the method learned nothing."**
[REBUTTAL — already in text]

The recovered design delivers the requested (EA, IPF) to 0.5/0.6%: for
design purposes the target is met. The 8° deviation is measured
ill-posedness — the LC1-60° specimen is the design table's outlier and the
smooth map places its performance at 68° — and the near-full-range credible
interval is the framework *honestly reporting* that this target does not
identify a unique angle. A method that returned a tight interval here
would be wrong, not better.

**R2.10 — "GP-BO over one continuous variable and a binary flag is
overkill; a grid solves this in seconds."** [FIXED — framing added]

Correct that the grid is affordable here — that is exactly what makes the
Table 11 grid *anchor* possible. The added sentence in Section 5.7 frames
the small space as a feature for verification, with the sample-efficient
search (11–20 evaluations) being the component that transfers to richer
design spaces where grids fail.

**R2.11 — "Differentiating a learned potential is not new — Hamiltonian
neural networks did this in 2019."** [FIXED — citation + positioning]

Agreed; the construction has clear antecedents, now cited (Hamiltonian
neural networks) in the Introduction. The claimed novelty is the
combination: the construction deployed on experimental crashworthiness
data, under an honest held-out-design protocol, inside a *verified*
inverse loop — none of which follows from the architectural trick alone.

**R2.12 — "Why Softplus activations?"** [FIXED]

One clause added in Section 3.2: the force is a first input-derivative of
the network and the curvature regulariser differentiates it again, so a
smooth (C^∞) activation is required; ReLU-family activations would make the
curvature loss degenerate.

**R2.13 — "A two-parameter fit to six points with R² = 0.95 proves
nothing (mechanics section)."** [FIXED — clarified]

Clarified in Section 5.1: with two parameters on six points the absolute
R² is secondary; the inference rests on the *ranking* among the five
candidate kinematic forms, which is robust, and on the independent
corroboration by the master-curve collapse.

**R2.14 — "The design space is one angle. Real crashworthiness design has
thickness, cell size, material variables."** [FIXED — limitation added]

Added as Limitation (vi): a single continuous geometric variable is
demonstrated; nothing in the framework is dimension-specific (the GP-BO,
conformal, and verification layers are dimension-agnostic), but the
demonstration at higher dimensionality is future work.

**R2.15 — "Ensemble members share data and architecture; your t-tests are
invalid."** [ALREADY ADDRESSED]

The manuscript already treats bootstrap CIs as the inferential statistic
and reports t/p as descriptive only (Table 3 note, STATISTICAL_TESTING_
POLICY.txt in the artifact set).

**R2.16 — "Data availability says 'available at GitHub' — reviewers need
a frozen version."** [ALREADY ADDRESSED / ACTION FOR AUTHORS]

The repository is tagged (v1.1-paper) and the trained bundles are archived
in-repo; minting a Zenodo DOI for the tag before submission is recommended
and remains an author action.

---

## Summary of manuscript changes in this pass

1. New Table 4 (conventional-ML baselines on the held-out split, freshly
   regenerated) + supporting paragraph in Section 5.2; subsequent tables
   renumbered 5–11.
2. Removed the unsupported R² > 0.97 interpolation figure (Section 3.5);
   the within-curve-interpolation argument is now structural.
3. Interpolation-vs-extrapolation terminology made explicit (Section 3.5).
4. DDNS-violates-the-tautology argument added (Section 5.3).
5. HPO fold-selection caveat strengthened (Section 3.7).
6. Classifier-retention justification added (Section 3.8).
7. Grid-anchor framing of GP-BO added (Section 5.7).
8. Softplus smoothness rationale added (Section 3.2).
9. Hamiltonian-NN antecedent cited and positioned (Introduction);
   references renumbered.
10. Limitation (vi): single design variable.
11. Coverage-estimate sampling caveat added (Section 5.4).
12. Mechanics degrees-of-freedom clarification (Section 5.1).
