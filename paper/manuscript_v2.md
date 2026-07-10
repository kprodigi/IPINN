---
title: "Physics-Informed Inverse Design and Crashworthiness Optimization of Hexagonal Composite Ring Structures"
author: "Nahid Sarker^a^, Elsadig Mahdi^b^, Monzure-Khoda Kazi^a,\\*^"
---

*^a^Karen M. Swindler Department of Chemical and Biological Engineering, South Dakota School of Mines and Technology, SD, 57701, Rapid City, USA*

*^b^Department of Mechanical and Industrial Engineering, College of Engineering, Qatar University, PO. Box-2713, Doha, Qatar*

\* Corresponding author: Monzure-Khoda Kazi. *E-mail:* Kazi.Khoda@sdsmt.edu, *Phone:* +1 979 721 0640.

**Abstract**

Crashworthy composite energy absorbers require designs that maximize energy absorption while limiting peak crushing loads, yet optimization is hampered by complex failure mechanisms, scarce experimental data, and costly nonlinear simulations. This study presents a physics-informed inverse design framework for hexagonal glass/epoxy composite ring structures subjected to quasi-static compression. A hard-constrained physics-informed neural network (Hard-PINN) parameterizes the absorbed energy with a single-output network and recovers the crushing force by automatic differentiation, so the work–energy identity holds exactly at every point of the design space by construction. Using a dataset of twelve experimental crushing curves spanning six interior angles and two loading configurations, and evaluating under a held-out-angle protocol in which every θ\* = 60° measurement is withheld from training, the Hard-PINN attains an ensemble load R² of 0.822, outperforming a soft-constrained PINN (0.788) and an unconstrained data-driven surrogate (0.718), with all pairwise differences significant under bootstrap confidence intervals. Ensemble uncertainty is recalibrated by curve-level split-conformal correction, with honestly reported held-out coverage (86% at 2σ for the Hard-PINN). At the design level, the deployed surrogate reproduces the energy absorption and initial peak force of all twelve experimental configurations with mean absolute percentage errors of 4.4% and 5.8%. At the held-out angle, these design errors are 1.4- to 4.6-fold below the error of simply interpolating between neighboring experiments (the model-free floor). The surrogate is coupled to Gaussian-process Bayesian optimization with an ensemble loading-configuration plausibility penalty, and the inverse loop is verified against known ground truth: it recovers the generating loading configuration for five of five targets, achieves sub-degree angle recovery for off-grid verification targets (Δθ = 0.36° and 0.01°), and correctly flags a deliberately infeasible target as unattainable. Variance decomposition and objective-decomposition diagnostics make both the forward map and each inverse solution interpretable, and a data-only mechanics analysis corroborates the learned design trends with a kinematic LC2 plateau-force law (R² = 0.95). The framework provides a physically consistent, verifiable, and experimentally grounded design tool for crashworthy composite structures under severe data scarcity.

**Keywords:** Physics-informed neural networks; inverse design; Bayesian optimization; composite materials; crashworthiness; uncertainty quantification.

**Highlights**

- Hard-PINN recovers crushing force exactly as the gradient of a learned energy field
- Hard constraint lifts held-out-angle load R² to 0.82 vs 0.79 (soft) and 0.72 (none)
- Curve-level split-conformal calibration reports honest held-out coverage
- Verified inverse design: 5/5 loading-configuration recovery and sub-degree off-grid round trips
- Framework refuses an infeasible target and quantifies inverse ill-posedness

# Introduction

Crashworthy structural design is a central problem in automotive, aerospace, and civil engineering: controlled dissipation of impact energy governs occupant survivability and structural integrity [1, 2]. Woven-roving glass fiber/epoxy composites have emerged as promising candidates for energy-absorbing components such as crash boxes, honeycombs, and structural reinforcements because of their exceptional specific strength, low density, and favorable cost-to-performance ratio [3]. Among the available geometric topologies, hexagonal cellular structures combine superior stiffness-to-weight ratios with the characteristic plateau-stress response that is critical for progressive energy absorption during crushing [4]. Hexagonal composite ring structures are therefore strong candidates for next-generation crashworthy components.

Optimizing hexagonal composite rings for crashworthiness is nonetheless challenging because of their multiscale, history-dependent failure mechanisms: progressive crushing proceeds through micro-cracking, fiber pull-out, inter-ply delamination, and matrix fragmentation [5, 6]. Capturing these modes numerically requires costly explicit nonlinear finite-element simulation [7], while physical crash testing is expensive and slow, restricting the data that can practically be generated [8]. These constraints frustrate purely data-driven machine-learning approaches, which typically demand large training corpora and operate as opaque black-box models with no intrinsic guarantee of thermodynamic consistency [9, 10].

The inverse design problem amplifies these difficulties. The map from design parameters to crashworthiness response is ill-posed: distinct geometric and loading configurations can yield indistinguishable energy-absorption envelopes [11, 12], and the strongly nonlinear response — buckling, progressive folding, fracture — creates many local optima that confound gradient-based search [13, 14]. Exhaustive enumeration is infeasible in realistic design spaces [7, 15], and each candidate evaluation costs an expensive simulation or a destructive test, ruling out evaluation-hungry population metaheuristics such as Differential Evolution [16] or Particle Swarm Optimization [17]. Surrogate-assisted optimization has therefore become the dominant paradigm for expensive crashworthiness problems [18].

Hybrid frameworks that combine physics-based constraints with data-driven learning offer a pragmatic answer to these issues. Physics-Informed Neural Networks (PINNs), introduced by Raissi et al. [19] and extended to solid-mechanics applications [20, 21], embed governing equations into the network training loss so that the surrogate remains consistent with the underlying physics while requiring fewer training samples than a purely data-driven alternative. Most existing PINN formulations enforce the physics through soft penalties: the residual of the governing equation is added to the loss with a tunable weight, which encourages but does not guarantee compliance and can leave the model vulnerable to non-physical predictions in extrapolation regimes [22, 23]. Enforcing physics by differentiating a learned potential is not new in itself; Hamiltonian and Lagrangian neural networks [24] use the same idea. The hard-constraint PINNs of [22, 23, 25] enforce algebraic constraints through projection or completion layers; the present construction instead makes the force the exact derivative of the energy network. What is contributed here is the deployment of that construction on experimental crashworthiness data under an honest held-out-angle protocol, inside an inverse loop whose answers are verified rather than merely converged.

A second, less discussed weakness of the surrogate-based inverse-design literature is that inverse solutions are rarely *verified*. Reported inverse-design errors typically measure how closely the surrogate's prediction at the recovered design matches the requested target — a quantity that is small whenever the optimizer converges, regardless of whether the recovered design is physically correct. Under severe data scarcity this distinction matters: an inverse loop can "match" a target by exploiting surrogate bias. A credible inverse-design demonstration requires targets whose generating design is known, so that the recovered parameters can be compared against ground truth; verification targets located off the training grid, so that recovery cannot reduce to memorization; and deliberately infeasible targets, so that the framework demonstrates it can refuse impossible requests rather than silently returning a poor compromise.

This work presents a complete, verifiable pipeline for the forward prediction and inverse design of hexagonal composite-ring energy absorbers under quasi-static crushing, organized around a Hard-Constrained Physics-Informed Neural Network (Hard-PINN). The Hard-PINN parameterizes the cumulative absorbed energy Ê(d, θ, LC) with a single-output multilayer perceptron and recovers the crushing force as F̂ = ∂Ê/∂d through automatic differentiation, so the work–energy identity holds by construction at every input point — not merely at the training samples. Two baselines trained on identical data with identical tuning isolate the contribution of the hard constraint: an unconstrained Data-Driven Neural Surrogate (DDNS) with two independent outputs, and a Soft-Constrained PINN (Soft-PINN) that imposes the identity through a weighted penalty. All three are M = 20-member bootstrap ensembles with curve-level split-conformal calibration of their uncertainty, and the inverse module couples the full-data Hard-PINN to multi-start Gaussian-process Bayesian optimization (GP-BO) over the joint (θ, LC) design space, guided by an ensemble plausibility classifier.

This work extends the experimental and machine-learning foundations established for composite hexagonal ring crashworthiness [5, 26–28], advancing from purely data-driven surrogates to a physics-constrained paradigm that embeds the governing thermodynamic identity directly in the network architecture. The novel contributions of this study are:

1. *Hard-constrained physics-informed surrogate for crashworthiness.* A single-output energy network that enforces the work–energy theorem exactly through automatic differentiation, so the physics identity holds regardless of loss-weight tuning — the source of sensitivity in soft-constraint PINN formulations. It is complemented by physically motivated soft regularizers (force non-negativity, angle-smoothness, and curvature control) that encode thermodynamic and geometric priors supporting generalization to unseen interior angles.

2. *Honest held-out-angle validation with calibrated uncertainty.* A controlled three-way comparison of data-driven, soft-constrained, and hard-constrained surrogates under the held-out-angle protocol of Section 4.5 (all θ\* = 60° measurements withheld). Differences are tested with bootstrap statistics; uncertainty bands are conformally calibrated on curves never used to fit the calibration factors; and design-level accuracy is benchmarked against model-free interpolation floors.

3. *Verifiable inverse design.* A GP-BO inverse loop evaluated against known ground truth: targets generated from measured configurations (recovering the true loading configuration in five of five cases), off-grid round-trip verification targets recovered to sub-degree accuracy, and an infeasibility probe that the framework correctly refuses. To the authors' knowledge, this is the first crashworthiness inverse-design study to report all three verification modes together with ill-posedness diagnostics (solution multiplicity, posterior credible intervals, and multi-start dispersion).

4. *Interpretability and mechanics corroboration.* An exact variance decomposition of the learned forward map attributing design-metric variability to angle, loading configuration, and their interaction; a per-target decomposition of the inverse objective at the optimum; and an independent, data-only mechanics analysis (crush-mode signatures, densification kinematics, and master-curve collapse) that corroborates the surrogate-learned design trends with interpretable kinematic laws.

The remainder of this paper is organized as follows. Section 2 formulates the problem and fixes notation. Section 3 describes the specimens, the tests, the resulting dataset, and the measured crushing behavior. Section 4 presents the methodology: preprocessing, architectures, losses, training, ensembling, the validation protocol, conformal calibration, hyperparameter optimization, the inverse-design and verification modules, and the explainability layer. Section 5 reports the mechanics analysis, forward-prediction accuracy, physics verification, uncertainty calibration, design-level accuracy, verified inverse design, the multi-objective trade-off, and limitations. Section 6 summarizes conclusions and future work.

# Problem statement

Hexagonal composite ring structures subjected to quasi-static crushing exhibit coupled load–displacement and energy–displacement responses governed by geometry and loading orientation. The structural response is defined by three inputs: the loading configuration LC ∈ {LC1, LC2}, where LC1 denotes side-based (lateral) loading and LC2 denotes vertex-based (axial) loading; the interior hexagonal angle θ ∈ [45°, 70°]; and the crushing displacement d ∈ [0, d~end~(LC)] mm.

For each triplet (LC, θ, d) the measured outputs are the instantaneous crushing load F(d; θ, LC) (kN) and the absorbed energy E(d; θ, LC) (J). These quantities satisfy the work–energy consistency relation

*E(d; θ, LC) = ∫~0~^d^ F(d′; θ, LC) dd′,  equivalently  ∂E/∂d = F.*  (1)

In words: the absorbed energy is the area under the load–displacement curve, so the load equals the slope of the energy curve at every displacement. Dimensional consistency is maintained through the convention 1 kN·mm ≡ 1 J. Throughout the paper, hats denote learned network quantities (Ê, F̂), while unhatted symbols refer to experimental measurements.

*Forward-design objective.* Given θ, LC, and a displacement history d ∈ [0, d~end~(LC)], the forward problem is to learn a surrogate Ŝ(d; θ, LC) = (F̂, Ê) that reproduces the full crushing curves with high fidelity and with explicit — soft or hard — enforcement of Eq. (1).

*Derived crashworthiness metrics.* Two scalar indicators are extracted at an evaluation displacement d~eval~. The energy absorbed is

*EA(θ, LC; d~eval~) = E(d~eval~; θ, LC) − E(0; θ, LC).*  (2)

The initial peak force (IPF) is identified from F(d) over d ∈ [d~min~, d~eval~] as the first local maximum with prominence ≥ 5% of the load range and width ≥ 2 samples, with d~min~ = 0.5 mm filtering numerical transients; if no qualifying peak exists, the maximum within the first 25% of d~eval~ is used as a fallback to avoid misidentifying the densification peak. A higher EA indicates greater energy-absorbing capacity; a lower IPF indicates a more stable progressive crush. The mean crushing force F̄ = EA/d~eval~ and the crushing-force efficiency CFE = F̄/IPF are reported as supplementary indicators.

Because the two loading configurations have different natural stroke lengths (d~end~(LC1) = 80 mm, d~end~(LC2) = 130 mm), a comparison of EA at the respective d~end~ would confound intrinsic material response with crush-path advantage. All cross-LC inverse-design comparisons are therefore performed at the common displacement d~common~ = 80 mm; energy at the common displacement is labeled EA@80 mm and full-stroke values are labeled EA~full~.

*Inverse-design objectives.* Two inverse problems are formulated at d~eval~ = d~common~: (i) target matching — determine (θ\*, LC\*) ∈ [45°, 70°] × {LC1, LC2} that reproduces a prescribed target (EA~target~, IPF~target~); and (ii) multi-objective characterization — identify the Pareto-optimal set of designs (those for which neither objective can be improved without worsening the other) that simultaneously maximize EA and minimize IPF.

*Key research questions.* RQ1: Does hard physics enforcement improve generalization to unseen geometry relative to soft and no enforcement? RQ2: Do bootstrap ensembles, after curve-level split-conformal recalibration, deliver honestly quantified uncertainty? RQ3: Can GP-BO recover *verifiably correct* designs — matching known ground truth, succeeding off the training grid, and refusing infeasible targets? RQ4: Can decomposition diagnostics and an independent mechanics analysis make both the forward map and each inverse solution interpretable?

# Experimental methods and data

This section describes the specimens, the quasi-static tests, and the data screening that produced the twelve crushing curves on which every surrogate of Section 4 is trained and validated.

## Materials and specimen fabrication

Hexagonal thin-walled composite rings were fabricated from E-glass woven roving fabric impregnated with an epoxy resin system (resin-to-hardener ratio 100:17 by volume). Six plies of glass fabric were drawn through a resin bath and wet-wound onto a hexagonal mandrel under controlled tension, winding speed, and resin content. The shells were cured for 24 h at room temperature (≈ 32 °C) and sectioned into ring specimens of 20 mm height. The lay-up orientation was [0/90] and the fiber volume fraction 55 ± 7%. Specimens were produced for each of the twelve angle–loading combinations formed by six interior angles θ ∈ {45°, 50°, 55°, 60°, 65°, 70°} crossed with two loading configurations [5].

## Quasi-static compression tests

Specimens were compressed between parallel hardened-steel plates on a universal testing machine of 100 kN capacity at a constant crosshead speed of 2.5 mm/min, with force–displacement histories recorded continuously. In LC1 the plates contact the flat sides of the hexagon (side-based loading); in LC2 they contact opposing vertices (vertex-based loading). The stroke was 80 mm for LC1 and 130 mm for LC2, capturing the full crushing response up to densification in each configuration. Replicate tests in the underlying experimental campaign verified repeatability [5]; the modeling dataset used here comprises one representative crushing curve per configuration — twelve curves in total (Fig. 1), resampled to uniform displacement increments (≈ 640–1,050 points per curve; N ≈ 10,500 observations; load resolution 0.01 kN). Because no replicate curves are included, specimen-to-specimen scatter is not identifiable from these data, and all uncertainty bands reported in this study are epistemic (model) uncertainty. The energy channel is the cumulative trapezoidal integral of the load channel, so Eq. (1) holds in the data by construction; the value of physics enforcement in the surrogates is therefore an inductive-bias (regularization) benefit rather than validation against an independent measurement. Data-quality screening is automated in the released pipeline (corrupted logger rows dropped with logged counts; one 10.6 mm gap in the LC1, 50° record handled by piecewise integration).

![](build/figs/Fig_dataset_overview.png){width=16cm}

Fig. 1. Experimental dataset overview: (a) measured load–displacement curves for all six angles under both loading configurations; (b) energy absorbed EA and (c) initial peak force IPF versus interior angle by loading configuration.

## Crushing behavior

Under LC1 (stable crushing) all angles show a well-defined initial peak within 10–20 mm followed by decline to a stable plateau, as hinges form at the flat segments. Under LC2 (progressive crushing) the response has two phases: a low-load phase (0.2–0.5 kN) while the inclined walls bend and fragment, followed by a sharp load rise as collapsed material compacts under the advancing plates — the highest forces occur in this densification phase (θ = 70° reaches 1.42 kN near 91 mm). Three energy-absorption modes were observed in the underlying campaign — brittle shear cracking, plastic deformation, and Euler-type buckling — with prevalence governed by the thickness-to-side-length ratio [5]. Table 1 reports the crashworthiness indicators extracted from the measured curves at the natural stroke of each configuration.

Table 1. Experimental crashworthiness metrics. EA@80 mm is the energy absorbed to the common 80 mm displacement used for all cross-LC comparisons; EA~full~ is the energy over the natural stroke; IPF is the initial peak force per the Section 2 definition (for LC2 the global load maximum occurs later, during densification, and is deliberately not reported as IPF); F̄ = EA~full~/stroke; CFE = F̄/IPF.

| θ (°) | LC | Stroke (mm) | EA@80 mm (J) | EA~full~ (J) | IPF (kN) | F̄ (kN) | CFE |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 45 | LC1 | 0–80 | 28.34 | 28.34 | 0.468 | 0.354 | 0.757 |
| 45 | LC2 | 0–130 | 25.17 | 36.84 | 0.390 | 0.283 | 0.727 |
| 50 | LC1 | 0–80 | 28.17 | 28.17 | 0.490 | 0.352 | 0.719 |
| 50 | LC2 | 0–130 | 28.78 | 41.59 | 0.448 | 0.320 | 0.713 |
| 55 | LC1 | 0–80 | 23.41 | 23.41 | 0.440 | 0.293 | 0.665 |
| 55 | LC2 | 0–130 | 34.14 | 71.89 | 0.772 | 0.553 | 0.717 |
| 60 | LC1 | 0–80 | 30.24 | 30.24 | 0.672 | 0.378 | 0.563 |
| 60 | LC2 | 0–130 | 37.90 | 60.20 | 0.490 | 0.463 | 0.945 |
| 65 | LC1 | 0–80 | 26.62 | 26.62 | 0.580 | 0.333 | 0.574 |
| 65 | LC2 | 0–130 | 37.12 | 52.95 | 0.539 | 0.407 | 0.755 |
| 70 | LC1 | 0–80 | 33.81 | 33.81 | 0.768 | 0.423 | 0.550 |
| 70 | LC2 | 0–130 | 63.40 | 98.85 | 0.810 | 0.760 | 0.939 |

Two properties of Table 1 shape everything downstream. First, the design trends are *jagged*: EA and IPF are not monotone in θ (e.g., the LC1 minimum at 55° and the pronounced LC2 spike at 55°), reflecting genuine mode competition in single-specimen data. Second, the two loading configurations produce distinct (EA, IPF) signatures — the discriminative basis for the plausibility classifier of Section 4.8. These measured curves and indicators are the ground truth for every subsequent analysis.

# Methodology

The Inverse Physics-Informed Neural Network (IPINN) framework comprises two coupled modules (Fig. 2). The forward module learns the crushing response from experimental data and returns the load–displacement and energy–displacement curves as functions of (d, θ, LC). The inverse module repeatedly queries the validated forward surrogate to identify design parameters that match prescribed crashworthiness targets and to characterize the EA–IPF trade-off. A unified preprocessing pipeline is applied across all surrogates: a periodic angle embedding (sin θ, cos θ), one-hot encoding of LC, and standardization of inputs and outputs to zero mean and unit variance using training-set statistics.

To isolate the effect of physics integration, three forward-model variants — DDNS, Soft-PINN, and Hard-PINN — are trained under identical preprocessing, identical data splits, and identical evaluation metrics (Fig. 3). Fidelity is assessed under the held-out-angle protocol of Section 4.5, and epistemic uncertainty is quantified through bootstrap ensembling with Tukey-fence convergence filtering (Section 4.4) and curve-level split-conformal calibration (Section 4.6).

![](build/figs/Fig_framework_schematic.png){width=16cm}

Fig. 2. Integrated forward and inverse design methodology for crashworthiness optimization of hexagonal composite rings.

![](build/figs/Fig_architecture_schematic.png){width=16cm}

Fig. 3. Surrogate architectures and physics-enforcement strategies: DDNS (independent dual heads, no coupling), Soft-PINN (dual heads coupled by a weighted physics penalty), and Hard-PINN (single energy output; force recovered exactly by automatic differentiation).

## Data representation and preprocessing

The experimental dataset D = {(d~i~, θ~i~, LC~i~, F~i~, E~i~)} spans six interior angles θ ∈ {45°, 50°, 55°, 60°, 65°, 70°} and two loading configurations (Section 3). Because the interior angle is geometrically periodic, θ is embedded through its trigonometric components (sin θ, cos θ), which vary smoothly for any θ; LC is one-hot encoded. All continuous variables are standardized using training-set statistics,

*d~s~ = (d − μ~d~)/σ~d~,  F~n~ = (F − μ~F~)/σ~F~,  E~n~ = (E − μ~E~)/σ~E~,*  (3)

yielding the input feature vector x = [d~s~, sin θ, cos θ, 1~LC1~, 1~LC2~] ∈ R^5^. A direct consequence of operating in normalized coordinates is that the physical identity ∂E/∂d = F must be re-expressed in the network's coordinate system. Applying the chain rule to Eq. (3) gives

*F~n~ = G · (∂Ê~n~/∂d~s~),  G ≡ σ~E~/(σ~F~ σ~d~),*  (4)

where G is the gradient-scaling constant determined once from training-set statistics. This is the identity that the Hard-PINN enforces by construction in its autograd graph.

## Forward surrogate architectures

All three surrogates are fully connected multilayer perceptrons with Softplus activations and Kaiming-normal initialization; optional dropout is applied between hidden layers. Softplus is not incidental: the Hard-PINN differentiates the network once to recover F̂ and a second time for its curvature regularizer, so an infinitely differentiable activation is required — piecewise-linear activations would render the curvature loss degenerate. The critical distinction is how — and to what degree — the work–energy identity is embedded in the model.

*DDNS.* A two-headed MLP predicts standardized load and energy as independent outputs (F̂~n~, Ê~n~) with no physical coupling. Its tuned configuration (Table S1; 11,170 parameters) is selected by the hyperparameter search of Section 4.7. The objective is purely data-driven,

*L~DDNS~ = w~F~ · SmoothL1(F̂~n~, F~n~) + w~E~ · MSE(Ê~n~, E~n~),*  (5)

with w~F~ ≈ 3.57, w~E~ ≈ 3.45. The DDNS acts as the negative control: any improvement claimed by the physics-informed variants must exceed what a similarly tuned, unconstrained surrogate achieves from the same data.

*Soft-PINN.* The Soft-PINN retains the two-headed architecture but adds an explicit penalty for thermodynamic inconsistency. With the residual r(x) = F~n~ − G ∂Ê~n~/∂d~s~, the physics loss is the scale-invariant mean square L~phys~ = E~x~[(r/σ~F~)²], evaluated not only at training samples but at collocation points — extra input locations where the physics term can be checked without any measured data — sampled jointly over the full feature domain, including angles held out by the validation protocol, so that the constraint regularizes predictions where no data exist. Three further regularizers encode physical priors: a monotonicity loss penalizing negative energy gradients, an angle-smoothness loss on both heads, and a boundary-condition penalty enforcing E(0) = 0 and F(0) = 0. Its tuned configuration (Table S1) is the largest of the three models, at 34,690 parameters.

*Hard-PINN.* A soft constraint can reduce but not eliminate thermodynamic inconsistency, because the physics loss competes with the data losses and is satisfied only to the precision permitted by its weight. The Hard-PINN resolves this architecturally: the network parameterizes only the standardized energy Ê~n~(x) as a single scalar output, and the standardized force is recovered by automatic differentiation as F̂~n~ = G ∂Ê~n~/∂d~s~. Because Eq. (4) is satisfied by construction at every point in the input space — at training, at validation, and at every inverse-design query — the explicit physics-residual loss is unnecessary and is removed. Three collocation-based regularizers shape the learned energy surface: a monotonicity loss applied to the autograd-recovered force in physical units (penalizing F̂ < 0), an angle-smoothness loss on both Ê and F̂, and a curvature loss on ∂²Ê/∂d² that discourages oscillatory force profiles in data-sparse regions; collectively these also drive E(0) ≈ 0 and F(0) ≈ 0. Its tuned configuration (Table S1) is the smallest of the three models, at 9,089 parameters.

## Training procedure

For the DDNS and Soft-PINN a standard schedule suffices: Adam with a ReduceLROnPlateau scheduler on the validation load R², early stopping with patience 15 evaluations, and a maximum budget of 800 epochs. The Hard-PINN poses a different optimization challenge that arises from its defining strength: because F̂ is a derivative of Ê with respect to an input, the predicted force is sensitive to weight-space perturbations early in training, and a full base learning rate can amplify these into catastrophic updates. A warmup–cosine learning-rate schedule combined with stochastic weight averaging and gradient clipping stabilizes training (details in Supplementary Section S1).

## Ensemble construction and convergence filtering

Each surrogate is trained as a bootstrap ensemble of M = 20 members, each initialized with a distinct seed and trained on a bootstrap resample of the training rows (a random redraw of the rows with replacement, so each member sees a slightly different dataset). The ensemble mean and standard deviation of any scalar prediction provide the point estimate and its epistemic (model) uncertainty. Because individual members can converge to markedly poorer basins — particularly for the Hard-PINN — members are filtered by the Tukey fence on their training-set load R²: members below Q1 − 1.5 IQR (the standard boxplot outlier rule, where Q1 is the lower quartile and IQR the interquartile range) are discarded as failed training runs. The filter is bypassed when the across-member IQR is below 0.01, i.e. when the ensemble has clearly converged. The procedure is documented transparently via the reported M~total~ (members trained) and M~eff~ (members retained).

## Validation protocol and metrics

Generalization is assessed with a held-out-angle protocol: every measurement at θ\* = 60° is withheld as the validation set, and training spans θ ∈ {45°, 50°, 55°, 65°, 70°} only. The model must therefore reconstruct two complete, previously unseen crushing curves (LC1 and LC2 at 60°; 1,684 points) from geometric generalization alone. All preprocessors are fitted on the training partition only.

A random row-wise 80/20 split — the prevalent protocol in the surrogate crashworthiness literature — is deliberately *not* reported as evidence of generalization: because each curve contributes hundreds of densely sampled rows, a random row split places nearly identical neighboring points of the *same* curve on both sides of the partition, so the scores measure within-curve interpolation, approach the interpolation ceiling for every surrogate, and say nothing about the design generalization that inverse design requires. The term held-out *angle* is also used deliberately: θ\* = 60° is interior to the design range, so the protocol tests geometric interpolation at an unobserved design rather than extrapolation beyond the observed range — boundary angles are harder still (Section 5.9).

As external reference points, four conventional ML surrogates — ridge regression, a 100-tree random forest, gradient-boosted trees (XGBoost), and an anisotropic Matérn-5/2 Gaussian process with additive noise kernel, fitted on a 2,000-point subsample for tractability — are evaluated on identical features under the identical split (Section 5.2).

Performance is reported through the coefficient of determination R², RMSE, and MAE on each output, together with the thermodynamic-consistency residual \|F̂ − ∂Ê/∂d\| evaluated on validation inputs for all three surrogates (Section 5.3). Two levels of granularity are distinguished throughout: *curve-level* accuracy on the held-out load and energy signals, and *design-level* accuracy of the derived metrics (EA, IPF) that drive inverse design. At the design level, the deployed full-data surrogate is additionally audited against the measured values of all twelve configurations and benchmarked against a model-free floor — linear interpolation of the neighboring experimental design values over θ — so that its contribution beyond trivially interpolating the design table is quantified (Section 5.5).

## Curve-level split-conformal calibration

Bootstrap ensembles are typically over-confident because between-member variance underestimates predictive uncertainty when members share data and inductive biases. A split-conformal recalibration [29] is therefore applied. In plain terms, it multiplies the ensemble's uncertainty bands by a single factor, chosen on held-out data, so that the stated fraction of measurements actually falls inside the bands. One refinement matters here: the split is performed at the *curve* level, not the row level. The held-out curves are partitioned into a calibration subset and an evaluation subset; multiplicative factors c~1σ~ = Q~0.683~({z~i~}) and c~2σ~ = Q~0.954~({z~i~}) are estimated from the normalized residuals z~i~ = \|y~i~ − μ~i~\|/σ~i~ (each measurement's distance from the ensemble mean, in units of the ensemble standard deviation; Q~q~ is the empirical q-quantile) of the calibration curves only, and the resulting coverage is *measured on curves never used to fit the factors*. Both factors are reported because the residuals are heavier-tailed than Gaussian; a row-level split would leak within-curve correlation and overstate coverage, so the curve-level protocol yields the honest estimates of Section 5.4. Conformal factors are computed independently for the load and energy channels.

## Hyperparameter optimization

The hyperparameters of every surrogate are tuned independently using Optuna's multivariate Tree-structured Parzen Estimator with a budget of 150 trials per approach; each trial trains three independently seeded members, and the trial objective is the mean validation load R² across survivors (protocol details in Supplementary Section S2). The selected configurations are listed in Table S1 (Supplementary Section S2).

One protocol limitation is disclosed: the search objective was evaluated on the same θ\* = 60° fold that is used for the reported comparison, so the reported absolute scores benefit from tuning-fold selection to a degree shared by all three approaches; the codebase now defaults to an inner held-out angle (55°) for any future search. All three approaches received identical treatment (the same fold, the same 150-trial budget), but identical treatment does not guarantee identical benefit, so a residual effect on the gap magnitudes cannot be excluded; the ranking itself, however, is corroborated by a tuning-independent diagnostic: the conformal correction factors (Section 5.4), which are smallest for the Hard-PINN. The physics residual of Section 5.3 orders the models the same way, but that ordering follows directly from the enforcement strength and is not independent evidence of held-out accuracy.

## Inverse design

The inverse-design module is built on the Hard-PINN, which delivers the most physically consistent predictions by construction. A separate Hard-PINN ensemble (M = 20) is retrained on the complete dataset — all six angles and both loading configurations — using the optimized hyperparameters; this full-data ensemble serves exclusively as the forward surrogate for inverse design.

*Crashworthiness indicators and plausibility screening.* From the ensemble-predicted curves at a candidate (θ, LC), EA and IPF are extracted per ensemble member to propagate uncertainty through the nonlinear peak-detection operation. Because the optimizer searches jointly over θ and LC, it may propose a design whose predicted (EA, IPF) signature is implausible for the claimed loading configuration. A calibrated soft-voting ensemble classifier P~ens~(LC \| EA@80 mm, IPF, θ) — Gaussian Naive Bayes, RBF-kernel SVM, random forest, and a small MLP — detects such proposals through the penalty

*φ(θ, LC) = −log P~ens~(LC | EA, IPF, θ),*  (6)

added to the objective with a weight λ auto-tuned by leave-one-out cross-validation (λ = 0.0054 in production). With only twelve design-level observations the classifier is necessarily weak (leave-one-out AUC = 0.53, barely above the 0.50 of random guessing; Section 5.7), and it is therefore framed as a gentle plausibility regularizer rather than an oracle; its ablation-verified neutrality on the recovered solutions and its λ-sensitivity are quantified in Section 5.7. The penalty is retained despite its weakness because it costs nothing where the fit terms dominate, it guards against implausible (EA, IPF) proposals in sparsely observed corners of the design space, and it strengthens automatically, with no code change, as experiments accumulate.

*Target-matching objective.* For a target (EA~t~, IPF~t~) the optimizer minimizes

*J(θ, LC) = w~EA~ (EA(θ, LC) − EA~t~)² + w~IPF~ (IPF(θ, LC) − IPF~t~)² + λ φ(θ, LC),*  (7)

with w~EA~ = 1/EA~t~² and w~IPF~ = 1/IPF~t~², rendering both errors dimensionless. Candidate designs whose ensemble-mean EA is non-positive are rejected with a large finite penalty, and every sweep evaluation is screened by a physical-plausibility audit (negative-force fraction, energy non-monotonicity, EA sign, IPF-fallback usage; Section 5.7).

*GP-BO over the joint design space.* J is minimized by Gaussian-process Bayesian optimization (GP-BO). In plain terms, the optimizer fits a statistical model of the objective from the evaluations so far and, at each step, tests the design most likely to improve on the best value yet seen. The statistical model is a single Matérn-5/2 Gaussian process [30] over the joint (θ, LC) space, and the next design to test is chosen by the Expected Improvement criterion. Bayesian optimization of this kind is now widely used for sample-efficient material and structural design [31]. Each run draws 5 space-filling initial points and up to 15 sequential acquisitions (n~calls~ = 20) with early termination on stagnation; GP-BO is restarted 5 times with independent seeds and the best restart is reported, for a budget of 100 surrogate evaluations per target. Cross-restart angle spread and LC-unanimity are recorded as transparency diagnostics, and every GP-BO solution is anchored against an exhaustive dense-grid evaluation of the same objective (402 evaluations) that certifies how close the sample-efficient search comes to the global grid optimum.

*Multi-objective characterization.* Beyond matching individual targets, the EA–IPF trade-off is mapped by evaluating the deployed surrogate on a dense grid of 501 angles × 2 loading configurations and scalarizing the two objectives — collapsing them into a single number — in two ways: a weighted sum with EA-priority weight α ∈ [0, 1], and the Chebyshev form, which minimizes the larger of the two weighted deviations and can reach trade-off points that a weighted sum misses. Pareto-dominance filtering of the same landscape provides the reference non-dominated set.

*Solution-landscape diagnostics.* To make the ill-posedness of the inverse problem quantitative, J(θ, LC) is evaluated on a dense grid so that the number of local minima within a factor 1.5 of the global best (the solution multiplicity index, SMI), the local sensitivity dJ/dθ, and a pseudo-posterior P(θ \| target) ∝ exp(−J/T) — where the temperature T sets how sharply low objective values map to high probability, so angles matching the target better are treated as proportionally more probable — with its 95% credible interval, are reported alongside every inverse solution.

## Verification protocol for the inverse loop

A small target-matching error certifies only optimizer convergence, not correctness. Three verification modes therefore accompany every inverse-design result.

*Ground-truth recovery.* The five strategic targets are generated from the *measured* (EA, IPF) values of specific experimental configurations at displacement-fair quantiles, so each target carries a known generating design (θ~true~, LC~true~). The recovered design is scored by Δθ = \|θ\* − θ~true~\|, the loading-configuration match, and a bound-activity flag indicating whether the solution sits on the search boundary.

*Off-grid round trips.* Two verification targets are synthesized from the surrogate at angles that exist in no experimental row (V1: θ = 52.5°, LC1; V2: θ = 62.5°, LC2). Recovery of these targets tests the invertibility of the learned map away from the training grid and cannot be achieved by memorization.

*Infeasibility probe.* One target (V3) is constructed to be unattainable (EA 30% above the maximum of the predicted landscape combined with IPF 30% below its minimum). The framework must flag it as such: a solution is declared attainable only if the combined relative error is below 5%.

## Explainability layer

Two diagnostics make the framework's decisions inspectable. First, an exact functional-ANOVA variance decomposition of the learned forward map on a dense, balanced (θ, LC) grid attributes the variance of each design metric to the angle main effect, the loading-configuration main effect, and their interaction — a global explanation of what drives the design space. Second, every inverse solution is accompanied by a decomposition of the objective at the optimum into its EA-fit, IPF-fit, and classifier-penalty terms, exposing what the optimizer actually traded off. The independent mechanics analysis of Section 5.1 closes the loop by testing whether the surrogate-learned trends admit interpretable kinematic explanations.

## Data-only mechanics analysis

Independently of any neural network, the twelve curves are characterized by per-specimen crush signatures and tested for simple kinematic structure. The signatures are the initial stiffness, the plateau force (the median load between 10 mm and 40% of the stroke), the densification onset d~dens~ (the first displacement at which the load sustains 1.5× the plateau), the crushing-force efficiency, and an oscillation-intensity measure. The design trend of each signature is fitted against five candidate angle functions {θ, sin θ, cos θ, sin(θ/2), sin θ cos θ} by two-parameter least squares, and a master-curve collapse test normalizes each curve by its plateau force and densification onset to ask whether the specimens share one underlying shape. The findings are reported in Section 5.1.

## Implementation details

All networks are implemented in PyTorch (≥ 2.0) and trained on NVIDIA H100 GPUs with deterministic execution settings (base seed 2026). Average per-member training times in the production configuration were 165 s (DDNS), 1,417 s (Soft-PINN), and 14,402 s (Hard-PINN) — the computational price of the second-order automatic-differentiation graph, incurred once at training time and irrelevant at inverse-design query time. Statistical analysis uses SciPy; the plausibility classifier is built with scikit-learn; Optuna coordinates the distributed hyperparameter search; GP-BO uses scikit-optimize. Every saved model bundle is stamped with the git commit, an SHA-256 fingerprint of the input data, and a configuration hash, and the figure/table regeneration path verifies this provenance before producing outputs. The complete codebase, data, and trained ensemble bundles are openly available (see Data availability).

# Results and discussion

## Mechanics of the design space: crush-mode signatures and kinematic trends

Before any learning, the data-only mechanics analysis of Section 4.11 characterizes what the twelve curves themselves say about the design space (Fig. 4). The per-specimen signatures separate the two loading configurations cleanly: LC1 specimens never densify within their 80 mm window, whereas LC2 specimens at θ ∈ {55°, 60°, 70°} densify at 70.4, 78.4, and 61.1 mm respectively.

![](build/figs/Fig_mode_signatures.png){width=16cm}

Fig. 4. Crush-mode signatures across the measured design space: (a) crush-force efficiency versus densification onset per specimen; (b) plateau force and IPF design trends versus interior angle.

Two quantitative findings corroborate the surrogate analyses that follow. First, the LC2 plateau force follows a simple kinematic law: among candidate angle functions {θ, sin θ, cos θ, sin(θ/2), sin θ cos θ}, the projected-wall-obliquity form F~plateau~ = a sin θ cos θ + b fits with R² = 0.947 across all six angles — including the 70° rise — whereas no candidate exceeds R² = 0.33 for LC1, consistent with hinge-dominated (angle-insensitive) collapse of the flat sides. With two parameters fitted to six points the absolute R² is read cautiously; the inference rests on the ranking among the candidate forms, which is decisive, and on the independent master-curve corroboration below. Second, the θ = 70° ring under LC2 densifies earliest (61.1 mm), so a substantially larger fraction of its 130 mm stroke occurs in the compaction regime; this kinematic-stroke mechanism explains the sharp EA rise at LC2, 70° (Table 1) as a regime transition rather than measurement noise. A master-curve collapse test supports the same conclusion: mechanics-based normalization halves the LC2 pairwise curve dispersion (Fig. S1, Supplementary Section S3). These mechanics results establish independently of any neural network that the design space contains a genuine LC2 regime transition near the upper angle boundary — the single most demanding feature any surrogate must capture.

## Forward prediction at a held-out angle

Table 2 reports the forward-model comparison under the held-out-angle protocol, in which both 60° curves are reconstructed without any 60° supervision. The Hard-PINN attains the highest ensemble load R² (0.822), followed by the Soft-PINN (0.788) and the DDNS (0.718); the same ordering holds for the mean single-member scores (0.805, 0.772, 0.699) and for RMSE and MAE (parity plots in Fig. S2). Energy prediction is uniformly strong (R² ≥ 0.980) because the cumulative energy surface is intrinsically smooth. The DDNS lost two members to the Tukey convergence fence (M~eff~ = 18); both physics-informed ensembles retained all twenty.

Table 2. Forward performance under the held-out-angle protocol (θ\* = 60°; 1,684 held-out points). Ensemble R² is computed from the ensemble-mean prediction; the 95% CI is the bootstrap interval of single-member R².

| Model | M~eff~/M~total~ | Load R² (ens.) | Load R² (member mean) | Member 95% CI | Energy R² (ens.) | Load RMSE (kN) | Load MAE (kN) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| DDNS | 18/20 | 0.7175 | 0.6991 | [0.681, 0.717] | 0.9803 | 0.0819 | 0.0600 |
| Soft-PINN | 20/20 | 0.7875 | 0.7719 | [0.758, 0.786] | 0.9911 | 0.0711 | 0.0501 |
| Hard-PINN | 20/20 | **0.8217** | **0.8046** | [0.791, 0.818] | 0.9888 | **0.0651** | **0.0463** |

![](build/figs/Fig_unseen_load_curves.png){width=16cm}

Fig. 5. Load predictions at the held-out angle θ\* = 60° for LC1 (a) and LC2 (b): experiment, ensemble means of the three surrogates, and conformally calibrated ±2σ bands (Section 5.4).

The physics-constraint hierarchy DDNS < Soft-PINN < Hard-PINN is monotone in the strength of enforcement, and it is achieved by the *smallest* model of the three (9,089 parameters versus 11,170 and 34,690), ruling out capacity as the operative mechanism and pointing directly to the architectural constraint. The reconstructed curves (Fig. 5) show that all three surrogates capture the overall crush profile; the Hard-PINN best tracks the LC2 densification rise, though all models smooth the sharpest compaction spike — an honest reflection of predicting a regime transition from neighboring angles only. Per-configuration scores decompose the ensemble figures: for the Hard-PINN, load R² = 0.776 (LC1) and 0.823 (LC2), with energy R² = 0.955 and 0.991.

Statistical robustness of the ranking is assessed on the member-level score distributions (Table 3). Because ensemble members share training data, the bootstrap confidence interval of the difference in mean member R² is the inferential statistic; Welch's *t* and Cohen's *d* are reported descriptively. All three pairwise differences exclude zero with large effect sizes, and the ordering survives Bonferroni correction.

Table 3. Pairwise comparison of member-level load R² (held-out-angle protocol). The bootstrap CI of ΔR² is the inferential statistic; *t* and *p* are descriptive because members are not fully independent.

| Comparison | ΔR² | 95% bootstrap CI | Excludes 0 | Cohen's d | t | p (Bonferroni) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| DDNS vs Soft-PINN | −0.073 | [−0.093, −0.052] | Yes | −2.24 | −6.81 | < 0.001 |
| DDNS vs Hard-PINN | −0.105 | [−0.125, −0.085] | Yes | −3.24 | −9.86 | < 0.001 |
| Soft-PINN vs Hard-PINN | −0.033 | [−0.050, −0.016] | Yes | −1.13 | −3.56 | 0.003 |

Context matters for the absolute scores: most published crashworthiness surrogates report only random-split interpolation accuracy (Section 4.5). The held-out-angle protocol is deliberately harder, and 0.82 at a wholly unseen geometry — with the LC2 curve containing a regime transition — should be read against the model-free floors of Section 5.5, not interpolation ceilings.

The same protocol defeats conventional machine-learning surrogates outright (Table 4). Trained on identical features and the identical split, ridge regression attains a held-out load R² of 0.10, a 100-tree random forest 0.21, gradient-boosted trees −1.01 — worse than predicting the mean — and the subsampled Gaussian process 0.30. The mechanism is structural: tree ensembles partition the training feature space and cannot synthesize behavior at an unseen angle, and the Gaussian process falls back to its default (prior) prediction when queried far from any training data. Every neural surrogate clears every conventional baseline by a wide margin — the unconstrained DDNS beats the best baseline by 0.42 R² — showing that on these data only the neural surrogates achieve geometric generalization; the architectural physics constraint then contributes a further 0.10 on top. Energy prediction remains high for all baselines (R² ≥ 0.91): the discriminating signal lives in the load channel.

Table 4. Conventional ML baselines under the held-out-angle protocol (identical features and split; neural ensemble results repeated from Table 2 for reference).

| Model | Load R² | Energy R² | Load RMSE (kN) | Capacity |
|:---|:---:|:---:|:---:|:---:|
| Ridge regression | 0.104 | 0.971 | 0.146 | 6 params |
| Random forest | 0.215 | 0.975 | 0.137 | 100 trees |
| XGBoost | −1.015 | 0.906 | 0.219 | 100 trees |
| Gaussian process (Matérn-5/2) | 0.302 | 0.909 | 0.129 | N = 2,000 |
| DDNS (Table 2) | 0.718 | 0.980 | 0.082 | 11,170 params |
| Soft-PINN (Table 2) | 0.788 | 0.991 | 0.071 | 34,690 params |
| **Hard-PINN (Table 2)** | **0.822** | **0.989** | **0.065** | **9,089 params** |

## Physics-constraint verification

Table 5 quantifies thermodynamic consistency as the residual \|F̂ − ∂Ê/∂d\| evaluated across the validation inputs (Fig. 6). The Hard-PINN's residual is zero to machine precision — at every point, for every member — as guaranteed by its architecture; no penalty tuning is involved. The Soft-PINN reduces the mean residual roughly 3.7-fold relative to the DDNS (0.020 versus 0.075 kN, i.e. ≈ 5% versus ≈ 19% of the mean plateau load) but cannot eliminate it, because the penalty competes with data losses and is enforced only at sampled points. Neither baseline produced negative forces on the validation set, but only the Hard-PINN couples its outputs so that such violations are structurally linked to the energy field. One further point sharpens the comparison: because the energy channel of the dataset is itself the integral of the measured load (Section 3.2), the training data satisfy Eq. (1) identically — yet the unconstrained DDNS still violates the identity by ≈ 19% of the mean plateau load at validation inputs. Model-level consistency does not follow from data-level consistency; it must be imposed, and Table 5 quantifies what each imposition strategy buys.

Table 5. Thermodynamic-consistency residual \|F̂ − ∂Ê/∂d\| on validation inputs.

| Model | Constraint | Mean (kN) | Max (kN) | Std (kN) |
|:---|:---|:---:|:---:|:---:|
| DDNS | none | 0.0748 | 0.1673 | 0.0337 |
| Soft-PINN | penalty | 0.0204 | 0.0379 | 0.0046 |
| Hard-PINN | architectural | **0.0000** | **0.0000** | **0.0000** |

![](build/figs/Fig_physics_verification.png){width=16cm}

Fig. 6. Thermodynamic-consistency verification: (a) residual distributions; (b) computed ∂Ê/∂d versus predicted F̂; (c) mean violation magnitude (log scale) — the Hard-PINN bar sits at machine precision.

## Uncertainty calibration with honest held-out coverage

All raw bootstrap ensembles are severely over-confident at the held-out angle: 2σ bands cover only 41–52% of held-out observations instead of the nominal 95.4% (Table 6). This is expected — members trained on the same five angles agree with each other about how to extrapolate even when they are collectively wrong — and it is the reason calibrated bands, not raw ensemble bands, are used everywhere downstream. After curve-level split-conformal correction (factors fitted on calibration curves only), the coverage measured on *unseen* curves reaches 98.6% (DDNS), 74.3% (Soft-PINN), and 86.1% (Hard-PINN) at the 2σ level. These honest numbers are less flattering than the in-sample coverages a row-level split would report, and they show residual model-dependence: the DDNS band is now conservative (its large factor of 4.10 over-inflates), while the Soft-PINN remains under-covered. The Hard-PINN requires the smallest corrections of the three on both channels (load 2.51, energy 2.70, versus energy factors of 6.28 and 5.68 for DDNS and Soft-PINN) — its wider raw member spread is genuine predictive diversity rather than artificial consensus. These coverage numbers come with a caveat: they are computed from only a few held-out curves whose points are strongly correlated, so they carry non-trivial sampling uncertainty and are best read as a ranking of the three models rather than as coverage guarantees. The reliability diagram (Fig. 7) visualizes both the raw over-confidence and the corrected curves.

Table 6. Curve-level split-conformal calibration at the held-out angle (load channel). Factors are fitted on calibration curves; coverage is measured on held-out curves.

| Model | Raw 1σ cov. | Raw 2σ cov. | Factor c~2σ~ | Corrected 1σ cov. | Corrected 2σ cov. |
|:---|:---:|:---:|:---:|:---:|:---:|
| DDNS | 0.172 | 0.405 | 4.097 | 0.725 | 0.986 |
| Soft-PINN | 0.276 | 0.505 | 2.621 | 0.539 | 0.743 |
| Hard-PINN | 0.225 | 0.520 | 2.508 | 0.617 | 0.861 |

![](build/figs/Fig_reliability_diagram.png){width=16cm}

Fig. 7. Reliability diagram at the held-out angle: raw ensemble coverage (solid) versus curve-level conformally corrected coverage (dotted) per surrogate; 2σ conformal factors 4.10 (DDNS), 2.62 (Soft-PINN), and 2.51 (Hard-PINN).

Two practical consequences follow. First, bootstrap ensembles quantify member disagreement, not distribution shift [32], so conformal correction is not optional under geometric generalization. Second, a separate design-level audit shows that ensemble spreads of EA and IPF underestimate realized errors by ≈ 4.2–4.5× (worst at the LC2 regime transition), so design-level uncertainty statements in Sections 5.6–5.8 carry conformally inflated bands.

## Design-level accuracy against model-free floors

Inverse design consumes EA and IPF, so the deployed full-data surrogate is audited at the design level (Table 7). Across all twelve experimental configurations the mean absolute percentage error is 4.36% for EA@80 mm and 5.75% for IPF (LC1: 4.05/5.31%; LC2: 4.66/6.20%), with the largest single-cell errors at exactly the configurations the mechanics analysis flags as anomalous: the LC1, 60° specimen, whose IPF (0.672 kN) spikes far above both neighbors, and the LC2 densification-affected cells.

The held-out-angle floor comparison is the sharper test. A model-free predictor that linearly interpolates the neighboring experimental design values over θ errs at θ\* = 60° by 17.3% (EA) and 24.1% (IPF) for LC1, and 6.0% and 33.8% for LC2. The held-out-angle surrogate cuts these errors to 9.7/17.6% (LC1) and 3.2/7.3% (LC2) — a 1.4- to 4.6-fold skill margin over trivially interpolating the design table, on jagged single-specimen trends where part of the floor error is irreducible specimen idiosyncrasy.

Table 7. Design-level accuracy of the deployed surrogate over all twelve experimental configurations (EA at the common 80 mm displacement).

| θ (°) | LC | EA exp (J) | EA pred (J) | EA err (%) | IPF exp (kN) | IPF pred (kN) | IPF err (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 45 | LC1 | 28.34 | 28.42 | 0.26 | 0.468 | 0.468 | 0.13 |
| 50 | LC1 | 28.17 | 27.20 | 3.45 | 0.490 | 0.471 | 3.82 |
| 55 | LC1 | 23.41 | 24.96 | 6.60 | 0.440 | 0.475 | 7.97 |
| 60 | LC1 | 30.24 | 27.30 | 9.72 | 0.672 | 0.553 | 17.62 |
| 65 | LC1 | 26.62 | 27.54 | 3.46 | 0.580 | 0.587 | 1.29 |
| 70 | LC1 | 33.81 | 33.54 | 0.81 | 0.768 | 0.761 | 1.01 |
| 45 | LC2 | 25.17 | 24.94 | 0.92 | 0.390 | 0.386 | 1.02 |
| 50 | LC2 | 28.78 | 30.07 | 4.47 | 0.448 | 0.430 | 4.02 |
| 55 | LC2 | 34.14 | 33.22 | 2.70 | 0.772 | 0.857 | 11.12 |
| 60 | LC2 | 37.90 | 36.68 | 3.22 | 0.490 | 0.526 | 7.34 |
| 65 | LC2 | 37.12 | 41.55 | 11.95 | 0.539 | 0.586 | 8.64 |
| 70 | LC2 | 63.40 | 60.41 | 4.72 | 0.810 | 0.769 | 5.08 |
| **MAPE** | **all** | | | **4.36** | | | **5.75** |

## Design-space structure and its explanation

With design-level accuracy established, Fig. 8 maps what design space the surrogate has learned: EA@80 mm and IPF versus θ per loading configuration, with ±1σ ensemble bands and experimental anchors. The learned surfaces reproduce the mechanics-identified structure — the flat, hinge-dominated LC1 response, the monotone LC2 growth toward high angles, and the accelerating LC2 EA above ≈ 65°.

![](build/figs/Fig_design_space.png){width=16cm}

Fig. 8. Design-space predictions of the deployed Hard-PINN (EA@80 mm and IPF versus θ, per loading configuration, with ±1σ ensemble bands and experimental markers).

The exact variance decomposition (Section 4.10) quantifies what drives this space (Table 8). At the common displacement, EA variance is shared almost equally between angle (39.2%) and loading configuration (37.3%) with a substantial interaction (23.5%) — geometry and loading cannot be optimized independently. IPF is angle-dominated (59.2%, interaction 36.4%): peak force is primarily a geometric property, modulated by loading. Full-stroke energy, by contrast, is loading-dominated (74.4%) — chiefly the stroke-length disparity — which is precisely why the displacement-fair EA@80 mm metric is used for all design comparisons: it removes a nuisance factor that would otherwise dominate 74% of the objective variance.

Table 8. Exact variance decomposition of the learned design metrics over the balanced (θ, LC) grid (501 angles × 2 LCs).

| Metric | Angle main effect | LC main effect | Interaction |
|:---|:---:|:---:|:---:|
| EA@80 mm | 0.392 | 0.373 | 0.235 |
| IPF | 0.592 | 0.044 | 0.364 |
| EA (full stroke) | 0.142 | 0.744 | 0.114 |

## Verified inverse design

The inverse loop is now subjected to the three-mode verification protocol of Section 4.9.

*Ground-truth recovery.* Table 9 reports the five strategic targets, each generated from a measured configuration so that ground truth is known. The GP-BO loop recovers the correct loading configuration in **five of five** cases and matches the target metrics to 0.3–3.5% (EA) and 0.1–1.0% (IPF) (Fig. 9). Angle recovery is exact for the two boundary-anchored targets (T2 → 45.0°, T4 → 70.0°; the truth lies on the search bound, so the reported bound-activity flag is expected), 1.5° for T1, and 4.6° for T5. Multi-seed robustness runs (5 outer seeds × 5 restarts per target) show tight dispersion for four targets (θ spread ≤ 0.35°; Table S2).

Table 9. Ground-truth recovery for the five strategic inverse-design targets.

| Target | EA~t~ (J) | IPF~t~ (kN) | True (θ, LC) | Recovered (θ, LC) | Δθ (°) | LC match | EA err (%) | IPF err (%) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| T1 | 26.62 | 0.580 | 65.0, LC1 | 63.5, LC1 | 1.52 | Yes | 3.5 | 0.9 |
| T2 | 28.34 | 0.468 | 45.0, LC1 | 45.0, LC1 | 0.00 | Yes | 0.3 | 0.1 |
| T3 | 30.24 | 0.672 | 60.0, LC1 | 68.1, LC1 | 8.12 | Yes | 0.5 | 0.6 |
| T4 | 33.81 | 0.768 | 70.0, LC1 | 70.0, LC1 | 0.00 | Yes | 0.8 | 1.0 |
| T5 | 37.12 | 0.539 | 65.0, LC2 | 60.4, LC2 | 4.63 | Yes | 0.3 | 0.7 |

The one large angular deviation — T3, 68.1° recovered for a target generated at 60° — is not an optimizer failure but measured ill-posedness with a specific mechanistic origin. The LC1, 60° specimen is the design table's outlier: its IPF (0.672 kN) sits far above both neighbors (0.44 and 0.58 kN), and the surrogate — fitting a smooth angle trend — reproduces that cell worst of all twelve (Table 7). To match the *measured* target values, the optimizer therefore correctly found the angle at which the smooth learned map delivers them: 68.1°, with 0.5/0.6% residual. The diagnostics tell this story quantitatively: the T3 posterior 95% credible interval spans [45.9°, 69.7°] (landscapes in Fig. S3) — nearly the whole design range — flagging the target as weakly identifying before any claim is staked on the point estimate.

T4 shows the complementary degeneracy across loading configurations. The dense grid finds a marginally lower objective on the *other* branch (LC2 at 57.0°, J = 0.00244) than on the recovered branch (LC1 at 70.0°, J = 0.00287): two nearly indistinguishable solutions exist in different loading regimes. The multi-start dispersion for T4 (θ spread 5.9°) reflects exactly this bimodality, and the recovered branch is the true one. Reporting these degeneracies (solution multiplicity, credible intervals, restart dispersion, grid anchors) alongside every solution is, we argue, as important a deliverable as the solutions themselves.

![](build/figs/Fig_inverse_parity_uncertainty.png){width=16cm}

Fig. 9. Inverse-design parity with ensemble uncertainty: recovered versus target EA (a) and IPF (b) for T1–T5 with conformally inflated error bars.

*Off-grid round trips and infeasibility (Table 10).* The two off-grid verification targets are recovered with sub-degree accuracy — V1 (truth 52.5°, LC1): Δθ = 0.36°; V2 (truth 62.5°, LC2): Δθ = 0.01° — with combined relative errors of 0.7% and 0.02%; since neither angle exists in any training row, these round trips certify invertibility away from the data grid. The infeasibility probe V3 (EA 30% above the attainable maximum, IPF 30% below the attainable minimum) terminates with a combined relative error of 81% — two orders of magnitude above the 5% attainability threshold — and is correctly declared *not attainable* — the refusal behavior promised in Section 1, now demonstrated with an 81% residual against the 5% threshold. All three verification verdicts are correct.

Table 10. Verification suite: off-grid round trips and infeasibility probe. Rel. error is the combined relative target-matching error at the returned solution; all three verdicts are correct.

| ID | Family | Truth | Recovered | Δθ (°) | Rel. error | Verdict |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| V1 | off-grid round trip | 52.5°, LC1 | 52.9°, LC1 | 0.36 | 0.007 | attainable |
| V2 | off-grid round trip | 62.5°, LC2 | 62.5°, LC2 | 0.01 | 0.0002 | attainable |
| V3 | infeasibility probe | — | — | — | 0.811 | not attainable |

*Efficiency and transparency.* Every GP-BO solution is anchored against an exhaustive dense-grid evaluation of the identical objective and against 25 independent optimizer replicates (Table S2, Supplementary Section S5): the search needed 11–20 sequential evaluations per target and lands on the grid optimum to ≤ 1.0° for four of five targets — the T4 discrepancy being the cross-LC degeneracy above (ΔJ ≈ 4 × 10^−4^) — with replicate dispersion ≤ 0.35° except for the degenerate T4.

A physical-plausibility audit of all 1,002 design-sweep evaluations recorded zero violations of any kind, and the objective decomposition at each optimum confirms the classifier steered rather than distorted the solutions: no LC decision flips, angle shifts ≤ 0.4°, and target-matching accuracy flat for λ ≤ 0.02 (Supplementary Section S5). Fig. S4 closes the loop by overlaying the predicted load–displacement curve at each recovered design on the nearest experimental curve of the same loading configuration.

## Multi-objective trade-off

Executing the multi-objective protocol of Section 4.8, Fig. 10 maps the EA–IPF trade-off. Three findings carry design significance. First, the displacement-fair Pareto front is dominated by LC2 across the entire weight range — vertex loading offers superior EA-versus-IPF exchange independent of its longer stroke. Second, the front exhibits sharp regime structure rather than a continuum: IPF-priority weights (α ≤ 0.3) select the 45° boundary, a balanced regime selects ≈ 52°, and EA-priority weights (α ≥ 0.6) jump to the 70° boundary — small changes in designer priorities near the transitions trigger discrete changes in the recommended configuration. Third, spanning the front trades a 2.4× EA gain (24.9 → 60.4 J) against a 2.0× IPF increase (0.39 → 0.77 kN); the knee at (31.7 J, 0.42 kN; θ ≈ 51.9°, LC2) offers a 27% EA gain for a 9% IPF concession and is the recommended balanced design. These conclusions echo the variance decomposition (Table 8): with EA and IPF respectively angle- and interaction-dominated at fixed stroke, the loading configuration acts as a regime selector while the angle tunes the trade-off within a regime.

![](build/figs/Fig_pareto_tradeoff.png){width=16cm}

Fig. 10. Multi-objective EA–IPF trade-off: (a) optimal angle versus trade-off weight α with LC-conditional fronts; (b) delivered EA and IPF versus α; (c) Pareto front with the recommended knee design (31.7 J, 0.42 kN at θ ≈ 51.9°, LC2); (d) scalarized objective versus α.

## Limitations

Six limitations bound the claims of this study.

1. *Single specimen per configuration.* The dataset contains one representative curve per configuration; specimen-to-specimen scatter is not identifiable, and all reported uncertainty is epistemic. The jagged design trends in Table 1 (e.g., the LC1, 60° IPF spike) may partly reflect specimen idiosyncrasy, which caps achievable design-level accuracy at unseen angles — the model-free floors of Section 5.5 quantify this ceiling.
2. *One held-out angle.* The protocol tests generalization at one interior angle; boundary angles (45°, 70°) are harder — the mechanics analysis explains why for the LC2 regime transition — and leave-one-angle-out scores there are substantially weaker: a data-scarcity wall that added specimens or high-fidelity simulation would be needed to breach.
3. *Tuning-fold overlap.* The hyperparameter-search fold coincides with the reported comparison fold (Section 4.7); the released code defaults to an inner tuning fold.
4. *Derived energy channel.* The energy channel is computed from the load channel, so physics enforcement here is an inductive bias rather than independent validation; the benefit is nonetheless measurable (Tables 2 and 5).
5. *Weak plausibility classifier.* The classifier is weak at n = 12 (LOO AUC 0.53) and is deployed accordingly — as a small, ablation-verified regularizer, not an oracle.
6. *Low design-space dimensionality.* The demonstrated design space has one continuous geometric variable and one categorical loading variable; nothing in the framework is dimension-specific — the surrogate, conformal, GP-BO, and verification layers are all dimension-agnostic — but its behavior on richer parameterizations (wall thickness, cell size, lay-up) remains to be demonstrated.

# Conclusions

This study developed and verified a physics-informed forward-prediction and inverse-design framework for hexagonal glass/epoxy composite ring energy absorbers, organized around a hard-constrained PINN in which the crushing force is the exact derivative of a learned energy field. The principal conclusions are:

1. *Hard physics enforcement wins where it matters.* Under a held-out-angle protocol requiring reconstruction of two complete unseen crushing curves, the physics-constraint hierarchy is monotone: Hard-PINN (ensemble load R² = 0.822) > Soft-PINN (0.788) > DDNS (0.718), with all pairwise bootstrap confidence intervals excluding zero and the advantage achieved by the smallest model of the three. The Hard-PINN's thermodynamic residual is zero to machine precision everywhere, by construction. Every neural surrogate beats the best conventional baseline (Gaussian process, R² = 0.30) by more than 0.4 R² (Table 4), and although the training data satisfy the work–energy identity exactly, the unconstrained baseline still violates it by ≈ 19% of mean plateau load — consistency must be imposed, not inherited.

2. *Honest uncertainty is attainable but sobering.* Raw bootstrap ensembles cover only 41–52% of held-out observations at the nominal 95% level; curve-level split-conformal correction with genuinely held-out coverage restores 86% (Hard-PINN) — and the Hard-PINN needs the smallest corrections of the three on both output channels. Design-level ensemble spreads require ≈ 4.4× inflation. Studies reporting raw ensemble bands, or conformal coverage evaluated on rows that informed the calibration, will systematically overstate reliability.

3. *Inverse design can and should be verified, not just converged.* The GP-BO loop recovered the generating loading configuration for five of five ground-truth targets and matched targets to ≤ 3.5%; recovered off-grid verification targets to 0.36° and 0.01°; and correctly refused a deliberately infeasible target. The diagnostics exposed — rather than hid — two genuine degeneracies (a weakly identifying target with a near-full-range credible interval, and a cross-LC near-degenerate pair), which is the behavior required of a trustworthy design tool on scarce data.

4. *The design space is interpretable.* Variance decomposition attributes fixed-stroke EA almost equally to angle and loading with a strong interaction, identifies IPF as angle-dominated, and shows full-stroke EA to be 74% loading-dominated — justifying the displacement-fair metric. An independent mechanics analysis corroborates the learned trends: the LC2 plateau force follows a sin θ cos θ obliquity law (R² = 0.95), and the early densification of the 70° ring (61 mm) explains the LC2 energy surge as a kinematic regime transition. The Pareto front is LC2-dominated with sharp regime transitions and a knee at (31.7 J, 0.42 kN).

Future work will proceed along four axes: replicate specimens (or validated finite-element data) to separate aleatoric from epistemic uncertainty and strengthen boundary-angle generalization; monotonicity-by-construction variants and low-order separable angle bases (both released with the code) once budget-matched tuning is complete; distance-aware or Bayesian uncertainty to complement conformal correction under distribution shift; and transfer of the verification protocol — none of whose ingredients is specific to hexagonal rings — to other material systems and geometry classes.

**CRediT authorship contribution statement**

**Nahid Sarker:** Conceptualization, Software, Methodology, Data curation, Validation, Visualization, Writing – original draft. **Elsadig Mahdi:** Investigation, Supervision, Resources, Validation. **Monzure-Khoda Kazi:** Conceptualization, Methodology, Software, Supervision, Validation, Writing – review & editing, Funding acquisition.

**Data availability**

The complete dataset, source code, trained ensemble bundles, and all scripts required to reproduce every table and figure in this paper are openly available at https://github.com/kprodigi/IPINN.

**Declaration of competing interest**

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

**Acknowledgments**

This work was supported by the South Dakota Board of Regents through Faculty Startup Funding [grant number 4FSU69].

**Declaration of generative AI and AI-assisted technologies in the writing process**

During the preparation of this work, the authors used Grammarly solely for grammar and spelling checks. After using the tool, the authors thoroughly reviewed and edited the content as needed and take full responsibility for the final content of the publication.

**Supplementary material**

Supplementary material associated with this article comprises Sections S1–S6, Tables S1–S2, and Figures S1–S4: the Hard-PINN training-stabilization details, the hyperparameter-search protocol and selected configurations, the master-curve collapse test, extended forward-prediction diagnostics, the optimizer-transparency and physical-plausibility audits, and a catalog of further archived diagnostics. All supplementary artifacts regenerate deterministically from the released code, data, and model bundles (see Data availability).

**References**

[1] Alghamdi AAA. Collapsible impact energy absorbers: an overview. Thin-Walled Structures. 2001;39:189–213.

[2] Lu G, Yu T. Energy absorption of structures and materials. Elsevier; 2003.

[3] Lau STW, Said MR, Yaakob MY. On the effect of geometrical designs and failure modes in composite axial crushing: A literature review. Composite Structures. 2012;94:803–12.

[4] Palanivelu S, Paepegem WV, Degrieck J, Vantomme J, Kakogiannis D, Ackeren JV, et al. Crushing and energy absorption performance of different geometrical shapes of small-scale glass/polyester composite tubes under quasi-static loading conditions. Composite Structures. 2011;93:992–1007.

[5] Mahdi E, Hamouda AMS. Energy absorption capability of composite hexagonal ring systems. Materials & Design. 2012;34:201–10.

[6] Ghasemnejad H, Hadavinia H, Aboutorabi A. Effect of delamination failure in crashworthiness analysis of hybrid composite box structures. Materials & Design. 2010;31:1105–16.

[7] Fang J, Sun G, Qiu N, Kim NH, Li Q. On design optimization for structural crashworthiness and its state of the art. Structural and Multidisciplinary Optimization. 2017;55:1091–119.

[8] Sarker N, Kazi M-K. Physics-Informed Inverse Design of Hexagonal Composite and Aluminium Structures for Crashworthiness Optimization. 2025 AIChE Annual Meeting: AIChE; 2025.

[9] Kazi M-K, Eljack F, Mahdi E. Predictive ANN models for varying filler content for cotton fiber/PVC composites based on experimental load displacement curves. Composite Structures. 2020;254:112885.

[10] Kazi M-K, Eljack F, Mahdi E. Optimal filler content for cotton fiber/PP composite based on mechanical properties using artificial neural network. Composite Structures. 2020;251:112654.

[11] Hou L, Zhang H, Peng Y, Wang S, Yao S, Li Z, et al. An integrated multi-objective optimization method with application to train crashworthiness design. Structural and Multidisciplinary Optimization. 2021;63:1513–32.

[12] Boursier Niutta C, Wehrle EJ, Duddeck F, Belingardi G. Surrogate modeling in the design optimization of structures with discontinuous responses with respect to the design variables — a new approach for crashworthiness design. In: Schumacher A, Vietor T, Fiebig S, Bletzinger K-U, Maute K, editors. Advances in Structural and Multidisciplinary Optimization. Cham: Springer; 2018. p. 242–58.

[13] Iniguez-Rabago A, Li Y, Overvelde JTB. Exploring multistability in prismatic metamaterials through local actuation. Nature Communications. 2019;10:5577.

[14] Panter JR, Chen J, Zhang T, Kusumaatmaja H. Harnessing energy landscape exploration to control the buckling of cylindrical shells. Communications Physics. 2019;2:151.

[15] Lookman T, Balachandran PV, Xue D, Yuan R. Active learning in materials science with emphasis on adaptive sampling using uncertainties for targeted design. npj Computational Materials. 2019;5:21.

[16] Storn R, Price K. Differential Evolution — a simple and efficient heuristic for global optimization over continuous spaces. Journal of Global Optimization. 1997;11:341–59.

[17] Kennedy J, Eberhart R. Particle swarm optimization. Proceedings of ICNN'95 — International Conference on Neural Networks; 1995. p. 1942–8.

[18] Jin Y. Surrogate-assisted evolutionary computation: Recent advances and future challenges. Swarm and Evolutionary Computation. 2011;1:61–70.

[19] Raissi M, Perdikaris P, Karniadakis GE. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics. 2019;378:686–707.

[20] Haghighat E, Raissi M, Moure A, Gomez H, Juanes R. A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics. Computer Methods in Applied Mechanics and Engineering. 2021;379:113741.

[21] Hu H, Qi L, Chao X. Physics-informed Neural Networks (PINN) for computational solid mechanics: Numerical frameworks and applications. Thin-Walled Structures. 2024;205:112495.

[22] Iftakher A, Golder R, Roy BN, Faruque Hasan MM. Physics-informed neural networks with hard nonlinear equality and inequality constraints. Computers & Chemical Engineering. 2026;204:109418.

[23] Golder R, Roy BN, Hasan M. DAE-HardNet: A physics constrained neural network enforcing differential-algebraic hard constraints. arXiv preprint arXiv:2512.05881; 2025.

[24] Greydanus S, Dzamba M, Yosinski J. Hamiltonian neural networks. Advances in Neural Information Processing Systems 32 (NeurIPS); 2019. p. 15353–63.

[25] Chen H, Constante Flores GE, Li C. Physics-informed neural networks with hard linear equality constraints. Computers & Chemical Engineering. 2024;189:108764.

[26] Kazi M-K, Eljack F, Mahdi E. Data-driven modeling to predict the load vs. displacement curves of targeted composite materials for industry 4.0 and smart manufacturing. Composite Structures. 2021;258:113207.

[27] Kazi M-K, Eljack F, Mahdi E. Design of composite rectangular tubes for optimum crashworthiness performance via experimental and ANN techniques. Composite Structures. 2022;279:114858.

[28] Kazi M-K, Mahdi E. Crashworthiness optimization of composite hexagonal ring system using random forest classification and artificial neural network. Composites Part C: Open Access. 2024;13:100440.

[29] Lei J, G'Sell M, Rinaldo A, Tibshirani RJ, Wasserman L. Distribution-free predictive inference for regression. Journal of the American Statistical Association. 2018;113(523):1094–1111.

[30] Rasmussen CE, Williams CKI. Gaussian Processes for Machine Learning. The MIT Press; 2006.

[31] Cardoso Coelho RP, Carvalho Alves AF, Nogueira Pires TM, Andrade Pires FM. A composite Bayesian optimisation framework for material and structural design. Computer Methods in Applied Mechanics and Engineering. 2025;434:117516.

[32] Lakshminarayanan B, Pritzel A, Blundell C. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in Neural Information Processing Systems 30 (NeurIPS); 2017. p. 6402–13.
