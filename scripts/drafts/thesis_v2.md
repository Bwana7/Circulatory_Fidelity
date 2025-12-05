# Circulatory Fidelity: Stability Constraints on Hierarchical Variational Inference

**A Computational Framework Linking Structured Approximations to Dopaminergic Precision Regulation**

---

**Author:** Aaron Lowry  
**Date:** November 2025  
**Status:** Working Draft

---

## Abstract

This thesis introduces *Circulatory Fidelity* (CF), a measure quantifying the statistical dependency preserved between levels of a hierarchical generative model during approximate inference. Using the Hierarchical Gaussian Filter (HGF) as a model system, we analyze the dynamical stability of variational updates under different approximation schemes.

We demonstrate that mean-field variational inference—which assumes independence between hierarchical levels—exhibits period-doubling bifurcations leading to deterministic chaos when environmental volatility exceeds a critical threshold. Structured approximations that preserve cross-level dependencies remain stable across a broader parameter range. This stability difference is characterized via Lyapunov exponent analysis and bifurcation diagrams.

We propose that biological inference systems may face analogous trade-offs between computational cost and dynamical stability. As a candidate neural implementation, we suggest that tonic dopamine concentration could modulate the precision (gain) of hierarchical message passing, though this mapping remains speculative and requires empirical validation.

The framework generates several testable predictions regarding the relationship between neuromodulatory state and belief dynamics, which we outline as proposals for future experimental work.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Theoretical Framework](#2-theoretical-framework)
3. [Dynamical Systems Analysis](#3-dynamical-systems-analysis)
4. [Resource-Rational Extensions](#4-resource-rational-extensions)
5. [A Candidate Neural Implementation](#5-a-candidate-neural-implementation)
6. [Computational Methods](#6-computational-methods)
7. [Proposed Experimental Tests](#7-proposed-experimental-tests)
8. [Discussion](#8-discussion)
9. [Conclusion](#9-conclusion)
10. [References](#10-references)

---

## Notation

| Symbol | Definition | Domain |
|--------|------------|--------|
| z | Log-volatility (Level 2 state) | ℝ |
| x | Hidden state (Level 1) | ℝ |
| y | Observation | ℝ |
| κ | Coupling strength | > 0 |
| ω | Baseline log-volatility | ℝ |
| ϑ | Volatility of volatility | > 0 |
| π_u | Observation precision | > 0 |
| γ | Precision weight | > 0 |
| μ_z, μ_x | Posterior means | ℝ |
| σ²_z, σ²_x | Posterior variances | > 0 |
| I(·;·) | Mutual information | ≥ 0 |
| H(·) | Entropy | ≥ 0 |
| λ_max | Maximal Lyapunov exponent | ℝ |

---

## 1. Introduction

### 1.1 The Problem

Biological agents infer hidden causes of sensory observations across multiple timescales. A canonical example: estimating both the current state of an environment and how quickly that environment is changing. These two quantities—state and volatility—interact: beliefs about volatility determine how much weight to place on new observations when updating state estimates.

The Hierarchical Gaussian Filter (HGF) formalizes this structure (Mathys et al., 2011, 2014). In the HGF, a higher level encodes log-volatility, which parameterizes the expected rate of change at the lower level. Exact Bayesian inference in such models is generally intractable, motivating variational approximations.

### 1.2 Variational Approximations

Variational inference replaces the intractable true posterior p(x,z|y) with a tractable approximation q(x,z), chosen to minimize the Kullback-Leibler divergence from the true posterior. The most common simplification is the *mean-field approximation*:

$$q(x,z) = q(x)q(z)$$

This factorization assumes independence between levels, dramatically reducing computational complexity. However, this independence assumption discards information about how states at different levels covary.

An alternative is the *structured approximation*:

$$q(x,z) = q(z)q(x|z)$$

which preserves the conditional dependency of lower-level states on higher-level states. This comes at increased computational cost but retains more information about the joint posterior structure.

### 1.3 This Thesis

We investigate the dynamical consequences of these approximation choices. Our central empirical finding (from simulation) is that mean-field variational updates can become dynamically unstable—exhibiting bifurcations and chaos—under conditions where structured approximations remain stable.

We formalize this stability difference through a quantity we term *Circulatory Fidelity* (CF), measuring the mutual information preserved between hierarchical levels. We then explore the implications of this finding for understanding biological inference systems, proposing (speculatively) that neuromodulatory mechanisms may have evolved partly to maintain stable inference dynamics.

**Scope and limitations:** This is primarily a computational and theoretical analysis. The neural implementation we propose is speculative and intended to generate testable hypotheses rather than to make strong claims about biological mechanism. Throughout, we distinguish between what we can demonstrate formally, what we observe in simulation, and what we conjecture.

---

## 2. Theoretical Framework

### 2.1 The Hierarchical Gaussian Filter

We work with a two-level HGF defined by the following generative model:

**Level 2 (Volatility):**
$$z_t \mid z_{t-1} \sim \mathcal{N}(z_{t-1}, \vartheta^{-1})$$

**Level 1 (Hidden state):**
$$x_t \mid x_{t-1}, z_t \sim \mathcal{N}(x_{t-1}, \exp(\kappa z_t + \omega))$$

**Observations:**
$$y_t \mid x_t \sim \mathcal{N}(x_t, \pi_u^{-1})$$

The parameter ϑ controls how quickly volatility itself changes (sometimes called the "hazard rate" or "meta-volatility"). The coupling κ determines how strongly volatility modulates state transitions. The baseline ω sets the typical level of volatility when z = 0.

### 2.2 Variational Updates

Under the mean-field approximation q(x,z) = q(x)q(z), the variational updates take the form of coupled fixed-point equations. At each timestep, given observation y_t, the posterior parameters are updated according to:

**Level 1 update:**
$$\mu_x^{(t)} = \mu_x^{(t-1)} + \frac{\pi_u}{\pi_u + \hat{\pi}_x} (y_t - \mu_x^{(t-1)})$$

where $\hat{\pi}_x = \exp(-\kappa \mu_z^{(t-1)} - \omega)$ is the expected precision at level 1.

**Level 2 update:**
$$\mu_z^{(t)} = \mu_z^{(t-1)} + \frac{\kappa}{2} \frac{\hat{\pi}_x}{\vartheta + \frac{\kappa^2}{2}\hat{\pi}_x} \left( (y_t - \mu_x^{(t-1)})^2 \hat{\pi}_x - 1 \right)$$

These update equations define a discrete dynamical system. Our analysis focuses on the stability properties of this system.

### 2.2.1 Structured Variational Update

Under the structured approximation $q(x,z) = q(z)q(x|z)$, the updates differ from mean-field by incorporating cross-level coupling. The key modification is to the Level 2 update, which gains an additional damping term:

$$\mu_z^{(t)} = \mu_z^{(t-1)} + K_z \cdot \nu - \underbrace{\gamma_{zx} \cdot \text{Cov}_q(z, x) \cdot \delta_x}_{\text{coupling term}}$$

where $K_z$ is the standard gain, $\nu$ is the volatility prediction error, and $\gamma_{zx}$ is a coupling coefficient.

**Status of the coupling coefficient:** The specific form of $\gamma_{zx}$ used in our simulations is a **modeling approximation**, not a first-principles derivation. We use:

$$\gamma_{zx} = \frac{\kappa \pi_x}{4}$$

**Justification for this choice:**

1. **Dimensional consistency:** The coefficient has units of precision, matching the update equation structure.

2. **Correct limiting behavior:** As $\kappa \to 0$ (no coupling between levels), $\gamma_{zx} \to 0$, recovering the mean-field update. As precision $\pi_x$ increases (more confident predictions), the damping effect strengthens.

3. **Empirical calibration:** The factor of 1/4 was chosen to produce stable dynamics across the parameter range of interest, consistent with the heuristic that structured approximations should prevent the instabilities observed under mean-field.

**What a full derivation would require:** A principled derivation would compute the natural gradient on the structured variational family, yielding the FIM-weighted update. The off-diagonal FIM terms $G_{zx}$ would then determine the coupling coefficient exactly. We leave this complete derivation for future work, noting that our approximation captures the qualitative stabilizing effect of cross-level coupling even if the precise coefficient differs from the exact natural gradient.

*See Appendix A.3 for further discussion of this approximation.*

### 2.3 Circulatory Fidelity

**Definition 1 (Circulatory Fidelity).** For a joint approximate posterior q(x,z) with marginal entropies H_q(x) and H_q(z), Circulatory Fidelity is:

$$\text{CF} = \frac{I_q(z; x)}{\min(H_q(z), H_q(x))}$$

where I_q(z;x) = H_q(z) + H_q(x) - H_q(z,x) is the mutual information between z and x under q.

**Properties:**
- CF ∈ [0, 1]
- CF = 0 if and only if z and x are independent under q
- CF = 1 if and only if one variable is a deterministic function of the other

**Proposition 3 (CF Under Mean-Field).** Under the mean-field approximation, CF = 0.

*Proof:* Under $q(x,z) = q(x)q(z)$, the joint entropy decomposes: $H(x,z) = H(x) + H(z)$. Therefore $I(x;z) = H(x) + H(z) - H(x,z) = 0$. Since the numerator is zero, CF = 0. ∎

**Corollary.** Any structured approximation with non-trivial conditional dependency has CF > 0.

**Interpretation:** CF measures what fraction of the uncertainty in one level can be "explained" by the other level. The mean-field approximation *provably* discards all such shared information.

**Remark 2 (Choice of Normalization).** We normalize by $\min(H(z), H(x))$ rather than the joint entropy $H(z,x)$. This choice has two advantages:

1. **Interpretability:** With this normalization, CF = 1 when one variable is a deterministic function of the other (i.e., all uncertainty in the less-uncertain variable is "explained" by the other). With joint entropy normalization, CF < 1 even for deterministic relationships when the marginal entropies differ.

2. **Behavior in high-volatility regimes:** The joint entropy $H(z,x) = H(z) + H(x) - I(z;x)$ grows with marginal entropies. Using it as denominator would cause CF to shrink in high-volatility regimes precisely when the dependency structure matters most. The $\min(H(z), H(x))$ normalization avoids this artifact.

Formally, CF with our normalization is the **coefficient of constraint** from information theory, analogous to the correlation coefficient in linear regression.

### 2.4 Information Geometry

The space of Gaussian distributions forms a Riemannian manifold with the Fisher Information Matrix (FIM) as its metric. For the HGF posterior, the FIM structure differs between approximation schemes.

**Proposition 1 (FIM Block-Diagonality).** Under mean-field q(x,z) = q(x)q(z), the Fisher Information Matrix is block-diagonal:

$$G_{\text{MF}} = \begin{pmatrix} G_{zz} & 0 \\ 0 & G_{xx} \end{pmatrix}$$

Under structured q(x,z) = q(z)q(x|z), the FIM generically contains non-zero off-diagonal terms:

$$G_{\text{struct}} = \begin{pmatrix} G_{zz} & G_{zx} \\ G_{xz} & G_{xx} \end{pmatrix}$$

*Proof:* Under mean-field, $\ln q(x,z) = \ln q(x) + \ln q(z)$. Derivatives with respect to z-parameters depend only on z; derivatives with respect to x-parameters depend only on x. Cross-terms in the FIM factor as products of expectations, each of which vanishes because the score function has zero mean for exponential families. Under structured approximation, the conditional $q(x|z)$ prevents this factorization. See Appendix A.1 for complete proof. ∎

**Interpretation:** The off-diagonal FIM terms G_zx encode information about how to jointly adjust beliefs at both levels in response to evidence. Their *provable* absence under mean-field means this coordinated adjustment mechanism is structurally precluded by the approximation.

---

## 3. Dynamical Systems Analysis

### 3.1 The Update Map

The mean-field variational updates define a discrete map:

$$\mathbf{s}^{(t+1)} = F(\mathbf{s}^{(t)}, y_t)$$

where s = (μ_z, μ_x, σ²_z, σ²_x) is the state vector. For fixed input statistics (e.g., y_t drawn i.i.d. from a fixed distribution), this becomes an iterated function system whose stability we can analyze.

### 3.2 Local Stability Analysis

**Proposition 2 (Deterministic Skeleton Stability).** Consider the mean-field HGF z-dynamics under the deterministic skeleton approximation (replacing stochastic observations with their expected values). The linearized dynamics around equilibrium have Jacobian eigenvalue:

$$\lambda = \frac{\vartheta}{\vartheta + \alpha}, \quad \text{where } \alpha = \frac{\kappa^2 \pi_x^*}{2}$$

Since $\vartheta > 0$ and $\alpha > 0$, we have $0 < \lambda < 1$, implying **local stability** of the deterministic skeleton for all parameter values.

*Proof:* See Appendix A.2 for complete derivation.

**Key insight:** This stability result for the deterministic skeleton appears to contradict the observed instabilities. The resolution is that the instability arises from **stochastic fluctuations**, not from the deterministic dynamics.

**Stochastic instability mechanism:** In the full stochastic system, large prediction errors can drive large updates. Even when the mean dynamics are stable, the *variance* of the state can grow if the gain $K_z$ is large. Specifically, the update:

$$\mu_z^{(t+1)} = \mu_z^{(t)} + K_z \cdot ((\delta^{(t)})^2 \pi_x - 1)$$

has variance that depends on the fourth moment of $\delta$, which can exceed the mean-squared response, leading to variance growth.

**Epistemic status:** The deterministic stability is *derived*. The stochastic instability mechanism is *hypothesized* based on numerical observations but not yet analytically proven. A complete analysis would require studying the evolution of the full probability distribution of $\mu_z$, which is beyond our present scope.

### 3.3 Bifurcation Structure

**Empirical Observation 1.** In numerical simulations with parameters κ = 1, ω = -2, π_u = 10, we observe the following:

- For ϑ < ϑ_c, the mean-field dynamics converge to a stable fixed point
- At ϑ ≈ ϑ_c, a period-doubling bifurcation occurs
- For ϑ_c < ϑ < ϑ_chaos, successive period-doublings are observed
- For ϑ > ϑ_chaos, the dynamics appear chaotic (positive Lyapunov exponent)

The critical values depend on parameters and noise realization. In our simulations:

$$\vartheta_c \in [0.04, 0.06], \quad \vartheta_{\text{chaos}} \in [0.10, 0.15]$$

We report ranges rather than point estimates to reflect sensitivity to initial conditions and noise.

---

**Remark 1 (Resolution of the Stability Paradox).** The deterministic skeleton analysis (Proposition 2) predicts stability for all $\vartheta > 0$, yet simulations clearly show instabilities. This apparent contradiction is resolved by recognizing that **deterministic and stochastic stability are distinct phenomena:**

| Analysis | Object of Study | Prediction |
|----------|-----------------|------------|
| Deterministic skeleton | Dynamics of *expected* state | Stable for all ϑ > 0 |
| Full stochastic system | Dynamics of *state distribution* | Unstable for small ϑ |

The instability arises not from the fixed point being unstable, but from **variance growth**: large prediction errors (which occur with non-zero probability) drive large updates, and when the gain $K_z$ is high, the variance of the state grows even though its mean remains stable.

This is analogous to noise-induced phenomena in other dynamical systems (e.g., stochastic resonance, noise-induced transitions). The key insight is that analyzing only the deterministic dynamics can miss instabilities that arise from the interaction of nonlinearity and noise.

**Implications:**
1. Linear stability analysis is insufficient for stochastic systems
2. The observed bifurcations are properties of the *stochastic* HGF, not artifacts
3. Proving stochastic instability requires analyzing the evolution of the full probability distribution

We identify the analytical characterization of this stochastic instability as an open problem (Section 8.4).

---

### 3.4 Lyapunov Exponent Computation

**Definition 2 (Maximal Lyapunov Exponent).** The maximal Lyapunov exponent characterizes the average rate of separation of nearby trajectories:

$$\lambda_{\max} = \lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^{T} \ln \frac{\|\delta \mathbf{s}^{(t)}\|}{\|\delta \mathbf{s}^{(t-1)}\|}$$

where δs^(t) is the separation vector, periodically renormalized to prevent overflow.

**Empirical Observation 2.** For ϑ = 0.15 (within the chaotic regime):

| Approximation | λ_max | 95% CI |
|---------------|-------|--------|
| Mean-field | +0.8 to +1.5 | varies with seed |
| Structured | -0.05 to +0.02 | typically < 0 |

The structured approximation yields λ_max near zero or slightly negative across tested parameters, indicating stable or marginally stable dynamics. The mean-field approximation yields positive λ_max in the high-ϑ regime, indicating sensitive dependence on initial conditions.

### 3.5 Interpretation

The structured approximation's stability advantage can be understood intuitively: by maintaining cross-level correlations, it provides a "damping" effect where beliefs at each level constrain each other. Under mean-field, the levels update independently, allowing oscillations to grow unchecked.

However, we caution against over-interpreting these results:

1. Our analysis is limited to the two-level HGF; behavior in deeper hierarchies may differ
2. The specific bifurcation thresholds are parameter-dependent
3. Biological inference systems likely differ in important ways from the idealized models studied here

---

## 4. Resource-Rational Extensions

### 4.1 Motivation

The structured approximation provides greater stability but requires more computation—it must track the conditional distribution q(x|z) rather than just marginals. This suggests a trade-off: agents with limited computational resources might prefer mean-field approximations when volatility is low (and stability is not threatened) but switch to structured approximations when volatility is high.

We formalize this intuition by augmenting the variational objective with a cost term, following the resource-rationality framework of Lieder and Griffiths (2020).

### 4.2 Resource-Rational Free Energy

**Definition 3 (Resource-Rational Free Energy).** 

$$F_{\text{RR}} = F_{\text{VFE}} + \beta \cdot C(q)$$

where:
- F_VFE is the standard variational free energy
- C(q) is a computational cost function
- β > 0 weights the cost-accuracy trade-off

For the mean-field vs. structured comparison, we propose:

$$C(q) = \begin{cases} 0 & \text{if } q = q(x)q(z) \\ c_0 \cdot \text{CF}(q) & \text{if } q = q(z)q(x|z) \end{cases}$$

where c_0 captures the baseline cost of maintaining correlations.

**Interpretation:** This formalizes the intuition that preserving more mutual information (higher CF) requires more computational resources. The optimal approximation scheme minimizes F_RR, balancing accuracy against cost.

### 4.3 Relationship to Thermodynamics

There is a suggestive connection between computational costs and thermodynamic dissipation. Landauer's principle establishes that erasing one bit of information requires at minimum k_B T ln(2) joules. However, the relationship between this bound and the cost of biological inference is not straightforward:

1. Inference updates may not constitute logically irreversible operations in Landauer's sense
2. The relevant costs may be computational (time, memory) rather than energetic
3. Empirical estimates of neural signaling costs (~10⁴ ATP/bit; Laughlin et al., 1998) measure information *transmission*, not belief *updating*

We therefore frame our cost term as *resource-rational* rather than thermodynamic. The mathematical form may be similar, but the interpretation is epistemic (bounded rationality) rather than physical (thermodynamic necessity). Making stronger thermodynamic claims would require demonstrating that specific inference algorithms have computable minimum dissipation, which is beyond our present scope.

### 4.4 Optimal Precision

If we model the cost of high-precision inference as:

$$C(\gamma) = \gamma \ln(\gamma / \gamma_0)$$

then the resource-rational optimal precision is:

$$\gamma^* = \gamma_0 \exp\left(-1 - \frac{1}{\beta}\frac{\partial F_{\text{VFE}}}{\partial \gamma}\right)$$

This predicts that precision should increase when prediction errors are large (∂F_VFE/∂γ more negative) but saturate due to the cost term.

**Testable implication:** If this trade-off operates in biological systems, we should observe bounded precision that does not increase indefinitely with prediction error magnitude.

---

## 5. A Candidate Neural Implementation

### 5.1 Prefatory Remarks

This section is speculative. We propose a mapping between the computational framework and neural mechanisms, but this mapping is a hypothesis for future testing, not a claim about established neuroscience. The goal is to generate experimentally tractable predictions.

### 5.2 Dopamine and Precision

Several theoretical proposals suggest that dopamine modulates the precision or gain of neural computations (Friston et al., 2012; Schwartenbeck et al., 2015). The basic idea: dopaminergic input adjusts how strongly prediction errors influence belief updates.

We propose (tentatively) that tonic dopamine concentration could implement the precision parameter γ in hierarchical inference:

- Higher tonic dopamine → higher γ → stronger weighting of prediction errors
- Lower tonic dopamine → lower γ → greater reliance on prior beliefs

### 5.3 Transfer Function

If this mapping holds, there should be a transfer function relating dopamine concentration D to precision γ. Rather than positing an arbitrary sigmoid, we derive the functional form from receptor binding kinetics.

**Derivation from mass-action kinetics:** Consider dopamine (D) binding to D2 receptors (R) with dissociation constant K_d:

$$R + D \rightleftharpoons RD, \quad K_d = \frac{[R][D]}{[RD]}$$

At equilibrium, the fractional receptor occupancy θ is given by the Hill equation:

$$\theta = \frac{D^n}{K_d^n + D^n}$$

where n is the Hill coefficient (n ≈ 1 for D2 receptors; Richfield et al., 1989).

**Linking assumption:** We assume that D2 receptor occupancy modulates neural gain, and that this gain corresponds to precision. Specifically, if precision scales linearly with occupancy:

$$\gamma(D) = \gamma_{\max} \cdot \theta = \gamma_{\max} \cdot \frac{D^n}{K_d^n + D^n}$$

**Parameter constraints:**

| Parameter | Value | Status |
|-----------|-------|--------|
| K_d | ~20 nM | Constrained by D2 receptor pharmacology |
| n | ~1 | Constrained by receptor biophysics |
| γ_max | To be fit | Free parameter (depends on downstream circuitry) |

**What is derived vs. assumed:**
- *Derived:* The Hill equation form follows from equilibrium thermodynamics of receptor-ligand binding
- *Derived:* K_d ≈ 20 nM is the relevant concentration scale (from measured D2 affinity)
- *Assumed:* That occupancy maps linearly to precision (plausible but not proven)
- *Free:* γ_max must be determined from behavioral data

This derivation grounds the transfer function in established pharmacology. The inflection point of the curve (where sensitivity is maximal) occurs at D = K_d ≈ 20 nM, which lies within the physiological range of tonic dopamine (10-100 nM). This is not a coincidence of curve-fitting but a consequence of evolutionary tuning of receptor affinity to the relevant concentration range.

*See Supplementary Material (docs/transfer_function_derivation.md) for complete derivation including mass-action kinetics and discussion of D1/D2 receptor dichotomy.*

### 5.4 Phasic vs. Tonic Dopamine

Dopamine has been implicated in seemingly contradictory functions: reward prediction error signaling (Schultz et al., 1997) and precision/confidence modulation (Friston et al., 2012). These may not be contradictory if:

- *Phasic* dopamine (rapid bursts) signals prediction errors
- *Tonic* dopamine (baseline levels) modulates precision

This phasic/tonic distinction maps onto the D1/D2 receptor dichotomy:

- D2 receptors (K_d ≈ 10-30 nM) have affinity in the tonic range
- D1 receptors (K_d ≈ 1-10 μM) require phasic burst concentrations

We offer this as a potentially unifying interpretation, not as a settled conclusion.

---

## 6. Computational Methods

### 6.1 Implementation

We implement the HGF and variational inference using custom Julia code. The core update equations follow Mathys et al. (2014) with modifications for the structured approximation based on message-passing methods.

**Simplifications:** For tractability, our implementation makes several simplifications relative to the full theoretical model:

1. We use Gaussian approximations throughout
2. The structured approximation uses a first-order approximation for coupling terms
3. Lyapunov exponents are computed for the mean dynamics, ignoring posterior variance evolution

These simplifications mean our simulation results characterize the simplified model, which approximates but does not exactly match the full theoretical framework.

### 6.2 Lyapunov Computation

We compute Lyapunov exponents using the Benettin algorithm:

1. Initialize reference and perturbed trajectories with separation ε = 10⁻⁸
2. Evolve both trajectories under identical inputs
3. Every k steps (k = 10), measure separation, add log(separation/ε) to running sum, renormalize perturbation
4. Average over trajectory length, discarding initial transient (1000 steps)

Error estimates are obtained by computing exponents over multiple trajectory segments and reporting the standard error.

### 6.3 Code Availability

Full implementation code is provided in the accompanying repository. We include:

- Core model specification and update functions
- Lyapunov exponent computation
- Bifurcation diagram generation
- Utility functions for colored noise generation

---

## 7. Proposed Experimental Tests

The following are proposals for future empirical work, not accomplished experiments.

### 7.1 Computational Phenotyping

**Proposal:** Fit HGF models to behavioral data from reversal learning tasks under different pharmacological conditions. Compare mean-field vs. structured model fits.

**Prediction:** If CF is neurally implemented, dopaminergic manipulations should shift the preference between model classes:
- D2 antagonists → reduced CF → better fit by mean-field models
- Dopamine precursors → increased CF → better fit by structured models

**Caveat:** Many factors influence model fit. This prediction requires controlling for other pharmacological effects and demonstrates correlation, not mechanism.

### 7.2 Dynamical Signatures

**Proposal:** If mean-field dynamics exhibit period-doubling, this might manifest as quasi-periodic structure in belief trajectories. One could analyze trial-by-trial learning rates for oscillatory signatures.

**Prediction:** The power spectrum of learning rate time series might show structure in conditions associated with low CF.

**Strong caveat:** This is highly speculative. Biological noise may completely obscure any bifurcation structure. Null results would not falsify the theory, as they could reflect noise rather than absence of the underlying dynamics.

### 7.3 Clinical Populations

**Proposal:** Compare CF-related measures across clinical populations with known dopaminergic differences.

**Speculative predictions:**

| Condition | Putative DA State | CF Prediction | Behavioral Prediction |
|-----------|-------------------|---------------|----------------------|
| Parkinson's (off medication) | Low tonic DA | Low CF | Over-reliance on priors |
| Schizophrenia | Elevated striatal DA | High CF | Potentially unstable inference |

**Strong caveat:** These clinical populations differ in many ways beyond dopamine. Any observed differences could have multiple explanations. These are hypotheses to investigate, not predictions we are confident will hold.

### 7.4 Crucial Experiment: Spectral Signatures of Period-Doubling

The preceding proposals test whether CF-related measures correlate with pharmacological state. However, the editorial challenge is sharper: what does CF predict that alternatives do not?

**The distinguishing prediction:** CF theory predicts not merely that reduced dopamine causes noisier inference (which any theory predicts), but that it causes **qualitatively different dynamics**—specifically, quasi-periodic oscillations reflecting period-doubling bifurcations.

**Experimental design:** Within-subject pharmacological study using D2 antagonist (sulpiride 400mg) vs. placebo in a volatile reversal learning task.

**Primary outcome:** Power spectral density of trial-by-trial learning rates (estimated via HGF).

**CF prediction:** Under sulpiride (reduced CF), spectral power in the 0.02-0.10 cycles/trial band should increase, with emergent peaks at frequencies related by ratio ~2 (period-doubling signature).

**Why this is crucial:** Alternative frameworks (predictive coding, generic Bayesian brain) predict quantitative degradation (more noise, slower learning) but not **specific frequency structure**. The period-doubling signature is unique to CF because it arises from the dynamical instability of mean-field inference.

| Prediction | CF Theory | Alternatives |
|------------|-----------|--------------|
| Increased variability | ✓ | ✓ |
| Oscillatory structure at specific frequencies | ✓ | ✗ |

**Falsification criteria:** CF would be falsified if sulpiride produces expected behavioral effects but no spectral signatures, or if signatures appear but frequency ratios are inconsistent with period-doubling.

*See `experiments/crucial_experiment_protocol.md` for complete protocol including power analysis, controls, and simulation-based predictions.*

---

## 8. Discussion

### 8.1 Summary of Contributions

1. **Formal definition of Circulatory Fidelity:** A normalized measure of cross-level dependency in hierarchical approximate inference, with **proven** properties (CF = 0 under mean-field, CF > 0 under structured)

2. **Information-geometric characterization:** **Proof** that the Fisher Information Matrix is block-diagonal under mean-field, precluding coordinated belief updates

3. **Stability analysis:** Derivation showing the deterministic skeleton is stable; **numerical observation** that the full stochastic system exhibits period-doubling and chaos at high volatility

4. **Stochastic instability hypothesis:** Proposal that the observed instabilities arise from variance growth in the stochastic dynamics, not from fixed-point instability

5. **Resource-rational framing:** An account of why agents might trade off computational cost against stability

6. **Neural implementation hypothesis:** A speculative but testable proposal linking the framework to dopaminergic neuromodulation, with transfer function derived from receptor kinetics

7. **Crucial experiment:** A protocol testing the unique prediction of CF—spectral signatures of period-doubling in belief dynamics under D2 antagonism—that distinguishes CF from alternative frameworks

### 8.2 What This Work Does Not Show

We want to be explicit about the limitations:

1. **Not a proof of biological mechanism:** The dopamine-precision mapping is a hypothesis, not an established fact. While the transfer function form is derived from receptor kinetics, the assumption that occupancy maps to precision remains to be validated.

2. **Not a complete theory:** We analyze a two-level Gaussian model; real hierarchies are deeper and non-Gaussian

3. **Not thermodynamically grounded:** Despite suggestive connections, we have not derived the cost function from physical first principles

4. **Not empirically validated:** The predictions remain untested

5. **Not a novel demonstration of structured vs. mean-field advantages:** The superiority of structured approximations is known (Parr et al., 2019). Our contribution is the dynamical systems characterization.

### 8.3 Relationship to Prior Work

The advantages of structured over mean-field variational inference are well-established (e.g., Parr et al., 2019; Schwöbel et al., 2018). Our contribution is to characterize this advantage in dynamical systems terms and propose a specific neural implementation.

The resource-rationality framework (Lieder & Griffiths, 2020) provides the normative foundation for our cost-accuracy trade-off. We apply this general framework to the specific case of hierarchical inference.

The dopamine-precision hypothesis has been proposed by others (Friston et al., 2012). Our contribution is to connect it to the stability analysis and derive specific (if speculative) predictions.

**What distinguishes CF from prior work:** Previous frameworks predict that structured approximations are "better" (lower error, higher accuracy). CF theory additionally predicts **specific dynamical signatures**—period-doubling, characteristic frequencies, bifurcation thresholds—that should manifest in belief dynamics when the system operates near the mean-field regime. The crucial experiment (Section 7.4) is designed to test this unique prediction.

### 8.4 Future Directions

**Theoretical:**
- Extend analysis to deeper hierarchies
- Develop continuous-time formulation
- **Prove the stochastic instability mechanism:** Derive conditions under which variance of the HGF state grows, explaining the observed bifurcations
- Establish connection between FIM structure and dynamical stability

**Computational:**
- Systematic parameter sweeps
- Comparison with particle filtering and MCMC baselines
- Analysis of approximation error vs. stability trade-off

**Empirical:**
- Computational phenotyping studies
- Pharmacological manipulations in healthy volunteers
- Clinical population comparisons

---

## 9. Conclusion

This thesis has introduced Circulatory Fidelity as a measure of cross-level dependency in hierarchical variational inference and analyzed its relationship to dynamical stability. Our main findings, from computational analysis, are:

1. Mean-field variational inference in the HGF exhibits period-doubling bifurcations and chaos at high volatility
2. Structured approximations that maintain CF > 0 remain stable across a broader parameter range
3. This stability difference can be understood geometrically through the Fisher Information Matrix structure

We have proposed, speculatively, that biological inference systems may implement mechanisms analogous to CF maintenance, potentially through dopaminergic precision modulation. This proposal generates testable predictions that await empirical investigation.

The framework is offered as a theoretical tool for generating hypotheses, not as a completed theory of neural computation. Its value will ultimately be determined by whether the predictions it generates are borne out by experiment.

---

## 10. References

Frässle, S., et al. (2021). TAPAS: An open-source software package for translational neuromodeling and computational psychiatry. *Frontiers in Psychiatry*, 12, 680811.

Friston, K. (2010). The free-energy principle: A unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.

Friston, K., et al. (2012). Dopamine, affordance and active inference. *PLoS Computational Biology*, 8(1), e1002327.

Laughlin, S. B., et al. (1998). The metabolic cost of neural information. *Nature Neuroscience*, 1(1), 36-41.

Lieder, F., & Griffiths, T. L. (2020). Resource-rational analysis: Understanding human cognition as the optimal use of limited computational resources. *Behavioral and Brain Sciences*, 43, e1.

Mathys, C., et al. (2011). A Bayesian foundation for individual learning under uncertainty. *Frontiers in Human Neuroscience*, 5, 39.

Mathys, C., et al. (2014). Uncertainty in perception and the Hierarchical Gaussian Filter. *Frontiers in Human Neuroscience*, 8, 825.

Parr, T., et al. (2019). Neuronal message passing using mean-field, Bethe, and marginal approximations. *Scientific Reports*, 9, 1889.

Richfield, E. K., et al. (1989). Anatomical and affinity state comparisons between dopamine D1 and D2 receptors in the rat central nervous system. *Neuroscience*, 30(3), 767-777.

Schultz, W., et al. (1997). A neural substrate of prediction and reward. *Science*, 275(5306), 1593-1599.

Schwartenbeck, P., et al. (2015). Optimal inference with suboptimal models: Addiction and active Bayesian inference. *Medical Hypotheses*, 84(2), 109-117.

Schwöbel, S., et al. (2018). Active inference, belief propagation, and the Bethe approximation. *Neural Computation*, 30(9), 2530-2567.

Seamans, J. K., & Yang, C. R. (2004). The principal features and mechanisms of dopamine modulation in the prefrontal cortex. *Progress in Neurobiology*, 74(1), 1-58.

Seeman, P., et al. (2006). Dopamine supersensitivity correlates with D2High states, implying many paths to psychosis. *PNAS*, 103(9), 3440-3445.

Servan-Schreiber, D., et al. (1990). A network model of catecholamine effects: Gain, signal-to-noise ratio, and behavior. *Science*, 249(4971), 892-895.

Surmeier, D. J., et al. (2007). D1 and D2 dopamine-receptor modulation of striatal glutamatergic signaling in striatal medium spiny neurons. *Trends in Neurosciences*, 30(5), 228-235.

---

## Appendix A: Proof Details

*Complete proofs with full derivations are provided in the supplementary document `docs/proofs.md`. Here we summarize the key results and their epistemic status.*

### A.1 Proposition 1 (FIM Block-Diagonality) — PROVEN

**Statement:** Under the mean-field approximation $q(x,z) = q(x)q(z)$ with Gaussian marginals, the Fisher Information Matrix is block-diagonal.

**Proof:** The FIM is defined as $G_{ij} = \mathbb{E}_q[\partial_i \ln q \cdot \partial_j \ln q]$.

Under mean-field: $\ln q(x,z) = \ln q(z) + \ln q(x)$

For cross-terms between z-parameters and x-parameters:
$$G_{\theta_z, \theta_x} = \mathbb{E}_{q(z)q(x)}\left[ \frac{\partial \ln q(z)}{\partial \theta_z} \cdot \frac{\partial \ln q(x)}{\partial \theta_x} \right]$$

By independence under $q$:
$$= \mathbb{E}_{q(z)}\left[ \frac{\partial \ln q(z)}{\partial \theta_z} \right] \cdot \mathbb{E}_{q(x)}\left[ \frac{\partial \ln q(x)}{\partial \theta_x} \right]$$

The score function has zero mean for exponential families:
$$\mathbb{E}_{q}\left[ \frac{\partial \ln q}{\partial \theta} \right] = \frac{\partial}{\partial \theta} \int q \, d\cdot = \frac{\partial}{\partial \theta} 1 = 0$$

Therefore all cross-terms vanish: $G_{\theta_z, \theta_x} = 0$. ∎

### A.2 Proposition 2 (Local Stability) — DERIVED WITH CAVEATS

**Statement:** Under the deterministic skeleton approximation, the linearized mean-field z-dynamics around equilibrium have Jacobian:
$$\lambda = \frac{\vartheta}{\vartheta + \alpha}$$
where $\alpha = \kappa^2 \pi_x^* / 2$.

**Key finding:** Under the deterministic skeleton, $0 < \lambda < 1$ for all $\vartheta > 0$, implying *stability*.

**Resolution of apparent contradiction:** The instability observed in simulations arises from the *stochastic* dynamics, not the deterministic skeleton. Large prediction errors drive large updates; when the gain is high, the *variance* of the state can grow even if the mean is stable. This stochastic instability mechanism is not captured by standard linearization.

**Epistemic status:** The deterministic stability result is derived. The stochastic instability is observed numerically but not yet proven analytically.

### A.3 Proposition 3 (CF Under Mean-Field) — PROVEN

**Statement:** Under mean-field, $\text{CF} = 0$.

**Proof:** By definition, $\text{CF} = I_q(z;x) / \min(H_q(z), H_q(x))$.

Mutual information: $I(z;x) = H(z) + H(x) - H(z,x)$

Under independence: $H(z,x) = H(z) + H(x)$

Therefore: $I(z;x) = H(z) + H(x) - H(z) - H(x) = 0$

Thus $\text{CF} = 0/\min(H(z), H(x)) = 0$. ∎

### A.4 On Bifurcations — NUMERICAL OBSERVATIONS

The period-doubling bifurcations and chaos reported in Section 3 are **numerical observations**, not proven results. The analytical machinery to prove these claims would require:

1. Reduction of the stochastic HGF dynamics to a form amenable to bifurcation theory
2. Verification of non-degeneracy conditions at the bifurcation point
3. Center manifold analysis for the full nonlinear system

We report the observed bifurcation thresholds as empirical findings with acknowledged uncertainty:
- First bifurcation: $\vartheta_c \in [0.04, 0.06]$
- Chaos onset: $\vartheta_{\text{chaos}} \in [0.10, 0.15]$

### A.5 On the Structured Update Approximation — MODELING CHOICE

**Context:** Section 2.2.1 introduces a structured variational update with coupling coefficient $\gamma_{zx} = \kappa \pi_x / 4$. This appendix clarifies the status of this coefficient.

**What would a first-principles derivation yield?**

Under the structured approximation $q(x,z) = q(z)q(x|z)$, the natural gradient descent update on the variational parameters is:

$$\Delta \boldsymbol{\theta} = -\eta \mathbf{G}^{-1} \nabla_{\boldsymbol{\theta}} F$$

where $\mathbf{G}$ is the Fisher Information Matrix and $F$ is the variational free energy. The off-diagonal terms $G_{zx}$ in the FIM for the structured family would determine the exact coupling between levels.

For the HGF with Gaussian posteriors, computing $G_{zx}$ requires:
1. Parameterizing $q(x|z) = \mathcal{N}(\mu_{x|z}(z), \sigma_{x|z}^2)$ with explicit z-dependence
2. Computing the expected outer product of score functions
3. Inverting the resulting FIM

This calculation is tractable but lengthy, and the result depends on the specific parameterization of the conditional.

**Our approximation:**

We bypass the full derivation by using a heuristic coupling coefficient:

$$\gamma_{zx} = \frac{\kappa \pi_x}{4}$$

This is chosen to satisfy:
1. **Correct scaling:** Proportional to coupling strength $\kappa$ and precision $\pi_x$
2. **Stability:** The factor of 1/4 ensures the damping is not so strong as to prevent learning, nor so weak as to fail to stabilize
3. **Empirical validation:** With this coefficient, the structured update remains stable across the parameter range where mean-field exhibits bifurcations

**Epistemic status:** This is a **modeling approximation**, not a derived result. The qualitative prediction (structured approximations are more stable) does not depend on the exact value of $\gamma_{zx}$, only on it being positive. The quantitative predictions (specific Lyapunov exponents, bifurcation thresholds) would change if the coefficient were derived exactly.

**Future work:** A complete derivation of the natural gradient on the structured variational family would either validate our approximation or provide the correct coefficient.

### A.6 Summary of Epistemic Status

| Claim | Status |
|-------|--------|
| FIM block-diagonal under mean-field | **Proven** |
| CF = 0 under mean-field | **Proven** |
| CF > 0 under structured | **Proven** |
| Deterministic skeleton stable | **Derived** |
| Stochastic instability mechanism | **Hypothesized** |
| Period-doubling occurs | **Observed** |
| Specific bifurcation thresholds | **Observed** |
| Structured prevents bifurcation | **Observed** |
| Structured update coefficient (κπ_x/4) | **Modeling approximation** |

---

## Appendix B: Simulation Parameters

Default parameters used throughout unless otherwise specified:

| Parameter | Symbol | Value | Justification |
|-----------|--------|-------|---------------|
| Coupling strength | κ | 1.0 | Standard value from Mathys et al. (2014) |
| Baseline log-volatility | ω | -2.0 | Gives reasonable volatility range |
| Observation precision | π_u | 10.0 | Moderate observation noise |
| Volatility of volatility | ϑ | varies | Primary parameter of interest |
| Simulation length | T | 10,000 | Sufficient for Lyapunov convergence |
| Transient discarded | - | 1,000 | Remove initial transients |
| Lyapunov renormalization | - | every 10 steps | Standard practice |
| Random seed | - | 42 | Reproducibility |

These values are chosen for computational convenience and should not be interpreted as fitted to biological data.

---

*End of document*
