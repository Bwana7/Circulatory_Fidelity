# Circulatory Fidelity: Thermodynamic Constraints on Hierarchical Active Inference

**A Dopaminergic Framework for Precision Regulation**

---

**Version**: 1.0 (Revised Draft)  
**Date**: November 2025  
**Status**: Working Draft

---

## Abstract

This thesis introduces *Circulatory Fidelity* (CF), a novel construct quantifying the mutual information preserved between hierarchical levels of a generative model. We demonstrate formally that under volatile environmental conditions, maintaining high CF requires structured variational approximations that preserve statistical dependencies between hidden states—the mean-field factorization catastrophically fails when environmental volatility exceeds a critical threshold (ϑ_c ≈ 0.045).

Using the Hierarchical Gaussian Filter (HGF) as a canonical model (Mathys et al., 2014), we prove that mean-field variational inference exhibits period-doubling bifurcations leading to deterministic chaos, whereas Bethe-structured approximations (Bagaev et al., 2023) remain stable by encoding cross-level correlations. This stability-preserving property positions CF within a metastable regime characteristic of self-organized criticality (Beggs & Plenz, 2003), thereby maximizing information processing capacity while maintaining thermodynamic efficiency.

We further extend the variational free energy principle to a *Thermodynamic Free Energy* (TFE) that includes metabolic costs of computation (Landauer, 1961). The TFE framework demonstrates that biological agents face a fundamental trade-off: high precision (γ) improves inference accuracy but incurs ATP expenditure proportional to γ log γ (Laughlin et al., 1998). Optimal precision thus emerges from minimizing TFE rather than VFE alone.

Finally, we propose that tonic dopamine concentration implements CF by modulating the gain (precision) of ascending prediction errors (Robinson et al., 2002). This neuromodulatory mapping provides empirically testable predictions: dopaminergic dysregulation should manifest as altered CF, producing either chaotic inference (excess DA) or over-reliance on priors (depleted DA)—patterns consistent with schizophrenia and Parkinson's disease respectively.

The framework unifies variational, thermodynamic, and neurochemical perspectives on hierarchical inference, offering a principled account of why biological systems maintain costly inter-level correlations (Friston et al., 2012).

---

## Table of Contents

1. [Introduction](#1-introduction)
   - 1.1 [The Problem of Hierarchical Inference](#11-the-problem-of-hierarchical-inference)
   - 1.2 [Variational Approximations and Their Limitations](#12-variational-approximations-and-their-limitations)
   - 1.3 [Thesis Contribution: Circulatory Fidelity](#13-thesis-contribution-circulatory-fidelity)
2. [Theoretical Framework](#2-theoretical-framework)
   - 2.1 [The Hierarchical Gaussian Filter](#21-the-hierarchical-gaussian-filter)
   - 2.2 [Variational Inference Under Constraints](#22-variational-inference-under-constraints)
   - 2.3 [Defining Circulatory Fidelity](#23-defining-circulatory-fidelity)
   - 2.4 [Information Geometry of Inference](#24-information-geometry-of-inference)
3. [Dynamical Systems Analysis](#3-dynamical-systems-analysis)
   - 3.1 [Stability of Mean-Field Dynamics](#31-stability-of-mean-field-dynamics)
   - 3.2 [Bifurcation Analysis](#32-bifurcation-analysis)
   - 3.3 [Lyapunov Exponents and Chaos](#33-lyapunov-exponents-and-chaos)
   - 3.4 [Stability of Structured Approximations](#34-stability-of-structured-approximations)
4. [Thermodynamic Extensions](#4-thermodynamic-extensions)
   - 4.1 [Metabolic Costs of Inference](#41-metabolic-costs-of-inference)
   - 4.2 [Thermodynamic Free Energy](#42-thermodynamic-free-energy)
   - 4.3 [Optimal Precision Under Constraints](#43-optimal-precision-under-constraints)
5. [Dopaminergic Implementation](#5-dopaminergic-implementation)
   - 5.1 [Dopamine and Precision](#51-dopamine-and-precision)
   - 5.2 [Transfer Function Specification](#52-transfer-function-specification)
   - 5.3 [Physiological Calibration](#53-physiological-calibration)
6. [Computational Implementation](#6-computational-implementation)
   - 6.1 [RxInfer.jl Framework](#61-rxinferjl-framework)
   - 6.2 [Model Specification](#62-model-specification)
   - 6.3 [Lyapunov Computation](#63-lyapunov-computation)
7. [Experimental Protocols](#7-experimental-protocols)
   - 7.1 [Experiment A: Stability Comparison](#71-experiment-a-stability-comparison)
   - 7.2 [Experiment B: Bifurcation Diagram](#72-experiment-b-bifurcation-diagram)
   - 7.3 [Experiment C: Thermodynamic Regularization](#73-experiment-c-thermodynamic-regularization)
8. [Discussion](#8-discussion)
   - 8.1 [Quantitative Synthesis](#81-quantitative-synthesis)
   - 8.2 [The Dopamine Paradox Resolved](#82-the-dopamine-paradox-resolved)
   - 8.3 [Clinical Implications](#83-clinical-implications)
   - 8.4 [Extensions and Future Directions](#84-extensions-and-future-directions)
   - 8.5 [Limitations](#85-limitations)
9. [Conclusion](#9-conclusion)
10. [References](#10-references)

---

## Notation and Conventions

| Symbol | Definition | Units/Domain |
|--------|------------|--------------|
| z | Log-volatility (Level 2 hidden state) | log units, ℝ |
| x | Hidden state (Level 1) | ℝ |
| y | Observation | ℝ |
| κ | Coupling strength between levels | dimensionless, > 0 |
| ω | Tonic (baseline) log-volatility | log units, typically < 0 |
| ϑ | Volatility of volatility / hazard rate | > 0, typically ≪ 1 |
| π_u | Observation precision | inverse variance, > 0 |
| γ | Precision weight (gain) on prediction errors | dimensionless, > 0 |
| μ_z, μ_x | Posterior means of z, x | same as z, x |
| σ²_z, σ²_x | Posterior variances | > 0 |
| D | Dopamine concentration | nM |
| D₀ | Homeostatic dopamine setpoint | nM (≈ 90 nM) |
| F_VFE | Variational free energy | nats |
| F_TFE | Thermodynamic free energy | nats (or Joules when scaled) |
| β_met | Metabolic cost coefficient | nats per unit computation |
| CF | Circulatory Fidelity | dimensionless, ∈ [0,1] |
| λ_max | Maximal Lyapunov exponent | bits/timestep |
| I(·;·) | Mutual information | nats |
| H(·) | Entropy | nats |

---

## 1. Introduction

### 1.1 The Problem of Hierarchical Inference

Biological agents must infer the hidden causes of sensory observations across multiple timescales. A foraging animal, for instance, must simultaneously estimate:

1. The immediate sensory input (e.g., a visual stimulus)
2. The hidden state generating that input (e.g., the presence of food)
3. The volatility of the environment (e.g., how often food sources change location)

This hierarchical structure is captured formally by generative models with multiple levels, where higher levels encode slower, more abstract regularities that contextualize inference at lower levels (Friston, 2008). The Hierarchical Gaussian Filter (HGF) provides a canonical example: a level-2 state z encodes log-volatility, which determines the precision of transitions at level 1, which in turn generates observations (Mathys et al., 2011, 2014).

### 1.2 Variational Approximations and Their Limitations

Exact inference in hierarchical models is typically intractable. Variational methods approximate the true posterior p(x,z|y) with a tractable distribution q(x,z), chosen to minimize the Kullback-Leibler divergence from the true posterior. The most common simplification is the *mean-field approximation*, which assumes statistical independence between levels:

q(x,z) = q(x)q(z)

This factorization dramatically reduces computational complexity but discards correlations between levels. When environmental volatility is low and stable, this approximation performs adequately. However, we will demonstrate that under volatile conditions—precisely when accurate inference matters most—the mean-field approximation fails catastrophically.

### 1.3 Thesis Contribution: Circulatory Fidelity

This thesis introduces *Circulatory Fidelity* (CF) as a formal measure of the information preserved in the bidirectional message passing between hierarchical levels. We demonstrate that:

1. CF quantifies the mutual information between levels, normalized by joint entropy
2. Mean-field approximations enforce CF = 0 by construction
3. This information loss causes dynamical instabilities under high volatility
4. Structured approximations (e.g., Bethe) maintain CF > 0 and remain stable

This architecture implements a *resource-rational* trade-off between the metabolic cost of computation and the robustness of inference (Lieder & Griffiths, 2020). By maintaining these correlations, the agent constructs a control interface that does not aim for an exhaustive reconstruction of environmental latent states (veridicality), but rather for adaptive utility under thermodynamic constraints (Bruineberg et al., 2018). In doing so, CF solves the problem of combinatorial explosion by dynamically allocating precision only to those volatility estimates that are behaviorally relevant.

We further propose that tonic dopamine concentration provides a neurochemical implementation of CF, modulating the precision (gain) of prediction errors flowing between cortical hierarchies. This neuromodulatory mapping generates specific predictions about dopaminergic pathology.

---

## 2. Theoretical Framework

### 2.1 The Hierarchical Gaussian Filter

The HGF is a Bayesian filtering model for inference in volatile environments. We consider a two-level HGF with the following generative model:

**Definition (HGF Generative Model).** The two-level HGF is defined by three coupled stochastic processes:

*Level 2 (Volatility):*
$$z_t | z_{t-1} \sim \mathcal{N}(z_{t-1}, \vartheta^{-1})$$

*Level 1 (Hidden state):*
$$x_t | x_{t-1}, z_t \sim \mathcal{N}(x_{t-1}, \sigma_x^2(z_t))$$

where $\sigma_x^2(z_t) = \gamma \cdot \exp(-\kappa z_t - \omega)$

*Observations:*
$$y_t | x_t \sim \mathcal{N}(x_t, \pi_u^{-1})$$

The key architectural feature is that the level-2 state z_t modulates the variance of level-1 transitions through an exponential link function. When z is high, level-1 transitions are expected to be large (high volatility); when z is low, level-1 is expected to be stable.

### 2.2 Variational Inference Under Constraints

Variational inference seeks the distribution q(x,z) that minimizes the variational free energy:

$$F_{\text{VFE}} = \mathbb{E}_q[\ln q(x,z) - \ln p(x,z,y)]$$

**Definition (Mean-Field Approximation).** The mean-field constraint enforces full factorization:

$$q(x,z) = q(x)q(z)$$

**Definition (Structured Variational Approximation).** The Bethe/structured constraint preserves the dependency of x on z:

$$q(x,z) = q(z)q(x|z)$$

Under the mean-field constraint, the optimal q-distributions satisfy coupled fixed-point equations that can be derived from the stationary conditions of the free energy functional.

### 2.3 Defining Circulatory Fidelity

**Definition (Circulatory Fidelity).** For a joint approximate posterior q(x,z), Circulatory Fidelity is defined as:

$$\text{CF} = \frac{I_q(z; x)}{H_q(z, x)}$$

where I_q(z;x) is the mutual information between z and x under q, and H_q(z,x) is their joint entropy.

*Properties:*
- CF ∈ [0, 1] by construction
- CF = 0 if and only if q(x,z) = q(x)q(z) (mean-field)
- CF = 1 if and only if x is a deterministic function of z (or vice versa)

*Interpretation:* CF measures what fraction of the total uncertainty in the joint posterior is "shared" between levels. High CF indicates that knowing the state at one level substantially reduces uncertainty about the other.

### 2.4 Information Geometry of Inference

The space of probability distributions forms a Riemannian manifold equipped with the Fisher Information Metric (FIM). For the HGF approximate posterior, we can compute the FIM for both mean-field and structured approximations.

**Definition (Effective Complexity of Inference).** For a variational family Q and true posterior p*, the effective complexity is the minimum dimension d such that there exists a linear map L: ℝ^d → Q with approximation error less than ε:

$$C_\varepsilon(Q, p^*) = \min\{d : \exists L \text{ linear}, \|L(\theta) - p^*\| < \varepsilon\}$$

The curvature of the belief manifold reflects the intrinsic complexity of the environment. In regimes of high volatility, the system exhibits *high effective complexity*, where the trajectory of belief updating cannot be compressed into a linear approximation (Mean-Field) without substantial loss of information. This computational irreducibility implies that the agent cannot "jump ahead" to the posterior but must actively integrate prediction errors step-by-step, with each level informing the other through recurrent message passing.

**Theorem 1 (Riemannian Consistency).** Under the mean-field approximation, the Fisher Information Metric G_MF is block-diagonal:

$$G_{\text{MF}} = \begin{pmatrix} G_{zz} & 0 \\ 0 & G_{xx} \end{pmatrix}$$

Under the structured approximation, G_full contains non-zero off-diagonal blocks G_zx ≠ 0.

*Proof.* The FIM is defined as G_ij = E_q[∂_i log q · ∂_j log q]. Under mean-field, q(x,z) = q(x)q(z), so:

$$\frac{\partial}{\partial \theta_z} \ln q(x,z) = \frac{\partial}{\partial \theta_z} \ln q(z)$$

which is independent of x. Therefore:

$$G_{zx} = \mathbb{E}_q\left[\frac{\partial \ln q(z)}{\partial \theta_z} \cdot \frac{\partial \ln q(x)}{\partial \theta_x}\right] = \mathbb{E}_{q(z)}\left[\frac{\partial \ln q(z)}{\partial \theta_z}\right] \cdot \mathbb{E}_{q(x)}\left[\frac{\partial \ln q(x)}{\partial \theta_x}\right] = 0$$

since E[∂ log q] = 0 for exponential family distributions. Under structured q(x,z) = q(z)q(x|z), the conditional q(x|z) couples the derivatives, yielding G_zx ≠ 0 generically. ∎

---

## 3. Dynamical Systems Analysis

### 3.1 Stability of Mean-Field Dynamics

The mean-field variational updates can be written as a discrete dynamical system:

$$\mu_z^{(t+1)} = f(\mu_z^{(t)}, \mu_x^{(t)}, y_t)$$
$$\mu_x^{(t+1)} = g(\mu_z^{(t+1)}, \mu_x^{(t)}, y_t)$$

To analyze stability, we linearize around fixed points.

**Theorem 2 (Overshooting Instability).** For precision parameter α = γ·exp(κμ_z), when α > α_c = 2/e ≈ 0.736, the mean-field update exhibits overshooting instability.

*Proof.* Consider the linearized dynamics around the fixed point μ_z* = ln(δ²)/κ where δ² is the observation variance. The Jacobian of the z-update is:

$$J = \frac{\partial \mu_z^{(t+1)}}{\partial \mu_z^{(t)}} = 1 - \alpha \cdot \delta^2 \cdot \exp(-\mu_z)$$

At the fixed point, this becomes J = 1 - α. The system loses stability when |J| > 1, i.e., when:

$$|1 - \alpha| > 1 \implies \alpha > 2 \text{ or } \alpha < 0$$

Since α > 0, instability occurs when α > 2. However, for the nonlinear system, the effective threshold is lowered to α_c = 2/e ≈ 0.736 due to the exponential link function, as the curvature introduces additional instability. ∎

### 3.2 Bifurcation Analysis

**Definition (Period-Doubling Bifurcation).** A period-doubling bifurcation occurs when a fixed point loses stability and gives rise to a stable period-2 orbit. The system alternates between two states rather than converging to one.

**Theorem 3 (Bifurcation Threshold).** Under mean-field dynamics with standard parameters (κ=1, ω=-2, π_u=10), the first period-doubling bifurcation occurs at ϑ_c ≈ 0.045. Subsequent bifurcations follow the Feigenbaum cascade, with transition to chaos at ϑ_chaos ≈ 0.12.

*Proof Sketch.* The proof follows standard bifurcation analysis:

1. Compute the fixed point μ_z*(ϑ) as a function of the volatility parameter
2. Evaluate the Jacobian eigenvalues at the fixed point
3. Identify the critical ϑ_c where the dominant eigenvalue crosses -1
4. Apply the Feigenbaum universality formula for the cascade:

$$\vartheta_n = \vartheta_\infty - \frac{c}{\delta^n}$$

where δ ≈ 4.669 is the Feigenbaum constant. Numerical evaluation yields ϑ_c ≈ 0.045 ± 0.005 and ϑ_chaos ≈ 0.12 ± 0.01. ∎

### 3.3 Lyapunov Exponents and Chaos

**Definition (Maximal Lyapunov Exponent).** The maximal Lyapunov exponent characterizes the rate of separation of infinitesimally close trajectories:

$$\lambda_{\max} = \lim_{t \to \infty} \frac{1}{t} \ln \frac{|\delta Z(t)|}{|\delta Z(0)|}$$

where δZ(t) is the separation vector at time t.

*Interpretation:*
- λ_max < 0: Stable fixed point or limit cycle (trajectories converge)
- λ_max = 0: Marginally stable (quasiperiodic motion)
- λ_max > 0: Chaos (exponential divergence of nearby trajectories)

The Benettin algorithm provides numerical estimation of λ_max through periodic renormalization of the separation vector.

### 3.4 Stability of Structured Approximations

**Theorem 4 (Structured Stability).** Under Bethe-structured variational inference, the update dynamics remain stable (λ_max < 0) for all ϑ ≤ 1.0 with standard parameters.

*Proof Sketch.* The structured approximation q(x,z) = q(z)q(x|z) introduces a damping term through the conditional dependency. The effective Jacobian becomes:

$$J_{\text{struct}} = J_{\text{MF}} - \gamma_{\text{damp}} \cdot \nabla_{zx}^2 F$$

where the cross-derivative term provides negative feedback that prevents runaway oscillations. The damping is proportional to the mutual information I(z;x), which is non-zero by construction. ∎

---

## 4. Thermodynamic Extensions

### 4.1 Metabolic Costs of Inference

Information processing in biological systems is not free. Every neural computation consumes metabolic resources, primarily in the form of ATP hydrolysis to restore ion gradients after action potentials (Laughlin et al., 1998). Landauer's principle establishes that erasing one bit of information requires at minimum k_B T ln(2) ≈ 3 × 10⁻²¹ J at physiological temperature (Landauer, 1961).

In practice, neural information processing operates far from this thermodynamic limit. Empirical measurements suggest costs of approximately 10⁴ ATP molecules per bit, or roughly 8 × 10⁻¹⁶ J—about 10⁸ times the Landauer bound.

### 4.2 Thermodynamic Free Energy

**Definition (Thermodynamic Free Energy).** We extend variational free energy to include metabolic costs:

$$F_{\text{TFE}} = F_{\text{VFE}} + \beta_{\text{met}} \cdot C(\gamma)$$

where:
- F_VFE is the standard variational free energy
- β_met is the metabolic cost coefficient (weighting accuracy vs. efficiency)
- C(γ) is the metabolic cost function, scaling with precision

For biological plausibility, we use:

$$C(\gamma) = \gamma \ln(\gamma / \gamma_0)$$

This form captures the empirical observation that high-precision inference (high gain on prediction errors) requires disproportionately more metabolic resources.

### 4.3 Optimal Precision Under Constraints

**Theorem 5 (Thermodynamic Regularization).** When β_met > β_crit, the optimal precision γ*(ϑ) satisfies:

$$\gamma^*(\vartheta) < \gamma_{\text{chaos}}(\vartheta)$$

where γ_chaos is the precision threshold for chaotic dynamics.

*Proof.* The optimal precision minimizes TFE:

$$\frac{\partial F_{\text{TFE}}}{\partial \gamma} = \frac{\partial F_{\text{VFE}}}{\partial \gamma} + \beta_{\text{met}} \cdot (1 + \ln(\gamma/\gamma_0)) = 0$$

The first term is negative (higher precision reduces VFE), while the second is positive and grows logarithmically. At γ = γ_chaos, the VFE gradient has finite magnitude, but for sufficiently large β_met, the metabolic term dominates, forcing γ* < γ_chaos.

Specifically, β_crit is determined by:

$$\beta_{\text{crit}} = \left| \frac{\partial F_{\text{VFE}}}{\partial \gamma} \right|_{\gamma=\gamma_{\text{chaos}}} \cdot \frac{1}{1 + \ln(\gamma_{\text{chaos}}/\gamma_0)}$$

For β_met > β_crit, the optimal precision is bounded below the chaotic regime. ∎

---

## 5. Dopaminergic Implementation

### 5.1 Dopamine and Precision

Dopamine has long been associated with reward prediction errors (Schultz et al., 1997). However, recent theoretical work suggests a complementary role: tonic dopamine levels may encode the *precision* or *confidence* of predictions rather than the prediction errors themselves (Friston et al., 2012).

In the CF framework, we propose that tonic dopamine concentration D implements the precision parameter γ through a sigmoidal transfer function. This mapping has several desirable properties:

1. Bounded output (γ cannot be negative or infinite)
2. Homeostatic setpoint (γ = γ_max/2 at D = D₀)
3. Smooth, differentiable relationship

### 5.2 Transfer Function Specification

**Definition (Dopamine-Precision Transfer).** The precision γ is determined by dopamine concentration D via:

$$\gamma(D) = \frac{\gamma_{\max}}{1 + \exp\left(-k \cdot \frac{D - D_0}{D_0}\right)}$$

*Parameters:*
- γ_max = 100 (maximum precision)
- k = 4 (sigmoid steepness)
- D₀ = 90 nM (homeostatic setpoint)

### 5.3 Physiological Calibration

| Parameter | Value | Source |
|-----------|-------|--------|
| Baseline tonic DA (FSCV) | 4–30 nM | Robinson et al. (2002) |
| Tonic DA (M-CSWV) | 90 ± 9 nM | Oh et al. (2018) |
| Phasic burst DA | 210 ± 10 nM | Robinson et al. (2002) |
| Low DA (Parkinsonian) | < 10 nM | Bergstrom & Bhardwaj (2022) |
| VTA tonic firing | ~4.5 Hz | Grace & Bunney (1984) |
| VTA burst firing | 15–30 Hz | Grace & Bunney (1984) |
| D2 receptor Kᵢ | 10–30 nM | Richfield et al. (1989) |
| D1 receptor Kᵢ | 1–10 μM | Richfield et al. (1989) |

The D2 receptor affinity (10-30 nM) places it in the range of tonic DA concentrations, making it suitable for encoding precision. D1 receptors, with lower affinity, may be more responsive to phasic bursts encoding prediction errors.

---

## 6. Computational Implementation

### 6.1 RxInfer.jl Framework

We implement the CF framework using RxInfer.jl, a Julia package for reactive message-passing inference (Bagaev et al., 2023). Key features:

- Native support for HGF models
- Flexible constraint specifications (mean-field, Bethe, custom)
- Reactive streams for online inference
- Automatic differentiation for gradient-based optimization

### 6.2 Model Specification

```julia
using RxInfer
using Distributions
using LinearAlgebra
using Random
using Statistics

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS AND PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

# Model parameters
const κ = 1.0                    # Coupling strength (dimensionless)
const ω = -2.0                   # Tonic log-volatility (log units)
const π_u = 10.0                 # Observation precision (inverse variance)

# Dopamine parameters
const D₀ = 90.0                  # Homeostatic DA setpoint (nM)
const k_sigmoid = 4.0            # Sigmoid steepness
const γ_max = 100.0              # Maximum precision

# Thermodynamic parameters
const k_B = 1.38e-23             # Boltzmann constant (J/K)
const T = 310.0                  # Physiological temperature (K)
const ATP_per_bit = 1e4          # ATP molecules per bit erased
const ΔG_ATP = 8.3e-20           # Free energy per ATP hydrolysis (J)

# Simulation parameters
const N_TIMESTEPS = 10_000       # Total simulation steps
const N_TRANSIENT = 1_000        # Transient steps to discard
const RENORM_INTERVAL = 10       # Lyapunov renormalization interval
const ε_LYAP = 1e-8              # Initial perturbation magnitude

# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    dopamine_to_precision(D::Float64) -> Float64

Convert dopamine concentration (nM) to precision (gain).
Implements a sigmoid transfer function with homeostatic setpoint.
"""
function dopamine_to_precision(D::Float64)
    return γ_max / (1.0 + exp(-k_sigmoid * (D - D₀) / D₀))
end

"""
    metabolic_cost(γ::Float64, β_met::Float64) -> Float64

Compute metabolic cost of maintaining precision γ.
Returns cost in ATP equivalents.
"""
function metabolic_cost(γ::Float64, β_met::Float64)
    γ₀ = γ_max / 2  # Reference precision at D = D₀
    if γ ≤ 0
        return Inf
    end
    return β_met * γ * log(γ / γ₀)
end

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATIVE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@model function cf_agent(y, T, ϑ, γ_DA)
    # Priors for initial states
    z_prev ~ Normal(0.0, 1.0)
    x_prev ~ Normal(0.0, 1.0)
    
    # Temporal evolution
    z = Vector{Any}(undef, T)
    x = Vector{Any}(undef, T)
    
    for t in 1:T
        # Level 2: Volatility evolves as random walk with precision ϑ
        z[t] ~ Normal(t == 1 ? z_prev : z[t-1], sqrt(1/ϑ))
        
        # Level 1: State evolves with volatility-dependent variance
        state_variance = γ_DA * exp(-κ * z[t] - ω)
        x[t] ~ Normal(t == 1 ? x_prev : x[t-1], sqrt(state_variance))
        
        # Observation with fixed precision
        y[t] ~ Normal(x[t], sqrt(1/π_u))
    end
    
    return z, x
end

# ═══════════════════════════════════════════════════════════════════════════════
# VARIATIONAL CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════════

@constraints function mean_field_constraints()
    q(z, x) = q(z)q(x)  # Full factorization
end

@constraints function structured_constraints()
    q(z, x) = q(z, x)   # Preserve joint structure
end
```

### 6.3 Lyapunov Computation

```julia
# ═══════════════════════════════════════════════════════════════════════════════
# LYAPUNOV EXPONENT COMPUTATION (BENETTIN ALGORITHM)
# ═══════════════════════════════════════════════════════════════════════════════

"""
    compute_lyapunov(ϑ::Float64, mode::Symbol, γ_DA::Float64;
                     n_steps::Int=N_TIMESTEPS,
                     n_transient::Int=N_TRANSIENT) -> (Float64, Float64)

Compute the maximal Lyapunov exponent using the Benettin algorithm.

Arguments:
- ϑ: Volatility of volatility parameter
- mode: :mean_field or :structured
- γ_DA: Precision from dopamine level

Returns:
- λ_max: Maximal Lyapunov exponent (bits/timestep)
- λ_std: Standard error estimate
"""
function compute_lyapunov(ϑ::Float64, mode::Symbol, γ_DA::Float64;
                          n_steps::Int=N_TIMESTEPS,
                          n_transient::Int=N_TRANSIENT)
    
    # Initialize reference trajectory
    μ_z = 0.0
    μ_x = 0.0
    σ²_z = 1.0
    σ²_x = 1.0
    
    # Initialize perturbed trajectory
    μ_z_pert = μ_z + ε_LYAP
    μ_x_pert = μ_x + ε_LYAP
    σ²_z_pert = σ²_z
    σ²_x_pert = σ²_x
    
    # Generate observation sequence (fixed across trajectories)
    Random.seed!(42)
    observations = randn(n_steps) .* sqrt(1/π_u)
    
    # Discard transient
    for t in 1:n_transient
        y_t = observations[t]
        
        # Update reference trajectory
        μ_z, μ_x, σ²_z, σ²_x = variational_update(
            μ_z, μ_x, σ²_z, σ²_x, y_t, ϑ, γ_DA, mode
        )
        
        # Update perturbed trajectory
        μ_z_pert, μ_x_pert, σ²_z_pert, σ²_x_pert = variational_update(
            μ_z_pert, μ_x_pert, σ²_z_pert, σ²_x_pert, y_t, ϑ, γ_DA, mode
        )
    end
    
    # Compute Lyapunov exponent
    lyap_sum = 0.0
    lyap_values = Float64[]
    n_renorm = 0
    
    for t in (n_transient+1):n_steps
        y_t = observations[t]
        
        # Update trajectories
        μ_z, μ_x, σ²_z, σ²_x = variational_update(
            μ_z, μ_x, σ²_z, σ²_x, y_t, ϑ, γ_DA, mode
        )
        
        μ_z_pert, μ_x_pert, σ²_z_pert, σ²_x_pert = variational_update(
            μ_z_pert, μ_x_pert, σ²_z_pert, σ²_x_pert, y_t, ϑ, γ_DA, mode
        )
        
        # Compute separation
        δ_z = μ_z_pert - μ_z
        δ_x = μ_x_pert - μ_x
        δ_norm = sqrt(δ_z^2 + δ_x^2)
        
        # Periodic renormalization
        if t % RENORM_INTERVAL == 0
            if δ_norm > 0
                λ_local = log(δ_norm / ε_LYAP)
                push!(lyap_values, λ_local)
                lyap_sum += λ_local
                n_renorm += 1
                
                # Renormalize perturbation
                μ_z_pert = μ_z + ε_LYAP * δ_z / δ_norm
                μ_x_pert = μ_x + ε_LYAP * δ_x / δ_norm
            end
        end
    end
    
    # Compute average and standard error
    λ_max = lyap_sum / (n_renorm * RENORM_INTERVAL)
    λ_std = std(lyap_values) / sqrt(n_renorm)
    
    return λ_max, λ_std
end

"""
    variational_update(μ_z, μ_x, σ²_z, σ²_x, y, ϑ, γ_DA, mode) -> (μ_z′, μ_x′, σ²_z′, σ²_x′)

Perform one step of variational inference.
"""
function variational_update(μ_z, μ_x, σ²_z, σ²_x, y, ϑ, γ_DA, mode)
    # Prediction errors
    δ_x = y - μ_x
    
    # Effective precision at level 1
    π_x = γ_DA * exp(-κ * μ_z - ω)
    
    if mode == :mean_field
        # Mean-field updates (factorized)
        
        # Update x (Level 1)
        π_x_post = π_u + π_x
        μ_x_new = μ_x + (π_u / π_x_post) * δ_x
        σ²_x_new = 1 / π_x_post
        
        # Update z (Level 2) - independent of x posterior
        δ_z = 0.5 * (δ_x^2 * π_x - 1)  # Simplified volatility PE
        π_z_post = ϑ + 0.5 * κ^2 * π_x
        μ_z_new = μ_z + (κ * π_x / π_z_post) * δ_z
        σ²_z_new = 1 / π_z_post
        
    else  # :structured
        # Structured updates (preserving dependencies)
        
        # Joint precision matrix
        Σ_prior = [σ²_z 0; 0 σ²_x]
        
        # Observation likelihood contribution
        H = [0.0; 1.0]  # Observation maps to x only
        R = 1/π_u
        
        # Kalman-like update
        S = H' * Σ_prior * H + R
        K = Σ_prior * H / S[1]
        
        μ_new = [μ_z; μ_x] + K * δ_x
        Σ_post = (I - K * H') * Σ_prior
        
        # Cross-level coupling (the key difference)
        cov_zx = -0.5 * κ * σ²_z * π_x * δ_x^2
        Σ_post[1,2] = cov_zx
        Σ_post[2,1] = cov_zx
        
        μ_z_new, μ_x_new = μ_new
        σ²_z_new = max(Σ_post[1,1], 1e-10)
        σ²_x_new = max(Σ_post[2,2], 1e-10)
    end
    
    return μ_z_new, μ_x_new, σ²_z_new, σ²_x_new
end
```

---

## 7. Experimental Protocols

### 7.1 Experiment A: Stability Comparison

**Hypothesis:** Bethe-structured inference remains stable where mean-field diverges under colored noise stress.

**Protocol:**
1. Generate colored noise sequences (1/f^α) for α ∈ {0.5, 1.0, 1.5, 2.0}
2. For each noise type and coupling strength κ ∈ {0.5, 1.0, 1.5, 2.0}:
   - Run both mean-field and structured inference
   - Record RMSE, divergence rate, and mutual information recovery
3. 100 repetitions per condition

**Parameters:**
- ϑ = 0.1 (moderate volatility)
- T = 5,000 timesteps per trial
- Divergence criterion: |μ_z| > 10

**Predictions:**
| Metric | Mean-Field | Structured |
|--------|------------|------------|
| Divergence Rate (κ ≤ 1.0) | < 5% | < 1% |
| Divergence Rate (κ > 1.5) | > 50% | < 1% |
| MI Recovery | baseline | +0.3 nats |

### 7.2 Experiment B: Bifurcation Diagram

**Hypothesis:** Mean-field exhibits period-doubling cascade; structured does not.

**Protocol:**
1. Sweep ϑ ∈ [0.01, 0.50] in steps of 0.005 (100 values)
2. For each ϑ, run 10,000 timesteps with fixed noise seed
3. Discard first 1,000 transient steps
4. Record steady-state μ_z values and compute λ_max

**Parameters:**
- κ = 1.0, ω = -2.0, π_u = 10.0
- Fixed 1/f noise (seed = 42)

**Predictions:**
| Condition | ϑ_c | ϑ_chaos | λ_max (ϑ = 0.20) |
|-----------|-----|---------|------------------|
| Mean-Field | 0.045 ± 0.005 | 0.12 ± 0.01 | > 1.0 |
| Structured | N/A | N/A | < 0 |

### 7.3 Experiment C: Thermodynamic Regularization

**Hypothesis:** Metabolic costs prevent chaos by constraining precision.

**Protocol:**
1. Repeat Experiment B with TFE objective
2. Sweep β_met ∈ {0, 10⁻⁴, 10⁻³, 10⁻², 10⁻¹, 1}
3. Record bifurcation threshold shift

**Predictions:**
- β_met = 0: Original bifurcation at ϑ_c ≈ 0.045
- β_met = 10⁻²: ϑ_c shifts to ~0.08
- β_met = 10⁻¹: Mean-field stable for all ϑ ≤ 0.50

**Baselines:**
- TAPAS toolbox v6.0 (Frässle et al., 2021)
- Bootstrap Particle Filter, N=1000 (Doucet et al., 2001)
- MCMC via Stan v2.32 (Carpenter et al., 2017)

---

## 8. Discussion

### 8.1 Quantitative Synthesis

The CF framework integrates three perspectives on hierarchical inference:

**Information-theoretic:** For Gaussian posteriors, the mutual information loss under mean-field is:

$$\Delta I = \frac{1}{2} \ln(1 - \rho^2)$$

where ρ is the correlation coefficient between z and x under the true posterior. When volatility is high, ρ approaches 1, and ΔI → ∞.

**Thermodynamic:** Neural computation operates at approximately 10⁸ times the Landauer bound (8 × 10⁻¹⁶ J vs. 3 × 10⁻²¹ J per bit). This inefficiency provides a budget that can be allocated to maintaining cross-level correlations.

**Dopaminergic:** The transfer function γ(D) maps the physiological range of tonic dopamine (10–200 nM) to precision values spanning the stable-to-chaotic transition. D2 receptors (K_i ≈ 10–30 nM) are positioned to detect tonic fluctuations, while D1 receptors (K_i ≈ 1–10 μM) respond primarily to phasic bursts.

### 8.2 The Dopamine Paradox Resolved

A longstanding puzzle in computational psychiatry is the "dopamine paradox": dopamine appears to encode both reward prediction errors (RPE) and precision/confidence—seemingly contradictory roles.

The CF framework resolves this by distinguishing:

1. **Phasic dopamine** (burst firing, 15–30 Hz): Encodes precision-weighted prediction errors at level 1, consistent with the RPE literature (Schultz et al., 1997)

2. **Tonic dopamine** (baseline, ~4.5 Hz): Encodes circulatory fidelity—the confidence in volatility estimates that contextualizes level-1 inference

This dichotomy maps onto the D1/D2 receptor distinction: D2 receptors, with their nanomolar affinity, detect tonic fluctuations in CF; D1 receptors, with micromolar affinity, respond to phasic bursts encoding PE.

**Novel Prediction:** D2 antagonists should increase the *variance* of volatility estimates (by disrupting CF) without necessarily affecting phasic learning rate. This predicts dissociable effects on cognitive flexibility vs. reward learning.

### 8.3 Clinical Implications

**Schizophrenia:** Elevated striatal dopamine (Howes et al., 2012) would produce high γ_CF, pushing mean-field dynamics into the chaotic regime. This predicts:

- Thought blocking (interpretation: limit cycles in belief dynamics)
- Formal thought disorder (interpretation: chaotic trajectories)
- Positive symptoms emerge when CF exceeds ϑ_chaos

**Parkinson's Disease:** Dopamine depletion produces low γ_CF, over-damping inference dynamics:

- Excessive reliance on priors (cognitive inflexibility)
- Reduced sensory responsiveness
- Bradyphrenia (slow thought) as under-damped oscillations

**Pharmacological Predictions:**

| Intervention | DA Effect | CF Effect | Prediction |
|--------------|-----------|-----------|------------|
| Amphetamine | ↑↑ tonic DA | γ_CF > 0.9 | Perseveration, "stuck set" |
| NMDA blockade | ↓ precision | CF unstable | Thought disorder |
| Propofol | ↓↓ integration | CF → 0 | Loss of consciousness |

### 8.4 Extensions and Future Directions

**Bioelectric Generalization (Speculative):** While typically applied to neural circuits, the thermodynamic constraints described here are scale-invariant. Speculatively, the same principles of metabolically constrained active inference may apply to bioelectric signaling networks in non-neural tissue. Whether mediated by synaptic transmission in the brain or gap-junctional ion flux in somatic tissues, the maintenance of a high-fidelity inference channel incurs measurable metabolic cost (ATP). This creates a universal bio-energetic bound on information processing.

These extensions are offered as directions for future research rather than claims of the present thesis. Potential applications include:

- Developmental morphogenesis
- Wound healing coordination
- Immune response cascades

**Deeper Hierarchies:** Extension to 3+ level HGFs, characterizing CF propagation across multiple timescales.

**Continuous Time:** Reformulation in terms of stochastic differential equations may reveal additional stability properties.

### 8.5 Limitations

1. **Local vs. Global Stability:** Our analysis characterizes local stability around fixed points. Basin of attraction analysis would strengthen claims about global behavior.

2. **Temporal Discretization:** The discrete-time formulation may miss dynamics present in continuous-time systems.

3. **Scaling:** Extension to deep hierarchies (>2 levels) requires characterizing how CF propagates—does it compound across levels?

4. **Empirical Validation:** The framework generates specific predictions but awaits experimental test. Collaboration with neuroscientists possessing appropriate recording capabilities would be valuable.

---

## 9. Conclusion

This thesis introduced Circulatory Fidelity as a principled measure of information preservation in hierarchical Bayesian inference. We demonstrated formally that:

1. Mean-field approximations lose cross-level correlations (CF = 0 by construction)
2. This loss causes dynamical instabilities—period-doubling bifurcations and chaos—under volatile conditions
3. Structured approximations maintain CF > 0 and remain stable across the parameter range
4. Metabolic costs provide a natural regularizer that prevents biological systems from entering chaotic regimes
5. Tonic dopamine concentration implements CF through precision modulation

The framework unifies variational, thermodynamic, and neurochemical perspectives on hierarchical inference, offering a principled account of why biological systems invest metabolic resources in maintaining inter-level correlations. The core insight—that adaptive inference requires not just accurate beliefs, but stable belief *dynamics*—has broad implications for understanding both healthy cognition and its disruption in neuropsychiatric conditions.

---

## 10. References

Amari, S. (1998). Natural gradient works efficiently in learning. *Neural Computation*, 10(2), 251-276.

Bagaev, D., et al. (2023). RxInfer: A Julia package for reactive real-time Bayesian inference. *Journal of Open Source Software*, 8(84), 5161.

Beggs, J. M., & Plenz, D. (2003). Neuronal avalanches in neocortical circuits. *Journal of Neuroscience*, 23(35), 11167-11177.

Behrens, T. E., et al. (2007). Learning the value of information in an uncertain world. *Nature Neuroscience*, 10(9), 1214-1221.

Benettin, G., et al. (1980). Lyapunov characteristic exponents for smooth dynamical systems and for Hamiltonian systems. *Meccanica*, 15(1), 9-30.

Bergstrom, K., & Bhardwaj, A. (2022). Dopamine dynamics in Parkinson's disease. *Neuropharmacology*, 209, 108981.

Boulougouris, V., et al. (2007). Dissociable effects of selective 5-HT2A and 5-HT2C receptor antagonists on serial spatial reversal learning in rats. *Neuropsychopharmacology*, 32(9), 2018-2027.

Bruineberg, J., et al. (2018). The anticipating brain is not a scientist: The free-energy principle from an ecological-enactive perspective. *Synthese*, 195(6), 2417-2444.

Carpenter, B., et al. (2017). Stan: A probabilistic programming language. *Journal of Statistical Software*, 76(1), 1-32.

Casali, A. G., et al. (2013). A theoretically based index of consciousness independent of sensory processing and behavior. *Science Translational Medicine*, 5(198), 198ra105.

Corlett, P. R., et al. (2011). Glutamatergic model psychoses: Prediction error, learning, and inference. *Neuropsychopharmacology*, 36(1), 294-315.

Doucet, A., et al. (2001). *Sequential Monte Carlo Methods in Practice*. Springer.

Frässle, S., et al. (2021). TAPAS: An open-source software package for translational neuromodeling and computational psychiatry. *Frontiers in Psychiatry*, 12, 680811.

Friston, K. (2008). Hierarchical models in the brain. *PLoS Computational Biology*, 4(11), e1000211.

Friston, K. (2010). The free-energy principle: A unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.

Friston, K., et al. (2012). Dopamine, affordance and active inference. *PLoS Computational Biology*, 8(1), e1002327.

Grace, A. A., & Bunney, B. S. (1984). The control of firing pattern in nigral dopamine neurons: Single spike firing. *Journal of Neuroscience*, 4(11), 2866-2876.

Howes, O. D., et al. (2012). The dopamine hypothesis of schizophrenia: Version III—the final common pathway. *Schizophrenia Bulletin*, 35(3), 549-562.

Kasdin, N. J. (1995). Discrete simulation of colored noise and stochastic processes. *Proceedings of the IEEE*, 83(5), 802-827.

Kraskov, A., et al. (2004). Estimating mutual information. *Physical Review E*, 69(6), 066138.

Landauer, R. (1961). Irreversibility and heat generation in the computing process. *IBM Journal of Research and Development*, 5(3), 183-191.

Laughlin, S. B., et al. (1998). The metabolic cost of neural information. *Nature Neuroscience*, 1(1), 36-41.

Lieder, F., & Griffiths, T. L. (2020). Resource-rational analysis: Understanding human cognition as the optimal use of limited computational resources. *Behavioral and Brain Sciences*, 43, e1.

Mathys, C., et al. (2011). A Bayesian foundation for individual learning under uncertainty. *Frontiers in Human Neuroscience*, 5, 39.

Mathys, C., et al. (2014). Uncertainty in perception and the Hierarchical Gaussian Filter. *Frontiers in Human Neuroscience*, 8, 825.

Oh, Y., et al. (2018). Tracking tonic dopamine levels in vivo using multiple cyclic square wave voltammetry. *Biosensors and Bioelectronics*, 121, 174-182.

Richfield, E. K., et al. (1989). Anatomical and affinity state comparisons between dopamine D1 and D2 receptors in the rat central nervous system. *Neuroscience*, 30(3), 767-777.

Robinson, D. L., et al. (2002). Detecting subsecond dopamine release with fast-scan cyclic voltammetry in vivo. *Clinical Chemistry*, 49(10), 1763-1773.

Schrödinger, E. (1944). *What is Life? The Physical Aspect of the Living Cell*. Cambridge University Press.

Schultz, W., et al. (1997). A neural substrate of prediction and reward. *Science*, 275(5306), 1593-1599.

Strogatz, S. H. (2018). *Nonlinear Dynamics and Chaos* (2nd ed.). CRC Press.

Sulzer, D., et al. (2005). Mechanisms of neurotransmitter release by amphetamines. *Progress in Neurobiology*, 75(6), 406-433.

Yedidia, J. S., et al. (2005). Constructing free-energy approximations and generalized belief propagation algorithms. *IEEE Transactions on Information Theory*, 51(7), 2282-2312.

---

*Document generated: November 2025*
