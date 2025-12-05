# Notation and Conventions

This document defines all symbols used in the Circulatory Fidelity framework.

## State Variables

| Symbol | Definition | Units/Domain | Typical Values |
|--------|------------|--------------|----------------|
| z | Log-volatility (Level 2 hidden state) | log units, ℝ | [-5, 5] |
| x | Hidden state (Level 1) | ℝ | [-10, 10] |
| y | Observation | ℝ | [-10, 10] |
| μ_z | Posterior mean of z | log units | - |
| μ_x | Posterior mean of x | same as x | - |
| σ²_z | Posterior variance of z | > 0 | [0.01, 10] |
| σ²_x | Posterior variance of x | > 0 | [0.01, 10] |

## Model Parameters

| Symbol | Definition | Units/Domain | Default |
|--------|------------|--------------|---------|
| κ | Coupling strength between levels | dimensionless, > 0 | 1.0 |
| ω | Tonic (baseline) log-volatility | log units, typically < 0 | -2.0 |
| ϑ | Volatility of volatility / hazard rate | > 0, typically ≪ 1 | 0.1 |
| π_u | Observation precision | inverse variance, > 0 | 10.0 |

## Three-Level Model Parameters

| Symbol | Definition | Units/Domain | Default |
|--------|------------|--------------|---------|
| z₁ | Level 1: Hidden state | ℝ | - |
| z₂ | Level 2: Log-volatility | ℝ | - |
| z₃ | Level 3: Meta-log-volatility | ℝ | - |
| κ₂ | Level 2→1 coupling | dimensionless, > 0 | 1.0 |
| κ₃ | Level 3→2 coupling | dimensionless, > 0 | 1.0 |
| ω₂ | Level 2 baseline | log units | -2.0 |
| ω₃ | Level 3 baseline | log units | -2.0 |
| ϑ₃ | Level 3 volatility | > 0 | 0.1 |

## Dopamine Parameters

| Symbol | Definition | Units/Domain | Default |
|--------|------------|--------------|---------|
| D | Dopamine concentration | nM | - |
| D₀ | Homeostatic dopamine setpoint | nM | 90.0 |
| γ | Precision weight (gain) | dimensionless, > 0 | - |
| γ_max | Maximum precision | dimensionless | 100.0 |
| k_sigmoid | Sigmoid steepness | dimensionless | 4.0 |

## Information-Theoretic Quantities

| Symbol | Definition | Units |
|--------|------------|-------|
| CF | Circulatory Fidelity | dimensionless, ∈ [0,1] |
| I(z;x) | Mutual information between z and x | nats |
| H(z,x) | Joint entropy of z and x | nats |
| H(z) | Marginal entropy of z | nats |
| H(x) | Marginal entropy of x | nats |

## Dynamical Systems

| Symbol | Definition | Units |
|--------|------------|-------|
| λ_max | Maximal Lyapunov exponent | bits/timestep |
| ϑ_c | Critical volatility (first bifurcation) | same as ϑ |
| ϑ_chaos | Chaos onset volatility | same as ϑ |
| T | Number of timesteps | integer |

## Thermodynamic / Resource-Rational Quantities

| Symbol | Definition | Units |
|--------|------------|-------|
| F_VFE | Variational free energy | nats |
| F_RR | Resource-rational free energy | nats |
| β | Cost-accuracy trade-off weight | dimensionless |
| C(q) | Computational cost function | nats |
| I(z;x) | Mutual information (cost basis) | nats |

## Subscripts and Superscripts

| Notation | Meaning |
|----------|---------|
| (t) | Time index |
| _prev | Previous timestep |
| _new | Updated value |
| _MF | Mean-field approximation |
| _struct | Structured approximation |
| * | Optimal value |

## Distributions

| Notation | Distribution |
|----------|--------------|
| 𝒩(μ, σ²) | Gaussian with mean μ and variance σ² |
| q(·) | Approximate posterior |
| p(·) | Generative model / true distribution |

## Matrix Notation

| Symbol | Definition |
|--------|------------|
| G | Fisher Information Metric |
| Σ | Covariance matrix |
| Λ | Precision matrix (inverse covariance) |
| J | Jacobian matrix |
| I | Identity matrix |

## Key Equations

### Circulatory Fidelity
```
CF = I(z;x) / min(H(z), H(x))
```
Note: This normalization corresponds to the "uncertainty coefficient" from classical information theory (Coombs et al., 1970).

### Pairwise CF (Three-Level)
```
CF₁₂ = I(z₁;z₂) / min(H(z₁), H(z₂))
CF₂₃ = I(z₂;z₃) / min(H(z₂), H(z₃))
```

### HGF Generative Model
```
z_t | z_{t-1} ~ 𝒩(z_{t-1}, 1/ϑ)
x_t | x_{t-1}, z_t ~ 𝒩(x_{t-1}, γ·exp(-κz_t - ω))
y_t | x_t ~ 𝒩(x_t, 1/π_u)
```

### Resource-Rational Free Energy
```
F_RR = F_VFE + β · I(z;x)
```
where I(z;x) is the mutual information between hierarchical levels (computational cost).

### Dopamine-Precision Transfer
```
γ(D) = γ_max / (1 + exp(-k · (D - D₀) / D₀))
```

### Lyapunov Exponent
```
λ_max = lim_{t→∞} (1/t) ln(|δZ(t)| / |δZ(0)|)
```
