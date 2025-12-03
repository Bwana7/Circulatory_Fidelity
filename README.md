# Circulatory Fidelity

**Thermodynamic Constraints on Hierarchical Active Inference: A Dopaminergic Framework for Precision Regulation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia](https://img.shields.io/badge/Julia-1.9+-blue.svg)](https://julialang.org/)
[![RxInfer](https://img.shields.io/badge/RxInfer-3.0+-green.svg)](https://rxinfer.ml/)

## Overview

This repository contains the theoretical framework, computational implementation, and experimental protocols for **Circulatory Fidelity (CF)**—a novel construct quantifying the mutual information preserved between hierarchical levels of a generative model under metabolic constraints.

### Key Contributions

1. **Formal Definition of Circulatory Fidelity**: CF = I(z;x)/min(H(z),H(x)), measuring information preservation in hierarchical inference
2. **Stability Analysis**: Demonstration that mean-field approximations exhibit period-doubling bifurcations under volatile conditions
3. **Three-Level Extension**: Analysis showing 85× amplification of mean-field instability in deeper hierarchies
4. **Interface Criticality**: Discovery that lower hierarchical interfaces are more critical for stability
5. **Resource-Rational Cost Function**: Derivation of C(q) = c₀·I(z;x) from operational principles
6. **Dopaminergic Implementation**: Speculative mapping of CF to tonic dopamine regulation of precision

## Repository Structure

```
Circulatory_Fidelity/
├── README.md                    # This file
├── LICENSE                      # MIT License
│
├── drafts/                      # Thesis manuscripts
│   ├── thesis_v2.tex           # LaTeX thesis (primary)
│   ├── thesis_v2.pdf           # Compiled PDF
│   ├── references.bib          # Bibliography
│   └── changelog.md            # Version history
│
├── src/                         # Source code (Julia - primary)
│   ├── CirculatoryFidelity.jl  # Main module
│   ├── models.jl               # Two-level HGF models
│   ├── three_level_models.jl   # Three-level HGF extension
│   ├── constraints.jl          # Variational constraints
│   ├── lyapunov.jl             # Dynamical systems analysis
│   ├── thermodynamics.jl       # Cost function computations
│   └── utils.jl                # Helper functions
│
├── experiments/                 # Experimental scripts
│   ├── exp_a_stability.jl      # Two-level stability analysis
│   ├── exp_three_level_stability.jl  # Three-level analysis (Julia)
│   ├── three_level_robust.py   # Three-level analysis (Python)
│   └── crucial_experiment_protocol.md  # Empirical test protocol
│
├── docs/                        # Documentation
│   ├── notation.md             # Symbol definitions
│   ├── proofs.md               # Mathematical proofs
│   ├── transfer_function_derivation.md  # Hill equation derivation
│   ├── multilevel_extension_analysis.md # Three-level theory
│   └── three_level_preliminary_results.md # Results summary
│
└── data/                        # Reference data
    └── physiological_values.csv # Empirical calibration
```

## Implementation Languages

**Julia** is the primary implementation language, chosen for:
- Performance in numerical simulations
- Excellent scientific computing ecosystem (DifferentialEquations.jl, ForwardDiff.jl)
- Growing adoption in computational neuroscience
- Native Unicode support for mathematical notation

**Python** implementations are provided for environments where Julia is unavailable, but Julia is recommended for serious use.

## Quick Start

### Prerequisites

```julia
using Pkg
Pkg.add(["Distributions", "LinearAlgebra", "Random", "Statistics"])
# Optional, for full RxInfer integration:
Pkg.add("RxInfer")
```

### Two-Level Analysis

```julia
using CirculatoryFidelity

# Define model parameters
params = CFParameters(
    κ = 1.0,      # Coupling strength
    ω = -2.0,     # Tonic log-volatility
    ϑ = 0.1,      # Volatility of volatility
    π_u = 10.0    # Observation precision
)

# Compute Lyapunov exponent
λ_max = compute_lyapunov(params, n_steps=10000)
```

### Three-Level Analysis

```julia
using CirculatoryFidelity.ThreeLevelHGF

# Define three-level parameters
p = ThreeLevelParams(
    κ₂ = 1.0,     # Level 2→1 coupling
    κ₃ = 1.0,     # Level 3→2 coupling
    ω₂ = -2.0,    # Level 2 baseline
    ω₃ = -2.0,    # Level 3 baseline
    ϑ₃ = 0.1,     # Meta-volatility
    π_u = 10.0    # Observation precision
)

# Simulate with mean-field approximation
results_mf = simulate(p, 10000, update_meanfield!)

# Simulate with structured approximation
results_st = simulate(p, 10000, (s,y,p) -> update_structured!(s,y,p))

# Compare stability
stats_mf = analyze_stability(results_mf)
stats_st = analyze_stability(results_st)

println("Mean-field Var(μ₂): ", stats_mf[:var₂])
println("Structured Var(μ₂): ", stats_st[:var₂])
```

### Running the Full Three-Level Analysis

```bash
cd experiments
julia exp_three_level_stability.jl
```

## Core Equations

### Circulatory Fidelity
$$\text{CF} = \frac{I_q(z; x)}{\min(H_q(z), H_q(x))} \in [0, 1]$$

### Resource-Rational Free Energy
$$F_{\text{RR}} = F_{\text{VFE}} + \beta \cdot I_q(z; x)$$

### Pairwise CF (Three-Level)
$$\text{CF}_{12} = \frac{I_q(z_1; z_2)}{\min(H_q(z_1), H_q(z_2))}, \quad \text{CF}_{23} = \frac{I_q(z_2; z_3)}{\min(H_q(z_2), H_q(z_3))}$$

## Key Results

### Two-Level Stability

| Regime | Mean-Field | Structured |
|--------|------------|------------|
| Low volatility (ϑ > 0.15) | Stable | Stable |
| Moderate (0.06 < ϑ < 0.15) | Period-doubling | Stable |
| High volatility (ϑ < 0.06) | Chaotic | Stable |

### Three-Level Extension

| Finding | Result |
|---------|--------|
| Depth amplification | Mean-field instability increases **85×** from 2→3 levels |
| Interface criticality | Lower interface (1-2) provides **94%** of stability benefit |
| Cascade dynamics | Mean-field **freezes** level 3 (Var ≈ 0) |
| CF discrimination | Mean-field: CF ≈ 0; Structured: CF > 0 |

## Citation

```bibtex
@thesis{circulatory_fidelity_2025,
  title = {Circulatory Fidelity: Thermodynamic Constraints on 
           Hierarchical Active Inference},
  author = {Lowry, Aaron},
  year = {2025},
  type = {Working Draft}
}
```

## References

Key papers this work builds upon:

- Mathys, C. et al. (2014). Uncertainty in perception and the Hierarchical Gaussian Filter. *Frontiers in Human Neuroscience*, 8, 825.
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11, 127-138.
- Coombs, C. H. et al. (1970). Mathematical Psychology. Prentice-Hall. (Uncertainty coefficient)
- Lieder, F. & Griffiths, T. L. (2020). Resource-rational analysis. *Behavioral and Brain Sciences*, 43, e1.
- Laughlin, S. B. et al. (1998). The metabolic cost of neural information. *Nature Neuroscience*, 1(1), 36-41.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
