"""
    CirculatoryFidelity

A Julia package implementing the Circulatory Fidelity framework for
hierarchical active inference under thermodynamic constraints.

# Overview

Circulatory Fidelity (CF) quantifies the mutual information preserved between
hierarchical levels of a generative model. This package provides:

- HGF generative models with configurable parameters (2-level and 3-level)
- Mean-field and structured variational constraints
- Lyapunov exponent computation for stability analysis
- Thermodynamic free energy extensions
- Dopamine-precision transfer functions

# Quick Start

```julia
using CirculatoryFidelity

# Define parameters
params = CFParameters(κ=1.0, ω=-2.0, ϑ=0.1, π_u=10.0)

# Run inference
results = run_inference(observations, params, :structured)

# Analyze stability
λ_max = compute_lyapunov(params)
```

# Three-Level Extension

```julia
using CirculatoryFidelity.ThreeLevelHGF

# Define three-level parameters
p = ThreeLevelParams(κ₂=1.0, κ₃=1.0, ω₂=-2.0, ω₃=-2.0, ϑ₃=0.1)

# Compare approximation schemes
results = compare_approximations(p)
```

# References

- Mathys et al. (2014). Uncertainty in perception and the HGF.
- Bagaev et al. (2023). RxInfer.jl.
- Friston et al. (2012). Dopamine, affordance and active inference.
"""
module CirculatoryFidelity

using Distributions
using LinearAlgebra
using Random
using Statistics

# Core submodules (minimal dependencies)
include("three_level_models.jl")

# Try to load RxInfer-dependent modules
const HAS_RXINFER = try
    using RxInfer
    true
catch
    false
end

if HAS_RXINFER
    include("models.jl")
    include("constraints.jl")
end

# Always available
include("lyapunov.jl")
include("thermodynamics.jl")
include("utils.jl")

# Export three-level module
export ThreeLevelHGF

# Export main types (if RxInfer available)
if HAS_RXINFER
    export CFParameters, CFResults, TFEParameters
    export cf_agent, run_inference
    export mean_field_constraints, structured_constraints
end

# Export analysis functions
export compute_lyapunov, compute_circulatory_fidelity, run_bifurcation_analysis

# Export transfer functions
export dopamine_to_precision, metabolic_cost

# Export utilities
export generate_colored_noise, compute_mutual_information

end # module
