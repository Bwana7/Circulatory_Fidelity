# Circulatory Fidelity

**A Prior Predictive Diagnostic for Mean-Field Variational Inference**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Circulatory Fidelity (CF) is a normalized information-theoretic measure that quantifies structural coupling between variables in hierarchical models. It diagnoses whether mean-field variational inference (MFVI) will succeed or fail *before* running inference, directly from the prior predictive distribution.

**Key formula:**
```
CF(z, x) = I(z; x) / min(H(z), H(x))
```

where I(z;x) is mutual information and H(·) is differential entropy.

**Primary diagnostic:** We adopt the Linfoot correlation r_L = √(1 - exp(-2I)) which equals |ρ| for Gaussians and provides a universal [0,1] scale.

## Theoretical Contributions

1. **Computational Synergy Principle**: Synergistic dependencies arise if and only if the generative function is affine over GF(2), connecting to Siegenthaler's correlation immunity in cryptography.

2. **Asymmetry Paradox**: High CF predicts MFVI *failure* in filtering models (constitutive coupling) but hierarchy *redundancy* in pooling models (inductive coupling).

3. **Proximal Dominance Principle**: In deep hierarchies, proximal coupling causes up to 40× degradation while distal coupling causes none.

4. **Maximal Coupling Rule**: For non-stationary time series, MFVI suitability is determined by maximum windowed CF, not global average.

## Key Results

| Model | Finding | Correlation | N |
|-------|---------|-------------|---|
| SVF (Stochastic Volatility Filter) | High CF → degraded inference | r = 0.85 (aggregated) | 8,000 |
| HLM (Hierarchical Linear Model) | Low CF → no-pooling overfitting | r = -0.78 | 8,000 |
| Three-Layer Hierarchy | Proximal Dominance Principle | 40× vs 1× MSE ratio | 16,000 |
| Elementary Cellular Automata | 16/256 rules are affine (pure synergy) | 6.25% | 256 |

## Installation

### Python
```bash
pip install numpy scipy
# Clone repository and import from python/
from python.circulatory_fidelity import circulatory_fidelity_gaussian, circulatory_fidelity_ksg
from python.synergy_extension import synergy_screen
```

### Julia
```julia
using Pkg
Pkg.add(["NearestNeighbors", "SpecialFunctions"])
include("julia/src/CirculatoryFidelity.jl")
using .CirculatoryFidelity
```

## Quick Start

### Python
```python
from python.circulatory_fidelity import circulatory_fidelity_gaussian, circulatory_fidelity_ksg, linfoot_correlation

# Gaussian case (closed-form) - REQUIRES sigma parameters
cf = circulatory_fidelity_gaussian(rho=0.7, sigma_z=1.0, sigma_x=1.0)  # → 0.43

# Convert to Linfoot correlation
r_L = linfoot_correlation(rho=0.7)  # → 0.7 (equals |ρ| for Gaussians)

# Non-Gaussian case (KSG estimator)
cf = circulatory_fidelity_ksg(X, Y, k=5)

# Synergy screening
from python.synergy_extension import synergy_screen
result = synergy_screen(z1, z2, x)
print(f"Risk level: {result['risk_level']}")
```

### Julia
```julia
using .CirculatoryFidelity

# Gaussian case - REQUIRES sigma parameters
cf = circulatory_fidelity_gaussian(0.7, 1.0, 1.0)  # → 0.43

# Non-Gaussian case
cf = circulatory_fidelity_ksg(X, Y; k=5)
```

**Important**: CF requires `min(H(z), H(x)) > 0`. For Gaussians, this means σ > 0.2420.

## Interpretive Scale (Linfoot Correlation)

| Coupling Regime | r_L | Interpretation |
|-----------------|-----|----------------|
| Negligible | < 0.25 | MFVI safe |
| Weak | 0.25–0.35 | MFVI likely acceptable |
| Moderate | 0.35–0.55 | Caution warranted |
| Strong | > 0.55 | Structured inference recommended |

## Repository Structure

```
├── paper/
│   └── Circulatory_Fidelity_TMLR.tex      # Main manuscript (TMLR format)
├── python/
│   ├── __init__.py                         # Package initialization
│   ├── circulatory_fidelity.py            # Core CF implementation
│   └── synergy_extension.py               # Synergy detection tools
├── julia/
│   ├── Project.toml                        # Julia package manifest
│   ├── src/CirculatoryFidelity.jl         # Julia implementation
│   └── test/runtests.jl                   # Test suite
├── notebooks/
│   ├── 01_SVF_Case_Study.ipynb            # Stochastic volatility analysis
│   ├── 02_HLM_Case_Study.ipynb            # Hierarchical linear model
│   ├── 03_Deep_Hierarchy_Case_Study.ipynb # Three-layer hierarchy
│   ├── 04_KSG_NonGaussian_Demo.ipynb      # KSG estimator validation
│   └── 05_Synergy_Higher_Order.ipynb      # Synergy & Walsh-Hadamard
├── simulations/
│   ├── svf_validation.csv                 # 8,000 SVF simulations
│   ├── hlm_validation.csv                 # 8,000 HLM simulations
│   ├── three_layer_validation.csv         # 16,000 three-layer simulations
│   ├── ksg_validation.csv                 # KSG estimator validation
│   ├── threshold_calibration.csv          # Threshold derivation data
│   └── trigger_experiment.csv             # PCA failure demonstration
├── figures/
│   ├── fig1_bottleneck.pdf                # Information bottleneck
│   ├── fig2_workflow.pdf                  # Diagnostic workflow
│   ├── fig3_svf_results.pdf               # SVF validation results
│   ├── fig4_hlm_results.pdf               # HLM validation results
│   ├── fig5_geometry.pdf                  # Statistical manifold
│   ├── fig6_unified.pdf                   # Unified interpretation
│   ├── fig7_threelayer.pdf                # Three-layer results
│   ├── figS1_eurusd_analysis.pdf          # EUR/USD windowed CF
│   ├── figS2_hsb_analysis.pdf             # HSB data analysis
│   └── figS3_psis_comparison.pdf          # PSIS comparison
├── README.md
└── LICENSE
```

## Validation Summary

All simulations verify paper claims:

### Core Validations
- **SVF**: N=8,000 simulations
  - Pooled correlation: r = 0.27
  - Aggregated correlation: r = 0.85
  - Threshold CF > 0.10 separates degraded inference
  
- **HLM**: N=8,000 simulations  
  - Correlation: r = -0.78
  - CF = reliability (exact mathematical identity)
  
- **Three-Layer**: N=16,000 simulations
  - Proximal-only (κ₂₁=1.5): MSE ratio = 40×
  - Distal-only (κ₃₂=1.5): MSE ratio = 1.0×
  - Combined: MSE ratio = 47×

### Synergy Validation
- **ECA Survey**: 16/256 rules are affine over GF(2) (exactly 6.25%)
- **XOR Blind Spot**: CF(Z₁,X)≈0, CF(Z₂,X)≈0, but CF(Z₁·Z₂,X)>0.39
- **Walsh-Hadamard**: XOR has zero first-order coefficients (pure synergy)

### Thresholds (with 95% CI from bootstrap)
- **SVF**: CF > 0.10 indicates structured inference needed [CI: 0.09-0.11]
- **HLM**: CF < 0.4 indicates partial pooling needed

## Model Specifications

### Two-Level SVF (Variance Coupling)
```
x₃(t) ~ N(x₃(t-1), σ₃²)         # log-volatility
x₂(t) ~ N(x₂(t-1), exp(κ·x₃(t)))  # state (κ controls coupling)
y(t)  ~ N(x₂(t), σ_obs²)          # observation
```

### Three-Level SVF (Proximal Dominance)
```
Level 3 → Level 2: coupling κ₃₂
Level 2 → Level 1: coupling κ₂₁ (proximal)
Level 1 → y: observation
```

### HLM
```
y_ij ~ N(β_j, σ²)      # observation i in group j
β_j  ~ N(μ, τ²)        # group effect
CF   = reliability     # by construction
```

## Citation

```bibtex
@article{circulatory_fidelity_2025,
  title={Circulatory Fidelity: Quantifying Structural Coupling to Diagnose 
         Mean-Field Failure in Hierarchical Models},
  author={Anonymous},
  journal={Transactions on Machine Learning Research},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Requirements

### Python
- numpy >= 1.20
- scipy >= 1.7

### Julia
- Julia >= 1.6
- NearestNeighbors.jl
- SpecialFunctions.jl
