# Circulatory Fidelity

**A Prior Predictive Diagnostic for Mean-Field Variational Inference**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Circulatory Fidelity (CF) is a normalized information-theoretic measure that quantifies structural coupling between variables in hierarchical models. It diagnoses whether mean-field variational inference (MFVI) will succeed or fail *before* running inference.

**Key formula:**
```
CF(z, x) = I(z; x) / min(H(z), H(x))
```

where I(z;x) is mutual information and H(·) is differential entropy.

We also use the **Linfoot correlation** r_L = √(1 - exp(-2I)) as a universal [0,1] scale that equals |ρ| for Gaussians.

## Key Results

| Model | Finding | Correlation |
|-------|---------|-------------|
| SVF | High CF → degraded inference | r = 0.84 (95% CI: [0.67, 0.92]) |
| SVF | CF predicts log-likelihood gap | r = 0.86 |
| HLM | Low CF → no-pooling overfitting | r = -0.78 |
| Deep Hierarchy | Proximal Dominance Principle | MSE ratio 1.0× (distal) vs 40× (proximal) |

## Installation

### Python
```bash
pip install numpy scipy
# Then import from the python/ directory
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
from circulatory_fidelity import (
    circulatory_fidelity_gaussian, 
    circulatory_fidelity_copula,
    linfoot_correlation
)

# Gaussian case (closed-form) - REQUIRES sigma parameters
cf = circulatory_fidelity_gaussian(rho=0.7, sigma_z=1.0, sigma_x=1.0)  # → 0.43

# Non-Gaussian continuous (copula transform - RECOMMENDED)
cf = circulatory_fidelity_copula(X, Y)  # Conservative lower bound

# Linfoot correlation
r_L = linfoot_correlation(rho=0.7)  # → 0.7 for Gaussians
```

### Julia
```julia
using .CirculatoryFidelity

# Gaussian case - REQUIRES sigma parameters
cf = circulatory_fidelity_gaussian(0.7, 1.0, 1.0)  # → 0.43

# Non-Gaussian continuous (copula transform - RECOMMENDED)
cf = circulatory_fidelity_copula(X, Y)  # Conservative lower bound

# Linfoot correlation
r_L = linfoot_correlation(0.7)  # → 0.7

# Discrete/mixed variables only (use with caution - significant bias)
cf = circulatory_fidelity_ksg(X, Y; k=5)
```

**Important**: CF requires `min(H(z), H(x)) > 0`. For Gaussians, this means σ > 0.2420.

**Estimation Methods**:
- **Gaussian**: Use closed-form `circulatory_fidelity_gaussian()` (exact)
- **Non-Gaussian continuous**: Use `circulatory_fidelity_copula()` (conservative lower bound with closed-form SE)
- **Discrete/mixed**: Use `circulatory_fidelity_ksg()` with awareness of 30-45% negative bias

## Repository Structure

```
Circulatory_Fidelity/
├── paper/
│   ├── Circulatory_Fidelity_Manuscript.pdf   # Main paper (includes appendices)
│   ├── Circulatory_Fidelity_Manuscript.tex   # LaTeX source
│   └── tmlr.sty                              # TMLR style file
├── python/
│   ├── circulatory_fidelity.py               # Main CF module
│   ├── copula_cf_validation.py               # Copula validation experiments
│   ├── svf_psis_validation.py                # Log-likelihood gap validation (N=900)
│   ├── hierarchical_vae_dsprites.py          # dSprites experiment
│   └── deprecated/                           # Development versions (archived)
├── julia/
│   ├── Project.toml                          # Julia dependencies
│   ├── Manifest.toml                         # Julia lockfile
│   └── src/
│       └── CirculatoryFidelity.jl            # Julia implementation
│   └── test/
│       └── runtests.jl                       # Unit tests
├── notebooks/
│   ├── 01_SVF_Case_Study.ipynb               # Stochastic Volatility Filter
│   ├── 02_HLM_Case_Study.ipynb               # Hierarchical Linear Model
│   ├── 03_Deep_Hierarchy_Case_Study.ipynb    # Three-layer analysis
│   ├── 04_Copula_NonGaussian_Demo.ipynb      # Copula-based estimation
│   ├── 05_Synergy_Higher_Order.ipynb         # Synergy detection & XOR analysis
│   └── 06_CF_LogLik_Validation.ipynb         # Log-likelihood gap validation
├── simulations/
│   ├── svf_validation.csv                    # SVF simulations (N=8,000)
│   ├── hlm_validation.csv                    # HLM simulations (N=8,000)
│   ├── three_layer_validation.csv            # Three-layer (N=16,000)
│   ├── cf_psis_comprehensive_validation.csv  # Log-likelihood validation (N=900)
│   ├── dsprites_proximal_dominance.csv       # dSprites validation
│   ├── copula_validation.csv                 # Copula estimator validation
│   ├── threshold_calibration.csv             # Threshold calibration
│   ├── trigger_experiment.csv                # PCA failure mode validation
│   └── archive/                              # Intermediate files (archived)
├── figures/
│   ├── fig1_bottleneck.pdf                   # Information bottleneck
│   ├── fig2_workflow.pdf                     # Diagnostic workflow
│   ├── fig3_svf_results.pdf                  # SVF validation results
│   ├── fig4_hlm_results.pdf                  # HLM validation results
│   ├── fig6_unified.pdf                      # Unified interpretation
│   ├── fig7_threelayer.pdf                   # Three-layer results
│   ├── fig_copula_validation.pdf             # Copula validation
│   ├── fig_psis_validation.pdf               # Log-likelihood gap validation
│   ├── fig_dsprites_proximal.pdf             # dSprites results
│   ├── fig_eca_comparison.pdf                # Rule 30 vs 150 comparison
│   ├── fig_walsh.pdf                         # Walsh-Hadamard decomposition
│   ├── fig_windowed.pdf                      # Windowed CF analysis
│   ├── fig_proximal.pdf                      # Proximal dominance 3D
│   ├── fig_asymmetry.pdf                     # Filtering vs pooling
│   ├── fig_linfoot.pdf                       # Linfoot correlation
│   ├── fig_manifold.pdf                      # Information geometry
│   ├── fig_eca_grid.pdf                      # ECA survey grid
│   ├── figS1_eurusd_analysis.pdf             # EUR/USD case study
│   ├── figS2_hsb_analysis.pdf                # HSB case study
│   ├── figS3_psis_comparison.pdf             # PSIS comparison
│   └── generate_figures.py                   # Figure generation script
├── requirements.txt                          # Python dependencies
└── LICENSE                                   # MIT License
```

## Citation

```bibtex
@article{circulatory_fidelity_2026,
  title={Circulatory Fidelity: Quantifying Structural Coupling to Diagnose 
         Mean-Field Failure in Hierarchical Models},
  author={Aaron Lowry},
  year={2026}
}
```

## License

MIT License - see LICENSE file for details.

## Validation Summary

All simulations verify paper claims:
- **SVF**: 8,000 simulations, r = 0.84 (aggregated, 95% CI: [0.67, 0.92])
- **Log-Likelihood Gap Validation**: 900 simulations, CF predicts log-likelihood gap (r = 0.86)
  - Note: PSIS-k̂ is inappropriate for Gaussian posteriors (see Section 7.5)
- **HLM**: 8,000 simulations, r = -0.78
- **Three-Layer**: 16,000 simulations establishing Proximal Dominance Principle
  - Proximal-only (κ₂₁=1.5): MSE ratio = 40×
  - Distal-only (κ₃₂=1.5): MSE ratio = 1.0× (exactly, mathematically guaranteed)
  - Combined: MSE ratio = 47×
  - Distal amplification range: 1.01–2.26×
- **dSprites**: 8,000 images validating Proximal Dominance on real data
- **Copula Estimation**: Validated against closed-form Gaussian solutions (exact)

### Classification Performance (CF > 0.10 → ΔLL > 58)
- Sensitivity: 79%
- Specificity: 94%
- PPV: 93%
- NPV: 82%

### Thresholds (with 95% CI from bootstrap)
- **SVF**: CF > 0.10 indicates structured inference needed [CI: 0.09-0.11]
- **HLM**: CF < 0.4 indicates partial pooling needed

### Model Specifications
- **Two-level SVF**: Variance-coupling (κ modulates innovation scale)
- **Three-level SVF**: Variance-coupling extension for Proximal Dominance analysis
- **HLM**: CF = reliability by construction
