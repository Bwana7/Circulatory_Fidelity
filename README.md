# Circulatory Fidelity

**A Prior Predictive Diagnostic for Mean-Field Variational Inference**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18121821.svg)](https://doi.org/10.5281/zenodo.18121821)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org/)
[![Julia](https://img.shields.io/badge/Julia-1.9+-purple.svg)](https://julialang.org/)

## Overview

Circulatory Fidelity (CF) is a diagnostic framework for predicting when mean-field variational inference (MFVI) will fail in hierarchical Bayesian models. The primary diagnostic metric is **Inference Coupling (IC)**, which quantifies structural dependencies between latent variables and observables.

**Key insight**: High IC indicates tight coupling that MFVI's factorized approximation cannot capture, predicting inference failures *before* running expensive computations.

**Resources:**
- **GitHub**: https://github.com/Bwana7/Circulatory_Fidelity
- **Zenodo**: https://zenodo.org/records/18121821

## Quick Start

### Python

```python
from circulatory_fidelity import inference_coupling, diagnose

# Estimate IC between two variables
ic, se = inference_coupling(z_samples, x_samples)

# Full diagnostic workflow
result = diagnose(z, x, model_type='filtering')
print(f"IC = {result['ic']:.3f}, Risk: {result['risk_level']}")
```

### Julia

```julia
using CirculatoryFidelity

# Estimate IC
ic, se = inference_coupling(z, x)

# For Gaussians (closed-form)
ic = ic_gaussian(ρ)
```

## Key Concepts

### Inference Coupling (IC)

For Gaussian pairs:
```
IC = |ρ| (Linfoot correlation)
```

IC measures how much knowing X reduces uncertainty about Z, normalized to [0,1].

### Model Type Matters

| Model Type | High IC Means | Recommendation |
|------------|---------------|----------------|
| **Filtering** (SVF) | MFVI will fail | Use structured VI |
| **Pooling** (HLM) | Signal reliable | No-pooling acceptable |

### Recommended Thresholds (Interpretive Scale)

| IC Range | Coupling Regime | Interpretation |
|----------|-----------------|----------------|
| < 0.25 | Negligible | MFVI safe |
| 0.25–0.35 | Weak | MFVI likely acceptable |
| 0.35–0.55 | Moderate | Caution warranted |
| 0.55–0.70 | Strong | Consider structured inference |
| > 0.70 | Very strong | Structured inference required |

**Note**: These are general guidelines. For specific model classes (e.g., SVF, HLM), use domain-specific calibration. For pooling models, interpretation inverts: low IC indicates groups are similar (strong pooling needed).

## Installation

### Python

```bash
pip install numpy scipy
# Then copy circulatory_fidelity.py to your project
```

### Julia

```julia
using Pkg
Pkg.add(url="https://github.com/Bwana7/Circulatory_Fidelity")
```

## Estimation Methods

### Copula-Based (Recommended for All Applications)

```python
ic, se = inference_coupling(x, y, method='copula')  # default
```

Algorithm:
1. Rank-transform to uniform marginals
2. Apply probit (inverse normal CDF)
3. Compute Pearson correlation
4. IC = |ρ|

**Key insight**: The copula method is **exact for Gaussian data** (differences < 0.001 from direct Pearson) AND provides conservative estimates for non-Gaussian data. This enables a **unified workflow** without needing to verify distributional assumptions.

**Properties**:
- Exact for Gaussians (returns |ρ|)
- Conservative lower bound for non-Gaussians with monotonic dependence
- Closed-form standard errors
- Returns IC ≈ 0 for non-monotonic dependence (triggers Stage 2 protocol)

### Pearson (Alternative for Verified Gaussians)

```python
ic, se = inference_coupling(x, y, method='pearson')
```

Use only when Gaussianity has been verified. Mathematically equivalent to copula for Gaussian data, but biased for non-Gaussian marginals.

### KSG (For Validation)

```python
ic, se = inference_coupling(x, y, method='ksg')
```

Use when:
- Validating copula estimates
- Suspected non-monotonic dependence

## Time Series: The Maximal Coupling Rule

For non-stationary time series with potential regime changes, use **windowed IC**:

```python
from circulatory_fidelity import windowed_ic

# Compute IC in rolling windows (minimum recommended: 50)
result = windowed_ic(z_series, x_series, window_size=50)

print(f"IC_max = {result['ic_max']:.3f}")
print(f"IC_mean = {result['ic_mean']:.3f}")
print(f"SE per window = {result['se_per_window']:.3f}")
print(result['recommendation'])
```

**The Maximal Coupling Rule**: MFVI suitability depends on `IC_max`, not the global average. A single high-IC episode (e.g., volatility spike) can invalidate mean-field approximations for the entire trajectory.

**Minimum window size**: Window must be large enough for stable correlation estimates. SE ≈ 1/√(W-3). For W < 30, estimates have high variance and may produce noise-driven false positives. We recommend W >= 50.

### When to use windowed IC:
- Sequential/temporal models (SVF, state-space models)
- Models with potential regime changes
- Non-stationary time series

### When global IC suffices:
- Equilibrated hierarchical models
- Cross-sectional pooling problems
- Stationary processes

## High-Dimensional Data

⚠️ **IMPORTANT**: For high-dimensional vectors, use dimensionality reduction first:

```python
from circulatory_fidelity import reduce_dimensions_pls

# Reduce to 1D projections that preserve coupling
# Cross-validation enabled by default to prevent overfitting
z_reduced, x_reduced = reduce_dimensions_pls(Z, X, n_components=1, cross_validate=True)
ic, se = inference_coupling(z_reduced, x_reduced)
```

The Manifold Hypothesis justifies supervised reduction (PLS/CCA) for preserving diagnostic-relevant structure. Cross-validation ensures the extracted components represent genuine coupling rather than spurious correlation.

## Non-Monotonic Dependencies

The copula estimator is invariant to monotonic transformations but returns IC ≈ 0 for non-monotonic relationships (e.g., Y = X²). Use the non-monotonic check:

```python
from circulatory_fidelity import check_nonmonotonic_dependence

result = check_nonmonotonic_dependence(x, y)
if result['nonmonotonic_flag']:
    print("Non-monotonic dependence detected!")
    print(f"Linear IC: {result['ic_linear']:.3f}")
    print(f"Quadratic IC: {result['ic_quadratic']:.3f}")
```

## Repository Structure

```
circulatory_fidelity/
├── src/
│   ├── python/
│   │   ├── circulatory_fidelity.py    # Main Python implementation
│   │   └── generate_figures.py        # Figure generation
│   └── julia/
│       └── CirculatoryFidelity.jl     # Julia implementation
├── test/
│   └── runtests.jl                    # Julia tests
├── notebooks/
│   ├── 01_SVF_Case_Study.ipynb        # Stochastic volatility filter
│   ├── 02_HLM_Case_Study.ipynb        # Hierarchical linear models
│   ├── 03_Deep_Hierarchy_Case_Study.ipynb  # Proximal dominance
│   ├── 04_Estimation_Methods_Comparison.ipynb  # Copula vs KSG
│   ├── 05_Synergy_Higher_Order.ipynb  # Synergistic dependencies
│   ├── 06_IC_LogLik_Validation.ipynb  # Post-inference validation
│   └── data/                          # Notebook data files
├── data/                              # Validation datasets
├── figures/                           # Generated figures
├── paper/
│   ├── Circulatory_Fidelity_v1_1.tex  # Manuscript source
│   └── Circulatory_Fidelity_v1_1.pdf  # Compiled manuscript
├── Project.toml                       # Julia project file
├── README.md
└── LICENSE
```

## Key Results

### Validation Summary (N > 32,000 simulations)

| Model | Correlation (IC vs Failure) | N |
|-------|---------------------------|---|
| SVF | r = 0.83 | 8,000 |
| HLM | r = -0.76 | 8,000 |
| Three-Layer | r = 0.89 | 16,000 |
| IC vs Log-Lik Gap | r = 0.86 | 900 |

### Proximal Dominance Principle

For deep hierarchies with fully factorized MFVI:
- **Proximal coupling alone**: Up to 40× MSE degradation
- **Distal coupling alone**: Exactly 1.0× (zero degradation, mathematically guaranteed)
- **Both present**: Distal amplifies proximal failure by 1.01–2.26×

**Key insight**: Distal coupling causes zero degradation when proximal coupling is absent, but acts as a force multiplier when proximal coupling is present.

## Citation

```bibtex
@software{lowry_circulatory_fidelity_2025,
  author       = {Lowry, Aaron},
  title        = {Circulatory Fidelity: A Relational Theory of Information 
                  Flow in Hierarchical Models},
  year         = {2025},
  publisher    = {Zenodo},
  version      = {v1.1.0},
  doi          = {10.5281/zenodo.18121821},
  url          = {https://doi.org/10.5281/zenodo.18121821}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Version History

- **v1.1** (2025): IC as primary metric, copula-based estimation, synergy detection
- **v1.0** (2024): Initial release with CF = I(Z;X)/min(H(Z),H(X))
