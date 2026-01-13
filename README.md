# Circulatory Fidelity

**A Relational Theory of Information Flow in Hierarchical Models**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18121821.svg)](https://doi.org/10.5281/zenodo.18121821)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org/)
[![Julia](https://img.shields.io/badge/Julia-1.9+-purple.svg)](https://julialang.org/)

## Overview

Circulatory Fidelity (CF) is a diagnostic framework for predicting when mean-field variational inference (MFVI) will fail in hierarchical Bayesian models. The framework is grounded in the precision matrix decomposition Λ = D + R, which separates *nodal* structure (D, retained by MFVI) from *relational* structure (R, discarded by MFVI).

The primary diagnostic metric is **Inference Coupling (IC)**, which quantifies the information-theoretic cost of discarding relational structure:

```
IC = |ρ|  (for Gaussian pairs)
```

**Key insight**: IC predicts MFVI failure *before* running expensive inference—directly from prior predictive samples or model parameters.

### Resources

- **GitHub**: https://github.com/Bwana7/Circulatory_Fidelity
- **Zenodo**: https://zenodo.org/records/18121821
- **Contact**: circulatoryfidelity@gmail.com

---

## Theoretical Foundation

### The Precision Matrix Decomposition

For a multivariate Gaussian with precision matrix Λ:

```
Λ = D + R
```

- **D** (diagonal): Nodal structure — retained by MFVI
- **R** (off-diagonal): Relational structure — discarded by MFVI

MFVI approximates the true posterior p(θ|y) with a factorized distribution q(θ) = ∏ᵢ qᵢ(θᵢ), effectively keeping D but discarding R. When R carries substantial information, this approximation fails.

### Why IC Works

Building on the classical result that Gaussian mutual information depends only on correlation—not on marginal variances (Gel'fand & Yaglom, 1959):

```
I(Z;X) = -½ log(1 - ρ²)
```

IC inherits this property: **MFVI failure depends on relational structure (IC), not nodal structure (marginal variances)**. The companion Balance Factor (B) characterizes architectural asymmetry but adds ΔR² ≈ 0 for predicting MFVI failure—an empirical confirmation of the theoretical invariance.

### Connection to Linfoot Correlation

IC coincides with the Linfoot informational correlation (Linfoot, 1957) for Gaussian systems, inheriting desirable properties:
- Coordinate invariance
- Bounded on [0, 1]
- Extension to non-Gaussian distributions via copula estimation

---

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

---

## Key Concepts

### Model Type Matters: The Dependency Asymmetry

A key theoretical contribution is recognizing that **high IC has opposite implications** depending on the dependency structure:

| Model Type | Dependency Axis | High IC Means | Recommendation |
|------------|-----------------|---------------|----------------|
| **Filtering** (SVF, state-space) | Vertical (layers) | Constitutive coupling—MFVI will fail | Use structured VI |
| **Pooling** (HLM, random effects) | Horizontal (groups) | Inductive coupling—signal reliable | No-pooling acceptable |

This **Dependency Asymmetry** is resolved by a novel taxonomy distinguishing:
- **Constitutive coupling**: Variables co-determined by shared generative mechanism (filtering models)
- **Inductive coupling**: Variables co-vary due to shared statistical regularity (pooling models)

### Recommended Thresholds

| IC Range | Coupling Regime | Interpretation |
|----------|-----------------|----------------|
| < 0.25 | Negligible | MFVI safe |
| 0.25–0.35 | Weak | MFVI likely acceptable |
| 0.35–0.55 | Moderate | Caution warranted |
| 0.55–0.70 | Strong | Consider structured inference |
| > 0.70 | Very strong | Structured inference required |

**Note**: For specific model classes (e.g., SVF threshold IC = 0.10, HLM threshold varies with failure definition), use domain-specific calibration from the validation studies.

---

## The Two-Stage Diagnostic Protocol

### Detecting Synergistic Dependencies

Standard pairwise IC can miss **synergistic** dependencies where information emerges only from joint variable configurations (e.g., XOR functions). The **Computational Synergy Principle** (Theorem 1 in manuscript) establishes:

> Synergistic dependencies arise when the generative function is affine over GF(2), connecting Siegenthaler's correlation immunity from cryptography to information-theoretic synergy.

The **Two-Stage Protocol** addresses this:

```python
from circulatory_fidelity import two_stage_diagnostic

result = two_stage_diagnostic(z1, z2, x)

# Stage 1: Pairwise IC
print(f"Pairwise IC(z1, x): {result['ic_z1_x']:.3f}")
print(f"Pairwise IC(z2, x): {result['ic_z2_x']:.3f}")

# Stage 2: Interaction IC (only if Stage 1 shows low IC)
print(f"Interaction IC(z1·z2, x): {result['ic_interaction']:.3f}")

# Interpretation
if result['synergy_detected']:
    print("⚠️ XOR-type algebraic structure detected—MFVI inappropriate")
```

**Key signature**: Pairwise IC ≈ 0 combined with interaction IC > 0 **positively identifies** XOR-type algebraic structure.

---

## The Proximal Dominance Principle

For deep hierarchies (L ≥ 3 layers), the **Proximal Dominance Principle** (formalizing documented observations in the VAE literature) provides dramatic diagnostic simplification:

> MFVI failure is determined by coupling in the layer nearest observations. Distal coupling causes **exactly 1.0×** degradation when proximal coupling is absent (mathematically guaranteed), but acts as a **force multiplier** (1.01–2.26×) when proximal coupling is present.

**Practical implication**: IC analysis of only the proximal layer suffices, reducing diagnostic complexity from O(L²) to O(1).

```python
# For a 3-layer model: z³ → z² → z¹ → y
# Only need to check proximal coupling (z¹, y)
ic_proximal, se = inference_coupling(z1_samples, y_samples)

if ic_proximal > threshold:
    print("Proximal coupling detected—check distal layers for amplification")
else:
    print("Proximal coupling absent—MFVI safe regardless of distal structure")
```

---

## Time Series: The Maximal Coupling Rule

For non-stationary time series with potential regime changes:

```python
from circulatory_fidelity import windowed_ic

# Compute IC in rolling windows
result = windowed_ic(z_series, x_series, window_size=50)

print(f"IC_max = {result['ic_max']:.3f}")
print(f"IC_mean = {result['ic_mean']:.3f}")
```

**The Maximal Coupling Rule**: MFVI suitability depends on `IC_max`, not the global average. A single high-IC episode (e.g., volatility spike) can invalidate mean-field approximations for the entire trajectory.

---

## Installation

### Python

```bash
pip install numpy scipy
# Clone repository and import circulatory_fidelity.py
```

Or using pyproject.toml:
```bash
pip install -e .
```

### Julia

```julia
using Pkg
Pkg.add(url="https://github.com/Bwana7/Circulatory_Fidelity")
```

---

## Estimation Methods

### Copula-Based (Recommended Default)

```python
ic, se = inference_coupling(x, y, method='copula')  # default
```

**Algorithm**:
1. Rank-transform to uniform marginals
2. Apply probit (inverse normal CDF)
3. Compute Pearson correlation
4. IC = |ρ|

**Key insight**: The copula method is **exact for Gaussian data** AND provides **conservative estimates** for non-Gaussian data. This enables a **unified workflow** without needing to verify distributional assumptions.

### Pearson (Verified Gaussians Only)

```python
ic, se = inference_coupling(x, y, method='pearson')
```

### KSG (For Validation)

```python
ic, se = inference_coupling(x, y, method='ksg')
```

**⚠️ Warning**: KSG exhibits 30–45% negative bias. Use copula as primary method; KSG for validation only.

---

## Key Results

### Validation Summary (N > 32,000 simulations)

| Model | Metric | Correlation with IC | N |
|-------|--------|---------------------|---|
| SVF (Filtering) | MSE Ratio | r = 0.83 | 8,000 |
| HLM (Pooling) | MSE Ratio | r = −0.76 | 8,000 |
| Three-Layer Hierarchy | MSE Ratio | r = 0.89 | 16,000 |
| SVF | Log-Likelihood Gap | r = 0.86 | 900 |

### Proximal Dominance Quantification

For three-layer hierarchies with fully factorized MFVI:

| Coupling Configuration | MSE Degradation |
|------------------------|-----------------|
| Proximal only (κ₂₁ > 0, κ₃₂ = 0) | Up to 40× |
| Distal only (κ₂₁ = 0, κ₃₂ > 0) | Exactly 1.0× |
| Both present | Distal amplifies by 1.01–2.26× |

### Copula Estimation Accuracy

| True ρ | Copula IC | Error |
|--------|-----------|-------|
| 0.30 | 0.2999 | < 0.001 |
| 0.50 | 0.4998 | < 0.001 |
| 0.70 | 0.6999 | < 0.001 |
| 0.90 | 0.8999 | < 0.001 |

---

## Citation

```bibtex
@software{lowry_circulatory_fidelity_2026,
  author       = {Lowry, Aaron},
  title        = {Circulatory Fidelity: A Relational Theory of Information
	Flow in Hierarchical Models},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v1.1.0},
  doi          = {10.5281/zenodo.18121821},
  url          = {https://doi.org/10.5281/zenodo.18121821}
}
```

---

## Version History

### v1.1.0 (January 2026) — Current Release

**Major theoretical restructuring** to ensure proper attribution and academic rigor:

#### Terminology Changes
- **Primary metric renamed**: "Circulatory Fidelity" (CF) → "Inference Coupling" (IC)
- **Rationale**: IC = |ρ| is the Linfoot correlation, a standard metric; "CF" now refers to the overall diagnostic framework

#### Attribution Corrections
- **Relational Invariance**: Now properly attributed as application of Gel'fand & Yaglom (1959), not claimed as novel theorem
- **Linfoot Equivalence**: Removed from contributions; properly attributed to Linfoot (1957)
- **Proximal Dominance**: Acknowledges prior observations in VAE literature (Havtorn et al., 2021; Sønderby et al., 2016; Zhao et al., 2017); novel contribution is the O(L²) → O(1) complexity reduction claim

#### Genuine Novel Contributions (Retained)
1. **Computational Synergy Principle** — Novel interdisciplinary bridge connecting Siegenthaler's correlation immunity (cryptography) to information-theoretic synergy (PID)
2. **Dependency Asymmetry Taxonomy** — Novel synthesis distinguishing constitutive vs. inductive coupling
3. **Proximal Dominance Formalization** — Novel diagnostic complexity claim
4. **Maximal Coupling Rule** — Novel time-series diagnostic principle

#### Technical Improvements
- **Copula estimation**: Now recommended as unified default (exact for Gaussians, conservative for non-Gaussians)
- **Two-stage protocol**: Formalized for detecting synergistic dependencies
- **Comprehensive validation**: 32,000+ simulations across three model classes
- **DOIs added**: All key references now include DOI numbers

#### New Content
- Fluid–crystalline distinction in information topology (Rule 30 vs Rule 150)
- Broader applicability discussion (discrete Boolean domains, biological systems)
- IC vs PSIS-k̂ comparison (Appendix)

---

### v1.0.0 (December 2024) — Initial Release

- Original formulation: CF = I(Z;X) / min(H(Z), H(X))
- Entropy-normalized mutual information approach
- Initial validation on stochastic volatility models

#### Why v1.0 Was Superseded
The original entropy normalization was theoretically motivated but practically unnecessary:
1. For Gaussians, marginal entropies cancel in the normalization
2. The Linfoot correlation |ρ| is sufficient and more interpretable
3. Copula estimation enables unified workflow across distributions

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

- **Author**: Aaron Lowry
- **Email**: circulatoryfidelity@gmail.com
- **Repository**: https://github.com/Bwana7/Circulatory_Fidelity
