# Circulatory Fidelity

**A Pre-Inference Diagnostic for Mean-Field Variational Inference in Hierarchical Models**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18121821.svg)](https://doi.org/10.5281/zenodo.18121821)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org/)
[![Julia](https://img.shields.io/badge/Julia-1.9+-purple.svg)](https://julialang.org/)

## Overview

**Circulatory Fidelity (CF)** quantifies structural coupling between hierarchical levels of a generative model, enabling assessment of whether mean-field variational inference (MFVI) is appropriate *before* committing computational resources to posterior inference.

The key insight: MFVI assumes independence between latent variables, discarding potentially essential dependencies. CF measures exactly what this factorization assumption discards—using only prior predictive samples—and predicts whether the discarded structure is consequential for inference quality.

### Why "Circulatory Fidelity"?

Information *circulates* bidirectionally in hierarchical models: the likelihood propagates evidence upward while the prior propagates constraints downward. CF measures the *fidelity* of this circulation under approximation. Severing structural coupling (via mean-field factorization) is analogous to ligating a vessel—the consequences depend on whether the coupling is load-bearing or redundant.

## Key Contributions

The manuscript establishes four main results:

1. **Computational Synergy Principle**: Synergistic dependencies arise iff the generative function is non-affine over GF(2). A two-stage diagnostic (pairwise CF ≈ 0 combined with interaction CF > 0) positively identifies XOR-type structure where pairwise methods fail.

2. **Dependency Asymmetry**: High CF predicts MFVI failure in *filtering* models (stochastic volatility, state-space) but hierarchy redundancy in *pooling* models (hierarchical linear models). The same structural property breaks one model class and stabilizes another—resolved by distinguishing *constitutive* from *inductive* coupling.

3. **Proximal Dominance Principle**: In deep hierarchies, only the layer nearest observations matters for inference quality. Distal coupling causes exactly 1.0× degradation (mathematically guaranteed, not approximate), reducing diagnostic complexity from O(L²) to O(1) regardless of depth.

4. **Maximal Coupling Rule**: For non-stationary series, MFVI suitability depends on maximum windowed CF, not the global average.

## Primary Metric: Linfoot Correlation

We adopt the **Linfoot informational correlation** as the primary diagnostic:

```
r_L = √(1 - exp(-2·I(z;x)))
```

This transformation of mutual information provides:
- **Boundedness**: r_L ∈ [0, 1] for all distributions
- **Gaussian calibration**: r_L = |ρ| exactly for bivariate Gaussians
- **Coordinate invariance**: Unchanged under monotonic transformations

Interpretive scale:
| Coupling Regime | r_L | Interpretation |
|-----------------|-----|----------------|
| Negligible | < 0.25 | MFVI safe |
| Weak | 0.25–0.35 | MFVI likely acceptable |
| Moderate | 0.35–0.55 | Caution warranted |
| Strong | > 0.55 | Structured inference recommended |

## Validation

Empirical validation spans 32,000+ simulations across three model classes:

| Model Class | N | Primary Metric | Result |
|-------------|---|----------------|--------|
| Stochastic Volatility (SVF) | 8,000 | MSE prediction | r = 0.84 |
| Stochastic Volatility (SVF) | 900 | Log-likelihood gap | r = 0.86 |
| Hierarchical Linear Models (HLM) | 16,000 | MSE ratio | Dependency Asymmetry confirmed |
| Three-Layer Hierarchies | 16,000 | Proximal vs. distal | Proximal Dominance confirmed |

## Repository Structure

```
Circulatory_Fidelity/
├── README.md                    # This file
├── LICENSE                      # MIT License
│
├── drafts/                      # Manuscript
│   ├── Circulatory_Fidelity_Manuscript.tex
│   ├── Circulatory_Fidelity_Manuscript.pdf
│   └── references.bib
│
├── src/                         # Source code
│   ├── circulatory_fidelity.py  # Python implementation
│   └── CirculatoryFidelity.jl   # Julia implementation
│
├── experiments/                 # Validation notebooks
│   ├── 01_SVF_Case_Study.ipynb
│   ├── 02_HLM_Case_Study.ipynb
│   ├── 03_Deep_Hierarchy_Case_Study.ipynb
│   ├── 04_Copula_NonGaussian_Demo.ipynb
│   ├── 05_Synergy_Higher_Order.ipynb
│   └── 06_CF_LogLik_Validation.ipynb
│
└── data/                        # Validation datasets
    ├── svf_validation.csv
    ├── hlm_validation.csv
    └── three_layer_validation.csv
```

## Quick Start

### Python

```python
from circulatory_fidelity import compute_cf_linfoot, compute_cf_copula

# From samples (copula-based, recommended for non-Gaussian)
cf, se = compute_cf_copula(z_samples, x_samples)
print(f"CF = {cf:.3f} ± {1.96*se:.3f}")

# For Gaussian systems (closed-form)
cf = compute_cf_linfoot(correlation=0.7)
print(f"CF = {cf:.3f}")  # Returns 0.7 (Gaussian calibration)
```

### Julia

```julia
using CirculatoryFidelity

# From samples
cf, se = compute_cf_copula(z_samples, x_samples)

# For analytical ICC (hierarchical models)
cf = cf_from_icc(icc=0.5, n_groups=20, n_per_group=10)
```

## Practical Workflow

1. **Pre-inference**: Compute CF from prior predictive samples
2. **Decision**: 
   - If *filtering model* (SVF, state-space): CF > 0.10 → use structured inference
   - If *pooling model* (HLM): CF < 0.40 → partial pooling essential; CF > 0.40 → no-pooling may suffice
3. **Post-inference**: Validate with PSIS-k̂ (if applicable) or held-out log-likelihood

## Citation

```bibtex
@article{lowry2026circulatory,
  title={Circulatory Fidelity: Quantifying Structural Coupling to 
         Diagnose Mean-Field Failure in Hierarchical Models},
  author={Lowry, Aaron},
  journal={Preprint},
  year={2026},
  doi={10.5281/zenodo.18121821},
  url={https://zenodo.org/records/18121821}
}
```

## Related Work

CF builds on established results from:
- Information theory: Linfoot (1957), Cover & Thomas (2006)
- Normalized MI: Kvålseth (1987), Vinh et al. (2010)
- Variational inference diagnostics: Yao et al. (2018), Vehtari et al. (2017)
- Information geometry: Amari (2016)

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This work benefited from the probabilistic programming ecosystems of Stan, PyMC, NumPyro, and RxInfer.jl.
