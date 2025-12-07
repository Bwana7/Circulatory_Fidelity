# Circulatory Fidelity

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A Prior Predictive Diagnostic for Mean-Field Variational Inference**

## Overview

Circulatory Fidelity (CF) quantifies structural dependencies in hierarchical models that mean-field variational inference discards. Unlike post-hoc diagnostics, CF is computed *before* inference—directly from the generative model.

$$\text{CF}(z, x) = \frac{I(z; x)}{\min(H(z), H(x))} \in [0, 1]$$

## Key Results

| Model | r | Effect |
|-------|---|--------|
| **HGF** | +0.39 | High CF → MF fails (5.2× worse) |
| **HLM** | -0.72 | Low CF → No-pooling fails (1.8× worse) |

## Repository Structure

```
├── julia/                      # Julia implementation (primary)
│   ├── src/CirculatoryFidelity.jl
│   ├── demo.jl
│   └── Project.toml
├── python/                     # Python implementation
├── paper/                      # LaTeX manuscript (9 pages)
├── figures/                    # Publication figures
└── simulations/                # Empirical results (CSV)
```

## Quick Start (Julia)

```julia
using CirculatoryFidelity

# HGF: volatility-state coupling
params = HGFParams(coupling=0.8)
sim = simulate_hgf(params)
cf = compute_cf_hgf(sim)  # High → MF inappropriate

# HLM: group effect reliability
params = HLMParams(τ=0.3, σ=1.0)
cf = compute_cf_hlm(params)  # Low → partial pooling needed
```

## Workflow

```
1. Specify generative model
2. Simulate from prior predictive
3. Compute CF
4. CF indicates concern → use structured VI or MCMC
```

## Citation

```bibtex
@article{lowry2025circulatory,
  title={Circulatory Fidelity: A Prior Predictive Diagnostic 
         for Mean-Field Variational Inference},
  author={Lowry, Aaron},
  year={2025}
}
```

## License

MIT
