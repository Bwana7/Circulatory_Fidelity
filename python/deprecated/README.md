# Deprecated Development Files

This directory contains development versions of scripts created during the PSIS validation study. These files are retained for reference but are **not part of the official implementation**.

## Authoritative Files

The following files in the parent directory are the official implementations:

| File | Purpose |
|------|---------|
| `circulatory_fidelity.py` | Main CF computation module |
| `svf_psis_validation.py` | PSIS validation study (N=900 simulations) |
| `hierarchical_vae_dsprites.py` | dSprites Proximal Dominance validation |

## Development History

These deprecated files document the iterative development of the PSIS validation:

- `compute_psis_khat.py`, `v2.py`, `v3.py` - Initial attempts at PSIS-k̂ computation
- `compute_psis_final.py` - Intermediate version
- `compute_psis_honest.py` - Discovery that PSIS-k̂ is inappropriate for Gaussian posteriors
- `psis_validation.py`, `v2.py` - Early validation attempts

## Key Finding

The development process revealed that **PSIS-k̂ is not appropriate for Gaussian posteriors** because importance weights between Gaussians follow a log-normal distribution with light tails, yielding consistently negative k̂ values. This led to the manuscript revision replacing PSIS-k̂ with log-likelihood gap as the primary diagnostic.

See Section 7.5 of the manuscript for details.
