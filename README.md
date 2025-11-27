Circulatory Fidelity (CF) in Active Inference
(https://zenodo.org/badge/doi/placeholder.svg)](http://dx.doi.org/placeholder)

This repository contains the RxInfer.jl implementation of the Circulatory Fidelity (CF) agent described in Lowry (2025). The model introduces a Structured Variational Approximation to the Hierarchical Gaussian Filter (HGF) to prevent chaotic bifurcations in high-volatility environments.

Repository Contents
src/: Core model logic implementing the Factor Graph.

scripts/: Reproduction scripts for all figures in the paper.

data/: Placeholders for real-world datasets (see Data Availability).

Reproducing the Figures
To generate Figures 2, 3, and 4 (Stochastic Switching, Pharmacology, Propofol), run:julia julia scripts/generate_figures.jl


## Validation Against Real-World Data

We validate the CF model against two standard Computational Psychiatry datasets. 

### 1. The COBRE Dataset (Schizophrenia)
**Hypothesis:** Patients with Schizophrenia (Positive Symptoms) will exhibit higher `gamma_cf` (Rigidity) estimates than controls during reversal learning tasks.
*   **Source:**(https://openneuro.org/datasets/ds001168)
*   **Protocol:** Extract time-series from Dorsolateral Prefrontal Cortex (DLPFC). Fit the `hgf_agent` to the extracted BOLD signal volatility.
*   **Run Benchmark:** `julia scripts/benchmark_real_data.jl --dataset cobre`

### 2. Propofol Anaesthesia Dataset
**Hypothesis:** Deep sedation is characterized by a factorization of the joint posterior $q(z, \gamma)$, measurable as a drop in covariance between hierarchical levels.
*   **Source:** [OpenNeuro ds003352](https://openneuro.org/datasets/ds003352)
*   **Run Benchmark:** `julia scripts/benchmark_real_data.jl --dataset propofol`

## Requirements
*   Julia 1.9+
*   RxInfer.jl
*   Plots.jl
