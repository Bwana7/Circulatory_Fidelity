#!/usr/bin/env julia
"""
    demo.jl
    
    Demonstration of Circulatory Fidelity as a prior predictive diagnostic.
    
    This script reproduces the main results from the paper:
    - HGF: High CF predicts mean-field failure (r = 0.39)
    - HLM: Low CF predicts no-pooling failure (r = -0.72)
    
    Usage:
        julia --project=. demo.jl
    
    Output:
        - Console summary statistics
        - CSV files with full results
        - (Optional) Plots if Plots.jl is available
"""

using CirculatoryFidelity
using Statistics
using DataFrames
using Printf

println("="^70)
println("CIRCULATORY FIDELITY: DEMONSTRATION")
println("="^70)

#=============================================================================
    PART 1: Core CF Examples
=============================================================================#

println("\n[1] Core CF Computation")
println("-"^40)

# CF for different correlations
for ρ in [0.0, 0.3, 0.5, 0.7, 0.9]
    cf = CF(ρ, 1.0, 1.0)
    mi = mutual_information(ρ)
    @printf("  ρ = %.1f → MI = %.3f nats, CF = %.3f\n", ρ, mi, cf)
end

#=============================================================================
    PART 2: HGF Parameter Sweep
=============================================================================#

println("\n[2] HGF Parameter Sweep")
println("-"^40)

hgf_results = run_hgf_sweep(
    coupling_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0],
    n_sims = 100,
    seed = 42
)

# Summary by coupling
println("\nResults by coupling strength:")
@printf("  %-10s %-10s %-15s\n", "Coupling", "Mean CF", "Mean MSE Ratio")
@printf("  %-10s %-10s %-15s\n", "-"^8, "-"^8, "-"^13)

for κ in sort(unique(hgf_results.coupling))
    sub = hgf_results[hgf_results.coupling .== κ, :]
    @printf("  %-10.1f %-10.3f %-15.3f\n", κ, mean(sub.cf), mean(sub.mse_ratio))
end

# Correlation
valid_hgf = dropmissing(hgf_results)
r_hgf = cor(valid_hgf.cf, valid_hgf.mse_ratio)
@printf("\nCF-MSE_ratio correlation: r = %.3f\n", r_hgf)

# Median split analysis
med_cf = median(valid_hgf.cf)
low_cf = valid_hgf[valid_hgf.cf .< med_cf, :]
high_cf = valid_hgf[valid_hgf.cf .>= med_cf, :]
@printf("  Low-CF MSE ratio:  %.3f\n", mean(low_cf.mse_ratio))
@printf("  High-CF MSE ratio: %.3f (%.1f× worse)\n", 
        mean(high_cf.mse_ratio), mean(high_cf.mse_ratio) / mean(low_cf.mse_ratio))

#=============================================================================
    PART 3: HLM Parameter Sweep
=============================================================================#

println("\n[3] HLM Parameter Sweep")
println("-"^40)

hlm_results = run_hlm_sweep(
    τ_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0],
    n_sims = 100,
    seed = 42
)

# Summary by τ
println("\nResults by between-group SD (τ):")
@printf("  %-8s %-8s %-10s %-15s\n", "τ", "ICC", "CF", "MSE Ratio")
@printf("  %-8s %-8s %-10s %-15s\n", "-"^6, "-"^6, "-"^8, "-"^13)

for τ in sort(unique(hlm_results.τ))
    sub = hlm_results[hlm_results.τ .== τ, :]
    @printf("  %-8.1f %-8.2f %-10.3f %-15.3f\n", 
            τ, mean(sub.icc), mean(sub.cf), mean(sub.mse_ratio))
end

# Correlation
valid_hlm = dropmissing(hlm_results)
r_hlm = cor(valid_hlm.cf, valid_hlm.mse_ratio)
@printf("\nCF-MSE_ratio correlation: r = %.3f\n", r_hlm)

# Median split
med_cf_hlm = median(valid_hlm.cf)
low_cf_hlm = valid_hlm[valid_hlm.cf .< med_cf_hlm, :]
high_cf_hlm = valid_hlm[valid_hlm.cf .>= med_cf_hlm, :]
@printf("  Low-CF MSE ratio:  %.3f (%.1f× worse)\n", 
        mean(low_cf_hlm.mse_ratio), mean(low_cf_hlm.mse_ratio) / mean(high_cf_hlm.mse_ratio))
@printf("  High-CF MSE ratio: %.3f\n", mean(high_cf_hlm.mse_ratio))

#=============================================================================
    PART 4: Summary
=============================================================================#

println("\n" * "="^70)
println("SUMMARY")
println("="^70)

println("""
┌─────────┬──────────┬─────────────────────────────────────────────┐
│ Model   │ r        │ Interpretation                              │
├─────────┼──────────┼─────────────────────────────────────────────┤
│ HGF     │ +$(round(r_hgf, digits=2))    │ High CF → MF fails (discards coupling)      │
│ HLM     │ $(round(r_hlm, digits=2))    │ Low CF → No-pooling fails (overfits)        │
└─────────┴──────────┴─────────────────────────────────────────────┘

CF provides a prior predictive diagnostic: compute from model structure
BEFORE fitting data to assess mean-field appropriateness.
""")

#=============================================================================
    PART 5: Save Results
=============================================================================#

using CSV

CSV.write("hgf_results.csv", hgf_results)
CSV.write("hlm_results.csv", hlm_results)
println("Results saved to hgf_results.csv and hlm_results.csv")

println("\nDemo complete.")
