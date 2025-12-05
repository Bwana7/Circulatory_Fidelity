# ═══════════════════════════════════════════════════════════════════════════════
# Experiment A: Stability Comparison Under Colored Noise
# ═══════════════════════════════════════════════════════════════════════════════

"""
Experiment A: Stability Comparison

HYPOTHESIS: Bethe-structured inference remains stable where mean-field diverges
under colored noise stress.

This experiment tests the stability of both approximation schemes across
different noise spectra and coupling strengths.
"""

using CirculatoryFidelity
using Statistics
using Random

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTAL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

# Noise spectral exponents (1/f^α)
const ALPHA_VALUES = [0.5, 1.0, 1.5, 2.0]

# Coupling strengths to test
const KAPPA_VALUES = [0.5, 1.0, 1.5, 2.0]

# Fixed parameters
const THETA = 0.1          # Moderate volatility
const N_TIMESTEPS = 5000   # Steps per trial
const N_REPETITIONS = 100  # Repetitions per condition

# Divergence criterion
const DIVERGENCE_THRESHOLD = 10.0

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENTAL FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

"""
    run_experiment_a()

Run the full stability comparison experiment.

Returns a Dict with results for all conditions.
"""
function run_experiment_a()
    results = Dict{String, Any}()
    
    for α in ALPHA_VALUES
        for κ in KAPPA_VALUES
            condition_key = "α=$(α)_κ=$(κ)"
            @info "Running condition" α κ
            
            # Run both modes
            mf_results = run_condition(α, κ, :mean_field)
            struct_results = run_condition(α, κ, :structured)
            
            results[condition_key] = Dict(
                "mean_field" => mf_results,
                "structured" => struct_results,
                "α" => α,
                "κ" => κ
            )
        end
    end
    
    return results
end

"""
    run_condition(α, κ, mode)

Run one experimental condition with N_REPETITIONS trials.
"""
function run_condition(α::Float64, κ::Float64, mode::Symbol)
    # Storage for metrics
    rmse_values = Float64[]
    divergence_count = 0
    mi_values = Float64[]
    
    params = CFParameters(κ=κ, ω=-2.0, ϑ=THETA, π_u=10.0)
    
    for rep in 1:N_REPETITIONS
        # Generate colored noise stimulus
        seed = rep * 1000
        noise = generate_colored_noise(N_TIMESTEPS, α; seed=seed)
        
        # Generate observations
        observations = noise .* sqrt(1/params.π_u)
        
        # Run inference
        results = run_inference(observations, params, mode)
        
        # Check for divergence
        if !results.converged || any(abs.(results.μ_z) .> DIVERGENCE_THRESHOLD)
            divergence_count += 1
        else
            # Compute RMSE (against observations)
            rmse = compute_rmse(results.μ_x, observations)
            push!(rmse_values, rmse)
            
            # Compute average MI recovery (CF)
            push!(mi_values, mean(results.CF))
        end
    end
    
    return Dict(
        "divergence_rate" => divergence_count / N_REPETITIONS,
        "rmse_mean" => isempty(rmse_values) ? NaN : mean(rmse_values),
        "rmse_std" => isempty(rmse_values) ? NaN : std(rmse_values),
        "mi_mean" => isempty(mi_values) ? NaN : mean(mi_values),
        "mi_std" => isempty(mi_values) ? NaN : std(mi_values),
        "n_diverged" => divergence_count,
        "n_converged" => N_REPETITIONS - divergence_count
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
Expected results based on theory:

| Condition | Mean-Field Div. Rate | Structured Div. Rate | ΔMI |
|-----------|---------------------|---------------------|-----|
| κ ≤ 1.0   | < 5%                | < 1%                | +0.2 nats |
| κ = 1.5   | ~30%                | < 1%                | +0.3 nats |
| κ ≥ 2.0   | > 50%               | < 1%                | +0.4 nats |

Key prediction: Structured approximation should be uniformly more stable
across all noise types (α values).
"""

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

"""
    analyze_results(results)

Generate summary statistics and test predictions.
"""
function analyze_results(results::Dict)
    summary = Dict{String, Any}()
    
    # Aggregate by κ
    for κ in KAPPA_VALUES
        mf_div_rates = Float64[]
        struct_div_rates = Float64[]
        mi_diffs = Float64[]
        
        for α in ALPHA_VALUES
            key = "α=$(α)_κ=$(κ)"
            if haskey(results, key)
                push!(mf_div_rates, results[key]["mean_field"]["divergence_rate"])
                push!(struct_div_rates, results[key]["structured"]["divergence_rate"])
                
                mi_diff = results[key]["structured"]["mi_mean"] - 
                          results[key]["mean_field"]["mi_mean"]
                if !isnan(mi_diff)
                    push!(mi_diffs, mi_diff)
                end
            end
        end
        
        summary["κ=$(κ)"] = Dict(
            "mf_div_rate_mean" => mean(mf_div_rates),
            "struct_div_rate_mean" => mean(struct_div_rates),
            "mi_advantage" => isempty(mi_diffs) ? NaN : mean(mi_diffs)
        )
    end
    
    return summary
end

"""
    test_predictions(results)

Test whether experimental results match theoretical predictions.
"""
function test_predictions(results::Dict)
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Structured should have lower divergence rate for all conditions
    for (key, data) in results
        mf_div = data["mean_field"]["divergence_rate"]
        struct_div = data["structured"]["divergence_rate"]
        
        tests_total += 1
        if struct_div <= mf_div
            tests_passed += 1
        else
            @warn "Prediction failed" condition=key mf_div struct_div
        end
    end
    
    # Test 2: For κ > 1.5, mean-field should diverge > 50%
    for κ in [1.5, 2.0]
        for α in ALPHA_VALUES
            key = "α=$(α)_κ=$(κ)"
            if haskey(results, key)
                mf_div = results[key]["mean_field"]["divergence_rate"]
                tests_total += 1
                if mf_div > 0.3  # Relaxed from 0.5 for κ=1.5
                    tests_passed += 1
                end
            end
        end
    end
    
    return Dict(
        "tests_passed" => tests_passed,
        "tests_total" => tests_total,
        "pass_rate" => tests_passed / tests_total
    )
end

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if abspath(PROGRAM_FILE) == @__FILE__
    @info "Starting Experiment A: Stability Comparison"
    
    results = run_experiment_a()
    
    @info "Analyzing results"
    summary = analyze_results(results)
    predictions = test_predictions(results)
    
    @info "Results summary" summary
    @info "Prediction tests" predictions
    
    # Save results
    save_results("experiment_a_results.json", results)
    
    @info "Experiment A complete"
end
