# ═══════════════════════════════════════════════════════════════════════════════
# crucial_experiment_simulations.jl - Simulation-Based Predictions for Crucial Experiment
# ═══════════════════════════════════════════════════════════════════════════════

"""
This script generates quantitative predictions for the crucial experiment
testing CF theory's unique prediction: period-doubling signatures in belief dynamics.

We simulate the HGF under varying approximation quality (CF levels) and analyze
the spectral structure of trial-by-trial learning rates.
"""

using Random
using Statistics
using FFTW

# ─────────────────────────────────────────────────────────────────────────────
# Simulation Parameters (matching experimental design)
# ─────────────────────────────────────────────────────────────────────────────

const N_TRIALS = 400
const N_REVERSALS = 10
const REVERSAL_INTERVAL = 40  # trials between reversals (approximate)
const P_CORRECT = 0.70        # reward probability for correct choice
const N_SIMULATIONS = 1000    # simulations per condition

# HGF parameters
const κ = 1.0
const ω = -2.0
const π_u = 10.0

# ─────────────────────────────────────────────────────────────────────────────
# Task Generation
# ─────────────────────────────────────────────────────────────────────────────

"""
    generate_task(n_trials, n_reversals; seed=nothing)

Generate a volatile reversal learning task.

Returns:
- correct_side: Vector of correct responses (1 or 2)
- outcomes: Vector of outcomes given optimal choice
"""
function generate_task(n_trials::Int, n_reversals::Int; seed=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
    
    # Generate reversal points
    min_interval = 30
    max_interval = 50
    reversal_points = cumsum(rand(min_interval:max_interval, n_reversals))
    reversal_points = reversal_points[reversal_points .< n_trials]
    
    # Generate correct side
    correct_side = ones(Int, n_trials)
    current_side = 1
    for t in 1:n_trials
        if t in reversal_points
            current_side = 3 - current_side  # flip between 1 and 2
        end
        correct_side[t] = current_side
    end
    
    # Generate outcomes (for correct choices)
    outcomes = rand(n_trials) .< P_CORRECT
    
    return correct_side, outcomes, reversal_points
end

# ─────────────────────────────────────────────────────────────────────────────
# HGF Simulation
# ─────────────────────────────────────────────────────────────────────────────

"""
    simulate_hgf(outcomes, ϑ, cf_level; seed=nothing)

Simulate HGF inference on a sequence of outcomes.

Arguments:
- outcomes: Vector of observations (0 or 1)
- ϑ: Volatility of volatility parameter
- cf_level: 0.0 = pure mean-field, 1.0 = full structured

Returns:
- μ_x: Posterior means at level 1
- μ_z: Posterior means at level 2
- learning_rates: Trial-by-trial learning rates
"""
function simulate_hgf(outcomes::Vector, ϑ::Float64, cf_level::Float64; seed=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
    
    T = length(outcomes)
    
    # Initialize
    μ_x = zeros(T)
    μ_z = zeros(T)
    learning_rates = zeros(T)
    
    μ_x_current = 0.0
    μ_z_current = 0.0
    σ²_x = 1.0
    σ²_z = 1.0
    
    for t in 1:T
        y_t = outcomes[t] ? 1.0 : -1.0  # Convert to ±1
        
        # Expected precision at level 1
        π_x_hat = exp(-κ * μ_z_current - ω)
        
        # Prediction error
        δ_x = y_t - μ_x_current
        
        # Learning rate at level 1
        α_x = π_x_hat / (π_x_hat + π_u)
        learning_rates[t] = α_x
        
        # Level 1 update
        μ_x_new = μ_x_current + α_x * δ_x
        
        # Level 2 update (volatility)
        # Gain for volatility update
        K_z = (κ / 2) * π_x_hat / (ϑ + (κ^2 / 2) * π_x_hat)
        
        # Volatility prediction error
        ν = δ_x^2 * π_x_hat - 1
        
        # Mean-field update
        Δμ_z_mf = K_z * ν
        
        # Structured update includes damping from cross-level correlation
        # The damping factor represents the stabilizing effect of CF > 0
        # At cf_level = 1, full damping; at cf_level = 0, no damping
        damping = 1.0 - 0.5 * cf_level * (1.0 - exp(-abs(ν)))
        Δμ_z_struct = K_z * ν * damping
        
        # Interpolate between mean-field and structured
        Δμ_z = (1 - cf_level) * Δμ_z_mf + cf_level * Δμ_z_struct
        
        μ_z_new = μ_z_current + Δμ_z
        
        # Store
        μ_x[t] = μ_x_new
        μ_z[t] = μ_z_new
        
        # Update for next iteration
        μ_x_current = μ_x_new
        μ_z_current = μ_z_new
    end
    
    return μ_x, μ_z, learning_rates
end

# ─────────────────────────────────────────────────────────────────────────────
# Spectral Analysis
# ─────────────────────────────────────────────────────────────────────────────

"""
    compute_psd(x; window_size=50, overlap=0.5)

Compute power spectral density using Welch's method.

Returns:
- freqs: Frequency values (cycles/trial)
- psd: Power spectral density
"""
function compute_psd(x::Vector{Float64}; window_size::Int=50, overlap::Float64=0.5)
    n = length(x)
    
    # Detrend
    x_detrend = x .- mean(x)
    
    # Welch's method
    step = Int(round(window_size * (1 - overlap)))
    n_windows = (n - window_size) ÷ step + 1
    
    # Hanning window
    window = 0.5 .* (1 .- cos.(2π .* (0:window_size-1) ./ (window_size - 1)))
    
    # Accumulate periodograms
    psd_sum = zeros(window_size ÷ 2 + 1)
    
    for i in 1:n_windows
        start_idx = (i - 1) * step + 1
        segment = x_detrend[start_idx:start_idx+window_size-1] .* window
        
        # FFT
        fft_result = fft(segment)
        periodogram = abs.(fft_result[1:window_size÷2+1]).^2
        
        psd_sum .+= periodogram
    end
    
    psd = psd_sum ./ n_windows
    
    # Normalize
    psd = psd ./ sum(psd)
    
    # Frequencies
    freqs = (0:window_size÷2) ./ window_size
    
    return freqs, psd
end

"""
    compute_band_power(freqs, psd, band)

Compute power in a frequency band.
"""
function compute_band_power(freqs::Vector{Float64}, psd::Vector{Float64}, 
                            band::Tuple{Float64, Float64})
    mask = (freqs .>= band[1]) .& (freqs .<= band[2])
    return sum(psd[mask])
end

"""
    find_peaks(freqs, psd; min_prominence=0.01)

Find peaks in PSD.
"""
function find_peaks(freqs::Vector{Float64}, psd::Vector{Float64}; 
                    min_prominence::Float64=0.01)
    peaks = Int[]
    for i in 2:length(psd)-1
        if psd[i] > psd[i-1] && psd[i] > psd[i+1]
            # Check prominence
            left_min = minimum(psd[1:i])
            right_min = minimum(psd[i:end])
            prominence = psd[i] - max(left_min, right_min)
            if prominence > min_prominence
                push!(peaks, i)
            end
        end
    end
    return peaks
end

# ─────────────────────────────────────────────────────────────────────────────
# Autocorrelation Analysis
# ─────────────────────────────────────────────────────────────────────────────

"""
    compute_acf(x, max_lag)

Compute autocorrelation function.
"""
function compute_acf(x::Vector{Float64}, max_lag::Int)
    n = length(x)
    x_centered = x .- mean(x)
    var_x = var(x)
    
    acf = zeros(max_lag + 1)
    for lag in 0:max_lag
        acf[lag + 1] = sum(x_centered[1:n-lag] .* x_centered[lag+1:n]) / ((n - lag) * var_x)
    end
    
    return acf
end

# ─────────────────────────────────────────────────────────────────────────────
# Main Simulation
# ─────────────────────────────────────────────────────────────────────────────

"""
    run_predictions(; n_sims=N_SIMULATIONS, ϑ=0.08, seed=42)

Run simulations to generate predictions for crucial experiment.
"""
function run_predictions(; n_sims::Int=N_SIMULATIONS, ϑ::Float64=0.08, seed::Int=42)
    Random.seed!(seed)
    
    # CF levels to simulate
    cf_levels = [1.0, 0.5, 0.0]  # High, Medium, Low CF
    cf_labels = ["High CF (structured)", "Medium CF", "Low CF (mean-field)"]
    
    # Storage
    results = Dict{String, Dict{String, Vector{Float64}}}()
    
    for (cf, label) in zip(cf_levels, cf_labels)
        band_powers = Float64[]
        acf1_values = Float64[]
        acf2_values = Float64[]
        peak_freqs = Vector{Float64}[]
        
        for sim in 1:n_sims
            # Generate task
            _, outcomes_bool, _ = generate_task(N_TRIALS, N_REVERSALS; seed=seed+sim)
            outcomes = Float64.(outcomes_bool)
            
            # Simulate HGF
            _, _, learning_rates = simulate_hgf(outcomes, ϑ, cf; seed=seed+sim+1000)
            
            # Spectral analysis
            freqs, psd = compute_psd(learning_rates)
            bp = compute_band_power(freqs, psd, (0.02, 0.10))
            push!(band_powers, bp)
            
            # Peak detection
            peaks = find_peaks(freqs, psd)
            if length(peaks) > 0
                push!(peak_freqs, freqs[peaks])
            else
                push!(peak_freqs, Float64[])
            end
            
            # Autocorrelation
            acf = compute_acf(learning_rates, 5)
            push!(acf1_values, acf[2])  # lag 1
            push!(acf2_values, acf[3])  # lag 2
        end
        
        results[label] = Dict(
            "band_power" => band_powers,
            "acf1" => acf1_values,
            "acf2" => acf2_values,
            "peak_freqs" => vcat(peak_freqs...)
        )
    end
    
    return results
end

"""
    print_predictions(results)

Print formatted predictions table.
"""
function print_predictions(results::Dict)
    println("\n" * "="^70)
    println("SIMULATION-BASED PREDICTIONS FOR CRUCIAL EXPERIMENT")
    println("="^70)
    
    println("\nPrimary Outcome: Band Power in [0.02, 0.10] cycles/trial")
    println("-"^50)
    println("Condition\t\t\tMean ± SD")
    println("-"^50)
    
    for label in ["High CF (structured)", "Medium CF", "Low CF (mean-field)"]
        bp = results[label]["band_power"]
        println("$label\t\t$(round(mean(bp), digits=3)) ± $(round(std(bp), digits=3))")
    end
    
    println("\nSecondary Outcome: Autocorrelation Structure")
    println("-"^50)
    println("Condition\t\t\tACF(1)\t\tACF(2)")
    println("-"^50)
    
    for label in ["High CF (structured)", "Medium CF", "Low CF (mean-field)"]
        acf1 = results[label]["acf1"]
        acf2 = results[label]["acf2"]
        println("$label\t$(round(mean(acf1), digits=3)) ± $(round(std(acf1), digits=3))\t$(round(mean(acf2), digits=3)) ± $(round(std(acf2), digits=3))")
    end
    
    # Effect size calculation
    high_cf = results["High CF (structured)"]["band_power"]
    low_cf = results["Low CF (mean-field)"]["band_power"]
    pooled_sd = sqrt((var(high_cf) + var(low_cf)) / 2)
    cohens_d = (mean(low_cf) - mean(high_cf)) / pooled_sd
    
    println("\nEffect Size (Low CF vs High CF)")
    println("-"^50)
    println("Cohen's d = $(round(cohens_d, digits=2))")
    
    println("\n" * "="^70)
end

# ─────────────────────────────────────────────────────────────────────────────
# Run if executed directly
# ─────────────────────────────────────────────────────────────────────────────

if abspath(PROGRAM_FILE) == @__FILE__
    println("Running crucial experiment simulations...")
    results = run_predictions(n_sims=1000)
    print_predictions(results)
end
