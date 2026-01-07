"""
CirculatoryFidelity.jl v1.1

Circulatory Fidelity: A Prior Predictive Diagnostic for Mean-Field Variational Inference

This module provides tools for computing Inference Coupling (IC), the primary 
diagnostic metric for predicting MFVI failure.

    IC = |ρ|  (Linfoot correlation / informational correlation)

For Gaussian pairs: IC = |ρ| = √(1 - exp(-2·I(Z;X)))

Key insight: IC measures how much knowing X reduces uncertainty about Z,
normalized to [0,1]. High IC → MFVI will struggle with the joint posterior.

TERMINOLOGY (v1.1):
    - Circulatory Fidelity (CF) = overall diagnostic framework
    - Inference Coupling (IC) = primary diagnostic metric
    - Historical note: v1.0 used CF = I(Z;X)/min(H(Z),H(X)); the Relational 
      Invariance Theorem proves normalization unnecessary—marginal entropies cancel.

Reference
---------
"Circulatory Fidelity: Quantifying Structural Coupling to Diagnose 
Mean-Field Failure in Hierarchical Models" (2025)

License: MIT
"""
module CirculatoryFidelity

using Statistics
using SpecialFunctions: digamma
using NearestNeighbors
using LinearAlgebra
using Random
using StatsBase: ordinalrank

export inference_coupling, inference_coupling_copula, ic_gaussian
export mutual_information_gaussian, differential_entropy_gaussian
export SIGMA_MIN

# Legacy exports (with deprecation)
export circulatory_fidelity_gaussian, circulatory_fidelity_ksg

# Estimation methods
export entropy_ksg, mutual_information_ksg

# Model types
export SVFParams, HLMParams, ThreeLayerParams
export simulate_svf, simulate_hlm, simulate_three_layer
export compute_ic_svf, compute_ic_hlm, compute_ic_three_layer

# Legacy exports
export compute_cf_svf, compute_cf_hlm, compute_cf_three_layer

# Inference methods
export svf_mf_inference, svf_oracle_inference
export hlm_no_pooling, hlm_partial_pooling
export three_layer_mf_inference, three_layer_oracle_inference

# Minimum sigma for positive differential entropy
const SIGMA_MIN = 1.0 / sqrt(2π * ℯ)  # ≈ 0.2420

# =============================================================================
# PRIMARY API (v1.1): INFERENCE COUPLING
# =============================================================================

"""
    ic_gaussian(ρ)

Compute Inference Coupling for bivariate Gaussian with correlation ρ.

    IC = |ρ|

This is the primary diagnostic for MFVI failure prediction.
"""
function ic_gaussian(ρ::Real)
    return abs(clamp(ρ, -1.0, 1.0))
end


"""
    inference_coupling(x, y; method=:copula)

Estimate Inference Coupling between vectors x and y.

RECOMMENDED WORKFLOW: Use the default copula method for all applications.
The copula estimator is exact for Gaussian data AND provides conservative
estimates for non-Gaussian data, enabling a unified workflow without
needing to verify distributional assumptions.

Methods:
- `:copula` (default, recommended): Rank-transform → probit → Pearson correlation.
  Exact for Gaussians (|ρ| with <0.001 difference), conservative for non-Gaussians.
- `:ksg`: KSG mutual information estimator (for validation or non-monotonic dependence)

Returns: (ic, se) tuple where se is standard error.

# Example
```julia
x = randn(1000)
y = 0.7*x + 0.3*randn(1000)
ic, se = inference_coupling(x, y)  # copula method (recommended)
```
"""
function inference_coupling(x::AbstractVector, y::AbstractVector; method::Symbol=:copula)
    if method == :copula
        return inference_coupling_copula(x, y)
    elseif method == :ksg
        ic = _ic_from_mi_ksg(x, y)
        # Bootstrap SE for KSG (approximate)
        se = 1.0 / sqrt(length(x) - 3)
        return (ic, se)
    else
        error("Unknown method: $method. Use :copula or :ksg")
    end
end


"""
    inference_coupling_copula(x, y)

Copula-based IC estimation (RECOMMENDED for all applications).

Algorithm:
1. Rank-transform to uniform marginals: u = (rank(x) - 0.5) / n
2. Apply inverse normal CDF (probit): z = Φ⁻¹(u)
3. Compute Pearson correlation of transformed variables
4. IC = |ρ|

Returns: (ic, se) where se is Fisher transform standard error.

Key properties:
- EXACT for Gaussian data: Returns |ρ| with differences < 0.001 from direct 
  Pearson correlation. The rank→probit transformation preserves Gaussian structure.
- CONSERVATIVE for non-Gaussian data: Provides lower bound on true IC for 
  distributions with monotonic dependence structure.
- UNIFIED WORKFLOW: No need to verify Gaussianity before estimation.
- Returns ~0 for non-monotonic dependence (triggers Stage 2 synergy screening)
"""
function inference_coupling_copula(x::AbstractVector, y::AbstractVector)
    n = length(x)
    n == length(y) || error("x and y must have same length")
    n > 3 || error("Need at least 4 samples")
    
    # Rank transform to uniform
    u = (ordinalrank(x) .- 0.5) ./ n
    v = (ordinalrank(y) .- 0.5) ./ n
    
    # Probit transform (inverse normal CDF)
    # Using approximation: Φ⁻¹(p) ≈ sign(p-0.5) * √(-2log(min(p, 1-p)))
    z_x = _probit.(u)
    z_y = _probit.(v)
    
    # Pearson correlation
    ρ = cor(z_x, z_y)
    
    # IC = |ρ|
    ic = abs(ρ)
    
    # Fisher transform SE
    se = 1.0 / sqrt(n - 3)
    
    return (ic, se)
end


# Probit function (inverse normal CDF) - simple approximation
function _probit(p::Real)
    p = clamp(p, 1e-10, 1 - 1e-10)
    # Rational approximation to inverse normal CDF
    t = sqrt(-2 * log(min(p, 1 - p)))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    x = t - (c0 + c1*t + c2*t^2) / (1 + d1*t + d2*t^2 + d3*t^3)
    return p < 0.5 ? -x : x
end


# =============================================================================
# GAUSSIAN CASE (Closed-form)
# =============================================================================

"""
    mutual_information_gaussian(ρ)

Compute mutual information for bivariate Gaussian with correlation ρ.

    I(Z;X) = -0.5 * log(1 - ρ²)
"""
function mutual_information_gaussian(ρ::Real)
    ρ = clamp(ρ, -0.9999, 0.9999)
    return -0.5 * log(1 - ρ^2)
end


"""
    differential_entropy_gaussian(σ)

Compute differential entropy for univariate Gaussian.

    H(X) = 0.5 * log(2πeσ²)

Note: H(X) < 0 when σ < SIGMA_MIN ≈ 0.2420
"""
function differential_entropy_gaussian(σ::Real)
    σ > 0 || error("σ must be positive")
    return 0.5 * log(2π * ℯ * σ^2)
end


"""
    circulatory_fidelity_gaussian(ρ, σ_z, σ_x)

DEPRECATED: Use `ic_gaussian(ρ)` instead.

Legacy v1.0 definition: CF = I(z; x) / min(H(z), H(x))

The Relational Invariance Theorem (v1.1) proves that for MFVI diagnostics,
the normalization is unnecessary—IC = |ρ| provides equivalent information.
"""
function circulatory_fidelity_gaussian(ρ::Real, σ_z::Real, σ_x::Real)
    @warn "circulatory_fidelity_gaussian is deprecated. Use ic_gaussian(ρ) instead." maxlog=1
    mi = mutual_information_gaussian(ρ)
    h_z = differential_entropy_gaussian(σ_z)
    h_x = differential_entropy_gaussian(σ_x)
    h_min = min(h_z, h_x)
    
    if h_min <= 0
        return NaN
    end
    
    return clamp(mi / h_min, 0.0, 1.0)
end


# =============================================================================
# KSG ESTIMATOR (Non-Gaussian)
# =============================================================================

"""
    entropy_ksg(X; k=5)

Estimate differential entropy using Kozachenko-Leonenko estimator.
"""
function entropy_ksg(X::AbstractVector; k::Int=5)
    n = length(X)
    n > k || error("Need more samples than k")
    
    X_mat = reshape(X, n, 1)
    tree = KDTree(X_mat')
    
    idxs, dists = knn(tree, X_mat', k+1)
    eps = [d[end] for d in dists]
    eps = max.(eps, 1e-10)
    
    H = digamma(n) - digamma(k) + log(2) + mean(log.(2 .* eps))
    return H
end


"""
    mutual_information_ksg(X, Y; k=5)

Estimate mutual information using KSG estimator.

Note: Has 30-45% negative bias for small samples. Use copula-based estimation
as the primary method; KSG is retained for validation and non-monotonic cases.
"""
function mutual_information_ksg(X::AbstractVector, Y::AbstractVector; k::Int=5)
    n = length(X)
    n == length(Y) || error("X and Y must have same length")
    n > k || error("Need more samples than k")
    
    XY = hcat(X, Y)'
    tree_xy = KDTree(XY)
    tree_x = KDTree(reshape(X, 1, n))
    tree_y = KDTree(reshape(Y, 1, n))
    
    _, dists_xy = knn(tree_xy, XY, k+1)
    eps_xy = [d[end] for d in dists_xy]
    
    n_x = zeros(n)
    n_y = zeros(n)
    
    for i in 1:n
        eps_i = eps_xy[i]
        n_x[i] = length(inrange(tree_x, [X[i]], eps_i)) - 1
        n_y[i] = length(inrange(tree_y, [Y[i]], eps_i)) - 1
    end
    
    n_x = max.(n_x, 1)
    n_y = max.(n_y, 1)
    
    mi = digamma(k) - mean(digamma.(n_x .+ 1) .+ digamma.(n_y .+ 1)) + digamma(n)
    return max(0.0, mi)
end


function _ic_from_mi_ksg(X::AbstractVector, Y::AbstractVector; k::Int=5)
    mi = mutual_information_ksg(X, Y; k=k)
    return sqrt(1 - exp(-2 * mi))
end


"""
    circulatory_fidelity_ksg(X, Y; k=5)

DEPRECATED: Use `inference_coupling(X, Y; method=:ksg)` instead.
"""
function circulatory_fidelity_ksg(X::AbstractVector, Y::AbstractVector; k::Int=5)
    @warn "circulatory_fidelity_ksg is deprecated. Use inference_coupling(X, Y; method=:ksg)" maxlog=1
    mi = mutual_information_ksg(X, Y; k=k)
    h_x = entropy_ksg(X; k=k)
    h_y = entropy_ksg(Y; k=k)
    
    h_min = min(h_x, h_y)
    
    if h_min <= 0
        return NaN
    end
    
    return clamp(mi / h_min, 0.0, 1.0)
end


# =============================================================================
# MODEL PARAMETERS
# =============================================================================

Base.@kwdef struct SVFParams
    coupling::Float64 = 0.5
    base_volatility::Float64 = 0.5
    volatility_noise::Float64 = 0.3
    observation_noise::Float64 = 0.5
end

Base.@kwdef struct HLMParams
    n_groups::Int = 30
    n_per_group::Int = 10
    tau::Float64 = 1.0
    sigma::Float64 = 1.0
    mu::Float64 = 0.0
end

icc(p::HLMParams) = p.tau^2 / (p.tau^2 + p.sigma^2)
reliability(p::HLMParams) = p.tau^2 / (p.tau^2 + p.sigma^2 / p.n_per_group)

"""
Three-layer stochastic volatility parameters (variance-coupling).
"""
Base.@kwdef struct ThreeLayerParams
    kappa_32::Float64 = 0.5    # Distal coupling (x3 → x2 variance)
    kappa_21::Float64 = 0.5    # Proximal coupling (x2 → x1 variance)
    sigma_3::Float64 = 0.3     # Log-volatility random walk noise
    omega_2::Float64 = -0.5    # Base log-variance for layer 2
    omega_1::Float64 = -0.5    # Base log-variance for layer 1
    sigma_obs::Float64 = 0.5   # Observation noise
end


# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

function simulate_svf(params::SVFParams, T::Int=300; seed::Union{Int,Nothing}=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
    
    x3 = zeros(T)
    x2 = zeros(T)
    vol = zeros(T)
    y = zeros(T)
    
    vol[1] = params.base_volatility
    y[1] = randn() * params.observation_noise
    
    for t in 2:T
        x3[t] = x3[t-1] + randn() * params.volatility_noise
        log_vol = clamp(params.coupling * x3[t], -3, 3)
        vol[t] = clamp(params.base_volatility * exp(log_vol), 0.1, 5.0)
        x2[t] = x2[t-1] + randn() * vol[t]
        y[t] = x2[t] + randn() * params.observation_noise
    end
    
    return (x3=x3, x2=x2, y=y, vol=vol, params=params)
end


"""
    compute_ic_svf(sim)

Compute Inference Coupling for SVF simulation using copula-based estimation.
"""
function compute_ic_svf(sim)
    x3 = sim.x3[2:end]
    dx2 = diff(sim.x2)
    log_abs_dx2 = log.(abs.(dx2) .+ 1e-10)
    
    ic, _ = inference_coupling_copula(x3, log_abs_dx2)
    return ic
end

# Legacy alias
compute_cf_svf(sim) = compute_ic_svf(sim)


function svf_mf_inference(sim)
    T = length(sim.y)
    avg_vol = sim.params.base_volatility
    
    x2_est = zeros(T)
    var_est = ones(T)
    
    for t in 2:T
        pred_var = var_est[t-1] + avg_vol^2
        obs_var = sim.params.observation_noise^2
        K = pred_var / (pred_var + obs_var)
        x2_est[t] = x2_est[t-1] + K * (sim.y[t] - x2_est[t-1])
        var_est[t] = (1 - K) * pred_var
    end
    
    return (x2_est, mean((x2_est .- sim.x2).^2))
end


function svf_oracle_inference(sim)
    T = length(sim.y)
    
    x2_est = zeros(T)
    var_est = ones(T)
    
    for t in 2:T
        pred_var = var_est[t-1] + sim.vol[t]^2
        obs_var = sim.params.observation_noise^2
        K = pred_var / (pred_var + obs_var)
        x2_est[t] = x2_est[t-1] + K * (sim.y[t] - x2_est[t-1])
        var_est[t] = (1 - K) * pred_var
    end
    
    return (x2_est, mean((x2_est .- sim.x2).^2))
end


function simulate_hlm(params::HLMParams; seed::Union{Int,Nothing}=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
    
    theta_true = randn(params.n_groups) .* params.tau .+ params.mu
    y = zeros(params.n_groups, params.n_per_group)
    
    for j in 1:params.n_groups
        y[j, :] = randn(params.n_per_group) .* params.sigma .+ theta_true[j]
    end
    
    y_bar = mean(y, dims=2)[:]
    return (theta_true=theta_true, y=y, y_bar=y_bar, params=params)
end


"""
    compute_ic_hlm(params)

Compute IC for HLM: IC = √ICC (square root of intraclass correlation).
"""
function compute_ic_hlm(params::HLMParams)
    return sqrt(icc(params))
end

# Legacy alias
compute_cf_hlm(params::HLMParams) = compute_ic_hlm(params)


function hlm_no_pooling(sim)
    theta_np = sim.y_bar
    mse = mean((theta_np .- sim.theta_true).^2)
    return (theta_np, mse)
end


function hlm_partial_pooling(sim)
    y_bar = sim.y_bar
    grand_mean = mean(y_bar)
    λ = reliability(sim.params)
    theta_pp = grand_mean .+ λ .* (y_bar .- grand_mean)
    mse = mean((theta_pp .- sim.theta_true).^2)
    return (theta_pp, mse)
end


function simulate_three_layer(params::ThreeLayerParams, T::Int=300; seed::Union{Int,Nothing}=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
    
    x3 = zeros(T)
    x2 = zeros(T)
    x1 = zeros(T)
    y = zeros(T)
    vol_2 = zeros(T)
    vol_1 = zeros(T)
    
    vol_2[1] = exp(0.5 * params.omega_2)
    vol_1[1] = exp(0.5 * params.omega_1)
    y[1] = randn() * params.sigma_obs
    
    for t in 2:T
        x3[t] = x3[t-1] + randn() * params.sigma_3
        
        log_var_2 = clamp(params.kappa_32 * x3[t] + params.omega_2, -6, 6)
        vol_2[t] = exp(0.5 * log_var_2)
        x2[t] = x2[t-1] + randn() * vol_2[t]
        
        log_var_1 = clamp(params.kappa_21 * x2[t] + params.omega_1, -6, 6)
        vol_1[t] = exp(0.5 * log_var_1)
        x1[t] = x1[t-1] + randn() * vol_1[t]
        
        y[t] = x1[t] + randn() * params.sigma_obs
    end
    
    return (x3=x3, x2=x2, x1=x1, y=y, vol_2=vol_2, vol_1=vol_1, params=params)
end


"""
    compute_ic_three_layer(sim)

Compute IC for three-layer hierarchy using copula-based estimation.
Returns: (ic_32, ic_21) for distal and proximal coupling.
"""
function compute_ic_three_layer(sim)
    # IC_32: Distal coupling (x3 modulates x2 variance)
    x3 = sim.x3[2:end]
    dx2 = diff(sim.x2)
    log_abs_dx2 = log.(abs.(dx2) .+ 1e-10)
    ic_32, _ = inference_coupling_copula(x3, log_abs_dx2)
    
    # IC_21: Proximal coupling (x2 modulates x1 variance)
    x2 = sim.x2[2:end]
    dx1 = diff(sim.x1)
    log_abs_dx1 = log.(abs.(dx1) .+ 1e-10)
    ic_21, _ = inference_coupling_copula(x2, log_abs_dx1)
    
    return (max(0.0, isfinite(ic_32) ? ic_32 : 0.0), 
            max(0.0, isfinite(ic_21) ? ic_21 : 0.0))
end

# Legacy alias
compute_cf_three_layer(sim) = compute_ic_three_layer(sim)


function three_layer_mf_inference(sim)
    T = length(sim.y)
    params = sim.params
    
    avg_vol_1 = exp(0.5 * params.omega_1)
    process_var = avg_vol_1^2
    obs_var = params.sigma_obs^2
    
    x1_est = zeros(T)
    var_est = ones(T)
    
    for t in 2:T
        pred_var = var_est[t-1] + process_var
        K = pred_var / (pred_var + obs_var)
        x1_est[t] = x1_est[t-1] + K * (sim.y[t] - x1_est[t-1])
        var_est[t] = (1 - K) * pred_var
    end
    
    return mean((x1_est .- sim.x1).^2)
end


function three_layer_oracle_inference(sim)
    T = length(sim.y)
    params = sim.params
    
    obs_var = params.sigma_obs^2
    
    x1_est = zeros(T)
    var_est = ones(T)
    
    for t in 2:T
        process_var = sim.vol_1[t]^2
        pred_var = var_est[t-1] + process_var
        K = pred_var / (pred_var + obs_var)
        x1_est[t] = x1_est[t-1] + K * (sim.y[t] - x1_est[t-1])
        var_est[t] = (1 - K) * pred_var
    end
    
    return mean((x1_est .- sim.x1).^2)
end


# =============================================================================
# WINDOWED IC FOR TIME SERIES (Maximal Coupling Rule)
# =============================================================================

export windowed_ic, windowed_ic_envelope

"""
    windowed_ic(z, x; window_size=50, step_size=nothing, method=:copula)

Compute windowed Inference Coupling for non-stationary time series.

The **Maximal Coupling Rule**: For time series with potential regime changes,
MFVI suitability depends on IC_max, not the global average. A single
high-IC episode can invalidate mean-field approximations.

**IMPORTANT**: Window size must be large enough for stable correlation estimates.
The standard error is SE ≈ 1/√(W-3). For W < 30, estimates have high variance
and may produce noise-driven false positives. We recommend W >= 50.

# Arguments
- `z::AbstractVector`: First variable (time series)
- `x::AbstractVector`: Second variable (time series)
- `window_size::Int=50`: Size of each window (minimum recommended: 50)
- `step_size::Union{Int,Nothing}=nothing`: Step between windows (default: window_size÷4)
- `method::Symbol=:copula`: IC estimation method

# Returns
Named tuple with:
- `ic_max`: Maximum IC across windows (primary diagnostic)
- `ic_mean`: Mean IC across windows  
- `ic_std`: Standard deviation of IC
- `ic_series`: Vector of IC values
- `window_centers`: Vector of window center indices
- `n_windows`: Number of windows computed
- `se_per_window`: Standard error for each window estimate
- `recommendation`: Diagnostic recommendation

# Example
```julia
result = windowed_ic(z_series, x_series; window_size=50)
println("IC_max = \$(result.ic_max)")
```
"""
function windowed_ic(z::AbstractVector, x::AbstractVector; 
                     window_size::Int=50,
                     step_size::Union{Int,Nothing}=nothing,
                     method::Symbol=:copula)
    
    const MIN_WINDOW_SIZE = 30
    const RECOMMENDED_WINDOW_SIZE = 50
    
    T = length(z)
    
    length(x) == T || error("z and x must have same length")
    window_size ≤ T || error("window_size ($window_size) exceeds series length ($T)")
    
    # Warn about small window sizes
    if window_size < MIN_WINDOW_SIZE
        se_approx = 1.0 / sqrt(window_size - 3)
        @warn "window_size=$window_size is below minimum recommended ($MIN_WINDOW_SIZE). " *
              "SE ≈ $(round(se_approx, digits=3)) is high, which may produce noise-driven false positives. " *
              "Consider using window_size >= $RECOMMENDED_WINDOW_SIZE."
    end
    
    if step_size === nothing
        step_size = max(1, window_size ÷ 4)  # 75% overlap default
    end
    
    ic_values = Float64[]
    window_centers = Int[]
    
    start = 1
    while start + window_size - 1 ≤ T
        z_window = z[start:start + window_size - 1]
        x_window = x[start:start + window_size - 1]
        
        try
            ic, _ = inference_coupling_copula(z_window, x_window)
            if isfinite(ic)
                push!(ic_values, ic)
                push!(window_centers, start + window_size ÷ 2)
            end
        catch
            # Skip windows with estimation failures
        end
        
        start += step_size
    end
    
    if isempty(ic_values)
        return (
            ic_max = NaN,
            ic_mean = NaN,
            ic_std = NaN,
            ic_series = Float64[],
            window_centers = Int[],
            n_windows = 0,
            window_size = window_size,
            se_per_window = NaN,
            recommendation = "Insufficient data for windowed analysis"
        )
    end
    
    ic_max = maximum(ic_values)
    ic_mean = mean(ic_values)
    ic_std = std(ic_values)
    se_per_window = window_size > 3 ? 1.0 / sqrt(window_size - 3) : NaN
    
    # Recommendation based on IC_max (Maximal Coupling Rule)
    # Thresholds from manuscript interpretive scale (Section 2.7)
    reliability_note = window_size < RECOMMENDED_WINDOW_SIZE ? 
        " (Note: SE=$(round(se_per_window, digits=3)) with W=$window_size; consider larger windows)" : ""
    
    recommendation = if ic_max < 0.25
        "MFVI safe - negligible coupling (IC_max < 0.25)$reliability_note"
    elseif ic_max < 0.35
        "MFVI likely acceptable - weak coupling (IC_max < 0.35)$reliability_note"
    elseif ic_max < 0.55
        "Caution warranted - moderate coupling (IC_max < 0.55); validate post-inference$reliability_note"
    elseif ic_max < 0.70
        "Consider structured inference - strong coupling (IC_max < 0.70)$reliability_note"
    else
        "Structured inference required - very strong coupling (IC_max >= 0.70)$reliability_note"
    end
    
    return (
        ic_max = ic_max,
        ic_mean = ic_mean,
        ic_std = ic_std,
        ic_series = ic_values,
        window_centers = window_centers,
        n_windows = length(ic_values),
        window_size = window_size,
        se_per_window = se_per_window,
        recommendation = recommendation
    )
end


"""
    windowed_ic_envelope(z, x; window_sizes=nothing, method=:copula)

Compute IC_max envelope across multiple window sizes.

Useful when regime structure is unknown. Sensitivity to window size
indicates regime structure warranting investigation.

# Returns
Named tuple with `ic_max_envelope`, `window_sizes`, `overall_ic_max`, `sensitivity`
"""
function windowed_ic_envelope(z::AbstractVector, x::AbstractVector;
                              window_sizes::Union{Vector{Int},Nothing}=nothing,
                              method::Symbol=:copula)
    T = length(z)
    
    if window_sizes === nothing
        min_w = max(20, T ÷ 20)
        max_w = min(T ÷ 2, T ÷ 3)
        window_sizes = [w for w in [25, 50, 100, 200, 500] if min_w ≤ w ≤ max_w]
        if isempty(window_sizes)
            window_sizes = [min(50, T ÷ 2)]
        end
    end
    
    ic_max_values = Float64[]
    valid_sizes = Int[]
    
    for w in window_sizes
        w > T && continue
        result = windowed_ic(z, x; window_size=w, method=method)
        if isfinite(result.ic_max)
            push!(ic_max_values, result.ic_max)
            push!(valid_sizes, w)
        end
    end
    
    if isempty(ic_max_values)
        return (
            ic_max_envelope = Float64[],
            window_sizes = Int[],
            overall_ic_max = NaN,
            sensitivity = NaN
        )
    end
    
    return (
        ic_max_envelope = ic_max_values,
        window_sizes = valid_sizes,
        overall_ic_max = maximum(ic_max_values),
        sensitivity = std(ic_max_values)
    )
end

end # module
