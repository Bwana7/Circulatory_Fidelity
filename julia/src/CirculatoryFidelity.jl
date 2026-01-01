"""
CirculatoryFidelity.jl

Circulatory Fidelity: A Prior Predictive Diagnostic for Mean-Field Variational Inference

This module provides tools for computing Circulatory Fidelity (CF), a normalized
information-theoretic measure that quantifies structural coupling between variables.

    CF(z, x) = I(z; x) / min(H(z), H(x))

IMPORTANT: CF requires positive marginal differential entropy.
For Gaussians, this requires Ïƒ > 1/âˆš(2Ï€e) â‰ˆ 0.2420.

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

export mutual_information_gaussian, differential_entropy_gaussian
export circulatory_fidelity_gaussian, SIGMA_MIN
export linfoot_correlation
export mutual_information_copula, circulatory_fidelity_copula, copula_correlation
export entropy_ksg, mutual_information_ksg, circulatory_fidelity_ksg
export SVFParams, HLMParams, ThreeLayerParams
export simulate_svf, simulate_hlm, simulate_three_layer
export compute_cf_svf, compute_cf_hlm, compute_cf_three_layer
export svf_mf_inference, svf_oracle_inference
export svf_kalman_filter, svf_fit_mfvi, svf_fit_oracle
export compute_log_likelihood_gap
export hlm_no_pooling, hlm_partial_pooling
export three_layer_mf_inference, three_layer_oracle_inference
export run_svf_validation, run_hlm_validation, run_three_layer_validation

# Minimum sigma for positive differential entropy
const SIGMA_MIN = 1.0 / sqrt(2Ï€ * â„¯)  # â‰ˆ 0.2420

# =============================================================================
# GAUSSIAN CASE (Closed-form)
# =============================================================================

"""
    mutual_information_gaussian(Ï)

Compute mutual information for bivariate Gaussian with correlation Ï.
"""
function mutual_information_gaussian(Ï::Real)
    Ï = clamp(Ï, -0.9999, 0.9999)
    return -0.5 * log(1 - Ï^2)
end

"""
    differential_entropy_gaussian(Ïƒ)

Compute differential entropy for univariate Gaussian.
Note: H(X) < 0 when Ïƒ < SIGMA_MIN â‰ˆ 0.2420
"""
function differential_entropy_gaussian(Ïƒ::Real)
    Ïƒ > 0 || error("Ïƒ must be positive")
    return 0.5 * log(2Ï€ * â„¯ * Ïƒ^2)
end

"""
    circulatory_fidelity_gaussian(Ï, Ïƒ_z, Ïƒ_x)

Compute CF for bivariate Gaussian (closed-form).
BOTH Ïƒ_z AND Ïƒ_x ARE REQUIRED PARAMETERS.

CF = I(z; x) / min(H(z), H(x))
Returns NaN if min(H(z), H(x)) <= 0.
"""
function circulatory_fidelity_gaussian(Ï::Real, Ïƒ_z::Real, Ïƒ_x::Real)
    mi = mutual_information_gaussian(Ï)
    h_z = differential_entropy_gaussian(Ïƒ_z)
    h_x = differential_entropy_gaussian(Ïƒ_x)
    h_min = min(h_z, h_x)
    
    if h_min <= 0
        @warn "min(H(z), H(x)) = $h_min <= 0. CF undefined. Ensure Ïƒ > $SIGMA_MIN"
        return NaN
    end
    
    return clamp(mi / h_min, 0.0, 1.0)
end

"""
    linfoot_correlation(ρ)

Compute Linfoot correlation r_L = √(1 - exp(-2I)) for Gaussian MI.
For Gaussians: r_L = |ρ| (exactly).
"""
function linfoot_correlation(ρ::Real)
    mi = mutual_information_gaussian(ρ)
    return sqrt(1 - exp(-2 * mi))
end

# =============================================================================
# COPULA-BASED ESTIMATION (Non-Gaussian Continuous)
# =============================================================================

using Distributions: Normal, quantile, cdf

"""
    mutual_information_copula(X, Y)

Estimate mutual information using Gaussian copula transform.
Provides a CONSERVATIVE LOWER BOUND for any continuous distribution.

The Gaussian copula minimizes MI among all copulas with the same correlation
parameter (Joe, 1989).
"""
function mutual_information_copula(X::AbstractVector, Y::AbstractVector)
    n = length(X)
    n == length(Y) || error("X and Y must have same length")
    
    # Rank transform to (0, 1)
    u = (sortperm(sortperm(X)) .- 0.5) ./ n
    v = (sortperm(sortperm(Y)) .- 0.5) ./ n
    
    # Transform to standard normal
    d = Normal(0, 1)
    z = quantile.(d, u)
    w = quantile.(d, v)
    
    # Copula correlation
    ρ_c = cor(z, w)
    
    if !isfinite(ρ_c)
        return 0.0
    end
    
    # Closed-form MI for Gaussian copula
    ρ_c = clamp(ρ_c, -0.9999, 0.9999)
    mi = -0.5 * log(1 - ρ_c^2)
    
    return max(0.0, mi)
end

"""
    circulatory_fidelity_copula(X, Y)

Compute CF using Gaussian copula transform.
RECOMMENDED for non-Gaussian continuous distributions.
Provides conservative lower bound with closed-form standard errors.
"""
function circulatory_fidelity_copula(X::AbstractVector, Y::AbstractVector)
    mi = mutual_information_copula(X, Y)
    
    # Standard normal entropy = 0.5 * log(2πe)
    h = 0.5 * log(2π * ℯ)
    
    return clamp(mi / h, 0.0, 1.0)
end

"""
    copula_correlation(X, Y) -> (ρ_c, se, ci_95)

Compute copula correlation with Fisher standard error and 95% CI.
"""
function copula_correlation(X::AbstractVector, Y::AbstractVector)
    n = length(X)
    
    # Rank transform
    u = (sortperm(sortperm(X)) .- 0.5) ./ n
    v = (sortperm(sortperm(Y)) .- 0.5) ./ n
    
    # Normal transform
    d = Normal(0, 1)
    z = quantile.(d, u)
    w = quantile.(d, v)
    
    # Copula correlation
    ρ_c = cor(z, w)
    
    # Fisher standard error
    se = n > 3 ? 1.0 / sqrt(n - 3) : Inf
    
    # 95% CI via Fisher transformation
    z_transform = atanh(ρ_c)
    ci_z = (z_transform - 1.96 * se, z_transform + 1.96 * se)
    ci_95 = (tanh(ci_z[1]), tanh(ci_z[2]))
    
    return (ρ_c, se, ci_95)
end

# =============================================================================
# KSG ESTIMATOR (Discrete/Mixed Variables Only)
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

"""
    circulatory_fidelity_ksg(X, Y; k=5)

Compute CF using KSG estimators (non-Gaussian case).
"""
function circulatory_fidelity_ksg(X::AbstractVector, Y::AbstractVector; k::Int=5)
    mi = mutual_information_ksg(X, Y; k=k)
    h_x = entropy_ksg(X; k=k)
    h_y = entropy_ksg(Y; k=k)
    
    h_min = min(h_x, h_y)
    
    if h_min <= 0
        @warn "min(H(X), H(Y)) = $h_min <= 0. CF undefined."
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
x_3 is phasic log-volatility (random walk)
x_2 has variance modulated by exp(Îº_32 * x_3 + Ï‰_2)
x_1 has variance modulated by exp(Îº_21 * x_2 + Ï‰_1)
"""
Base.@kwdef struct ThreeLayerParams
    kappa_32::Float64 = 0.5    # Distal coupling (x3 â†’ x2 variance)
    kappa_21::Float64 = 0.5    # Proximal coupling (x2 â†’ x1 variance)
    sigma_3::Float64 = 0.3     # Log-volatility random walk noise
    omega_2::Float64 = -0.5    # Base log-variance for layer 2
    omega_1::Float64 = -0.5    # Base log-variance for layer 1
    sigma_obs::Float64 = 0.5   # Observation noise
end

# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

function simulate_svf(params::SVFParams; T::Int=300)
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

function compute_cf_svf(sim)
    x3 = sim.x3[2:end]
    dx2 = diff(sim.x2)
    log_abs_dx2 = log.(abs.(dx2) .+ 1e-10)
    
    Ï = cor(x3, log_abs_dx2)
    if !isfinite(Ï)
        return NaN
    end
    
    Ïƒ_z = max(std(x3), 1.0)
    Ïƒ_x = max(std(log_abs_dx2), 1.0)
    
    return circulatory_fidelity_gaussian(Ï, Ïƒ_z, Ïƒ_x)
end

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

"""
    svf_kalman_filter(y, process_var, obs_var)

Kalman filter for local level model with log-likelihood computation.
Returns (filtered_states, filtered_variances, log_likelihood).

This is the core function for computing the log-likelihood gap diagnostic
(see Section 7 of the manuscript).
"""
function svf_kalman_filter(y::Vector{Float64}, process_var::Union{Float64, Vector{Float64}}, obs_var::Float64)
    T = length(y)
    proc_var = process_var isa Float64 ? fill(process_var, T) : process_var
    
    x_filt = zeros(T)
    P_filt = zeros(T)
    P_filt[1] = 1.0
    log_lik = 0.0
    
    for t in 2:T
        # Predict
        x_pred = x_filt[t-1]
        P_pred = P_filt[t-1] + proc_var[t]
        
        # Update  
        S = P_pred + obs_var  # Innovation variance
        K = P_pred / S        # Kalman gain
        innovation = y[t] - x_pred
        
        x_filt[t] = x_pred + K * innovation
        P_filt[t] = (1 - K) * P_pred
        
        # Log-likelihood contribution
        log_lik += -0.5 * (log(2π * S) + innovation^2 / S)
    end
    
    return (x_filtered=x_filt, P_filtered=P_filt, log_likelihood=log_lik)
end

"""
    svf_fit_mfvi(sim)

Fit MFVI (constant volatility assumption) to SVF simulation.
Returns (kalman_result, estimated_sigma, mse).
"""
function svf_fit_mfvi(sim)
    obs_var = sim.params.observation_noise^2
    
    # Grid search for optimal constant volatility
    best_sigma = sim.params.base_volatility
    best_ll = -Inf
    
    for σ in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        result = svf_kalman_filter(sim.y, σ^2, obs_var)
        if result.log_likelihood > best_ll
            best_ll = result.log_likelihood
            best_sigma = σ
        end
    end
    
    result = svf_kalman_filter(sim.y, best_sigma^2, obs_var)
    mse = mean((result.x_filtered .- sim.x2).^2)
    
    return (result=result, sigma_mf=best_sigma, mse=mse)
end

"""
    svf_fit_oracle(sim)

Fit oracle filter (knows true time-varying volatility) to SVF simulation.
Returns (kalman_result, mse).
"""
function svf_fit_oracle(sim)
    obs_var = sim.params.observation_noise^2
    process_var = sim.vol.^2
    
    result = svf_kalman_filter(sim.y, process_var, obs_var)
    mse = mean((result.x_filtered .- sim.x2).^2)
    
    return (result=result, mse=mse)
end

"""
    compute_log_likelihood_gap(sim)

Compute log-likelihood gap between oracle and MFVI posteriors.
This is the primary diagnostic metric (see Section 7, r = 0.86 correlation with CF).

ΔLL = ℓ_oracle - ℓ_MFVI
"""
function compute_log_likelihood_gap(sim)
    mfvi_fit = svf_fit_mfvi(sim)
    oracle_fit = svf_fit_oracle(sim)
    
    return oracle_fit.result.log_likelihood - mfvi_fit.result.log_likelihood
end

function simulate_hlm(params::HLMParams)
    theta_true = randn(params.n_groups) .* params.tau .+ params.mu
    y = zeros(params.n_groups, params.n_per_group)
    
    for j in 1:params.n_groups
        y[j, :] = randn(params.n_per_group) .* params.sigma .+ theta_true[j]
    end
    
    y_bar = mean(y, dims=2)[:]
    return (theta_true=theta_true, y=y, y_bar=y_bar, params=params)
end

compute_cf_hlm(params::HLMParams) = reliability(params)

function hlm_no_pooling(sim)
    theta_np = sim.y_bar
    mse = mean((theta_np .- sim.theta_true).^2)
    return (theta_np, mse)
end

function hlm_partial_pooling(sim)
    y_bar = sim.y_bar
    grand_mean = mean(y_bar)
    Î» = reliability(sim.params)
    theta_pp = grand_mean .+ Î» .* (y_bar .- grand_mean)
    mse = mean((theta_pp .- sim.theta_true).^2)
    return (theta_pp, mse)
end

"""
Simulate three-layer stochastic volatility hierarchy (VARIANCE-COUPLING).

Model:
    x_3(t) = x_3(t-1) + Îµ_3,  Îµ_3 ~ N(0, Ïƒ_3Â²)           [phasic log-vol]
    x_2(t) ~ N(x_2(t-1), exp(Îº_32 * x_3(t) + Ï‰_2))       [tonic log-vol]
    x_1(t) ~ N(x_1(t-1), exp(Îº_21 * x_2(t) + Ï‰_1))       [state]
    y(t) ~ N(x_1(t), Ïƒ_obsÂ²)                              [observation]
"""
function simulate_three_layer(params::ThreeLayerParams; T::Int=300)
    x3 = zeros(T)   # Phasic log-volatility
    x2 = zeros(T)   # Tonic log-volatility  
    x1 = zeros(T)   # State
    y = zeros(T)    # Observations
    vol_2 = zeros(T)  # Volatility at layer 2
    vol_1 = zeros(T)  # Volatility at layer 1
    
    # Initialize volatilities
    vol_2[1] = exp(0.5 * params.omega_2)
    vol_1[1] = exp(0.5 * params.omega_1)
    y[1] = randn() * params.sigma_obs
    
    for t in 2:T
        # Layer 3: Random walk (phasic log-volatility)
        x3[t] = x3[t-1] + randn() * params.sigma_3
        
        # Layer 2: Variance modulated by x3 (tonic log-volatility)
        log_var_2 = clamp(params.kappa_32 * x3[t] + params.omega_2, -6, 6)
        vol_2[t] = exp(0.5 * log_var_2)
        x2[t] = x2[t-1] + randn() * vol_2[t]
        
        # Layer 1: Variance modulated by x2 (state)
        log_var_1 = clamp(params.kappa_21 * x2[t] + params.omega_1, -6, 6)
        vol_1[t] = exp(0.5 * log_var_1)
        x1[t] = x1[t-1] + randn() * vol_1[t]
        
        # Observation
        y[t] = x1[t] + randn() * params.sigma_obs
    end
    
    return (x3=x3, x2=x2, x1=x1, y=y, vol_2=vol_2, vol_1=vol_1, params=params)
end

"""
Compute CF for three-layer stochastic volatility (variance-coupling).
CF_32: x3 vs log|Î”x2| (distal)
CF_21: x2 vs log|Î”x1| (proximal)
"""
function compute_cf_three_layer(sim)
    # CF_32: Distal coupling (x3 modulates x2 variance)
    x3 = sim.x3[2:end]
    dx2 = diff(sim.x2)
    log_abs_dx2 = log.(abs.(dx2) .+ 1e-10)
    Ï_32 = cor(x3, log_abs_dx2)
    
    # CF_21: Proximal coupling (x2 modulates x1 variance)
    x2 = sim.x2[2:end]
    dx1 = diff(sim.x1)
    log_abs_dx1 = log.(abs.(dx1) .+ 1e-10)
    Ï_21 = cor(x2, log_abs_dx1)
    
    cf_32 = isfinite(Ï_32) ? circulatory_fidelity_gaussian(Ï_32, max(std(x3), 1.0), max(std(log_abs_dx2), 1.0)) : 0.0
    cf_21 = isfinite(Ï_21) ? circulatory_fidelity_gaussian(Ï_21, max(std(x2), 1.0), max(std(log_abs_dx1), 1.0)) : 0.0
    
    return (max(0.0, isfinite(cf_32) ? cf_32 : 0.0), max(0.0, isfinite(cf_21) ? cf_21 : 0.0))
end

"""Mean-field inference: uses average volatility, ignoring coupling."""
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

"""Oracle inference: knows true volatility at each timestep."""
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

end # module
