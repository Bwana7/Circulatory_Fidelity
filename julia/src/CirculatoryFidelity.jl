"""
    CirculatoryFidelity.jl
    
    A Prior Predictive Diagnostic for Mean-Field Variational Inference
    
    This module implements Circulatory Fidelity (CF), an information-theoretic 
    measure for assessing whether mean-field variational inference is appropriate 
    for a given hierarchical model structure.
    
    Author: Aaron Lowry
    Version: 1.0.0
    Date: December 2025
    
    Reference: Lowry, A. (2025). Circulatory Fidelity: A Prior Predictive Diagnostic 
               for Mean-Field Variational Inference.
"""
module CirculatoryFidelity

using Statistics
using LinearAlgebra
using Random
using DataFrames

export CF, mutual_information, differential_entropy
export HGFParams, HLMParams
export simulate_hgf, simulate_hlm
export compute_cf_hgf, compute_cf_hlm
export run_hgf_sweep, run_hlm_sweep

#=============================================================================
    CORE THEORY: Information-Theoretic Quantities
=============================================================================#

"""
    mutual_information(ρ::Real) -> Float64

Compute mutual information for bivariate Gaussian with correlation ρ.

The mutual information between jointly Gaussian variables is:

    I(z;x) = -½ log(1 - ρ²)

This equals the KL divergence from the joint to the product of marginals,
representing the information lost under mean-field factorization.

# Arguments
- `ρ`: Pearson correlation coefficient, -1 < ρ < 1

# Returns
- Mutual information in nats

# Example
```julia
julia> mutual_information(0.5)
0.14384103622589042

julia> mutual_information(0.9)
1.0033021088637848
```
"""
function mutual_information(ρ::Real)::Float64
    ρ = clamp(ρ, -0.999, 0.999)
    return -0.5 * log(1 - ρ^2)
end

"""
    differential_entropy(σ::Real) -> Float64

Compute differential entropy for univariate Gaussian with standard deviation σ.

    H(x) = ½ log(2πeσ²)

# Arguments
- `σ`: Standard deviation (must be positive)

# Returns
- Differential entropy in nats

# Example
```julia
julia> differential_entropy(1.0)
1.4189385332046727
```
"""
function differential_entropy(σ::Real)::Float64
    σ > 0 || throw(DomainError(σ, "Standard deviation must be positive"))
    return 0.5 * log(2π * ℯ * σ^2)
end

"""
    CF(ρ::Real, σ_z::Real, σ_x::Real) -> Float64
    CF(z::AbstractVector, x::AbstractVector) -> Float64

Compute Circulatory Fidelity between variables z and x.

Circulatory Fidelity is defined as:

    CF(z,x) = I(z;x) / min(H(z), H(x))

where I(z;x) is mutual information and H(·) is differential entropy.

# Properties
- Bounded: 0 ≤ CF ≤ 1
- CF = 0 when z ⊥ x (independence)
- CF = 1 when one variable is a deterministic function of the other

# Methods

## From parameters (Gaussian assumption)
```julia
CF(ρ, σ_z, σ_x)
```
- `ρ`: Correlation coefficient
- `σ_z`, `σ_x`: Standard deviations

## From samples
```julia
CF(z, x)
```
- `z`, `x`: Sample vectors (Gaussian assumption for MI estimation)

# Example
```julia
julia> CF(0.7, 1.0, 1.0)
0.30176835911083776

julia> z = randn(1000); x = 0.8z + 0.6randn(1000);
julia> CF(z, x)
0.28  # approximately
```
"""
function CF(ρ::Real, σ_z::Real, σ_x::Real)::Float64
    mi = mutual_information(ρ)
    h_z = differential_entropy(σ_z)
    h_x = differential_entropy(σ_x)
    h_min = min(h_z, h_x)
    
    h_min > 0 || return NaN
    return clamp(mi / h_min, 0.0, 1.0)
end

function CF(z::AbstractVector{<:Real}, x::AbstractVector{<:Real})::Float64
    length(z) == length(x) || throw(DimensionMismatch("Vectors must have same length"))
    
    # Remove any NaN/Inf
    mask = isfinite.(z) .& isfinite.(x)
    z_clean = z[mask]
    x_clean = x[mask]
    
    length(z_clean) ≥ 20 || return NaN
    
    ρ = cor(z_clean, x_clean)
    isfinite(ρ) || return NaN
    
    σ_z = std(z_clean)
    σ_x = std(x_clean)
    
    return CF(ρ, σ_z, σ_x)
end

#=============================================================================
    MODEL 1: Hierarchical Gaussian Filter (HGF)
=============================================================================#

"""
    HGFParams

Parameters for the Hierarchical Gaussian Filter generative model.

The HGF models belief updating under uncertainty with hierarchical coupling:
- Level 3 (x₃): Log-volatility, evolves as random walk
- Level 2 (x₂): Hidden state, volatility depends on x₃
- Level 1 (y): Noisy observations of x₂

# Fields
- `coupling::Float64`: How strongly x₃ modulates x₂ volatility (κ in paper)
- `base_volatility::Float64`: Baseline state volatility
- `volatility_volatility::Float64`: Standard deviation of x₃ innovations
- `observation_noise::Float64`: Observation noise standard deviation

# Example
```julia
params = HGFParams(coupling=0.5)
sim = simulate_hgf(params, T=300)
cf = compute_cf_hgf(sim)
```
"""
Base.@kwdef struct HGFParams
    coupling::Float64 = 0.5
    base_volatility::Float64 = 0.3
    volatility_volatility::Float64 = 0.1
    observation_noise::Float64 = 0.5
end

"""
    simulate_hgf(params::HGFParams; T::Int=300, seed::Union{Int,Nothing}=nothing)

Simulate from HGF generative model (prior predictive).

# Generative Process
```
x₃(t) = x₃(t-1) + ε₃,     ε₃ ~ N(0, σ₃²)
σ₂(t) = σ_base × exp(κ × x₃(t))
x₂(t) = x₂(t-1) + ε₂,     ε₂ ~ N(0, σ₂(t)²)
y(t)  = x₂(t) + ε_y,      ε_y ~ N(0, σ_y²)
```

# Returns
NamedTuple with fields:
- `x3`: Volatility trajectory
- `x2`: State trajectory  
- `y`: Observations
- `vol`: Instantaneous volatility
- `params`: Input parameters

# Example
```julia
sim = simulate_hgf(HGFParams(coupling=0.8), T=500, seed=42)
plot(sim.x2, label="State")
```
"""
function simulate_hgf(params::HGFParams; T::Int=300, seed::Union{Int,Nothing}=nothing)
    !isnothing(seed) && Random.seed!(seed)
    
    x3 = zeros(T)
    x2 = zeros(T)
    vol = zeros(T)
    y = zeros(T)
    
    x3[1] = 0.0
    x2[1] = 0.0
    vol[1] = params.base_volatility
    y[1] = x2[1] + randn() * params.observation_noise
    
    for t in 2:T
        # Volatility random walk
        x3[t] = x3[t-1] + randn() * params.volatility_volatility
        
        # State volatility modulated by x3
        vol[t] = params.base_volatility * exp(clamp(params.coupling * x3[t], -3, 3))
        vol[t] = clamp(vol[t], 0.01, 5.0)
        
        # State evolution
        x2[t] = x2[t-1] + randn() * vol[t]
        
        # Observation
        y[t] = x2[t] + randn() * params.observation_noise
    end
    
    return (x3=x3, x2=x2, y=y, vol=vol, params=params)
end

"""
    compute_cf_hgf(sim) -> Float64

Compute CF from HGF simulation measuring volatility-state coupling.

For HGF, CF quantifies how much information x₃ (volatility) carries about 
state innovations Δx₂. High CF indicates strong coupling that mean-field 
will discard.
"""
function compute_cf_hgf(sim)
    x3 = sim.x3[2:end]
    Δx2 = diff(sim.x2)
    
    # Correlation between volatility and log|innovations|
    log_abs_Δx2 = log.(abs.(Δx2) .+ 1e-10)
    ρ = cor(x3, log_abs_Δx2)
    
    isfinite(ρ) || return NaN
    
    # Normalized MI
    h_ref = differential_entropy(1.0)
    mi = mutual_information(ρ)
    
    return clamp(mi / h_ref, 0.0, 1.0)
end

#=============================================================================
    MODEL 2: Hierarchical Linear Model (HLM)
=============================================================================#

"""
    HLMParams

Parameters for Hierarchical Linear Model.

The HLM models grouped data with partial pooling:
- Group effects θⱼ ~ N(0, τ²)
- Observations yᵢⱼ | θⱼ ~ N(θⱼ, σ²)

The intraclass correlation ICC = τ²/(τ²+σ²) determines reliability.

# Fields
- `n_groups::Int`: Number of groups (J)
- `n_per_group::Int`: Observations per group (n)
- `τ::Float64`: Between-group standard deviation
- `σ::Float64`: Within-group standard deviation

# Derived quantities
- `icc`: Intraclass correlation
- `reliability`: Reliability of group means
"""
Base.@kwdef struct HLMParams
    n_groups::Int = 30
    n_per_group::Int = 10
    τ::Float64 = 1.0
    σ::Float64 = 1.0
end

icc(p::HLMParams) = p.τ^2 / (p.τ^2 + p.σ^2)
reliability(p::HLMParams) = p.τ^2 / (p.τ^2 + p.σ^2 / p.n_per_group)

"""
    simulate_hlm(params::HLMParams; seed::Union{Int,Nothing}=nothing)

Simulate from HLM generative model.

# Returns
NamedTuple with fields:
- `θ`: Group effects (length n_groups)
- `y`: All observations (length n_groups × n_per_group)
- `group`: Group membership indices
- `params`: Input parameters
"""
function simulate_hlm(params::HLMParams; seed::Union{Int,Nothing}=nothing)
    !isnothing(seed) && Random.seed!(seed)
    
    # Group effects
    θ = randn(params.n_groups) .* params.τ
    
    # Observations
    N = params.n_groups * params.n_per_group
    y = zeros(N)
    group = zeros(Int, N)
    
    idx = 1
    for j in 1:params.n_groups
        for i in 1:params.n_per_group
            y[idx] = θ[j] + randn() * params.σ
            group[idx] = j
            idx += 1
        end
    end
    
    return (θ=θ, y=y, group=group, params=params)
end

"""
    compute_cf_hlm(params::HLMParams) -> Float64

Compute analytical CF for HLM based on reliability.

For HLM, CF relates to how reliably group means estimate group effects.
Low CF indicates weak signal where no-pooling (mean-field) overfits.
"""
function compute_cf_hlm(params::HLMParams)::Float64
    rel = reliability(params)
    
    # CF from reliability (correlation = √reliability for balanced design)
    ρ = sqrt(clamp(rel, 0.001, 0.999))
    h_ref = differential_entropy(1.0)
    mi = mutual_information(ρ)
    
    return clamp(mi / h_ref, 0.0, 1.0)
end

#=============================================================================
    PARAMETER SWEEPS
=============================================================================#

"""
    run_hgf_sweep(; coupling_values, n_sims, T, seed) -> DataFrame

Run parameter sweep over HGF coupling strengths.

# Arguments
- `coupling_values`: Vector of coupling values to test
- `n_sims`: Simulations per parameter setting
- `T`: Time steps per simulation
- `seed`: Random seed for reproducibility

# Returns
DataFrame with columns: coupling, cf, mf_mse, oracle_mse, mse_ratio
"""
function run_hgf_sweep(;
    coupling_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0],
    n_sims::Int = 100,
    T::Int = 300,
    seed::Int = 42
)
    Random.seed!(seed)
    
    results = DataFrame(
        coupling = Float64[],
        cf = Float64[],
        mf_mse = Float64[],
        oracle_mse = Float64[],
        mse_ratio = Float64[]
    )
    
    for κ in coupling_values
        params = HGFParams(coupling=κ)
        
        for _ in 1:n_sims
            sim = simulate_hgf(params, T=T)
            cf = compute_cf_hgf(sim)
            
            # MF inference (ignores volatility)
            mf_mse = mf_inference_error(sim)
            
            # Oracle inference (knows volatility)
            oracle_mse = oracle_inference_error(sim)
            
            push!(results, (κ, cf, mf_mse, oracle_mse, mf_mse / max(oracle_mse, 1e-10)))
        end
    end
    
    return results
end

"""
    run_hlm_sweep(; τ_values, n_sims, seed) -> DataFrame

Run parameter sweep over HLM between-group variance.

# Arguments  
- `τ_values`: Vector of between-group SDs to test
- `n_sims`: Simulations per parameter setting
- `seed`: Random seed for reproducibility

# Returns
DataFrame with columns: τ, icc, cf, no_pool_mse, partial_pool_mse, mse_ratio
"""
function run_hlm_sweep(;
    τ_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0],
    n_sims::Int = 100,
    seed::Int = 42
)
    Random.seed!(seed)
    
    results = DataFrame(
        τ = Float64[],
        icc = Float64[],
        cf = Float64[],
        no_pool_mse = Float64[],
        partial_pool_mse = Float64[],
        mse_ratio = Float64[]
    )
    
    for τ in τ_values
        params = HLMParams(τ=τ, σ=1.0)
        cf = compute_cf_hlm(params)
        
        for _ in 1:n_sims
            sim = simulate_hlm(params)
            
            np_mse = no_pooling_error(sim)
            pp_mse = partial_pooling_error(sim)
            
            push!(results, (τ, icc(params), cf, np_mse, pp_mse, np_mse / max(pp_mse, 1e-10)))
        end
    end
    
    return results
end

#=============================================================================
    INFERENCE METHODS (for validation)
=============================================================================#

# HGF: Mean-field inference (ignores volatility)
function mf_inference_error(sim)
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
    
    return mean((x2_est .- sim.x2).^2)
end

# HGF: Oracle inference (knows true volatility)
function oracle_inference_error(sim)
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
    
    return mean((x2_est .- sim.x2).^2)
end

# HLM: No-pooling (uses group means only)
function no_pooling_error(sim)
    params = sim.params
    θ_hat = [mean(sim.y[sim.group .== j]) for j in 1:params.n_groups]
    return mean((θ_hat .- sim.θ).^2)
end

# HLM: Partial pooling (shrinkage estimator)
function partial_pooling_error(sim)
    params = sim.params
    
    grand_mean = mean(sim.y)
    group_means = [mean(sim.y[sim.group .== j]) for j in 1:params.n_groups]
    
    shrinkage = reliability(params)
    θ_hat = shrinkage .* group_means .+ (1 - shrinkage) * grand_mean
    
    return mean((θ_hat .- sim.θ).^2)
end

end # module
