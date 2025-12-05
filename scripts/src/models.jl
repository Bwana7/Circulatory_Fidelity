# ═══════════════════════════════════════════════════════════════════════════════
# models.jl - Generative Models for Circulatory Fidelity
# ═══════════════════════════════════════════════════════════════════════════════

"""
    CFParameters

Parameters for the Circulatory Fidelity HGF model.

# Fields
- `κ::Float64`: Coupling strength between levels (dimensionless, > 0)
- `ω::Float64`: Tonic (baseline) log-volatility (log units, typically < 0)
- `ϑ::Float64`: Volatility of volatility / hazard rate (> 0, typically ≪ 1)
- `π_u::Float64`: Observation precision (inverse variance, > 0)
- `γ_max::Float64`: Maximum precision (free parameter, to be fit to data)

Note: The dopamine-precision transfer function uses K_d ≈ 20 nM (D2 receptor
dissociation constant) which is fixed by pharmacology, not a free parameter.
See docs/transfer_function_derivation.md for derivation.
"""
Base.@kwdef struct CFParameters
    κ::Float64 = 1.0
    ω::Float64 = -2.0
    ϑ::Float64 = 0.1
    π_u::Float64 = 10.0
    γ_max::Float64 = 100.0
end

"""
    CFResults

Results from running CF inference.

# Fields
- `μ_z::Vector{Float64}`: Posterior means of log-volatility
- `μ_x::Vector{Float64}`: Posterior means of hidden state
- `σ²_z::Vector{Float64}`: Posterior variances of z
- `σ²_x::Vector{Float64}`: Posterior variances of x
- `CF::Vector{Float64}`: Circulatory fidelity at each timestep
- `converged::Bool`: Whether inference converged
"""
struct CFResults
    μ_z::Vector{Float64}
    μ_x::Vector{Float64}
    σ²_z::Vector{Float64}
    σ²_x::Vector{Float64}
    CF::Vector{Float64}
    converged::Bool
end

"""
    cf_agent(y, T, params, γ_DA)

RxInfer model specification for the CF agent.

# Arguments
- `y`: Observations (vector of length T)
- `T`: Number of timesteps
- `params`: CFParameters struct
- `γ_DA`: Precision derived from dopamine level

# Returns
Tuple of (z, x) latent state vectors
"""
@model function cf_agent(y, T, params::CFParameters, γ_DA)
    # Unpack parameters
    κ = params.κ
    ω = params.ω
    ϑ = params.ϑ
    π_u = params.π_u
    
    # Priors for initial states
    z_prev ~ Normal(0.0, 1.0)
    x_prev ~ Normal(0.0, 1.0)
    
    # Temporal evolution
    z = Vector{Any}(undef, T)
    x = Vector{Any}(undef, T)
    
    for t in 1:T
        # Level 2: Volatility evolves as random walk with precision ϑ
        if t == 1
            z[t] ~ Normal(z_prev, sqrt(1/ϑ))
        else
            z[t] ~ Normal(z[t-1], sqrt(1/ϑ))
        end
        
        # Level 1: State evolves with volatility-dependent variance
        state_variance = γ_DA * exp(-κ * z[t] - ω)
        if t == 1
            x[t] ~ Normal(x_prev, sqrt(state_variance))
        else
            x[t] ~ Normal(x[t-1], sqrt(state_variance))
        end
        
        # Observation with fixed precision
        y[t] ~ Normal(x[t], sqrt(1/π_u))
    end
    
    return z, x
end

# Default baseline dopamine level (nM) - typical tonic concentration
const D_BASELINE = 50.0

"""
    run_inference(observations, params, mode; DA_level=nothing)

Run variational inference on observations.

# Arguments
- `observations::Vector{Float64}`: Observed data
- `params::CFParameters`: Model parameters
- `mode::Symbol`: Either `:mean_field` or `:structured`
- `DA_level::Union{Nothing,Float64}`: Dopamine level in nM (uses D_BASELINE=50 nM if nothing)

# Returns
CFResults struct with posterior estimates
"""
function run_inference(observations::Vector{Float64}, 
                       params::CFParameters, 
                       mode::Symbol;
                       DA_level::Union{Nothing,Float64}=nothing)
    
    T = length(observations)
    
    # Compute precision from dopamine (default to baseline tonic level)
    D = isnothing(DA_level) ? D_BASELINE : DA_level
    γ_DA = dopamine_to_precision(D, params)
    
    # Select constraints
    constraints = mode == :mean_field ? mean_field_constraints() : structured_constraints()
    
    # Initialize storage
    μ_z = zeros(T)
    μ_x = zeros(T)
    σ²_z = ones(T)
    σ²_x = ones(T)
    CF_vals = zeros(T)
    
    # Current state estimates
    curr_μ_z = 0.0
    curr_μ_x = 0.0
    curr_σ²_z = 1.0
    curr_σ²_x = 1.0
    
    converged = true
    
    for t in 1:T
        y_t = observations[t]
        
        # Perform variational update
        curr_μ_z, curr_μ_x, curr_σ²_z, curr_σ²_x, cov_zx = variational_update_full(
            curr_μ_z, curr_μ_x, curr_σ²_z, curr_σ²_x, 
            y_t, params, γ_DA, mode
        )
        
        # Store results
        μ_z[t] = curr_μ_z
        μ_x[t] = curr_μ_x
        σ²_z[t] = curr_σ²_z
        σ²_x[t] = curr_σ²_x
        
        # Compute CF
        CF_vals[t] = compute_circulatory_fidelity(curr_σ²_z, curr_σ²_x, cov_zx)
        
        # Check for divergence
        if abs(curr_μ_z) > 10 || abs(curr_μ_x) > 100
            converged = false
            # Continue but mark as diverged
        end
    end
    
    return CFResults(μ_z, μ_x, σ²_z, σ²_x, CF_vals, converged)
end

"""
    variational_update_full(μ_z, μ_x, σ²_z, σ²_x, y, params, γ_DA, mode)

Perform one step of variational inference, returning covariance.

# Returns
Tuple of (μ_z_new, μ_x_new, σ²_z_new, σ²_x_new, cov_zx)
"""
function variational_update_full(μ_z, μ_x, σ²_z, σ²_x, y, params, γ_DA, mode)
    κ = params.κ
    ω = params.ω
    ϑ = params.ϑ
    π_u = params.π_u
    
    # Prediction error at level 1
    δ_x = y - μ_x
    
    # Effective precision at level 1
    π_x = γ_DA * exp(-κ * μ_z - ω)
    
    cov_zx = 0.0  # Default for mean-field
    
    if mode == :mean_field
        # Mean-field updates (factorized)
        
        # Update x (Level 1)
        π_x_post = π_u + π_x
        μ_x_new = μ_x + (π_u / π_x_post) * δ_x
        σ²_x_new = 1 / π_x_post
        
        # Update z (Level 2) - independent of x posterior
        δ_z = 0.5 * (δ_x^2 * π_x - 1)
        π_z_post = ϑ + 0.5 * κ^2 * π_x
        μ_z_new = μ_z + (κ * π_x / π_z_post) * δ_z
        σ²_z_new = 1 / π_z_post
        
    else  # :structured
        # Structured updates (preserving dependencies)
        
        # Prior precision matrix
        Λ_prior = [1/σ²_z 0; 0 1/σ²_x]
        
        # Likelihood contribution (observation precision)
        H = [0.0; 1.0]  # Observation maps to x only
        
        # Posterior precision
        Λ_post = Λ_prior + π_u * (H * H')
        
        # Cross-level coupling term
        coupling = 0.5 * κ^2 * π_x * δ_x^2
        Λ_post[1,1] += coupling
        Λ_post[1,2] += 0.25 * κ * π_x
        Λ_post[2,1] += 0.25 * κ * π_x
        
        # Posterior covariance
        Σ_post = inv(Λ_post)
        
        # Posterior mean update
        η_prior = [μ_z / σ²_z; μ_x / σ²_x]
        η_obs = π_u * y * H
        η_post = η_prior + η_obs
        
        μ_new = Σ_post * η_post
        
        μ_z_new = μ_new[1]
        μ_x_new = μ_new[2]
        σ²_z_new = max(Σ_post[1,1], 1e-10)
        σ²_x_new = max(Σ_post[2,2], 1e-10)
        cov_zx = Σ_post[1,2]
    end
    
    return μ_z_new, μ_x_new, σ²_z_new, σ²_x_new, cov_zx
end

"""
    compute_circulatory_fidelity(σ²_z, σ²_x, cov_zx)

Compute Circulatory Fidelity from posterior moments.

CF = I(z;x) / H(z,x) where I is mutual information and H is joint entropy.

For Gaussian distributions:
- H(z,x) = 0.5 * log((2πe)² * det(Σ))
- I(z;x) = 0.5 * log(σ²_z * σ²_x / det(Σ))
- CF = I / H = log(σ²_z * σ²_x / det(Σ)) / log((2πe)² * det(Σ))

Simplified: CF = -log(1 - ρ²) / log((2πe)² * σ²_z * σ²_x * (1 - ρ²))
where ρ is the correlation coefficient.
"""
function compute_circulatory_fidelity(σ²_z::Float64, σ²_x::Float64, cov_zx::Float64)
    # Correlation coefficient
    ρ² = cov_zx^2 / (σ²_z * σ²_x)
    
    # Bound to avoid numerical issues
    ρ² = clamp(ρ², 0.0, 0.9999)
    
    if ρ² < 1e-10
        return 0.0  # No correlation = no CF
    end
    
    # Determinant of covariance matrix
    det_Σ = σ²_z * σ²_x * (1 - ρ²)
    
    # Joint entropy (in nats)
    H_joint = 0.5 * log((2π * ℯ)^2 * det_Σ)
    
    # Mutual information (in nats)
    I_zx = -0.5 * log(1 - ρ²)
    
    # CF = I / H (normalized)
    if H_joint > 0
        return clamp(I_zx / H_joint, 0.0, 1.0)
    else
        return 0.0
    end
end
