# ═══════════════════════════════════════════════════════════════════════════════
# three_level_models.jl - Three-Level HGF for Circulatory Fidelity Extension
# ═══════════════════════════════════════════════════════════════════════════════
#
# Extends the two-level CF analysis to three-level hierarchies.
# Follows the same structure as models.jl for consistency.
#
# Reference: Mathys et al. (2014) for three-level HGF specification
# ═══════════════════════════════════════════════════════════════════════════════

module ThreeLevelHGF

export ThreeLevelParams, ThreeLevelState, ThreeLevelResults
export update_meanfield!, update_structured!
export update_bottom_structured!, update_top_structured!
export simulate, compute_pairwise_cf, analyze_stability

using Statistics
using LinearAlgebra
using Random

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

"""
    ThreeLevelParams

Parameters for the three-level HGF.

# Fields
- `κ₂::Float64`: Level 2→1 coupling strength (> 0)
- `κ₃::Float64`: Level 3→2 coupling strength (> 0)
- `ω₂::Float64`: Level 2 baseline log-volatility (typically < 0)
- `ω₃::Float64`: Level 3 baseline log-volatility (typically < 0)
- `ϑ₃::Float64`: Level 3 volatility / meta-meta-volatility (> 0, typically ≪ 1)
- `π_u::Float64`: Observation precision (> 0)

# Generative Model
```
Level 3: z₃ₜ | z₃ₜ₋₁ ~ N(z₃ₜ₋₁, ϑ₃⁻¹)
Level 2: z₂ₜ | z₂ₜ₋₁, z₃ₜ ~ N(z₂ₜ₋₁, exp(κ₃z₃ₜ + ω₃))
Level 1: z₁ₜ | z₁ₜ₋₁, z₂ₜ ~ N(z₁ₜ₋₁, exp(κ₂z₂ₜ + ω₂))
Obs:     yₜ | z₁ₜ ~ N(z₁ₜ, π_u⁻¹)
```
"""
Base.@kwdef struct ThreeLevelParams
    κ₂::Float64 = 1.0
    κ₃::Float64 = 1.0
    ω₂::Float64 = -2.0
    ω₃::Float64 = -2.0
    ϑ₃::Float64 = 0.1
    π_u::Float64 = 10.0
end

# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

"""
    ThreeLevelState

Mutable state of the three-level HGF inference.

# Fields
- `μ₁, μ₂, μ₃`: Posterior means at each level
- `σ²₁, σ²₂, σ²₃`: Posterior variances at each level
"""
mutable struct ThreeLevelState
    μ₁::Float64
    μ₂::Float64
    μ₃::Float64
    σ²₁::Float64
    σ²₂::Float64
    σ²₃::Float64
end

# Default initial state
ThreeLevelState() = ThreeLevelState(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

# Copy constructor
Base.copy(s::ThreeLevelState) = ThreeLevelState(s.μ₁, s.μ₂, s.μ₃, s.σ²₁, s.σ²₂, s.σ²₃)

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────

"""
    ThreeLevelResults

Results from three-level HGF simulation.

# Fields
- `μ₁, μ₂, μ₃`: Time series of posterior means
- `σ²₁, σ²₂, σ²₃`: Time series of posterior variances
- `CF₁₂, CF₂₃`: Pairwise circulatory fidelity (computed post-hoc)
- `observations`: The observation sequence used
"""
struct ThreeLevelResults
    μ₁::Vector{Float64}
    μ₂::Vector{Float64}
    μ₃::Vector{Float64}
    σ²₁::Vector{Float64}
    σ²₂::Vector{Float64}
    σ²₃::Vector{Float64}
    CF₁₂::Float64
    CF₂₃::Float64
    observations::Vector{Float64}
end

# ─────────────────────────────────────────────────────────────────────────────
# Update Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    update_meanfield!(state, y, params)

Perform one mean-field variational update: q(z₁,z₂,z₃) = q(z₁)q(z₂)q(z₃).
Each level updates independently (no cross-level information flow).

# Arguments
- `state::ThreeLevelState`: Current state (modified in place)
- `y::Float64`: Current observation
- `params::ThreeLevelParams`: Model parameters

# Returns
- Tuple of prediction errors and gains: (δ₁, ν₂, ν₃, K₁, K₂, K₃)
"""
function update_meanfield!(state::ThreeLevelState, y::Float64, p::ThreeLevelParams)
    # Predicted precisions (inverse variances of transitions)
    π̂₁ = exp(-p.κ₂ * state.μ₂ - p.ω₂)
    π̂₂ = exp(-p.κ₃ * state.μ₃ - p.ω₃)
    
    # Numerical stability
    π̂₁ = clamp(π̂₁, 1e-6, 1e6)
    π̂₂ = clamp(π̂₂, 1e-6, 1e6)
    
    # ─── Level 1 Update ───
    δ₁ = y - state.μ₁
    K₁ = p.π_u / (p.π_u + π̂₁)
    
    state.μ₁ += K₁ * δ₁
    state.σ²₁ = 1.0 / (p.π_u + π̂₁)
    
    # ─── Level 2 Update ───
    # Volatility prediction error: weighted squared PE from level below
    ν₂ = (δ₁^2 * p.π_u * π̂₁ / (p.π_u + π̂₁)) + state.σ²₁ * π̂₁ - 1
    
    K₂_denom = π̂₂ + (p.κ₂^2 / 2) * π̂₁
    K₂ = K₂_denom > 1e-10 ? (p.κ₂ / 2) * π̂₁ / K₂_denom : 0.0
    
    state.μ₂ += K₂ * ν₂
    state.σ²₂ = K₂_denom > 1e-10 ? 1.0 / K₂_denom : state.σ²₂
    
    # ─── Level 3 Update ───
    # Meta-volatility prediction error
    ν₃ = (ν₂^2 * π̂₂ / max(K₂_denom, 1e-10)) + state.σ²₂ * π̂₂ - 1
    
    K₃_denom = p.ϑ₃ + (p.κ₃^2 / 2) * π̂₂
    K₃ = K₃_denom > 1e-10 ? (p.κ₃ / 2) * π̂₂ / K₃_denom : 0.0
    
    state.μ₃ += K₃ * ν₃
    state.σ²₃ = K₃_denom > 1e-10 ? 1.0 / K₃_denom : state.σ²₃
    
    # Prevent divergence
    state.μ₁ = clamp(state.μ₁, -100.0, 100.0)
    state.μ₂ = clamp(state.μ₂, -20.0, 20.0)
    state.μ₃ = clamp(state.μ₃, -20.0, 20.0)
    
    return (δ₁, ν₂, ν₃, K₁, K₂, K₃)
end

"""
    update_structured!(state, y, params; γ₁₂=0.3, γ₂₃=0.3)

Perform structured variational update with Markov dependencies.
q(z₁,z₂,z₃) = q(z₃)q(z₂|z₃)q(z₁|z₂)

Cross-level coupling provides damping that stabilizes dynamics.

# Arguments
- `state::ThreeLevelState`: Current state (modified in place)
- `y::Float64`: Current observation
- `params::ThreeLevelParams`: Model parameters
- `γ₁₂::Float64`: Coupling strength at 1-2 interface (default 0.3)
- `γ₂₃::Float64`: Coupling strength at 2-3 interface (default 0.3)
"""
function update_structured!(state::ThreeLevelState, y::Float64, p::ThreeLevelParams;
                           γ₁₂::Float64=0.3, γ₂₃::Float64=0.3)
    # Predicted precisions
    π̂₁ = clamp(exp(-p.κ₂ * state.μ₂ - p.ω₂), 1e-6, 1e6)
    π̂₂ = clamp(exp(-p.κ₃ * state.μ₃ - p.ω₃), 1e-6, 1e6)
    
    # ─── Level 1 Update (same as mean-field) ───
    δ₁ = y - state.μ₁
    K₁ = p.π_u / (p.π_u + π̂₁)
    
    state.μ₁ += K₁ * δ₁
    state.σ²₁ = 1.0 / (p.π_u + π̂₁)
    
    # ─── Level 2 Update (with damping) ───
    ν₂ = (δ₁^2 * p.π_u * π̂₁ / (p.π_u + π̂₁)) + state.σ²₁ * π̂₁ - 1
    
    K₂_denom = π̂₂ + (p.κ₂^2 / 2) * π̂₁
    K₂ = K₂_denom > 1e-10 ? (p.κ₂ / 2) * π̂₁ / K₂_denom : 0.0
    
    # Damping from cross-level coupling (key stability mechanism)
    damping₂ = 1.0 / (1.0 + γ₁₂ * abs(ν₂))
    
    state.μ₂ += K₂ * ν₂ * damping₂
    state.σ²₂ = K₂_denom > 1e-10 ? 1.0 / K₂_denom : state.σ²₂
    
    # ─── Level 3 Update (with damping) ───
    ν₃ = (ν₂^2 * π̂₂ / max(K₂_denom, 1e-10)) + state.σ²₂ * π̂₂ - 1
    
    K₃_denom = p.ϑ₃ + (p.κ₃^2 / 2) * π̂₂
    K₃ = K₃_denom > 1e-10 ? (p.κ₃ / 2) * π̂₂ / K₃_denom : 0.0
    
    damping₃ = 1.0 / (1.0 + γ₂₃ * abs(ν₃))
    
    state.μ₃ += K₃ * ν₃ * damping₃
    state.σ²₃ = K₃_denom > 1e-10 ? 1.0 / K₃_denom : state.σ²₃
    
    # Prevent divergence
    state.μ₁ = clamp(state.μ₁, -100.0, 100.0)
    state.μ₂ = clamp(state.μ₂, -20.0, 20.0)
    state.μ₃ = clamp(state.μ₃, -20.0, 20.0)
    
    return (δ₁, ν₂, ν₃, K₁, K₂, K₃)
end

"""
    update_bottom_structured!(state, y, params)

Bottom-structured only: q(z₃)q(z₂)q(z₁|z₂)
Maintains coupling only at the 1-2 interface (sensory-volatility).
"""
function update_bottom_structured!(state::ThreeLevelState, y::Float64, p::ThreeLevelParams)
    update_structured!(state, y, p; γ₁₂=0.3, γ₂₃=0.0)
end

"""
    update_top_structured!(state, y, params)

Top-structured only: q(z₃)q(z₂|z₃)q(z₁)
Maintains coupling only at the 2-3 interface (volatility-meta-volatility).
"""
function update_top_structured!(state::ThreeLevelState, y::Float64, p::ThreeLevelParams)
    update_structured!(state, y, p; γ₁₂=0.0, γ₂₃=0.3)
end

# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────

"""
    simulate(params, T, update_fn!; seed=42)

Simulate T timesteps of the three-level HGF.

# Arguments
- `params::ThreeLevelParams`: Model parameters
- `T::Int`: Number of timesteps
- `update_fn!::Function`: Update function to use (e.g., update_meanfield!)
- `seed::Int`: Random seed for reproducibility

# Returns
- `ThreeLevelResults`: Full simulation results including trajectories and CF
"""
function simulate(p::ThreeLevelParams, T::Int, update_fn!::Function; seed::Int=42)
    Random.seed!(seed)
    
    # Generate observations from a process with time-varying volatility
    # True volatility oscillates sinusoidally (challenging test case)
    true_vol = 0.5 .+ 0.3 .* sin.(2π .* (1:T) ./ 100)
    true_x = cumsum(sqrt.(true_vol) .* randn(T))
    observations = true_x .+ randn(T) ./ sqrt(p.π_u)
    
    # Initialize
    state = ThreeLevelState()
    
    # Storage
    μ₁ = zeros(T)
    μ₂ = zeros(T)
    μ₃ = zeros(T)
    σ²₁ = zeros(T)
    σ²₂ = zeros(T)
    σ²₃ = zeros(T)
    
    # Run simulation
    for t in 1:T
        update_fn!(state, observations[t], p)
        
        μ₁[t] = state.μ₁
        μ₂[t] = state.μ₂
        μ₃[t] = state.μ₃
        σ²₁[t] = state.σ²₁
        σ²₂[t] = state.σ²₂
        σ²₃[t] = state.σ²₃
    end
    
    # Compute pairwise CF from trajectories
    CF₁₂, CF₂₃ = compute_pairwise_cf(μ₁, μ₂, μ₃; burnin=min(2000, T÷4))
    
    return ThreeLevelResults(μ₁, μ₂, μ₃, σ²₁, σ²₂, σ²₃, CF₁₂, CF₂₃, observations)
end

# ─────────────────────────────────────────────────────────────────────────────
# Analysis Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    compute_pairwise_cf(μ₁, μ₂, μ₃; burnin=2000)

Compute pairwise Circulatory Fidelity from trajectory data.

For Gaussian variables, MI = -0.5 * log(1 - ρ²), and we normalize
by the minimum entropy to get CF ∈ [0, 1].

# Returns
- `(CF₁₂, CF₂₃)`: Tuple of CF values at each interface
"""
function compute_pairwise_cf(μ₁::Vector{Float64}, μ₂::Vector{Float64}, μ₃::Vector{Float64};
                             burnin::Int=2000)
    # Discard burnin period
    idx = (burnin+1):length(μ₁)
    if length(idx) < 100
        return (0.0, 0.0)
    end
    
    m₁, m₂, m₃ = μ₁[idx], μ₂[idx], μ₃[idx]
    
    # Compute correlations (handle degenerate cases)
    function safe_cor(x, y)
        if std(x) < 1e-10 || std(y) < 1e-10
            return 0.0
        end
        r = cor(x, y)
        return clamp(r, -0.9999, 0.9999)
    end
    
    ρ₁₂ = safe_cor(m₁, m₂)
    ρ₂₃ = safe_cor(m₂, m₃)
    
    # MI for Gaussians: I(X;Y) = -0.5 * log(1 - ρ²)
    I₁₂ = abs(ρ₁₂) > 0.01 ? -0.5 * log(1 - ρ₁₂^2) : 0.0
    I₂₃ = abs(ρ₂₃) > 0.01 ? -0.5 * log(1 - ρ₂₃^2) : 0.0
    
    # Differential entropy of Gaussian: H = 0.5 * log(2πe * var)
    H₁ = 0.5 * log(2π * ℯ * max(var(m₁), 1e-10))
    H₂ = 0.5 * log(2π * ℯ * max(var(m₂), 1e-10))
    H₃ = 0.5 * log(2π * ℯ * max(var(m₃), 1e-10))
    
    # Normalized CF
    CF₁₂ = min(H₁, H₂) > 0.1 ? I₁₂ / min(H₁, H₂) : 0.0
    CF₂₃ = min(H₂, H₃) > 0.1 ? I₂₃ / min(H₂, H₃) : 0.0
    
    return (CF₁₂, CF₂₃)
end

"""
    analyze_stability(results; burnin=2000)

Compute stability metrics from simulation results.

# Returns
- Dictionary with variance, rate of change, and autocorrelation at each level
"""
function analyze_stability(r::ThreeLevelResults; burnin::Int=2000)
    idx = (burnin+1):length(r.μ₁)
    
    # Variances (primary stability indicator)
    var₁ = var(r.μ₁[idx])
    var₂ = var(r.μ₂[idx])
    var₃ = var(r.μ₃[idx])
    
    # Rate of change (oscillation indicator)
    roc₂ = mean(abs.(diff(r.μ₂[idx])))
    roc₃ = mean(abs.(diff(r.μ₃[idx])))
    
    # Autocorrelation at lag 1 (negative indicates oscillation)
    function autocor(x)
        x = x .- mean(x)
        std(x) < 1e-10 && return 1.0
        return cor(x[1:end-1], x[2:end])
    end
    
    ac₂ = autocor(r.μ₂[idx])
    ac₃ = autocor(r.μ₃[idx])
    
    return Dict(
        :var₁ => var₁, :var₂ => var₂, :var₃ => var₃,
        :roc₂ => roc₂, :roc₃ => roc₃,
        :ac₂ => ac₂, :ac₃ => ac₃,
        :CF₁₂ => r.CF₁₂, :CF₂₃ => r.CF₂₃
    )
end

"""
    compute_lyapunov(params, update_fn!; T=10000, transient=2000, seed=42)

Compute maximal Lyapunov exponent using trajectory separation method.

# Returns
- `λ_max::Float64`: Maximal Lyapunov exponent (positive = chaos, negative = stable)
"""
function compute_lyapunov(p::ThreeLevelParams, update_fn!::Function;
                          T::Int=10000, transient::Int=2000, seed::Int=42)
    Random.seed!(seed)
    
    # Generate observation sequence
    observations = randn(T + transient) ./ sqrt(p.π_u)
    
    # Initialize reference and perturbed trajectories
    state_ref = ThreeLevelState()
    state_pert = ThreeLevelState()
    
    # Initial perturbation
    ε = 1e-8
    state_pert.μ₃ += ε
    
    # Discard transient
    for t in 1:transient
        update_fn!(state_ref, observations[t], p)
        update_fn!(state_pert, observations[t], p)
    end
    
    # Compute Lyapunov exponent via repeated renormalization
    lyap_sum = 0.0
    renorm_interval = 10
    n_renorms = 0
    
    for t in (transient+1):(T+transient)
        update_fn!(state_ref, observations[t], p)
        update_fn!(state_pert, observations[t], p)
        
        if t % renorm_interval == 0
            # Compute separation in state space
            sep = sqrt(
                (state_pert.μ₁ - state_ref.μ₁)^2 +
                (state_pert.μ₂ - state_ref.μ₂)^2 +
                (state_pert.μ₃ - state_ref.μ₃)^2
            )
            
            if sep > 1e-15 && isfinite(sep) && sep < 1e10
                lyap_sum += log(sep / ε)
                n_renorms += 1
                
                # Renormalize perturbation
                factor = ε / sep
                state_pert.μ₁ = state_ref.μ₁ + factor * (state_pert.μ₁ - state_ref.μ₁)
                state_pert.μ₂ = state_ref.μ₂ + factor * (state_pert.μ₂ - state_ref.μ₂)
                state_pert.μ₃ = state_ref.μ₃ + factor * (state_pert.μ₃ - state_ref.μ₃)
            end
        end
    end
    
    return n_renorms > 0 ? lyap_sum / (n_renorms * renorm_interval) : NaN
end

# ─────────────────────────────────────────────────────────────────────────────
# Comparison Functions
# ─────────────────────────────────────────────────────────────────────────────

"""
    compare_approximations(params; T=8000, seed=42)

Compare all four approximation schemes for given parameters.

# Returns
- Dictionary with results for each scheme
"""
function compare_approximations(p::ThreeLevelParams; T::Int=8000, seed::Int=42)
    schemes = [
        (:meanfield, update_meanfield!),
        (:structured, (s, y, p) -> update_structured!(s, y, p)),
        (:bottom_only, update_bottom_structured!),
        (:top_only, update_top_structured!)
    ]
    
    results = Dict{Symbol, Dict}()
    
    for (name, update_fn!) in schemes
        r = simulate(p, T, update_fn!; seed=seed)
        stats = analyze_stability(r)
        results[name] = stats
    end
    
    return results
end

"""
    compare_depth(ϑ₃; T=8000, seed=42)

Compare two-level vs three-level stability at given meta-volatility.

# Returns
- Dictionary with variance comparisons
"""
function compare_depth(ϑ₃::Float64; T::Int=8000, seed::Int=42)
    p = ThreeLevelParams(ϑ₃=ϑ₃)
    
    # Two-level: freeze μ₃ at 0
    function update_2level_mf!(state, y, p)
        update_meanfield!(state, y, p)
        state.μ₃ = 0.0
        state.σ²₃ = 1.0
    end
    
    function update_2level_struct!(state, y, p)
        update_structured!(state, y, p)
        state.μ₃ = 0.0
        state.σ²₃ = 1.0
    end
    
    r_2l_mf = simulate(p, T, update_2level_mf!; seed=seed)
    r_3l_mf = simulate(p, T, update_meanfield!; seed=seed)
    r_2l_st = simulate(p, T, update_2level_struct!; seed=seed)
    r_3l_st = simulate(p, T, (s,y,p) -> update_structured!(s,y,p); seed=seed)
    
    return Dict(
        :var₂_2L_MF => var(r_2l_mf.μ₂[2001:end]),
        :var₂_3L_MF => var(r_3l_mf.μ₂[2001:end]),
        :var₂_2L_ST => var(r_2l_st.μ₂[2001:end]),
        :var₂_3L_ST => var(r_3l_st.μ₂[2001:end])
    )
end

end # module
