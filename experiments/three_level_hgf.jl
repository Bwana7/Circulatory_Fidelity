# ═══════════════════════════════════════════════════════════════════════════════
# three_level_hgf.jl - Three-Level Hierarchical Gaussian Filter
# ═══════════════════════════════════════════════════════════════════════════════

"""
Three-level HGF implementation for exploring CF in deeper hierarchies.

Generative model:
    Level 3: z₃ₜ | z₃ₜ₋₁ ~ N(z₃ₜ₋₁, ϑ₃⁻¹)
    Level 2: z₂ₜ | z₂ₜ₋₁, z₃ₜ ~ N(z₂ₜ₋₁, exp(κ₃z₃ₜ + ω₃))
    Level 1: z₁ₜ | z₁ₜ₋₁, z₂ₜ ~ N(z₁ₜ₋₁, exp(κ₂z₂ₜ + ω₂))
    Obs:     yₜ | z₁ₜ ~ N(z₁ₜ, πᵤ⁻¹)
"""

using Random
using Statistics
using LinearAlgebra
using Printf

# ─────────────────────────────────────────────────────────────────────────────
# Model Parameters
# ─────────────────────────────────────────────────────────────────────────────

struct ThreeLevelHGFParams
    κ₂::Float64    # Level 2→1 coupling
    κ₃::Float64    # Level 3→2 coupling
    ω₂::Float64    # Level 2 baseline log-volatility
    ω₃::Float64    # Level 3 baseline log-volatility
    ϑ₃::Float64    # Level 3 meta-volatility
    πᵤ::Float64    # Observation precision
end

function default_three_level_params(; ϑ₃::Float64=0.1)
    ThreeLevelHGFParams(1.0, 1.0, -2.0, -2.0, ϑ₃, 10.0)
end

# Two-level params for comparison
struct TwoLevelHGFParams
    κ::Float64
    ω::Float64
    ϑ::Float64
    πᵤ::Float64
end

function default_two_level_params(; ϑ::Float64=0.1)
    TwoLevelHGFParams(1.0, -2.0, ϑ, 10.0)
end

# ─────────────────────────────────────────────────────────────────────────────
# State Representation
# ─────────────────────────────────────────────────────────────────────────────

mutable struct ThreeLevelState
    μ₁::Float64; μ₂::Float64; μ₃::Float64
    σ²₁::Float64; σ²₂::Float64; σ²₃::Float64
end

mutable struct TwoLevelState
    μ_x::Float64; μ_z::Float64
    σ²_x::Float64; σ²_z::Float64
end

initial_state_3L() = ThreeLevelState(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
initial_state_2L() = TwoLevelState(0.0, 0.0, 1.0, 1.0)

# ─────────────────────────────────────────────────────────────────────────────
# Two-Level Updates (for comparison)
# ─────────────────────────────────────────────────────────────────────────────

function update_2L_meanfield!(state::TwoLevelState, y::Float64, p::TwoLevelHGFParams)
    π̂_x = exp(-p.κ * state.μ_z - p.ω)
    
    δ = y - state.μ_x
    K_x = p.πᵤ / (p.πᵤ + π̂_x)
    state.μ_x = state.μ_x + K_x * δ
    state.σ²_x = 1.0 / (p.πᵤ + π̂_x)
    
    ν = (δ^2 * p.πᵤ * π̂_x / (p.πᵤ + π̂_x)) + state.σ²_x * π̂_x - 1
    K_z = (p.κ / 2) * π̂_x / (p.ϑ + (p.κ^2 / 2) * π̂_x)
    state.μ_z = state.μ_z + K_z * ν
    state.σ²_z = 1.0 / (p.ϑ + (p.κ^2 / 2) * π̂_x)
    
    return (δ, ν)
end

function update_2L_structured!(state::TwoLevelState, y::Float64, p::TwoLevelHGFParams; γ::Float64=0.25)
    π̂_x = exp(-p.κ * state.μ_z - p.ω)
    
    δ = y - state.μ_x
    K_x = p.πᵤ / (p.πᵤ + π̂_x)
    state.μ_x = state.μ_x + K_x * δ
    state.σ²_x = 1.0 / (p.πᵤ + π̂_x)
    
    ν = (δ^2 * p.πᵤ * π̂_x / (p.πᵤ + π̂_x)) + state.σ²_x * π̂_x - 1
    K_z = (p.κ / 2) * π̂_x / (p.ϑ + (p.κ^2 / 2) * π̂_x)
    damping = 1.0 - γ * p.κ * π̂_x * (1.0 - exp(-abs(ν)))
    state.μ_z = state.μ_z + K_z * ν * damping
    state.σ²_z = 1.0 / (p.ϑ + (p.κ^2 / 2) * π̂_x)
    
    return (δ, ν)
end

# ─────────────────────────────────────────────────────────────────────────────
# Three-Level Mean-Field Updates
# ─────────────────────────────────────────────────────────────────────────────

function update_3L_meanfield!(state::ThreeLevelState, y::Float64, p::ThreeLevelHGFParams)
    # Predicted precisions
    π̂₁ = exp(-p.κ₂ * state.μ₂ - p.ω₂)
    π̂₂ = exp(-p.κ₃ * state.μ₃ - p.ω₃)
    
    # Level 1
    δ₁ = y - state.μ₁
    K₁ = p.πᵤ / (p.πᵤ + π̂₁)
    state.μ₁ = state.μ₁ + K₁ * δ₁
    state.σ²₁ = 1.0 / (p.πᵤ + π̂₁)
    
    # Level 2
    w₁ = p.πᵤ * π̂₁ / (p.πᵤ + π̂₁)
    ν₂ = (δ₁^2 * w₁) + state.σ²₁ * π̂₁ - 1
    K₂ = (p.κ₂ / 2) * π̂₁ / (π̂₂ + (p.κ₂^2 / 2) * π̂₁)
    state.μ₂ = state.μ₂ + K₂ * ν₂
    state.σ²₂ = 1.0 / (π̂₂ + (p.κ₂^2 / 2) * π̂₁)
    
    # Level 3
    w₂ = π̂₂ / (π̂₂ + (p.κ₂^2 / 2) * π̂₁)
    ν₃ = (ν₂^2 * w₂ * π̂₂) + state.σ²₂ * π̂₂ - 1
    K₃ = (p.κ₃ / 2) * π̂₂ / (p.ϑ₃ + (p.κ₃^2 / 2) * π̂₂)
    state.μ₃ = state.μ₃ + K₃ * ν₃
    state.σ²₃ = 1.0 / (p.ϑ₃ + (p.κ₃^2 / 2) * π̂₂)
    
    return (δ₁, ν₂, ν₃)
end

# ─────────────────────────────────────────────────────────────────────────────
# Three-Level Structured Updates
# ─────────────────────────────────────────────────────────────────────────────

function update_3L_structured!(state::ThreeLevelState, y::Float64, p::ThreeLevelHGFParams;
                               γ₁₂::Float64=0.25, γ₂₃::Float64=0.25)
    π̂₁ = exp(-p.κ₂ * state.μ₂ - p.ω₂)
    π̂₂ = exp(-p.κ₃ * state.μ₃ - p.ω₃)
    
    # Level 1 (same as mean-field)
    δ₁ = y - state.μ₁
    K₁ = p.πᵤ / (p.πᵤ + π̂₁)
    state.μ₁ = state.μ₁ + K₁ * δ₁
    state.σ²₁ = 1.0 / (p.πᵤ + π̂₁)
    
    # Level 2 (with damping from 1-2 coupling)
    w₁ = p.πᵤ * π̂₁ / (p.πᵤ + π̂₁)
    ν₂ = (δ₁^2 * w₁) + state.σ²₁ * π̂₁ - 1
    K₂ = (p.κ₂ / 2) * π̂₁ / (π̂₂ + (p.κ₂^2 / 2) * π̂₁)
    damping₂ = 1.0 - γ₁₂ * p.κ₂ * π̂₁ * (1.0 - exp(-abs(ν₂)))
    state.μ₂ = state.μ₂ + K₂ * ν₂ * damping₂
    state.σ²₂ = 1.0 / (π̂₂ + (p.κ₂^2 / 2) * π̂₁)
    
    # Level 3 (with damping from 2-3 coupling)
    w₂ = π̂₂ / (π̂₂ + (p.κ₂^2 / 2) * π̂₁)
    ν₃ = (ν₂^2 * w₂ * π̂₂) + state.σ²₂ * π̂₂ - 1
    K₃ = (p.κ₃ / 2) * π̂₂ / (p.ϑ₃ + (p.κ₃^2 / 2) * π̂₂)
    damping₃ = 1.0 - γ₂₃ * p.κ₃ * π̂₂ * (1.0 - exp(-abs(ν₃)))
    state.μ₃ = state.μ₃ + K₃ * ν₃ * damping₃
    state.σ²₃ = 1.0 / (p.ϑ₃ + (p.κ₃^2 / 2) * π̂₂)
    
    return (δ₁, ν₂, ν₃)
end

# Partial structuring variants
function update_3L_bottom_structured!(state::ThreeLevelState, y::Float64, p::ThreeLevelHGFParams)
    update_3L_structured!(state, y, p; γ₁₂=0.25, γ₂₃=0.0)
end

function update_3L_top_structured!(state::ThreeLevelState, y::Float64, p::ThreeLevelHGFParams)
    update_3L_structured!(state, y, p; γ₁₂=0.0, γ₂₃=0.25)
end

# ─────────────────────────────────────────────────────────────────────────────
# Lyapunov Exponent Computation
# ─────────────────────────────────────────────────────────────────────────────

function compute_lyapunov_2L(p::TwoLevelHGFParams, update_fn!::Function;
                             T::Int=10000, transient::Int=1000, seed::Int=42)
    Random.seed!(seed)
    observations = randn(T + transient)
    
    state_ref = initial_state_2L()
    state_pert = initial_state_2L()
    ε = 1e-8
    state_pert.μ_z += ε
    
    for t in 1:transient
        update_fn!(state_ref, observations[t], p)
        update_fn!(state_pert, observations[t], p)
    end
    
    lyap_sum = 0.0
    renorm_interval = 10
    n_renorms = 0
    
    for t in (transient+1):(T+transient)
        update_fn!(state_ref, observations[t], p)
        update_fn!(state_pert, observations[t], p)
        
        if t % renorm_interval == 0
            sep = sqrt((state_pert.μ_x - state_ref.μ_x)^2 +
                       (state_pert.μ_z - state_ref.μ_z)^2)
            
            if sep > 1e-15 && isfinite(sep)
                lyap_sum += log(sep / ε)
                n_renorms += 1
                
                factor = ε / sep
                state_pert.μ_x = state_ref.μ_x + factor * (state_pert.μ_x - state_ref.μ_x)
                state_pert.μ_z = state_ref.μ_z + factor * (state_pert.μ_z - state_ref.μ_z)
            end
        end
    end
    
    return n_renorms > 0 ? lyap_sum / (n_renorms * renorm_interval) : NaN
end

function compute_lyapunov_3L(p::ThreeLevelHGFParams, update_fn!::Function;
                             T::Int=10000, transient::Int=1000, seed::Int=42)
    Random.seed!(seed)
    observations = randn(T + transient)
    
    state_ref = initial_state_3L()
    state_pert = initial_state_3L()
    ε = 1e-8
    state_pert.μ₃ += ε
    
    for t in 1:transient
        update_fn!(state_ref, observations[t], p)
        update_fn!(state_pert, observations[t], p)
    end
    
    lyap_sum = 0.0
    renorm_interval = 10
    n_renorms = 0
    
    for t in (transient+1):(T+transient)
        update_fn!(state_ref, observations[t], p)
        update_fn!(state_pert, observations[t], p)
        
        if t % renorm_interval == 0
            sep = sqrt((state_pert.μ₁ - state_ref.μ₁)^2 +
                       (state_pert.μ₂ - state_ref.μ₂)^2 +
                       (state_pert.μ₃ - state_ref.μ₃)^2)
            
            if sep > 1e-15 && isfinite(sep)
                lyap_sum += log(sep / ε)
                n_renorms += 1
                
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
# Trajectory Simulation for CF Computation
# ─────────────────────────────────────────────────────────────────────────────

function simulate_3L(p::ThreeLevelHGFParams, update_fn!::Function, T::Int; seed::Int=42)
    Random.seed!(seed)
    observations = randn(T)
    
    state = initial_state_3L()
    μ₁_traj = zeros(T)
    μ₂_traj = zeros(T)
    μ₃_traj = zeros(T)
    
    for t in 1:T
        update_fn!(state, observations[t], p)
        μ₁_traj[t] = state.μ₁
        μ₂_traj[t] = state.μ₂
        μ₃_traj[t] = state.μ₃
    end
    
    return (μ₁=μ₁_traj, μ₂=μ₂_traj, μ₃=μ₃_traj)
end

function compute_pairwise_cf(traj; transient::Int=1000)
    μ₁ = traj.μ₁[(transient+1):end]
    μ₂ = traj.μ₂[(transient+1):end]
    μ₃ = traj.μ₃[(transient+1):end]
    
    # Correlations
    ρ₁₂ = clamp(cor(μ₁, μ₂), -0.9999, 0.9999)
    ρ₂₃ = clamp(cor(μ₂, μ₃), -0.9999, 0.9999)
    ρ₁₃ = clamp(cor(μ₁, μ₃), -0.9999, 0.9999)
    
    # Mutual informations (Gaussian)
    I₁₂ = -0.5 * log(1 - ρ₁₂^2)
    I₂₃ = -0.5 * log(1 - ρ₂₃^2)
    I₁₃ = -0.5 * log(1 - ρ₁₃^2)
    
    # Entropies
    H₁ = 0.5 * log(2π * exp(1) * max(var(μ₁), 1e-10))
    H₂ = 0.5 * log(2π * exp(1) * max(var(μ₂), 1e-10))
    H₃ = 0.5 * log(2π * exp(1) * max(var(μ₃), 1e-10))
    
    # Normalized CFs
    CF₁₂ = I₁₂ / max(min(H₁, H₂), 1e-10)
    CF₂₃ = I₂₃ / max(min(H₂, H₃), 1e-10)
    CF₁₃ = I₁₃ / max(min(H₁, H₃), 1e-10)
    
    # Total correlation
    TC = I₁₂ + I₂₃ + I₁₃  # Approximation
    
    return (CF₁₂=CF₁₂, CF₂₃=CF₂₃, CF₁₃=CF₁₃, I₁₂=I₁₂, I₂₃=I₂₃, TC=TC,
            ρ₁₂=ρ₁₂, ρ₂₃=ρ₂₃, ρ₁₃=ρ₁₃)
end

# ─────────────────────────────────────────────────────────────────────────────
# Main Analysis Functions
# ─────────────────────────────────────────────────────────────────────────────

function run_depth_comparison(ϑ::Float64; n_seeds::Int=5)
    """Compare 2-level vs 3-level stability at matched parameters."""
    
    p2 = default_two_level_params(ϑ=ϑ)
    p3 = default_three_level_params(ϑ₃=ϑ)
    
    λ_2L_mf = Float64[]
    λ_2L_st = Float64[]
    λ_3L_mf = Float64[]
    λ_3L_st = Float64[]
    
    for seed in 1:n_seeds
        push!(λ_2L_mf, compute_lyapunov_2L(p2, update_2L_meanfield!; seed=seed))
        push!(λ_2L_st, compute_lyapunov_2L(p2, update_2L_structured!; seed=seed))
        push!(λ_3L_mf, compute_lyapunov_3L(p3, update_3L_meanfield!; seed=seed))
        push!(λ_3L_st, compute_lyapunov_3L(p3, update_3L_structured!; seed=seed))
    end
    
    return (
        λ_2L_mf = (mean=mean(λ_2L_mf), std=std(λ_2L_mf)),
        λ_2L_st = (mean=mean(λ_2L_st), std=std(λ_2L_st)),
        λ_3L_mf = (mean=mean(λ_3L_mf), std=std(λ_3L_mf)),
        λ_3L_st = (mean=mean(λ_3L_st), std=std(λ_3L_st))
    )
end

function run_partial_structuring_analysis(ϑ₃::Float64; n_seeds::Int=5)
    """Compare all four approximation schemes for 3-level."""
    
    p = default_three_level_params(ϑ₃=ϑ₃)
    
    results = Dict{Symbol, Vector{Float64}}(
        :meanfield => Float64[],
        :structured => Float64[],
        :bottom => Float64[],
        :top => Float64[]
    )
    
    for seed in 1:n_seeds
        push!(results[:meanfield], compute_lyapunov_3L(p, update_3L_meanfield!; seed=seed))
        push!(results[:structured], compute_lyapunov_3L(p, update_3L_structured!; seed=seed))
        push!(results[:bottom], compute_lyapunov_3L(p, update_3L_bottom_structured!; seed=seed))
        push!(results[:top], compute_lyapunov_3L(p, update_3L_top_structured!; seed=seed))
    end
    
    return Dict(k => (mean=mean(v), std=std(v)) for (k, v) in results)
end

function run_bifurcation_sweep(ϑ₃_values::Vector{Float64}; n_seeds::Int=3)
    """Sweep over ϑ₃ values to find bifurcation structure."""
    
    results = Dict(
        :ϑ₃ => ϑ₃_values,
        :λ_mf_mean => Float64[],
        :λ_mf_std => Float64[],
        :λ_st_mean => Float64[],
        :λ_st_std => Float64[],
        :λ_bottom_mean => Float64[],
        :λ_top_mean => Float64[]
    )
    
    for ϑ₃ in ϑ₃_values
        p = default_three_level_params(ϑ₃=ϑ₃)
        
        λ_mf = [compute_lyapunov_3L(p, update_3L_meanfield!; seed=s) for s in 1:n_seeds]
        λ_st = [compute_lyapunov_3L(p, update_3L_structured!; seed=s) for s in 1:n_seeds]
        λ_bot = [compute_lyapunov_3L(p, update_3L_bottom_structured!; seed=s) for s in 1:n_seeds]
        λ_top = [compute_lyapunov_3L(p, update_3L_top_structured!; seed=s) for s in 1:n_seeds]
        
        push!(results[:λ_mf_mean], mean(λ_mf))
        push!(results[:λ_mf_std], std(λ_mf))
        push!(results[:λ_st_mean], mean(λ_st))
        push!(results[:λ_st_std], std(λ_st))
        push!(results[:λ_bottom_mean], mean(λ_bot))
        push!(results[:λ_top_mean], mean(λ_top))
    end
    
    return results
end

function run_cf_tracking(ϑ₃::Float64; T::Int=11000)
    """Track pairwise CFs for different approximation schemes."""
    
    p = default_three_level_params(ϑ₃=ϑ₃)
    
    results = Dict{Symbol, NamedTuple}()
    
    for (name, update_fn!) in [
        (:meanfield, update_3L_meanfield!),
        (:structured, update_3L_structured!),
        (:bottom, update_3L_bottom_structured!),
        (:top, update_3L_top_structured!)
    ]
        traj = simulate_3L(p, update_fn!, T)
        cf = compute_pairwise_cf(traj)
        results[name] = cf
    end
    
    return results
end

# ─────────────────────────────────────────────────────────────────────────────
# Full Analysis Runner
# ─────────────────────────────────────────────────────────────────────────────

function run_full_analysis()
    println("═" ^ 70)
    println("THREE-LEVEL HGF: COMPLETE STABILITY ANALYSIS")
    println("═" ^ 70)
    
    # ─── Part 1: Depth Comparison ───
    println("\n" * "─" ^ 70)
    println("PART 1: DEPTH COMPARISON (2-Level vs 3-Level)")
    println("─" ^ 70)
    
    test_ϑ_values = [0.05, 0.08, 0.10, 0.15]
    
    println("\nϑ\t\t2L-MF\t\t2L-Struct\t3L-MF\t\t3L-Struct")
    println("─" ^ 70)
    
    depth_results = Dict()
    for ϑ in test_ϑ_values
        r = run_depth_comparison(ϑ; n_seeds=5)
        depth_results[ϑ] = r
        @printf("%.2f\t\t%.3f±%.2f\t%.3f±%.2f\t%.3f±%.2f\t%.3f±%.2f\n",
            ϑ, r.λ_2L_mf.mean, r.λ_2L_mf.std,
            r.λ_2L_st.mean, r.λ_2L_st.std,
            r.λ_3L_mf.mean, r.λ_3L_mf.std,
            r.λ_3L_st.mean, r.λ_3L_st.std)
    end
    
    # ─── Part 2: Partial Structuring ───
    println("\n" * "─" ^ 70)
    println("PART 2: PARTIAL STRUCTURING ANALYSIS (ϑ₃ = 0.10)")
    println("─" ^ 70)
    
    partial_results = run_partial_structuring_analysis(0.10; n_seeds=5)
    
    println("\nApproximation\t\tλ_max (mean ± std)")
    println("─" ^ 50)
    for (name, stats) in sort(collect(partial_results), by=x->x[2].mean, rev=true)
        @printf("%-20s\t%.3f ± %.3f\n", name, stats.mean, stats.std)
    end
    
    # ─── Part 3: Bifurcation Sweep ───
    println("\n" * "─" ^ 70)
    println("PART 3: BIFURCATION ANALYSIS (ϑ₃ sweep)")
    println("─" ^ 70)
    
    ϑ₃_sweep = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
    bif_results = run_bifurcation_sweep(ϑ₃_sweep; n_seeds=3)
    
    println("\nϑ₃\tMean-Field\tStructured\tBottom-Only\tTop-Only")
    println("─" ^ 70)
    for i in eachindex(ϑ₃_sweep)
        @printf("%.2f\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\n",
            bif_results[:ϑ₃][i],
            bif_results[:λ_mf_mean][i],
            bif_results[:λ_st_mean][i],
            bif_results[:λ_bottom_mean][i],
            bif_results[:λ_top_mean][i])
    end
    
    # ─── Part 4: CF Tracking ───
    println("\n" * "─" ^ 70)
    println("PART 4: PAIRWISE CF TRACKING (ϑ₃ = 0.10)")
    println("─" ^ 70)
    
    cf_results = run_cf_tracking(0.10)
    
    println("\nApproximation\t\tCF₁₂\t\tCF₂₃\t\tρ₁₂\t\tρ₂₃")
    println("─" ^ 70)
    for name in [:meanfield, :structured, :bottom, :top]
        cf = cf_results[name]
        @printf("%-20s\t%.3f\t\t%.3f\t\t%.3f\t\t%.3f\n",
            name, cf.CF₁₂, cf.CF₂₃, cf.ρ₁₂, cf.ρ₂₃)
    end
    
    # ─── Summary ───
    println("\n" * "═" ^ 70)
    println("SUMMARY OF FINDINGS")
    println("═" ^ 70)
    
    # Check if 3L is worse than 2L for mean-field
    worse_with_depth = any(ϑ -> depth_results[ϑ].λ_3L_mf.mean > depth_results[ϑ].λ_2L_mf.mean, test_ϑ_values)
    
    println("\n1. DEPTH EFFECT:")
    if worse_with_depth
        println("   ✓ Three-level mean-field shows HIGHER Lyapunov exponents")
        println("   → Instabilities WORSEN with hierarchical depth")
    else
        println("   → No clear worsening of instabilities with depth")
    end
    
    println("\n2. STRUCTURING EFFECT:")
    println("   Mean-field λ: $(round(partial_results[:meanfield].mean, digits=3))")
    println("   Structured λ:  $(round(partial_results[:structured].mean, digits=3))")
    println("   Bottom-only λ: $(round(partial_results[:bottom].mean, digits=3))")
    println("   Top-only λ:    $(round(partial_results[:top].mean, digits=3))")
    
    println("\n3. CRITICAL INTERFACE:")
    if partial_results[:bottom].mean < partial_results[:top].mean
        println("   → Bottom interface (1-2) appears MORE critical for stability")
    elseif partial_results[:top].mean < partial_results[:bottom].mean
        println("   → Top interface (2-3) appears MORE critical for stability")
    else
        println("   → Both interfaces contribute similarly to stability")
    end
    
    println("\n4. CF VALUES:")
    println("   Under mean-field: CF₁₂ ≈ $(round(cf_results[:meanfield].CF₁₂, digits=3)), CF₂₃ ≈ $(round(cf_results[:meanfield].CF₂₃, digits=3))")
    println("   Under structured: CF₁₂ ≈ $(round(cf_results[:structured].CF₁₂, digits=3)), CF₂₃ ≈ $(round(cf_results[:structured].CF₂₃, digits=3))")
    
    return (depth=depth_results, partial=partial_results, bifurcation=bif_results, cf=cf_results)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_full_analysis()
end
