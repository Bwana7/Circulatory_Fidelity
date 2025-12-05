# ═══════════════════════════════════════════════════════════════════════════════
# lyapunov.jl - Dynamical Systems Analysis for Circulatory Fidelity
# ═══════════════════════════════════════════════════════════════════════════════

# Simulation constants
const N_TIMESTEPS_DEFAULT = 10_000
const N_TRANSIENT_DEFAULT = 1_000
const RENORM_INTERVAL = 10
const ε_LYAP = 1e-8
const D_BASELINE_LYAP = 50.0  # Default baseline dopamine (nM)

"""
    compute_lyapunov(params::CFParameters;
                     mode::Symbol=:mean_field,
                     DA_level::Union{Nothing,Float64}=nothing,
                     n_steps::Int=N_TIMESTEPS_DEFAULT,
                     n_transient::Int=N_TRANSIENT_DEFAULT,
                     seed::Int=42)

Compute the maximal Lyapunov exponent using the Benettin algorithm.

The Lyapunov exponent characterizes the rate of separation of infinitesimally
close trajectories:

    λ_max = lim_{t→∞} (1/t) ln(|δZ(t)| / |δZ(0)|)

Interpretation:
- λ_max < 0: Stable fixed point or limit cycle
- λ_max = 0: Marginally stable (quasiperiodic)
- λ_max > 0: Chaos (exponential divergence)

# Arguments
- `params::CFParameters`: Model parameters
- `mode::Symbol`: :mean_field or :structured
- `DA_level`: Dopamine level in nM (uses 50 nM baseline if nothing)
- `n_steps::Int`: Total simulation steps
- `n_transient::Int`: Transient steps to discard
- `seed::Int`: Random seed for reproducibility

# Returns
Named tuple (λ_max, λ_std, trajectory)
- `λ_max`: Maximal Lyapunov exponent (bits/timestep)
- `λ_std`: Standard error estimate
- `trajectory`: Time series of μ_z values (for bifurcation diagrams)

# References
- Benettin et al. (1980). Lyapunov characteristic exponents.
"""
function compute_lyapunov(params::CFParameters;
                          mode::Symbol=:mean_field,
                          DA_level::Union{Nothing,Float64}=nothing,
                          n_steps::Int=N_TIMESTEPS_DEFAULT,
                          n_transient::Int=N_TRANSIENT_DEFAULT,
                          seed::Int=42)
    
    # Compute precision from dopamine (default to baseline tonic level)
    D = isnothing(DA_level) ? D_BASELINE_LYAP : DA_level
    γ_DA = dopamine_to_precision(D, params)
    
    # Initialize reference trajectory
    μ_z = 0.0
    μ_x = 0.0
    σ²_z = 1.0
    σ²_x = 1.0
    
    # Initialize perturbed trajectory
    μ_z_pert = μ_z + ε_LYAP
    μ_x_pert = μ_x + ε_LYAP
    σ²_z_pert = σ²_z
    σ²_x_pert = σ²_x
    
    # Generate observation sequence (fixed across trajectories)
    Random.seed!(seed)
    observations = randn(n_steps) .* sqrt(1/params.π_u)
    
    # Storage for trajectory (for bifurcation diagrams)
    trajectory = Float64[]
    
    # Discard transient
    for t in 1:n_transient
        y_t = observations[t]
        
        # Update reference trajectory
        μ_z, μ_x, σ²_z, σ²_x, _ = variational_update_full(
            μ_z, μ_x, σ²_z, σ²_x, y_t, params, γ_DA, mode
        )
        
        # Update perturbed trajectory
        μ_z_pert, μ_x_pert, σ²_z_pert, σ²_x_pert, _ = variational_update_full(
            μ_z_pert, μ_x_pert, σ²_z_pert, σ²_x_pert, y_t, params, γ_DA, mode
        )
    end
    
    # Compute Lyapunov exponent
    lyap_sum = 0.0
    lyap_values = Float64[]
    n_renorm = 0
    
    for t in (n_transient+1):n_steps
        y_t = observations[t]
        
        # Update trajectories
        μ_z, μ_x, σ²_z, σ²_x, _ = variational_update_full(
            μ_z, μ_x, σ²_z, σ²_x, y_t, params, γ_DA, mode
        )
        
        μ_z_pert, μ_x_pert, σ²_z_pert, σ²_x_pert, _ = variational_update_full(
            μ_z_pert, μ_x_pert, σ²_z_pert, σ²_x_pert, y_t, params, γ_DA, mode
        )
        
        # Store trajectory point
        push!(trajectory, μ_z)
        
        # Compute separation
        δ_z = μ_z_pert - μ_z
        δ_x = μ_x_pert - μ_x
        δ_norm = sqrt(δ_z^2 + δ_x^2)
        
        # Periodic renormalization
        if t % RENORM_INTERVAL == 0
            if δ_norm > 0 && isfinite(δ_norm)
                λ_local = log(δ_norm / ε_LYAP)
                push!(lyap_values, λ_local)
                lyap_sum += λ_local
                n_renorm += 1
                
                # Renormalize perturbation
                μ_z_pert = μ_z + ε_LYAP * δ_z / δ_norm
                μ_x_pert = μ_x + ε_LYAP * δ_x / δ_norm
            else
                # Trajectory diverged - reinitialize perturbation
                μ_z_pert = μ_z + ε_LYAP
                μ_x_pert = μ_x + ε_LYAP
            end
        end
    end
    
    # Compute average and standard error
    if n_renorm > 0
        λ_max = lyap_sum / (n_renorm * RENORM_INTERVAL)
        λ_std = length(lyap_values) > 1 ? std(lyap_values) / sqrt(n_renorm) : 0.0
    else
        λ_max = NaN
        λ_std = NaN
    end
    
    return (λ_max=λ_max, λ_std=λ_std, trajectory=trajectory)
end

"""
    run_bifurcation_analysis(params::CFParameters;
                             ϑ_range=0.01:0.005:0.50,
                             mode::Symbol=:mean_field,
                             DA_level::Union{Nothing,Float64}=nothing)

Generate bifurcation diagram by sweeping ϑ parameter.

# Arguments
- `params::CFParameters`: Base parameters (ϑ will be overridden)
- `ϑ_range`: Range of ϑ values to sweep
- `mode::Symbol`: :mean_field or :structured
- `DA_level`: Dopamine level in nM

# Returns
Dict with:
- `ϑ_values`: Parameter values
- `λ_max_values`: Lyapunov exponents
- `steady_states`: Steady-state μ_z values for each ϑ
- `bifurcation_points`: Detected bifurcation ϑ values
"""
function run_bifurcation_analysis(params::CFParameters;
                                  ϑ_range=0.01:0.005:0.50,
                                  mode::Symbol=:mean_field,
                                  DA_level::Union{Nothing,Float64}=nothing)
    
    n_points = length(ϑ_range)
    
    ϑ_values = collect(ϑ_range)
    λ_max_values = zeros(n_points)
    steady_states = Vector{Vector{Float64}}(undef, n_points)
    
    @info "Running bifurcation analysis" mode n_points
    
    for (i, ϑ) in enumerate(ϑ_range)
        # Create modified parameters
        p = CFParameters(
            κ = params.κ,
            ω = params.ω,
            ϑ = ϑ,
            π_u = params.π_u,
            γ_max = params.γ_max
        )
        
        # Compute Lyapunov exponent
        result = compute_lyapunov(p; mode=mode, DA_level=DA_level)
        
        λ_max_values[i] = result.λ_max
        
        # Extract steady-state values (last 1000 points)
        if length(result.trajectory) > 1000
            steady_states[i] = result.trajectory[end-999:end]
        else
            steady_states[i] = result.trajectory
        end
        
        if i % 10 == 0
            @info "Progress" completed=i total=n_points current_ϑ=ϑ λ_max=result.λ_max
        end
    end
    
    # Detect bifurcation points
    bifurcation_points = detect_bifurcations(ϑ_values, λ_max_values, steady_states)
    
    return Dict(
        "ϑ_values" => ϑ_values,
        "λ_max_values" => λ_max_values,
        "steady_states" => steady_states,
        "bifurcation_points" => bifurcation_points
    )
end

"""
    detect_bifurcations(ϑ_values, λ_max_values, steady_states)

Detect bifurcation points from sweep data.

Bifurcations are detected when:
1. λ_max crosses zero (stability change)
2. Number of distinct steady-state values changes (period-doubling)
"""
function detect_bifurcations(ϑ_values, λ_max_values, steady_states)
    bifurcations = Dict{String, Vector{Float64}}()
    bifurcations["stability_change"] = Float64[]
    bifurcations["period_doubling"] = Float64[]
    
    for i in 2:length(ϑ_values)
        # Check for λ_max crossing zero
        if λ_max_values[i-1] * λ_max_values[i] < 0
            # Linear interpolation to find crossing
            ϑ_cross = ϑ_values[i-1] + (ϑ_values[i] - ϑ_values[i-1]) * 
                      abs(λ_max_values[i-1]) / (abs(λ_max_values[i-1]) + abs(λ_max_values[i]))
            push!(bifurcations["stability_change"], ϑ_cross)
        end
        
        # Check for period-doubling (change in number of attractors)
        n_attractors_prev = count_attractors(steady_states[i-1])
        n_attractors_curr = count_attractors(steady_states[i])
        
        if n_attractors_curr > n_attractors_prev
            push!(bifurcations["period_doubling"], ϑ_values[i])
        end
    end
    
    return bifurcations
end

"""
    count_attractors(trajectory; tolerance=0.1)

Count the number of distinct attractor values in a trajectory.
"""
function count_attractors(trajectory; tolerance=0.1)
    if isempty(trajectory)
        return 0
    end
    
    # Cluster nearby values
    sorted = sort(trajectory)
    clusters = [sorted[1]]
    
    for v in sorted[2:end]
        if abs(v - clusters[end]) > tolerance
            push!(clusters, v)
        end
    end
    
    return length(clusters)
end

"""
    analyze_stability(params::CFParameters; mode::Symbol=:mean_field)

Comprehensive stability analysis for given parameters.

# Returns
Dict with stability metrics:
- `is_stable`: Boolean
- `λ_max`: Maximal Lyapunov exponent
- `regime`: :stable, :periodic, or :chaotic
- `period`: Estimated period if periodic (nothing if not)
"""
function analyze_stability(params::CFParameters; mode::Symbol=:mean_field)
    result = compute_lyapunov(params; mode=mode)
    
    analysis = Dict{String, Any}()
    analysis["λ_max"] = result.λ_max
    analysis["λ_std"] = result.λ_std
    
    if isnan(result.λ_max)
        analysis["is_stable"] = false
        analysis["regime"] = :diverged
        analysis["period"] = nothing
    elseif result.λ_max < -0.01
        analysis["is_stable"] = true
        analysis["regime"] = :stable
        analysis["period"] = nothing
    elseif abs(result.λ_max) <= 0.01
        analysis["is_stable"] = true
        analysis["regime"] = :periodic
        analysis["period"] = estimate_period(result.trajectory)
    else
        analysis["is_stable"] = false
        analysis["regime"] = :chaotic
        analysis["period"] = nothing
    end
    
    return analysis
end

"""
    estimate_period(trajectory)

Estimate the period of oscillation using autocorrelation.
"""
function estimate_period(trajectory)
    if length(trajectory) < 100
        return nothing
    end
    
    # Compute autocorrelation
    n = length(trajectory)
    max_lag = min(n ÷ 2, 500)
    
    mean_traj = mean(trajectory)
    var_traj = var(trajectory)
    
    if var_traj < 1e-10
        return nothing  # Constant trajectory
    end
    
    acf = zeros(max_lag)
    for lag in 1:max_lag
        cov_lag = mean((trajectory[1:n-lag] .- mean_traj) .* (trajectory[lag+1:n] .- mean_traj))
        acf[lag] = cov_lag / var_traj
    end
    
    # Find first peak after zero crossing
    for i in 2:max_lag-1
        if acf[i] > acf[i-1] && acf[i] > acf[i+1] && acf[i] > 0.5
            return i
        end
    end
    
    return nothing
end
