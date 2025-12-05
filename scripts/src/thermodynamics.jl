# ═══════════════════════════════════════════════════════════════════════════════
# thermodynamics.jl - Thermodynamic Free Energy Extensions
# ═══════════════════════════════════════════════════════════════════════════════

# Physical constants
const k_B = 1.38e-23             # Boltzmann constant (J/K)
const T_PHYS = 310.0             # Physiological temperature (K)
const ATP_PER_BIT = 1e4          # ATP molecules per bit erased (empirical)
const ΔG_ATP = 8.3e-20           # Free energy per ATP hydrolysis (J)
const D_BASELINE_THERMO = 50.0   # Default baseline dopamine (nM)

"""
    TFEParameters

Parameters for Thermodynamic Free Energy computation.

# Fields
- `β_met::Float64`: Metabolic cost coefficient (nats per unit computation)
- `γ₀::Float64`: Reference precision (typically γ_max/2)
- `T::Float64`: Temperature in Kelvin (default 310 K)
"""
Base.@kwdef struct TFEParameters
    β_met::Float64 = 1e-3
    γ₀::Float64 = 50.0
    T::Float64 = T_PHYS
end

"""
    landauer_bound()

Compute the Landauer bound: minimum energy to erase one bit at physiological temperature.

    E_min = k_B * T * ln(2)

# Returns
Energy in Joules (approximately 3 × 10⁻²¹ J at 310 K)
"""
function landauer_bound(T::Float64=T_PHYS)
    return k_B * T * log(2)
end

"""
    neural_cost_per_bit()

Estimate the actual metabolic cost per bit in neural computation.

Based on Laughlin et al. (1998): ~10⁴ ATP per bit.

# Returns
Energy in Joules (approximately 8 × 10⁻¹⁶ J)
"""
function neural_cost_per_bit()
    return ATP_PER_BIT * ΔG_ATP
end

"""
    thermodynamic_inefficiency()

Compute the ratio of actual neural cost to Landauer bound.

Neural computation is approximately 10⁸ times less efficient than
the thermodynamic limit.
"""
function thermodynamic_inefficiency()
    return neural_cost_per_bit() / landauer_bound()
end

"""
    metabolic_cost(γ::Float64, tfe_params::TFEParameters)

Compute metabolic cost of maintaining precision γ.

The cost function is:
    C(γ) = β_met * γ * ln(γ / γ₀)

This captures:
- Linear scaling with precision (more computation at higher γ)
- Logarithmic penalty for deviation from reference
- C(γ₀) = 0 (homeostatic precision is "free")

# Arguments
- `γ::Float64`: Precision (gain) value
- `tfe_params::TFEParameters`: TFE parameters

# Returns
Metabolic cost in nats
"""
function metabolic_cost(γ::Float64, tfe_params::TFEParameters)
    if γ ≤ 0
        return Inf
    end
    return tfe_params.β_met * γ * log(γ / tfe_params.γ₀)
end

"""
    dopamine_to_precision(D::Float64, params::CFParameters)

Convert dopamine concentration (nM) to precision (gain).

Derived from receptor binding kinetics (Hill equation):
    γ(D) = γ_max · D^n / (K_d^n + D^n)

where:
- K_d ≈ 20 nM is the D2 receptor dissociation constant (Richfield et al., 1989)
- n ≈ 1 is the Hill coefficient for D2 receptors
- γ_max is the maximum precision (free parameter)

The Hill equation form follows from equilibrium thermodynamics of receptor-ligand
binding. K_d is constrained by pharmacology; γ_max must be fit to behavioral data.

# Arguments
- `D::Float64`: Dopamine concentration in nM
- `params::CFParameters`: Model parameters (uses K_d, n, γ_max)

# Returns
Precision value γ ∈ (0, γ_max)

# References
- Richfield et al. (1989). Neuroscience, 30(3), 767-777.
- Seeman et al. (2006). PNAS, 103(9), 3440-3445.
"""
function dopamine_to_precision(D::Float64, params::CFParameters)
    # Hill equation derived from receptor binding kinetics
    # K_d ≈ 20 nM for D2 receptors (population average)
    K_d = 20.0  # nM - constrained by D2 receptor pharmacology
    n = 1.0     # Hill coefficient - approximately 1 for D2 receptors
    
    if D <= 0
        return 0.0
    end
    
    # Hill equation: θ = D^n / (K_d^n + D^n)
    # Precision = γ_max * θ (linear occupancy-to-precision mapping)
    occupancy = D^n / (K_d^n + D^n)
    return params.γ_max * occupancy
end

"""
    precision_to_dopamine(γ::Float64, params::CFParameters)

Inverse of dopamine_to_precision: find D given γ.

Inverts the Hill equation:
    γ = γ_max · D^n / (K_d^n + D^n)
    
Solving for D:
    D = K_d · (γ / (γ_max - γ))^(1/n)

# Arguments
- `γ::Float64`: Precision value
- `params::CFParameters`: Model parameters

# Returns
Dopamine concentration in nM
"""
function precision_to_dopamine(γ::Float64, params::CFParameters)
    K_d = 20.0  # nM - D2 receptor dissociation constant
    n = 1.0     # Hill coefficient
    
    if γ <= 0 || γ >= params.γ_max
        return NaN
    end
    
    # Inverse Hill equation
    return K_d * (γ / (params.γ_max - γ))^(1/n)
end

"""
    compute_TFE(vfe::Float64, γ::Float64, tfe_params::TFEParameters)

Compute Thermodynamic Free Energy.

    F_TFE = F_VFE + C(γ)

where C(γ) is the metabolic cost.

# Arguments
- `vfe::Float64`: Variational free energy
- `γ::Float64`: Current precision
- `tfe_params::TFEParameters`: TFE parameters

# Returns
Thermodynamic free energy in nats
"""
function compute_TFE(vfe::Float64, γ::Float64, tfe_params::TFEParameters)
    return vfe + metabolic_cost(γ, tfe_params)
end

"""
    optimal_precision(vfe_gradient::Float64, tfe_params::TFEParameters)

Find optimal precision that minimizes TFE.

At the optimum:
    ∂F_TFE/∂γ = ∂F_VFE/∂γ + β_met * (1 + ln(γ/γ₀)) = 0

Solving for γ:
    γ* = γ₀ * exp(-1 - ∂F_VFE/∂γ / β_met)

# Arguments
- `vfe_gradient::Float64`: Gradient of VFE with respect to γ (typically negative)
- `tfe_params::TFEParameters`: TFE parameters

# Returns
Optimal precision γ*
"""
function optimal_precision(vfe_gradient::Float64, tfe_params::TFEParameters)
    exponent = -1 - vfe_gradient / tfe_params.β_met
    return tfe_params.γ₀ * exp(exponent)
end

"""
    compute_critical_beta(params::CFParameters, ϑ_chaos::Float64)

Compute critical β_met above which mean-field remains stable.

# Arguments
- `params::CFParameters`: Model parameters
- `ϑ_chaos::Float64`: Volatility threshold for chaos (typically ~0.12)

# Returns
Critical metabolic cost coefficient β_crit
"""
function compute_critical_beta(params::CFParameters, ϑ_chaos::Float64=0.12)
    # At chaos threshold, we need metabolic penalty to dominate
    # This is an approximation based on the theory
    
    γ_DA = dopamine_to_precision(D_BASELINE_THERMO, params)
    γ₀ = params.γ_max / 2
    
    # Estimate VFE gradient at chaos boundary
    # (simplified - full computation requires numerical integration)
    vfe_grad_estimate = -ϑ_chaos * γ_DA
    
    # Critical β is where optimal γ equals chaos threshold γ
    # γ_chaos ≈ γ₀ * exp(-1 - vfe_grad / β_crit)
    # Solving: β_crit = -vfe_grad / (ln(γ_chaos/γ₀) + 1)
    
    γ_chaos = γ_DA * 2  # Rough estimate
    
    if γ_chaos / γ₀ > 0
        β_crit = abs(vfe_grad_estimate) / (log(γ_chaos / γ₀) + 1)
    else
        β_crit = 0.1  # Default fallback
    end
    
    return β_crit
end

"""
    tfe_regularized_update(μ_z, μ_x, σ²_z, σ²_x, y, params, tfe_params, mode)

Perform TFE-regularized variational update.

This modifies the standard update by incorporating metabolic costs
into the precision selection.
"""
function tfe_regularized_update(μ_z, μ_x, σ²_z, σ²_x, y, 
                                 params::CFParameters, 
                                 tfe_params::TFEParameters,
                                 mode::Symbol)
    # Standard VFE-based update
    γ_DA = dopamine_to_precision(D_BASELINE_THERMO, params)
    
    # Compute prediction error
    δ_x = y - μ_x
    
    # Estimate VFE gradient with respect to γ
    # ∂F_VFE/∂γ ≈ -0.5 * δ_x² * exp(-κ*μ_z - ω)
    vfe_gradient = -0.5 * δ_x^2 * exp(-params.κ * μ_z - params.ω)
    
    # Find TFE-optimal precision
    γ_optimal = optimal_precision(vfe_gradient, tfe_params)
    
    # Constrain to valid range
    γ_optimal = clamp(γ_optimal, 0.1, params.γ_max)
    
    # Use optimal precision for update
    μ_z_new, μ_x_new, σ²_z_new, σ²_x_new, cov_zx = variational_update_full(
        μ_z, μ_x, σ²_z, σ²_x, y, params, γ_optimal, mode
    )
    
    return μ_z_new, μ_x_new, σ²_z_new, σ²_x_new, γ_optimal
end

"""
    energy_budget_analysis(params::CFParameters, tfe_params::TFEParameters, 
                           n_timesteps::Int=1000)

Analyze energy budget over a simulation.

# Returns
Dict with:
- `total_cost`: Total metabolic cost (ATP equivalents)
- `cost_per_step`: Average cost per timestep
- `efficiency`: Information gained per unit energy
"""
function energy_budget_analysis(params::CFParameters, 
                                tfe_params::TFEParameters,
                                n_timesteps::Int=1000)
    
    # Run simulation
    Random.seed!(42)
    observations = randn(n_timesteps) .* sqrt(1/params.π_u)
    
    results = run_inference(observations, params, :structured)
    
    # Compute total metabolic cost
    γ_DA = dopamine_to_precision(D_BASELINE_THERMO, params)
    cost_per_step = metabolic_cost(γ_DA, tfe_params)
    total_cost = cost_per_step * n_timesteps
    
    # Compute information gained (mutual information recovered)
    mean_CF = mean(results.CF)
    
    # Convert to ATP equivalents
    cost_atp = total_cost * exp(1)  # Rough conversion from nats
    
    return Dict(
        "total_cost_nats" => total_cost,
        "total_cost_atp" => cost_atp,
        "cost_per_step" => cost_per_step,
        "mean_CF" => mean_CF,
        "efficiency" => mean_CF / cost_per_step,
        "landauer_bound_J" => landauer_bound(),
        "neural_cost_J" => neural_cost_per_bit(),
        "inefficiency_factor" => thermodynamic_inefficiency()
    )
end
