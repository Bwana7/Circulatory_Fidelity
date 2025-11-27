module CF_Model

using RxInfer, Distributions

export cf_agent, make_constraints

# ==============================================================================
#  The Generative Model
# ==============================================================================
# This implements the hierarchical structure defined in Equation 2.1 of the thesis.
# Level 3: Gamma_CF (Static Hyperparameter for Volatility Precision)
# Level 2: z (Log-Volatility Random Walk)
# Level 1: x (Perceptual State / Observation)

@model function cf_agent(y)
    # --- Level 3: Circulatory Fidelity ---
    # Modeled as a Gamma prior on the precision of the volatility transition.
    # In the "Closed Loop" case, this node remains active in the variational posterior.
    γ_cf ~ Gamma(shape = 1.0, rate = 1.0)

    # --- Level 2: Volatility States (z) ---
    # Initial state
    z[1] ~ Normal(mean = 0.0, precision = γ_cf)
    
    # --- Level 1: Perceptual States (x) ---
    # Initial state (variance depends on z)
    x[1] ~ Normal(mean = 0.0, precision = exp(z[1]))
    y[1] ~ Normal(mean = x[1], precision = 10.0) # Fixed sensory precision (high SNR)

    for t in 2:length(y)
        # Random walk for volatility, governed by CF
        z[t] ~ Normal(mean = z[t-1], precision = γ_cf)
        
        # State transition with volatility-dependent variance
        x[t] ~ Normal(mean = x[t-1], precision = exp(z[t]))
        
        # Observation
        y[t] ~ Normal(mean = x[t], precision = 10.0)
    end
end

# ==============================================================================
#  Constraint Factories (The Thesis "Intervention")
# ==============================================================================

"""
    make_constraints(mode::Symbol)

Returns the factorization constraints for the variational posterior.
- `:open_loop`: Standard Mean-Field (HGF). Factors q(z) and q(γ) separately.
- `:closed_loop`: Structured Variational (CF). Preserves q(z, γ) covariance.
"""
function make_constraints(mode::Symbol)
    if mode == :open_loop
        # The "Diagonal Deficit": Volatility and its precision are uncoupled.
        return @constraints begin
            q(z, x, γ_cf) = q(z)q(x)q(γ_cf)
        end
    elseif mode == :closed_loop
        # "Variational Closure": Volatility and precision are jointly inferred.
        return @constraints begin
            q(z, x, γ_cf) = q(z, γ_cf)q(x) 
        end
    else
        error("Unknown mode: $mode. Use :open_loop or :closed_loop")
    end
end

end # module
