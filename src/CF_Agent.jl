module CFAgent

using RxInfer, Distributions, LinearAlgebra

# ==============================================================================
#  The Generalized HGF with Circulatory Fidelity (Structured Variational)
# ==============================================================================

"""
    make_model(y; mode=:closed_loop)

Constructs the Hierarchical Gaussian Filter for active inference.
- `y`: Observation vector
- `mode`: :closed_loop (CF enabled) or :open_loop (Standard Mean Field)
"""
@model function hgf_agent(y, prior_cf_shape, prior_cf_rate)
    # --- Level 3: Circulatory Fidelity (Precision of Volatility) ---
    # In the closed loop model, this remains coupled to Level 2.
    γ_cf ~ Gamma(shape = prior_cf_shape, rate = prior_cf_rate)

    # --- Level 2: Volatility States (z) ---
    # We model log-volatility as a random walk governed by γ_cf
    z[1] ~ Normal(mean = 0.0, precision = γ_cf)
    
    # --- Level 1: Perceptual States (x) ---
    # Volatility determines the variance of x
    x[1] ~ Normal(mean = 0.0, precision = exp(z[1]))
    y[1] ~ Normal(mean = x[1], precision = 10.0) # Low observation noise

    for t in 2:length(y)
        # Level 2 Evolution (The "Circulatory" Step)
        z[t] ~ Normal(mean = z[t-1], precision = γ_cf)
        
        # Level 1 Evolution
        x[t] ~ Normal(mean = x[t-1], precision = exp(z[t]))
        
        # Observation
        y[t] ~ Normal(mean = x[t], precision = 10.0)
    end
end

# ==============================================================================
#  Variational Constraints (The Core Innovation)
# ==============================================================================

# 1. Standard HGF (Open Loop / Mean Field)
# Breaks the instantaneous link between volatility (z) and its precision (γ_cf)
function constraints_open_loop()
    return @constraints begin
        q(z, x, γ_cf) = q(z)q(x)q(γ_cf)
    end
end

# 2. Circulatory Fidelity (Closed Loop / Structured)
# Preserves the off-diagonal curvature (covariance) between z and γ_cf
function constraints_closed_loop()
    return @constraints begin
        q(z, x, γ_cf) = q(z, γ_cf)q(x) 
    end
end

end # module
