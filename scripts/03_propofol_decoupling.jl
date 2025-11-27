using RxInfer, Plots, LinearAlgebra
include("../src/CF_Model.jl")
using.CF_Model

# --- Propofol Simulation ---
# Hypothesis: Anaesthesia injects noise into the message passing interface.
# We simulate this by iteratively degrading the coupling strength.

noise_levels = range(0, 2.0, length=20)
mutual_info = Float64

# Dummy data for the simulation
data_segment = randn(50)

println("Simulating Propofol Dose-Response...")

for noise in noise_levels
    # In a full Factor Graph modification, we would add a Noise node on the edge.
    # Here, we approximate it by flattening the Gamma prior, which reduces the
    # effective information shared between z and γ.
    
    # We use a custom model per iteration to adjust the "effective coupling"
    @model function propofol_agent(y, noise_val)
        # Noise reduces the rate parameter, flattening the distribution
        γ_cf ~ Gamma(shape = 1.0, rate = 1.0 + noise_val*10) 
        z[1] ~ Normal(mean = 0.0, precision = γ_cf)
        x[1] ~ Normal(mean = 0.0, precision = exp(z[1]))
        y[1] ~ Normal(mean = x[1], precision = 10.0)
        for t in 2:length(y)
            z[t] ~ Normal(mean = z[t-1], precision = γ_cf)
            x[t] ~ Normal(mean = x[t-1], precision = exp(z[t]))
            y[t] ~ Normal(mean = x[t], precision = 10.0)
        end
    end

    res = infer(
        model = propofol_agent(noise_val=noise),
        data  = (y = data_segment,),
        constraints = make_constraints(:closed_loop)
    )
    
    # Metric: Variance of the Volatility Estimate
    # When the loop is closed (High MI), volatility is constrained (Low Variance).
    # When the loop breaks (Low MI), volatility random walks are unconstrained (High Variance).
    z_var = var(mean.(res.posteriors[:z]))
    push!(mutual_info, 1.0 / (1.0 + z_var)) # Proxy for Mutual Information
end

# --- Plotting Figure 4 ---
p = plot(noise_levels, mutual_info, 
    xlabel="Simulated Propofol Dose (Noise)", 
    ylabel="Information Integration (Proxy MI)",
    title="Figure 4: Propofol Induced Decoupling",
    legend=false, color=:darkred, lw=3)

savefig(p, "results/Figure4_Propofol.png")
println("Figure 4 saved.")
