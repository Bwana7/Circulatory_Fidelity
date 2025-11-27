using RxInfer, Plots, Random
include("../src/CF_Model.jl")
using.CF_Model

# --- Pharmacological Simulation ---
# We simulate Reversal Learning under two conditions:
# 1. Amphetamine: High Tonic DA -> High CF (Prior on γ_cf is clamped high)
# 2. Ketamine: NMDA Blockade -> Low CF (Prior on γ_cf is clamped low)

# Generate simple Reversal Data (Switch at t=50)
T = 100
obs = vcat(ones(50), -ones(50)) + randn(100) * 0.2

# 1. Amphetamine Model (Hyper-Rigid)
# We modify the model call to inject a strong prior on Gamma
@model function amphetamine_agent(y)
    # High shape/rate = High expected value, Low variance
    γ_cf ~ Gamma(shape = 100.0, rate = 1.0) 
    z[1] ~ Normal(mean = 0.0, precision = γ_cf)
    x[1] ~ Normal(mean = 0.0, precision = exp(z[1]))
    y[1] ~ Normal(mean = x[1], precision = 10.0)
    for t in 2:length(y)
        z[t] ~ Normal(mean = z[t-1], precision = γ_cf)
        x[t] ~ Normal(mean = x[t-1], precision = exp(z[t]))
        y[t] ~ Normal(mean = x[t], precision = 10.0)
    end
end

println("Simulating Amphetamine (Perseveration)...")
res_amph = infer(
    model = amphetamine_agent(),
    data  = (y = obs,),
    constraints = make_constraints(:closed_loop)
)

# 2. Ketamine Model (Hyper-Volatile)
@model function ketamine_agent(y)
    # Low shape = Low precision, high variance
    γ_cf ~ Gamma(shape = 0.1, rate = 1.0) 
    z[1] ~ Normal(mean = 0.0, precision = γ_cf)
    x[1] ~ Normal(mean = 0.0, precision = exp(z[1]))
    y[1] ~ Normal(mean = x[1], precision = 10.0)
    for t in 2:length(y)
        z[t] ~ Normal(mean = z[t-1], precision = γ_cf)
        x[t] ~ Normal(mean = x[t-1], precision = exp(z[t]))
        y[t] ~ Normal(mean = x[t], precision = 10.0)
    end
end

println("Simulating Ketamine (Stochastic Switching)...")
res_ket = infer(
    model = ketamine_agent(),
    data  = (y = obs,),
    constraints = make_constraints(:open_loop) # Ketamine also breaks the loop biologically
)

# --- Plotting Figure 3 ---
p = plot(obs, label="Data", color=:grey, alpha=0.3, title="Figure 3: Pharmacological Reversal")
vline!(p, [2], label="Reversal", color=:black)
plot!(p, mean.(res_amph.posteriors[:x]), label="Amphetamine (Perseveration)", color=:green, lw=2)
plot!(p, mean.(res_ket.posteriors[:x]), label="Ketamine (Hyper-Volatile)", color=:purple, lw=2)

savefig(p, "results/Figure3_Pharmacology.png")
println("Figure 3 saved.")
