using RxInfer, Plots, Random, Statistics
include("../src/CF_Model.jl")
using.CF_Model

# Reproducibility
rng = MersenneTwister(2025)

# --- Simulation Parameters ---
# Defined in Section 4.1 of the thesis
T = 200                 # Trials
HazardRate = 0.05       # Probability of state switch
NoiseStd = 0.5          # Observation noise

# Generate Synthetic Data (Stochastic Switching)
true_states = zeros(T)
current_state = 1.0
observations = zeros(T)

println("Generating data...")
for t in 1:T
    if rand(rng) < HazardRate
        global current_state *= -1.0
    end
    true_states[t] = current_state
    observations[t] = current_state + randn(rng) * NoiseStd
end

# --- Run Inference ---

println("Running Open Loop (Standard HGF)...")
# Note: We set a loose prior to simulate the instability of high volatility
result_open = infer(
    model = cf_agent(),
    data  = (y = observations,),
    constraints = make_constraints(:open_loop),
    iterations = 10
)

println("Running Closed Loop (Circulatory Fidelity)...")
result_closed = infer(
    model = cf_agent(),
    data  = (y = observations,),
    constraints = make_constraints(:closed_loop),
    iterations = 10
)

# --- Plotting Figure 2 ---
p = plot(observations, label="Observations", color=:grey, alpha=0.4, title="Figure 2: Stability Analysis")
plot!(p, true_states, label="True State", color=:black, lw=2, ls=:dash)
plot!(p, mean.(result_open.posteriors[:x]), label="Open Loop (Mean Field)", color=:red, lw=1.5)
plot!(p, mean.(result_closed.posteriors[:x]), label="Closed Loop (Structured)", color=:blue, lw=2)

savefig(p, "results/Figure2_StochasticSwitching.png")
println("Figure 2 saved.")
