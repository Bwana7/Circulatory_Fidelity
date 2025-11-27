using RxInfer, Plots, Random, Statistics
include("../src/CF_Agent.jl")
using.CFAgent

# Setup
rng = MersenneTwister(42)
default(titlefont=10, guidefont=8, lw=2)

# ==============================================================================
# EXPERIMENT 1: The Stochastic Switching Task (Figure 2)
# ==============================================================================
println("Running Stochastic Switching Simulation...")

# Generate Data (Hazard Rate = 0.05)
T = 200
true_states = zeros(T)
current_state = 1.0
observations = zeros(T)

for t in 1:T
    if rand() < 0.05
        global current_state *= -1.0 # Switch context
    end
    true_states[t] = current_state
    observations[t] = current_state + randn() * 0.5 # SNR ≈ 2
end

# Run Inference (Closed Loop vs Open Loop)
result_closed = infer(
    model = CFAgent.hgf_agent(prior_cf_shape=10.0, prior_cf_rate=10.0), # High CF
    data  = (y = observations,),
    constraints = CFAgent.constraints_closed_loop()
)

result_open = infer(
    model = CFAgent.hgf_agent(prior_cf_shape=0.1, prior_cf_rate=1.0), # Low CF/Open
    data  = (y = observations,),
    constraints = CFAgent.constraints_open_loop()
)

# Plotting Figure 2
p1 = plot(observations, label="Observations", color=:grey, alpha=0.5, title="Stochastic Switching Task")
plot!(p1, true_states, label="True Hidden State", color=:black, ls=:dash)
plot!(p1, mean.(result_closed.posteriors[:x]), label="Closed Loop (CF)", color=:blue)
plot!(p1, mean.(result_open.posteriors[:x]), label="Open Loop (Chaos)", color=:red)
savefig(p1, "Figure2_Trajectories.png")


# ==============================================================================
# EXPERIMENT 2: Pharmacological Reversal Learning (Figure 3)
# ==============================================================================
println("Running Pharmacological Simulation...")

# Simulate Amphetamine (Hyper-rigid / High Tonic DA)
# We fix the prior on Gamma to be extremely high and tight
result_amph = infer(
    model = CFAgent.hgf_agent(prior_cf_shape=100.0, prior_cf_rate=1.0), 
    data  = (y = observations,),
    constraints = CFAgent.constraints_closed_loop()
)

# Simulate Ketamine (Hyper-volatile / NMDA Blockade)
# We fix the prior on Gamma to be extremely low
result_ket = infer(
    model = CFAgent.hgf_agent(prior_cf_shape=0.01, prior_cf_rate=1.0), 
    data  = (y = observations,),
    constraints = CFAgent.constraints_open_loop()
)

# Plotting Figure 3
p2 = plot(title="Pharmacological Reversal Learning")
plot!(p2, mean.(result_amph.posteriors[:z]), label="Amphetamine (Perseveration)", color=:green)
plot!(p2, mean.(result_ket.posteriors[:z]), label="Ketamine (Stochastic Switching)", color=:purple)
plot!(p2, true_states, color=:black, ls=:dash, label="Truth")
savefig(p2, "Figure3_Pharmacology.png")


# ==============================================================================
# EXPERIMENT 3: Propofol & Mutual Information (Figure 4)
# ==============================================================================
println("Running Propofol Decoupling...")

# To calculate Mutual Information I(z; γ), we look at the covariance in the structured posterior.
# In RxInfer, the structured q(z, γ) returns a joint distribution.

noise_levels = 0.0:0.1:2.0
mi_values =

for noise in noise_levels
    # Simulate "noise injection" by flattening the prior correlation
    # Note: In a full factor graph, we would add a noise node on the edge.
    # Here we proxy it by degrading the prior shape parameter (reducing effective CF).
    
    res = infer(
        model = CFAgent.hgf_agent(prior_cf_shape=10.0 / (1+noise*10), prior_cf_rate=1.0),
        data = (y = observations[1:50],), # Short run
        constraints = CFAgent.constraints_closed_loop()
    )
    
    # Approx Mutual Info metric: Variance of the Volatility estimate
    # (High coupling = Controlled variance; Decoupled = Exploding variance)
    push!(mi_values, var(mean.(res.posteriors[:z])))
end

p3 = plot(noise_levels, mi_values, xlabel="Virtual Propofol Dose (Noise)", ylabel="Volatility Variance (De-integration)", title="Propofol Induced Decoupling")
savefig(p3, "Figure4_Propofol.png")

println("Figures Generated.")
