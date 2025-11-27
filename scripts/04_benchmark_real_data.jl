using RxInfer, CSV, DataFrames
include("../src/CF_Model.jl")
using.CF_Model

# --- Benchmark Script for COBRE/OpenNeuro ---
# Usage: julia scripts/04_benchmark_real_data.jl "path/to/subject_data.csv"

# 1. Load Data
# Assumes a CSV with a column 'bold_signal' representing the time-series
function load_data(filepath)
    df = CSV.read(filepath, DataFrame)
    return df.bold_signal
end

# 2. Fit Model
function fit_subject(data)
    println("Fitting CF Agent to subject data...")
    result = infer(
        model = cf_agent(),
        data  = (y = data,),
        constraints = make_constraints(:closed_loop),
        iterations = 15 # More iterations for real data
    )
    
    # Extract the inferred "Circulatory Fidelity" (Gamma posterior)
    gamma_posterior = result.posteriors[:γ_cf]
    
    # Return the mean precision (Circulatory Fidelity)
    return mean(gamma_posterior)
end

# Main Execution
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 1
        println("Usage: julia 04_benchmark_real_data.jl <csv_file>")
        # Default to a dummy run for testing
        data = randn(100) 
    else
        data = load_data(ARGS[1])
    end

    cf_metric = fit_subject(data)
    println("Estimated Circulatory Fidelity (CF): $cf_metric")
    
    # Save result
    open("results/subject_metrics.txt", "a") do io
        write(io, "Subject_CF: $cf_metric\n")
    end
end
