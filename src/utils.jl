# ═══════════════════════════════════════════════════════════════════════════════
# utils.jl - Utility Functions for Circulatory Fidelity
# ═══════════════════════════════════════════════════════════════════════════════

"""
    generate_colored_noise(n::Int, α::Float64; seed::Int=42)

Generate 1/f^α colored noise using the Kasdin-Walter algorithm.

# Arguments
- `n::Int`: Number of samples
- `α::Float64`: Spectral exponent
  - α = 0: White noise
  - α = 1: Pink noise (1/f)
  - α = 2: Brown noise (1/f²)
- `seed::Int`: Random seed

# Returns
Vector of n noise samples

# Reference
Kasdin, N.J. (1995). Discrete simulation of colored noise.
"""
function generate_colored_noise(n::Int, α::Float64; seed::Int=42)
    Random.seed!(seed)
    
    if α == 0
        return randn(n)
    end
    
    # Frequency domain method
    # Generate white noise
    white = randn(n)
    
    # FFT
    spectrum = fft(white)
    
    # Frequency vector
    freqs = fftfreq(n)
    
    # Apply 1/f^(α/2) filter (power spectrum goes as 1/f^α)
    for i in 1:n
        if freqs[i] != 0
            spectrum[i] *= abs(freqs[i])^(-α/2)
        end
    end
    
    # Inverse FFT and normalize
    colored = real(ifft(spectrum))
    colored = colored ./ std(colored)
    
    return colored
end

# Simple FFT frequency helper (if FFTW not available)
function fftfreq(n::Int)
    freqs = zeros(n)
    for i in 1:n
        if i <= n÷2 + 1
            freqs[i] = (i-1) / n
        else
            freqs[i] = (i-1-n) / n
        end
    end
    return freqs
end

# Fallback FFT implementation (basic DFT for small n)
function fft(x::Vector{<:Number})
    n = length(x)
    if n ≤ 32  # Use direct DFT for small arrays
        result = zeros(ComplexF64, n)
        for k in 0:n-1
            for j in 0:n-1
                result[k+1] += x[j+1] * exp(-2π * im * k * j / n)
            end
        end
        return result
    else
        # Try to use FFTW if available, otherwise recursive FFT
        try
            using FFTW
            return FFTW.fft(x)
        catch
            return recursive_fft(x)
        end
    end
end

function ifft(x::Vector{<:Number})
    n = length(x)
    result = conj.(fft(conj.(x))) ./ n
    return result
end

function recursive_fft(x::Vector{<:Number})
    n = length(x)
    if n == 1
        return ComplexF64[x[1]]
    end
    
    even = recursive_fft(x[1:2:end])
    odd = recursive_fft(x[2:2:end])
    
    result = zeros(ComplexF64, n)
    for k in 0:n÷2-1
        t = exp(-2π * im * k / n) * odd[k+1]
        result[k+1] = even[k+1] + t
        result[k+n÷2+1] = even[k+1] - t
    end
    
    return result
end

"""
    compute_mutual_information(x::Vector{Float64}, y::Vector{Float64}; k::Int=3)

Estimate mutual information using k-nearest neighbors (Kraskov estimator).

# Arguments
- `x::Vector{Float64}`: First variable
- `y::Vector{Float64}`: Second variable
- `k::Int`: Number of neighbors (default 3)

# Returns
Mutual information in nats

# Reference
Kraskov et al. (2004). Estimating mutual information.
"""
function compute_mutual_information(x::Vector{Float64}, y::Vector{Float64}; k::Int=3)
    n = length(x)
    @assert length(y) == n "Vectors must have same length"
    
    if n < k + 1
        return 0.0
    end
    
    # Combine into joint space
    joint = hcat(x, y)
    
    # For each point, find k-th nearest neighbor distance
    mi_estimate = 0.0
    
    for i in 1:n
        # Distances in joint space (Chebyshev/max norm)
        d_joint = [max(abs(x[i] - x[j]), abs(y[i] - y[j])) for j in 1:n if j != i]
        sort!(d_joint)
        ε = d_joint[k]  # k-th nearest neighbor distance
        
        # Count points within ε in marginals
        n_x = count(j -> j != i && abs(x[i] - x[j]) < ε, 1:n)
        n_y = count(j -> j != i && abs(y[i] - y[j]) < ε, 1:n)
        
        # Kraskov estimator contribution
        mi_estimate += digamma(k) - digamma(n_x + 1) - digamma(n_y + 1)
    end
    
    mi_estimate = mi_estimate / n + digamma(n)
    
    return max(0.0, mi_estimate)  # MI is non-negative
end

# Digamma function approximation
function digamma(x::Number)
    if x < 6
        return digamma(x + 1) - 1/x
    else
        # Asymptotic expansion
        return log(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4)
    end
end

"""
    compute_rmse(predicted::Vector{Float64}, actual::Vector{Float64})

Compute Root Mean Square Error.
"""
function compute_rmse(predicted::Vector{Float64}, actual::Vector{Float64})
    @assert length(predicted) == length(actual)
    return sqrt(mean((predicted .- actual).^2))
end

"""
    compute_correlation(x::Vector{Float64}, y::Vector{Float64})

Compute Pearson correlation coefficient.
"""
function compute_correlation(x::Vector{Float64}, y::Vector{Float64})
    @assert length(x) == length(y)
    return cor(x, y)
end

"""
    moving_average(x::Vector{Float64}, window::Int)

Compute moving average with given window size.
"""
function moving_average(x::Vector{Float64}, window::Int)
    n = length(x)
    result = zeros(n)
    
    for i in 1:n
        start_idx = max(1, i - window÷2)
        end_idx = min(n, i + window÷2)
        result[i] = mean(x[start_idx:end_idx])
    end
    
    return result
end

"""
    find_steady_state(trajectory::Vector{Float64}; tolerance::Float64=0.01)

Find steady-state value(s) from trajectory.

# Returns
Vector of distinct steady-state values
"""
function find_steady_state(trajectory::Vector{Float64}; tolerance::Float64=0.01)
    if isempty(trajectory)
        return Float64[]
    end
    
    # Use last portion of trajectory
    n = length(trajectory)
    last_portion = trajectory[max(1, n-1000):n]
    
    # Cluster values
    sorted = sort(last_portion)
    clusters = Float64[sorted[1]]
    
    for v in sorted
        if all(abs(v - c) > tolerance for c in clusters)
            push!(clusters, v)
        end
    end
    
    return sort(clusters)
end

"""
    generate_stimulus_sequence(n::Int, hazard_rate::Float64; seed::Int=42)

Generate a stimulus sequence with changepoints.

# Arguments
- `n::Int`: Number of timesteps
- `hazard_rate::Float64`: Probability of changepoint per timestep
- `seed::Int`: Random seed

# Returns
Tuple (stimuli, changepoints)
"""
function generate_stimulus_sequence(n::Int, hazard_rate::Float64; seed::Int=42)
    Random.seed!(seed)
    
    stimuli = zeros(n)
    changepoints = Int[]
    
    current_mean = randn()
    
    for t in 1:n
        # Check for changepoint
        if rand() < hazard_rate
            current_mean = randn()
            push!(changepoints, t)
        end
        
        # Generate stimulus
        stimuli[t] = current_mean + 0.1 * randn()
    end
    
    return stimuli, changepoints
end

"""
    save_results(filename::String, results::Dict)

Save results to a JSON file.
"""
function save_results(filename::String, results::Dict)
    open(filename, "w") do f
        # Simple JSON-like output
        println(f, "{")
        for (i, (k, v)) in enumerate(results)
            comma = i < length(results) ? "," : ""
            if v isa Vector
                println(f, "  \"$k\": $(v)$comma")
            elseif v isa String
                println(f, "  \"$k\": \"$v\"$comma")
            else
                println(f, "  \"$k\": $v$comma")
            end
        end
        println(f, "}")
    end
end

"""
    load_physiological_params()

Load physiological parameter values from calibration data.

# Returns
Dict with calibrated parameter values and sources
"""
function load_physiological_params()
    return Dict(
        "baseline_tonic_DA" => (value=17.0, range=(4.0, 30.0), unit="nM", source="Robinson et al. 2002"),
        "tonic_DA_MCSWV" => (value=90.0, std=9.0, unit="nM", source="Oh et al. 2018"),
        "phasic_burst_DA" => (value=210.0, std=10.0, unit="nM", source="Robinson et al. 2002"),
        "low_DA_parkinsonian" => (value=5.0, range=(0.0, 10.0), unit="nM", source="Bergstrom & Bhardwaj 2022"),
        "VTA_tonic_firing" => (value=4.5, unit="Hz", source="Grace & Bunney 1984"),
        "VTA_burst_firing" => (value=22.5, range=(15.0, 30.0), unit="Hz", source="Grace & Bunney 1984"),
        "D2_receptor_Ki" => (value=20.0, range=(10.0, 30.0), unit="nM", source="Richfield et al. 1989"),
        "D1_receptor_Ki" => (value=5000.0, range=(1000.0, 10000.0), unit="nM", source="Richfield et al. 1989")
    )
end
