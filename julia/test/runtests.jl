using Test
using CirculatoryFidelity

@testset "CirculatoryFidelity.jl" begin
    
    @testset "Gaussian CF" begin
        # Test MI
        @test mutual_information_gaussian(0.0) ≈ 0.0 atol=1e-10
        @test mutual_information_gaussian(0.5) ≈ 0.1438 atol=0.001
        
        # Test entropy
        @test differential_entropy_gaussian(1.0) ≈ 1.4189 atol=0.001
        
        # Test CF - CORRECTED: requires sigma_z and sigma_x parameters
        # CF = I(z;x) / min(H(z), H(x))
        @test circulatory_fidelity_gaussian(0.0, 1.0, 1.0) ≈ 0.0 atol=1e-10
        @test circulatory_fidelity_gaussian(0.7, 1.0, 1.0) ≈ 0.433 atol=0.01  # MI/H(σ=1)
        @test 0.0 ≤ circulatory_fidelity_gaussian(0.9, 1.0, 1.0) ≤ 1.0
        
        # Test SIGMA_MIN constraint
        @test SIGMA_MIN ≈ 0.2420 atol=0.001
        
        # Test Linfoot correlation: r_L = |ρ| for Gaussians
        @test linfoot_correlation(0.0) ≈ 0.0 atol=1e-10
        @test linfoot_correlation(0.5) ≈ 0.5 atol=0.001
        @test linfoot_correlation(0.9) ≈ 0.9 atol=0.001
        @test linfoot_correlation(-0.7) ≈ 0.7 atol=0.001
    end
    
    @testset "SVF Model" begin
        params = SVFParams(coupling=0.5)
        sim = simulate_svf(params; T=100)
        
        @test length(sim.x3) == 100
        @test length(sim.y) == 100
        
        cf = compute_cf_svf(sim)
        @test 0.0 ≤ cf ≤ 1.0
        
        _, mf_mse = svf_mf_inference(sim)
        _, oracle_mse = svf_oracle_inference(sim)
        @test mf_mse > 0
        @test oracle_mse > 0
    end
    
    @testset "SVF Log-Likelihood Gap" begin
        # Test Kalman filter with log-likelihood
        params = SVFParams(coupling=1.0)
        sim = simulate_svf(params; T=100)
        
        result = svf_kalman_filter(sim.y, 0.25, 0.25)
        @test length(result.x_filtered) == 100
        @test isfinite(result.log_likelihood)
        
        # Test log-likelihood gap (oracle should be better)
        mfvi = svf_fit_mfvi(sim)
        oracle = svf_fit_oracle(sim)
        @test oracle.result.log_likelihood >= mfvi.result.log_likelihood - 10  # Allow some tolerance
        
        # Test compute_log_likelihood_gap
        ll_gap = compute_log_likelihood_gap(sim)
        @test isfinite(ll_gap)
    end
    
    @testset "HLM Model" begin
        params = HLMParams(tau=1.0, sigma=1.0)
        @test CirculatoryFidelity.icc(params) ≈ 0.5 atol=0.01
        @test 0.0 < CirculatoryFidelity.reliability(params) < 1.0
        
        cf = compute_cf_hlm(params)
        @test 0.0 ≤ cf ≤ 1.0
        
        sim = simulate_hlm(params)
        _, np_mse = hlm_no_pooling(sim)
        _, pp_mse = hlm_partial_pooling(sim)
        @test np_mse > 0
        @test pp_mse > 0
    end
    
    @testset "Three-Layer Model" begin
        params = ThreeLayerParams(kappa_32=0.5, kappa_21=0.5)
        sim = simulate_three_layer(params; T=100)
        
        cf_32, cf_21 = compute_cf_three_layer(sim)
        @test 0.0 ≤ cf_32 ≤ 1.0
        @test 0.0 ≤ cf_21 ≤ 1.0
    end
end

println("All tests passed!")
