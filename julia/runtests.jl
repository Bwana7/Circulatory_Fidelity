using Test
using CirculatoryFidelity

@testset "CirculatoryFidelity.jl v1.1" begin
    
    @testset "Inference Coupling (IC)" begin
        # Test ic_gaussian
        @test ic_gaussian(0.0) ≈ 0.0 atol=1e-10
        @test ic_gaussian(0.7) ≈ 0.7 atol=1e-10
        @test ic_gaussian(-0.5) ≈ 0.5 atol=1e-10
        @test ic_gaussian(1.0) ≈ 1.0 atol=1e-10
        
        # Test copula-based estimation on correlated Gaussian
        n = 1000
        x = randn(n)
        y = 0.7 .* x .+ 0.3 .* randn(n)
        ic, se = inference_coupling(x, y)
        @test 0.5 < ic < 0.9  # Should be around 0.7
        @test se > 0
        @test se ≈ 1/sqrt(n-3) atol=0.001
    end
    
    @testset "Gaussian MI and Entropy" begin
        # Test MI
        @test mutual_information_gaussian(0.0) ≈ 0.0 atol=1e-10
        @test mutual_information_gaussian(0.5) ≈ 0.1438 atol=0.001
        
        # Test entropy
        @test differential_entropy_gaussian(1.0) ≈ 1.4189 atol=0.001
        
        # Test SIGMA_MIN constraint
        @test SIGMA_MIN ≈ 0.2420 atol=0.001
    end
    
    @testset "Legacy API (deprecated)" begin
        # Test that legacy functions still work but emit warnings
        # CF = I(z;x) / min(H(z), H(x))
        cf = @test_logs (:warn,) circulatory_fidelity_gaussian(0.0, 1.0, 1.0)
        @test cf ≈ 0.0 atol=1e-10
    end
    
    @testset "SVF Model" begin
        params = SVFParams(coupling=0.5)
        sim = simulate_svf(params, 100; seed=42)
        
        @test length(sim.x3) == 100
        @test length(sim.y) == 100
        
        ic = compute_ic_svf(sim)
        @test 0.0 ≤ ic ≤ 1.0
        
        _, mf_mse = svf_mf_inference(sim)
        _, oracle_mse = svf_oracle_inference(sim)
        @test mf_mse > 0
        @test oracle_mse > 0
    end
    
    @testset "HLM Model" begin
        params = HLMParams(tau=1.0, sigma=1.0)
        @test CirculatoryFidelity.icc(params) ≈ 0.5 atol=0.01
        @test 0.0 < CirculatoryFidelity.reliability(params) < 1.0
        
        # IC = √ICC for HLM
        ic = compute_ic_hlm(params)
        @test ic ≈ sqrt(0.5) atol=0.01
        
        sim = simulate_hlm(params; seed=42)
        _, np_mse = hlm_no_pooling(sim)
        _, pp_mse = hlm_partial_pooling(sim)
        @test np_mse > 0
        @test pp_mse > 0
    end
    
    @testset "Three-Layer Model" begin
        params = ThreeLayerParams(kappa_32=0.5, kappa_21=0.5)
        sim = simulate_three_layer(params, 100; seed=42)
        
        ic_32, ic_21 = compute_ic_three_layer(sim)
        @test 0.0 ≤ ic_32 ≤ 1.0
        @test 0.0 ≤ ic_21 ≤ 1.0
        
        # Proximal Dominance: distal-only coupling should have minimal effect
        params_distal_only = ThreeLayerParams(kappa_32=1.5, kappa_21=0.0)
        sim_distal = simulate_three_layer(params_distal_only, 300; seed=42)
        mf_mse = three_layer_mf_inference(sim_distal)
        oracle_mse = three_layer_oracle_inference(sim_distal)
        # MSE ratio should be close to 1 when kappa_21 = 0
        @test mf_mse / oracle_mse < 1.5  # Allowing some numerical tolerance
    end
end

println("All tests passed!")
