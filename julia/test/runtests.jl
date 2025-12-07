"""
    test_circulatory_fidelity.jl
    
    Unit tests for CirculatoryFidelity.jl
"""

using Test
include("src/CirculatoryFidelity.jl")
using .CirculatoryFidelity

@testset "CirculatoryFidelity.jl" begin

    @testset "Mutual Information" begin
        # MI should be 0 when ρ = 0
        @test mutual_information(0.0) ≈ 0.0 atol=1e-10
        
        # MI should increase with |ρ|
        @test mutual_information(0.5) < mutual_information(0.8)
        
        # MI should be symmetric
        @test mutual_information(0.5) ≈ mutual_information(-0.5)
        
        # MI should approach infinity as ρ → 1
        @test mutual_information(0.999) > 3.0
    end

    @testset "Differential Entropy" begin
        # Standard Gaussian entropy ≈ 1.419
        @test differential_entropy(1.0) ≈ 0.5 * log(2π * ℯ) atol=1e-10
        
        # Entropy increases with variance
        @test differential_entropy(1.0) < differential_entropy(2.0)
        
        # Should throw for non-positive σ
        @test_throws DomainError differential_entropy(0.0)
        @test_throws DomainError differential_entropy(-1.0)
    end

    @testset "CF Bounds" begin
        # CF should be in [0, 1]
        for ρ in [0.0, 0.3, 0.5, 0.7, 0.9]
            cf = CF(ρ, 1.0, 1.0)
            @test 0.0 ≤ cf ≤ 1.0
        end
        
        # CF = 0 when independent
        @test CF(0.0, 1.0, 1.0) ≈ 0.0 atol=1e-10
        
        # CF increases with |ρ|
        @test CF(0.3, 1.0, 1.0) < CF(0.7, 1.0, 1.0)
    end

    @testset "CF from Samples" begin
        # Generate correlated samples
        n = 10000
        z = randn(n)
        x = 0.7z + sqrt(1 - 0.7^2) * randn(n)
        
        cf_sample = CF(z, x)
        cf_true = CF(0.7, 1.0, 1.0)
        
        # Should be close to theoretical value
        @test abs(cf_sample - cf_true) < 0.1
    end

    @testset "HGF Simulation" begin
        params = HGFParams(coupling=0.5)
        sim = simulate_hgf(params, T=100, seed=42)
        
        @test length(sim.x3) == 100
        @test length(sim.x2) == 100
        @test length(sim.y) == 100
        @test all(isfinite.(sim.x2))
        
        # CF should be computable
        cf = compute_cf_hgf(sim)
        @test 0.0 ≤ cf ≤ 1.0
    end

    @testset "HLM Simulation" begin
        params = HLMParams(n_groups=20, n_per_group=10, τ=1.0, σ=1.0)
        sim = simulate_hlm(params, seed=42)
        
        @test length(sim.θ) == 20
        @test length(sim.y) == 200
        @test length(sim.group) == 200
        
        # CF should be computable
        cf = compute_cf_hlm(params)
        @test 0.0 ≤ cf ≤ 1.0
    end

    @testset "HLM Properties" begin
        # Higher τ → higher ICC
        p1 = HLMParams(τ=0.5, σ=1.0)
        p2 = HLMParams(τ=1.5, σ=1.0)
        
        @test CirculatoryFidelity.icc(p1) < CirculatoryFidelity.icc(p2)
        
        # Higher τ → higher reliability → higher CF
        @test compute_cf_hlm(p1) < compute_cf_hlm(p2)
    end

end

println("All tests passed!")
