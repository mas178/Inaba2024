using Random
using StatsBase
using Test: @testset, @test, @test_throws

include("../src/Simulation.jl")
using .Simulation: ModelPopulation, Param, C, D, interaction!, death_and_birth!

@testset "ar1 μ = 1.0" begin
    T = 10_000

    for (β, σ) in Iterators.product(0.1:0.2:0.9, 0.1:0.2:0.9)
        @test mean([mean(Simulation.ar1(β, σ, 1.0, T, MersenneTwister())) for _ = 1:1000]) ≈ 1.0 atol = 0.1
    end

    @test std(Simulation.ar1(0.1, 0.2, 1.0, T, MersenneTwister())) >
          std(Simulation.ar1(0.1, 0.1, 1.0, T, MersenneTwister()))
    @test std(Simulation.ar1(0.2, 0.1, 1.0, T, MersenneTwister())) >
          std(Simulation.ar1(0.1, 0.1, 1.0, T, MersenneTwister()))
    @test std(Simulation.ar1(0.2, 0.2, 1.0, T, MersenneTwister())) >
          std(Simulation.ar1(0.1, 0.1, 1.0, T, MersenneTwister()))
end

@testset "ar1 μ ≠ 1.0" begin
    T = 10_000
    β = 0.5
    sigma = 0.5

    μ = 2.0
    @test_throws AssertionError Simulation.ar1(β, sigma, μ, T, MersenneTwister())

    μ = 0.0
    @test_throws AssertionError Simulation.ar1(β, sigma, μ, T, MersenneTwister())

    μ = 1.8
    results = Simulation.ar1(β, sigma, μ, T, MersenneTwister())
    @test length(results) == T + 1
    @test mean(results) ≈ 1.8 atol = 0.02
    @test std(results) ≈ 0.178 atol = 0.01
    @test maximum(results) == 2.0
    @test minimum(results) == 1.6

    μ = 0.2
    results = Simulation.ar1(β, sigma, μ, T, MersenneTwister())
    @test length(results) == T + 1
    @test mean(results) ≈ 0.2 atol = 0.02
    @test std(results) ≈ 0.178 atol = 0.01
    @test maximum(results) == 0.4
    @test minimum(results) == 0.0

    μ = 1.3
    results = Simulation.ar1(β, sigma, μ, T, MersenneTwister())
    @test length(results) == T + 1
    @test mean(results) ≈ 1.3 atol = 0.02
    @test std(results) ≈ 0.451 atol = 0.01
    @test maximum(results) == 2.0
    @test minimum(results) ≈ 0.6

    μ = 0.7
    results = Simulation.ar1(β, sigma, μ, T, MersenneTwister())
    @test length(results) == T + 1
    @test mean(results) ≈ 0.7 atol = 0.02
    @test std(results) ≈ 0.451 atol = 0.01
    @test maximum(results) == 1.4
    @test minimum(results) == 0.0
end
