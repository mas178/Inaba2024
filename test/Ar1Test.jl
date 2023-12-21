using Random
using StatsBase
using Test: @testset, @test, @test_throws

include("../src/Simulation.jl")
using .Simulation: Simulation, C, D

@testset "ar1 μ = 1.0" begin
    T = 10_000

    for (β, σ) in Iterators.product(0.1:0.2:0.9, 0.1:0.2:0.9)
        @test mean([mean(Simulation.ar1(β, σ, 1.0, T, MersenneTwister())) for _ = 1:1000]) ≈ 1.0 atol = 0.1
    end

    @test std(Simulation.ar1(0.1, 0.2, 1.0, T, MersenneTwister())) >
          std(Simulation.ar1(0.1, 0.1, 1.0, T, MersenneTwister()))
    @test std(Simulation.ar1(0.5, 0.1, 1.0, T, MersenneTwister())) >
          std(Simulation.ar1(0.1, 0.1, 1.0, T, MersenneTwister()))
    @test std(Simulation.ar1(0.2, 0.2, 1.0, T, MersenneTwister())) >
          std(Simulation.ar1(0.1, 0.1, 1.0, T, MersenneTwister()))
end

@testset "ar1 μ ≠ 1.0" begin
    T = 50_000
    β = 0.5
    sigma = 0.5
    @show theoretical_sigma = sqrt(sigma^2 / (1 - β^2))

    μ = 1.8
    results = Simulation.ar1(β, sigma, μ, T, MersenneTwister())
    @test length(results) == T
    @test results[1] == μ
    @test mean(results) ≈ μ atol = μ * 0.05
    @test std(results) ≈ theoretical_sigma atol = 0.01

    μ = 0.2
    results = Simulation.ar1(β, sigma, μ, T, MersenneTwister())
    @test length(results) == T
    @test results[1] == μ
    @test mean(results) ≈ μ atol = μ * 0.1
    @test std(results) ≈ theoretical_sigma atol = 0.01

    μ = 1.3
    results = Simulation.ar1(β, sigma, μ, T, MersenneTwister())
    @test length(results) == T
    @test results[1] == μ
    @test mean(results) ≈ μ atol = μ * 0.05
    @test std(results) ≈ theoretical_sigma atol = 0.01

    μ = 0.7
    results = Simulation.ar1(β, sigma, μ, T, MersenneTwister())
    @test length(results) == T
    @test results[1] == μ
    @test mean(results) ≈ μ atol = μ * 0.05
    @test std(results) ≈ theoretical_sigma atol = 0.01
end

@testset "population" begin
    β = 0.1
    sigma = 100.0
    μ = 1000.0
    T = 50_000
    theoretical_sigma = sqrt(sigma^2 / (1 - β^2))
    N_vec = Simulation.ar1(β, sigma, μ, T, MersenneTwister())
    @test length(N_vec) == T
    @test N_vec[1] == μ
    @test mean(N_vec) ≈ μ atol = μ * 0.05
    @test std(N_vec) ≈ theoretical_sigma atol = theoretical_sigma * 0.05

    β = 0.9
    sigma = 100.0
    μ = 1000.0
    T = 10_000
    theoretical_sigma = sqrt(sigma^2 / (1 - β^2))
    N_vec = Simulation.ar1(β, sigma, μ, T, MersenneTwister())
    @test length(N_vec) == T
    @test N_vec[1] == μ
    @test mean(N_vec) ≈ μ atol = μ * 0.05
    @test std(N_vec) ≈ theoretical_sigma atol = theoretical_sigma * 0.05
end

@testset "get_N_vec" begin
    gen = 3000
    init_N = 1000
    β = 0.3
    sigma = 500
    theoretical_sigma = sqrt(sigma^2 / (1 - β^2))
    N_vec = Simulation.get_N_vec(Simulation.Param(generations = gen, β = β, sigma = sigma, initial_N = init_N))
    @test length(N_vec) == gen + 1
    @test N_vec[1] == init_N
    @test mean(N_vec) ≈ init_N atol = init_N * 0.1
    @test 400 < std(N_vec) < theoretical_sigma
    @test maximum(N_vec) ≤ init_N * 2
    @test minimum(N_vec) ≥ 3

    param = Simulation.Param(variability_mode = Simulation.PAYOFF)
    N_vec = Simulation.get_N_vec(param)
    @test N_vec == fill(param.initial_N, param.generations + 1)
end

@testset "get_death_birth_N_vec" begin
    param = Simulation.Param(β = 0.9, sigma = 400, initial_N = 1000)
    N_vec = Simulation.get_N_vec(param)
    death_birth_N_vec = Simulation.get_death_birth_N_vec(param, N_vec)

    for i = 1:(param.generations)
        @test N_vec[i + 1] == N_vec[i] - death_birth_N_vec[1][i] + death_birth_N_vec[2][i]
        @test N_vec[i] >= death_birth_N_vec[1][i]
        @test N_vec[i] - death_birth_N_vec[1][i] >= death_birth_N_vec[2][i]
    end
end

@testset "get_payoff_table_vec" begin
    param = Simulation.Param(generations = 10)
    payoff_table_vec = Simulation.get_payoff_table_vec(param)
    @test [pt[(C, C)] for pt in payoff_table_vec] == fill((1.0, 1.0), param.generations)
    @test [pt[(C, D)] for pt in payoff_table_vec] == fill((-0.1, 1.1), param.generations)
    @test [pt[(D, C)] for pt in payoff_table_vec] == fill((1.1, -0.1), param.generations)
    @test [pt[(D, D)] for pt in payoff_table_vec] == fill((0.0, 0.0), param.generations)

    param = Simulation.Param(generations = 1000, variability_mode = Simulation.PAYOFF)
    payoff_table_vec = Simulation.get_payoff_table_vec(param)
    @test [pt[(C, C)] for pt in payoff_table_vec] == fill((1.0, 1.0), param.generations)
    @test [pt[(C, D)][1] for pt in payoff_table_vec] ==
          [pt[(D, C)][2] for pt in payoff_table_vec] ==
          fill(-0.1, param.generations)
    @test [pt[(C, D)][2] for pt in payoff_table_vec] == [pt[(D, C)][1] for pt in payoff_table_vec]
    @test [pt[(D, D)] for pt in payoff_table_vec] == fill((0.0, 0.0), param.generations)

    T_vec = [pt[(C, D)][2] for pt in payoff_table_vec]
    @test length(T_vec) == param.generations
    @test T_vec[1] == param.initial_T
    @test mean(T_vec) ≈ param.initial_T atol = param.initial_T * 0.1
    @test std(T_vec) ≈ sqrt(param.sigma^2 / (1 - param.β^2)) atol = param.sigma * 0.1
end
