using Random: MersenneTwister
using StatsBase

using Graphs

using Test: @testset, @test

include("../src/Simulation.jl")
using .Simulation: Model, Param, C, D, POPULATION, PAYOFF, MUTATION
include("../src/Network.jl")
using .Network: nv, neighbors

function degree(weights::Matrix{Float16})::Vector{Int}
    return [count(x -> x > 0, weight_vec) for weight_vec in eachrow(weights)]
end

@testset "Model" begin
    @testset "default" begin
        # default models
        population_m = Model(Param(rng = MersenneTwister(10), variability_mode = POPULATION))
        payoff_m = Model(Param(rng = MersenneTwister(20), variability_mode = PAYOFF))
        mutation_m = Model(Param(rng = MersenneTwister(30), variability_mode = MUTATION))

        # param
        @test population_m.param.initial_N == payoff_m.param.initial_N == mutation_m.param.initial_N == 1_000
        @test population_m.param.initial_k == payoff_m.param.initial_k == mutation_m.param.initial_k == 10
        @test population_m.param.initial_T == payoff_m.param.initial_T == mutation_m.param.initial_T == 1.1
        @test population_m.param.S == payoff_m.param.S == mutation_m.param.S == -0.1
        @test population_m.param.initial_w == payoff_m.param.initial_w == mutation_m.param.initial_w == 0.5
        @test population_m.param.Δw == payoff_m.param.Δw == mutation_m.param.Δw == 0.1
        @test population_m.param.interaction_freqency ==
              payoff_m.param.interaction_freqency ==
              mutation_m.param.interaction_freqency ==
              1.0
        @test population_m.param.reproduction_rate ==
              payoff_m.param.reproduction_rate ==
              mutation_m.param.reproduction_rate ==
              0.1
        @test population_m.param.δ == payoff_m.param.δ == mutation_m.param.δ == 0.01
        @test population_m.param.initial_μ_s == payoff_m.param.initial_μ_s == mutation_m.param.initial_μ_s == 0.00
        @test population_m.param.initial_μ_c == payoff_m.param.initial_μ_c == mutation_m.param.initial_μ_c == 0.00
        @test population_m.param.β == payoff_m.param.β == mutation_m.param.β == 0.1
        @test population_m.param.sigma == payoff_m.param.sigma == mutation_m.param.sigma == 0.1
        @test population_m.param.generations == payoff_m.param.generations == mutation_m.param.generations == 100

        ## variability_mode
        @test population_m.param.variability_mode == POPULATION
        @test payoff_m.param.variability_mode == PAYOFF
        @test mutation_m.param.variability_mode == MUTATION

        ## random generator
        @test isa(population_m.param.rng, MersenneTwister)
        @test isa(payoff_m.param.rng, MersenneTwister)
        @test isa(population_m.param.rng, MersenneTwister)

        # generation
        @test population_m.generation == payoff_m.generation == mutation_m.generation == 1

        # environmental variables
        ## population
        @test population_m.death_N_vec == payoff_m.death_N_vec == mutation_m.death_N_vec == fill(100, 100)
        @test population_m.birth_N_vec == payoff_m.birth_N_vec == mutation_m.birth_N_vec == fill(100, 100)

        ## payoff_table
        @test population_m.payoff_table_vec == mutation_m.payoff_table_vec
        @test population_m.payoff_table_vec != payoff_m.payoff_table_vec
        @test [t[(C, C)] for t in population_m.payoff_table_vec] == fill((1.0, 1.0), 100)
        @test [t[(D, D)] for t in population_m.payoff_table_vec] == fill((0.0, 0.0), 100)
        @test [t[(C, D)] for t in population_m.payoff_table_vec] == fill((-0.1, 1.1), 100)
        @test [t[(D, C)] for t in population_m.payoff_table_vec] == fill((1.1, -0.1), 100)

        ### variable T
        @test [t[(C, D)][2] for t in payoff_m.payoff_table_vec] == [t[(D, C)][1] for t in payoff_m.payoff_table_vec]
        T_vec = [t[(C, D)][2] for t in payoff_m.payoff_table_vec]
        @test mean(T_vec) ≈ 1.1116489744902647
        @test std(T_vec) ≈ 0.09818350661601088
        @test maximum(T_vec) ≈ 1.3579004078889492
        @test minimum(T_vec) ≈ 0.8337367648951757

        ## mutation
        @test population_m.param.initial_μ_s == payoff_m.param.initial_μ_s == mutation_m.param.initial_μ_s == 0.0
        @test population_m.param.initial_μ_c == payoff_m.param.initial_μ_c == mutation_m.param.initial_μ_c == 0.0
        @test population_m.μ_s_vec == payoff_m.μ_s_vec == fill(Float16(0.0), 100)
        @test mutation_m.μ_s_vec != fill(Float16(0.0), 100)
        @test population_m.μ_c_vec == payoff_m.μ_c_vec == fill(Float16(0.0), 100)
        @test mutation_m.μ_c_vec != fill(Float16(0.0), 100)

        ### μ_s_vec
        @test mean(mutation_m.μ_s_vec) ≈ 0.04697
        @test std(mutation_m.μ_s_vec) ≈ 0.06384
        @test maximum(mutation_m.μ_s_vec) ≈ 0.3088
        @test minimum(mutation_m.μ_s_vec) ≈ 0.0

        ### μ_c_vec
        @test mean(mutation_m.μ_c_vec) ≈ 0.04343
        @test std(mutation_m.μ_c_vec) ≈ 0.0651
        @test maximum(mutation_m.μ_c_vec) ≈ 0.2668
        @test minimum(mutation_m.μ_c_vec) ≈ 0.0

        # agent's parameters
        ## strategy
        @test population_m.strategy_vec == payoff_m.strategy_vec == mutation_m.strategy_vec == fill(D, 1_000)

        ## payoff
        @test population_m.payoff_vec == payoff_m.payoff_vec == mutation_m.payoff_vec == fill(0.0, 1_000)

        ## graph
        @test nv(population_m.weights) == nv(payoff_m.weights) == nv(mutation_m.weights) == 1_000
        @test degree(population_m.weights) == degree(payoff_m.weights) == degree(mutation_m.weights) == fill(10, 1_000)
        for x = 1:1_000
            for y in neighbors(population_m.weights, x)
                @test population_m.weights[x, y] == population_m.weights[y, x] == Float16(0.5)
                @test payoff_m.weights[x, y] == payoff_m.weights[y, x] == Float16(0.5)
                @test mutation_m.weights[x, y] == mutation_m.weights[y, x] == Float16(0.5)
            end
        end
    end

    @testset "customized" begin
        # customized models
        common_params = Dict(
            :initial_N => 256,
            :initial_k => 100,
            :initial_T => 1.5,
            :S => 2.2,
            :initial_w => 0.123,
            :Δw => 0.19,
            :interaction_freqency => 0.111,
            :reproduction_rate => 0.56,
            :δ => 0.23,
            :initial_μ_s => 0.11,
            :initial_μ_c => 0.22,
            :β => 0.3,
            :sigma => 50,
            :generations => 10_000,
        )
        population_m = Model(Param(; common_params..., variability_mode = POPULATION, rng = MersenneTwister(40)))
        payoff_m = Model(Param(; common_params..., variability_mode = PAYOFF, rng = MersenneTwister(50)))
        mutation_m = Model(Param(; common_params..., variability_mode = MUTATION, rng = MersenneTwister(60)))

        # param
        @test population_m.param.initial_N == payoff_m.param.initial_N == mutation_m.param.initial_N == 256
        @test population_m.param.initial_k == payoff_m.param.initial_k == mutation_m.param.initial_k == 100
        @test population_m.param.initial_T == payoff_m.param.initial_T == mutation_m.param.initial_T == 1.5
        @test population_m.param.S == payoff_m.param.S == mutation_m.param.S == 2.2
        @test population_m.param.initial_w == payoff_m.param.initial_w == mutation_m.param.initial_w == Float16(0.123)
        @test population_m.param.Δw == payoff_m.param.Δw == mutation_m.param.Δw == 0.19
        @test population_m.param.interaction_freqency ==
              payoff_m.param.interaction_freqency ==
              mutation_m.param.interaction_freqency ==
              0.111
        @test population_m.param.reproduction_rate ==
              payoff_m.param.reproduction_rate ==
              mutation_m.param.reproduction_rate ==
              0.56
        @test population_m.param.δ == payoff_m.param.δ == mutation_m.param.δ == 0.23
        @test population_m.param.initial_μ_s == payoff_m.param.initial_μ_s == mutation_m.param.initial_μ_s == 0.11
        @test population_m.param.initial_μ_c == payoff_m.param.initial_μ_c == mutation_m.param.initial_μ_c == 0.22
        @test population_m.param.β == payoff_m.param.β == mutation_m.param.β == 0.3
        @test population_m.param.sigma == payoff_m.param.sigma == mutation_m.param.sigma == 50
        @test population_m.param.generations == payoff_m.param.generations == mutation_m.param.generations == 10_000

        ## variability_mode
        @test population_m.param.variability_mode == POPULATION
        @test payoff_m.param.variability_mode == PAYOFF
        @test mutation_m.param.variability_mode == MUTATION

        ## random generator
        @test isa(population_m.param.rng, MersenneTwister)
        @test isa(payoff_m.param.rng, MersenneTwister)
        @test isa(population_m.param.rng, MersenneTwister)

        # generation
        @test population_m.generation == payoff_m.generation == mutation_m.generation == 1

        # environmental variables
        ## population (payoff_m, mutation_m)
        @test payoff_m.death_N_vec == mutation_m.death_N_vec == fill(143, 10_000)
        @test payoff_m.birth_N_vec == mutation_m.birth_N_vec == fill(143, 10_000)
        ## population (population_m)
        diff_vec = [death_N - birth_N for (death_N, birth_N) in zip(population_m.death_N_vec, population_m.birth_N_vec)]
        @test mean(diff_vec) ≈ 0.0057
        @test std(diff_vec) ≈ 59.83938287352459
        ### death_N_vec
        @test mean(population_m.death_N_vec) ≈ 23.9762
        @test std(population_m.death_N_vec) ≈ 34.98015715658623
        @test maximum(population_m.death_N_vec) ≈ 218
        @test minimum(population_m.death_N_vec) ≈ 0
        ### birth_N_vec
        @test mean(population_m.birth_N_vec) ≈ 23.9705
        @test std(population_m.birth_N_vec) ≈ 34.750286156933974
        @test maximum(population_m.birth_N_vec) ≈ 194
        @test minimum(population_m.birth_N_vec) ≈ 0

        ## payoff_table
        @test population_m.payoff_table_vec == mutation_m.payoff_table_vec
        @test population_m.payoff_table_vec != payoff_m.payoff_table_vec
        @test [t[(C, C)] for t in population_m.payoff_table_vec] == fill((1.0, 1.0), 10_000)
        @test [t[(D, D)] for t in population_m.payoff_table_vec] == fill((0.0, 0.0), 10_000)
        @test [t[(C, D)] for t in population_m.payoff_table_vec] == fill((2.2, 1.5), 10_000)
        @test [t[(D, C)] for t in population_m.payoff_table_vec] == fill((1.5, 2.2), 10_000)

        ### variable T
        @test [t[(C, D)][2] for t in payoff_m.payoff_table_vec] == [t[(D, C)][1] for t in payoff_m.payoff_table_vec]
        T_vec = [t[(C, D)][2] for t in payoff_m.payoff_table_vec]
        @test mean(T_vec) ≈ 1.0179084050306493
        @test std(T_vec) ≈ 0.9944690954939563
        @test maximum(T_vec) ≈ 2.0
        @test minimum(T_vec) ≈ 0.0

        ## mutation
        @test population_m.param.initial_μ_s == payoff_m.param.initial_μ_s == mutation_m.param.initial_μ_s == 0.11
        @test population_m.param.initial_μ_c == payoff_m.param.initial_μ_c == mutation_m.param.initial_μ_c == 0.22
        @test population_m.μ_s_vec == payoff_m.μ_s_vec == fill(Float16(0.11), 10_000)
        @test mutation_m.μ_s_vec != fill(Float16(0.11), 10_000)
        @test population_m.μ_c_vec == payoff_m.μ_c_vec == fill(Float16(0.22), 10_000)
        @test mutation_m.μ_c_vec != fill(Float16(0.22), 10_000)

        ### μ_s_vec
        @test mean(mutation_m.μ_s_vec) ≈ 0.4956
        @test std(mutation_m.μ_s_vec) ≈ 0.4988
        @test maximum(mutation_m.μ_s_vec) ≈ 1
        @test minimum(mutation_m.μ_s_vec) ≈ 0

        ### μ_c_vec
        @test mean(mutation_m.μ_c_vec) ≈ 0.4995
        @test std(mutation_m.μ_c_vec) ≈ 0.4983
        @test maximum(mutation_m.μ_c_vec) ≈ 1
        @test minimum(mutation_m.μ_c_vec) ≈ 0

        # agent's parameters
        ## strategy
        @test population_m.strategy_vec == payoff_m.strategy_vec == mutation_m.strategy_vec == fill(D, 256)

        ## payoff
        @test population_m.payoff_vec == payoff_m.payoff_vec == mutation_m.payoff_vec == fill(0.0, 256)

        ## graph
        @test nv(population_m.weights) == nv(payoff_m.weights) == nv(mutation_m.weights) == 256
        @test degree(population_m.weights) == degree(payoff_m.weights) == degree(mutation_m.weights) == fill(100, 256)
        for x = 1:256
            for y in neighbors(population_m.weights, x)
                @test population_m.weights[x, y] == population_m.weights[y, x] == Float16(0.123)
                @test payoff_m.weights[x, y] == payoff_m.weights[y, x] == Float16(0.123)
                @test mutation_m.weights[x, y] == mutation_m.weights[y, x] == Float16(0.123)
            end
        end
    end
end
