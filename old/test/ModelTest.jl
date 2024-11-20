using Random: MersenneTwister
using StatsBase
using LinearAlgebra: diag

using Graphs

using Test: @testset, @test

include("../src/Simulation.jl")
using .Simulation: Model, Param, C, D, POPULATION, PAYOFF, STRATEGY_MUTATION, RELATIONSHIP_MUTATION
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
        s_mutation_m = Model(Param(rng = MersenneTwister(30), variability_mode = STRATEGY_MUTATION))
        r_mutation_m = Model(Param(rng = MersenneTwister(30), variability_mode = RELATIONSHIP_MUTATION))

        for model in [population_m, payoff_m, s_mutation_m, r_mutation_m]
            # param
            @test model.param.initial_N == 1_000
            @test model.param.initial_k == 10
            @test model.param.initial_T == 1.1
            @test model.param.S == -0.1
            @test model.param.initial_w == 0.5
            @test model.param.Δw == 0.1
            @test model.param.reproduction_rate == 0.1
            @test model.param.δ == 0.01
            @test model.param.initial_μ_s == 0.0
            @test model.param.initial_μ_r == 0.0
            @test model.param.β == 0.1
            @test model.param.σ == 0.1
            @test model.param.τ == 10
            @test model.param.generations == 100
            @test isa(model.param.rng, MersenneTwister)
            @test model.generation == 1

            # environmental variables
            ## population
            @test model.death_N_vec == fill(100, 100)
            @test model.birth_N_vec == fill(100, 100)

            # agent's parameters
            ## strategy
            @test model.strategy_vec == fill(D, 1_000)

            ## payoff
            @test model.payoff_vec == fill(0.0, 1_000)

            ## graph
            @test nv(model.weights) == 1_000
            @test degree(model.weights) == fill(10, 1_000)
            @test diag(model.weights) == fill(0.0, 1_000)
            @test model.weights == transpose(model.weights)
            for x = 1:1_000
                for y in neighbors(model.weights, x)
                    @test model.weights[x, y] == Float16(0.5)
                end
            end
        end

        ## variability_mode
        @test population_m.param.variability_mode == POPULATION
        @test payoff_m.param.variability_mode == PAYOFF
        @test s_mutation_m.param.variability_mode == STRATEGY_MUTATION
        @test r_mutation_m.param.variability_mode == RELATIONSHIP_MUTATION

        # environmental variables
        ## payoff_table
        @test population_m.payoff_table_vec == s_mutation_m.payoff_table_vec == r_mutation_m.payoff_table_vec
        @test population_m.payoff_table_vec != payoff_m.payoff_table_vec
        @test [t[(C, C)] for t in population_m.payoff_table_vec] == fill((1.0, 1.0), 100)
        @test [t[(D, D)] for t in population_m.payoff_table_vec] == fill((0.0, 0.0), 100)
        @test [t[(C, D)] for t in population_m.payoff_table_vec] == fill((-0.1, 1.1), 100)
        @test [t[(D, C)] for t in population_m.payoff_table_vec] == fill((1.1, -0.1), 100)

        ### variable T
        @test [t[(C, D)][2] for t in payoff_m.payoff_table_vec] == [t[(D, C)][1] for t in payoff_m.payoff_table_vec]
        T_vec = [t[(C, D)][2] for t in payoff_m.payoff_table_vec]
        @test mean(T_vec) ≈ 1.118881788959639
        @test std(T_vec) ≈ 0.08835749739372203
        @test maximum(T_vec) ≈ 1.2569887300872324
        @test minimum(T_vec) ≈ 0.9495949727697413

        ## mutation (μ_s & μ_r)
        @test population_m.μ_s_vec ==
              payoff_m.μ_s_vec ==
              r_mutation_m.μ_s_vec ==
              s_mutation_m.μ_s_vec ==
              fill(Float16(0.0), 100)
        @test population_m.μ_r_vec ==
              payoff_m.μ_r_vec ==
              s_mutation_m.μ_r_vec ==
              r_mutation_m.μ_r_vec ==
              fill(Float16(0.0), 100)
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
            :reproduction_rate => 0.56,
            :δ => 0.23,
            :initial_μ_s => 0.11,
            :initial_μ_r => 0.22,
            :β => 0.3,
            :σ => 50,
            :τ => 1,
            :generations => 10_000,
        )

        population_m = Model(Param(; common_params..., variability_mode = POPULATION, rng = MersenneTwister(40)))
        payoff_m = Model(Param(; common_params..., variability_mode = PAYOFF, rng = MersenneTwister(50)))
        s_mutation_m = Model(Param(; common_params..., variability_mode = STRATEGY_MUTATION, rng = MersenneTwister(60)))
        r_mutation_m =
            Model(Param(; common_params..., variability_mode = RELATIONSHIP_MUTATION, rng = MersenneTwister(60)))

        for model in [population_m, payoff_m, s_mutation_m, r_mutation_m]
            # param
            @test model.param.initial_N == 256
            @test model.param.initial_k == 100
            @test model.param.initial_T == 1.5
            @test model.param.S == 2.2
            @test model.param.initial_w == Float16(0.123)
            @test model.param.Δw == 0.19
            @test model.param.reproduction_rate == 0.56
            @test model.param.δ == 0.23
            @test model.param.initial_μ_s == 0.11
            @test model.param.initial_μ_r == 0.22
            @test model.param.β == 0.3
            @test model.param.σ == 50
            @test model.param.τ == 1
            @test model.param.generations == 10_000

            ## random generator
            @test isa(model.param.rng, MersenneTwister)

            # generation
            @test model.generation == 1

            # agent's parameters
            ## strategy
            @test model.strategy_vec == fill(D, 256)

            ## payoff
            @test model.payoff_vec == fill(0.0, 256)

            ## graph
            @test nv(model.weights) == 256
            @test degree(model.weights) == fill(100, 256)
            @test diag(model.weights) == fill(0.0, 256)
            @test model.weights == transpose(model.weights)
            for x = 1:256
                for y in neighbors(model.weights, x)
                    @test model.weights[x, y] == Float16(0.123)
                end
            end
        end

        ## variability_mode
        @test population_m.param.variability_mode == POPULATION
        @test payoff_m.param.variability_mode == PAYOFF
        @test s_mutation_m.param.variability_mode == STRATEGY_MUTATION

        # environmental variables
        ## population (payoff_m, s_mutation_m)
        @test payoff_m.death_N_vec == s_mutation_m.death_N_vec == fill(143, 10_000)
        @test payoff_m.birth_N_vec == s_mutation_m.birth_N_vec == fill(143, 10_000)
        ## population (population_m)
        diff_vec = [death_N - birth_N for (death_N, birth_N) in zip(population_m.death_N_vec, population_m.birth_N_vec)]
        @test mean(diff_vec) ≈ 0 atol = 1e-3
        @test std(diff_vec) ≈ 59.836 atol = 1e-3
        ### death_N_vec
        @test mean(population_m.death_N_vec) ≈ 23.9698
        @test std(population_m.death_N_vec) ≈ 34.97868835196489
        @test maximum(population_m.death_N_vec) ≈ 218
        @test minimum(population_m.death_N_vec) ≈ 0
        ### birth_N_vec
        @test mean(population_m.birth_N_vec) ≈ 23.9705
        @test std(population_m.birth_N_vec) ≈ 34.750286156933974
        @test maximum(population_m.birth_N_vec) ≈ 194
        @test minimum(population_m.birth_N_vec) ≈ 0

        ## payoff_table
        @test population_m.payoff_table_vec == s_mutation_m.payoff_table_vec
        @test population_m.payoff_table_vec != payoff_m.payoff_table_vec
        @test [t[(C, C)] for t in population_m.payoff_table_vec] == fill((1.0, 1.0), 10_000)
        @test [t[(D, D)] for t in population_m.payoff_table_vec] == fill((0.0, 0.0), 10_000)
        @test [t[(C, D)] for t in population_m.payoff_table_vec] == fill((2.2, 1.5), 10_000)
        @test [t[(D, C)] for t in population_m.payoff_table_vec] == fill((1.5, 2.2), 10_000)

        ### variable T
        @test [t[(C, D)][2] for t in payoff_m.payoff_table_vec] == [t[(D, C)][1] for t in payoff_m.payoff_table_vec]
        T_vec = [t[(C, D)][2] for t in payoff_m.payoff_table_vec]
        @test mean(T_vec) ≈ 1.5048853052966207
        @test std(T_vec) ≈ 0.498828983876384
        @test maximum(T_vec) ≈ 2.0
        @test minimum(T_vec) ≈ 1.0

        ## mutation (μ_s & μ_r)
        @test population_m.μ_s_vec == payoff_m.μ_s_vec == r_mutation_m.μ_s_vec == fill(Float16(0.11), 10_000)
        @test s_mutation_m.μ_s_vec != fill(Float16(0.11), 10_000)

        @test population_m.μ_r_vec == payoff_m.μ_r_vec == s_mutation_m.μ_r_vec == fill(Float16(0.22), 10_000)
        @test r_mutation_m.μ_r_vec != fill(Float16(0.22), 10_000)

        @test mean(s_mutation_m.μ_s_vec) == Float16(0.11)
        @test mean(r_mutation_m.μ_r_vec) == Float16(0.22)
        @test std(s_mutation_m.μ_s_vec) ≈ 0.11
        @test std(r_mutation_m.μ_r_vec) ≈ 0.22
        @test maximum(s_mutation_m.μ_s_vec) == Float16(0.22)
        @test maximum(r_mutation_m.μ_r_vec) == Float16(0.44)
        @test minimum(s_mutation_m.μ_s_vec) ≈ minimum(r_mutation_m.μ_r_vec) ≈ 0
    end
end
