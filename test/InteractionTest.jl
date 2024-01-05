using DataFrames: Not
using LinearAlgebra: Diagonal, diag
using Random
using StatsBase
using Test: @testset, @test

include("../src/Simulation.jl")
using .Simulation: Model, Param, C, D, interaction!, POPULATION, PAYOFF, MUTATION, VARIABILITY_MODE

@testset "C vs. C" begin
    for mode in keys(VARIABILITY_MODE)
        model = Model(Param(initial_N = 4, initial_k = 2, variability_mode = mode))
        model.strategy_vec[1] = model.strategy_vec[2] = C
        model.graph.weights[1, 2] =
            model.graph.weights[2, 1] = model.graph.weights[3, 4] = model.graph.weights[4, 3] = 0.8
        model.graph.weights[2, 3] =
            model.graph.weights[3, 2] = model.graph.weights[4, 1] = model.graph.weights[1, 4] = 0.001

        interaction!(model)

        @test model.payoff_vec[1] == model.payoff_vec[2] == 2.0
        @test model.graph.weights == transpose(model.graph.weights)
        @test model.graph.weights[1, 2] ≈ 0.968

        # check weight limit
        interaction!(model)
        @test model.graph.weights[1, 2] == Float16(1.0)
    end
end

@testset "C vs. D" begin
    for mode in keys(VARIABILITY_MODE)
        model = Model(Param(initial_N = 4, initial_k = 2, variability_mode = mode))
        model.strategy_vec[1] = C
        model.strategy_vec[2] = D
        model.graph.weights[1, 2] =
            model.graph.weights[2, 1] = model.graph.weights[3, 4] = model.graph.weights[4, 3] = 0.8
        model.graph.weights[2, 3] =
            model.graph.weights[3, 2] = model.graph.weights[4, 1] = model.graph.weights[1, 4] = 0.001

        interaction!(model)

        @test model.payoff_vec[1] == -0.2
        @test model.payoff_vec[2] == 2.2
        @test model.graph.weights == transpose(model.graph.weights)
        @test model.graph.weights[1, 2] ≈ 0.8 * 0.9 * 0.9
    end
end

@testset "D vs. C" begin
    for mode in keys(VARIABILITY_MODE)
        model = Model(Param(initial_N = 4, initial_k = 2, variability_mode = mode))
        model.strategy_vec[1] = D
        model.strategy_vec[2] = C
        model.graph.weights[1, 2] =
            model.graph.weights[2, 1] = model.graph.weights[3, 4] = model.graph.weights[4, 3] = 0.8
        model.graph.weights[2, 3] =
            model.graph.weights[3, 2] = model.graph.weights[4, 1] = model.graph.weights[1, 4] = 0.001

        interaction!(model)

        @test model.payoff_vec[1] == 2.2
        @test model.payoff_vec[2] == -0.2
        @test model.graph.weights == transpose(model.graph.weights)
        @test model.graph.weights[1, 2] ≈ 0.8 * 0.9 * 0.9
    end
end

@testset "D vs. D" begin
    for mode in keys(VARIABILITY_MODE)
        model = Model(Param(initial_N = 4, initial_k = 2, variability_mode = mode))
        model.strategy_vec[1] = model.strategy_vec[2] = D
        model.graph.weights[1, 2] =
            model.graph.weights[2, 1] = model.graph.weights[3, 4] = model.graph.weights[4, 3] = 0.8
        model.graph.weights[2, 3] =
            model.graph.weights[3, 2] = model.graph.weights[4, 1] = model.graph.weights[1, 4] = 0.001

        interaction!(model)

        @test model.payoff_vec[1] == model.payoff_vec[2] == 0.0
        @test model.graph.weights == transpose(model.graph.weights)
        @test model.graph.weights[1, 2] ≈ 0.8 * 0.9 * 0.9
    end
end

@testset "Many agents" begin
    for mode in keys(VARIABILITY_MODE)
        model = Model(Param(initial_N = 10, initial_k = 4, rng = MersenneTwister(1), variability_mode = mode))
        model.strategy_vec = repeat([C, D], 5)

        interaction!(model)

        @test diag(model.graph.weights) == fill(0.0, 10)
        @test model.graph.weights == transpose(model.graph.weights)

        if mode == POPULATION
            @test model.payoff_vec ≈ [-0.1, 0.0, 1.0, 1.1, 1.8, 1.1, 0.9, 1.1, -0.1, 2.2]
            @test Vector(model.graph.weights[1, 2:end]) == [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.45]
            @test Vector(model.graph.weights[2, 3:end]) == [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.45]
            @test Vector(model.graph.weights[3, 4:end]) == [0.5, 0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
            @test Vector(model.graph.weights[4, 5:end]) == [0.45, 0.45, 0.0, 0.0, 0.0, 0.0]
            @test Vector(model.graph.weights[5, 6:end]) == [0.45, 0.55, 0.0, 0.0, 0.0]
            @test Vector(model.graph.weights[6, 7:end]) == [0.5, 0.5, 0.0, 0.0]
            @test Vector(model.graph.weights[7, 8:end]) == [0.45, 0.5, 0.0]
            @test Vector(model.graph.weights[8, 9:end]) == [0.5, 0.45]
            @test Vector(model.graph.weights[9, 10:end]) == [0.45]
        elseif mode == PAYOFF
            @test model.payoff_vec ≈ [1.0, 0.0, 3.0, 0.0, 2.0, 0.0, 1.0, 2.2, 0.7, 1.1]
            @test Vector(model.graph.weights[1, 2:end]) == [0.5, 0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]
            @test Vector(model.graph.weights[2, 3:end]) == [0.5, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.45]
            @test Vector(model.graph.weights[3, 4:end]) ≈ [0.5, 0.605, 0.0, 0.0, 0.0, 0.0, 0.0]
            @test Vector(model.graph.weights[4, 5:end]) == [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
            @test Vector(model.graph.weights[5, 6:end]) == [0.5, 0.5, 0.0, 0.0, 0.0]
            @test Vector(model.graph.weights[6, 7:end]) == [0.5, 0.45, 0.0, 0.0]
            @test Vector(model.graph.weights[7, 8:end]) == [0.5, 0.55, 0.0]
            @test Vector(model.graph.weights[8, 9:end]) == [0.405, 0.5]
            @test Vector(model.graph.weights[9, 10:end]) == [0.45]
        elseif mode == MUTATION
            @test model.payoff_vec ≈ [1.8, 1.1, 2.0, 1.1, -0.1, 1.1, 0.9, 1.1, 0.8, 2.2]
            @test Vector(model.graph.weights[1, 2:end]) ≈ [0.45, 0.605, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.45]
            @test Vector(model.graph.weights[2, 3:end]) == [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
            @test Vector(model.graph.weights[3, 4:end]) == [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
            @test Vector(model.graph.weights[4, 5:end]) == [0.45, 0.45, 0.0, 0.0, 0.0, 0.0]
            @test Vector(model.graph.weights[5, 6:end]) == [0.5, 0.5, 0.0, 0.0, 0.0]
            @test Vector(model.graph.weights[6, 7:end]) == [0.45, 0.5, 0.0, 0.0]
            @test Vector(model.graph.weights[7, 8:end]) == [0.5, 0.55, 0.0]
            @test Vector(model.graph.weights[8, 9:end]) == [0.45, 0.5]
            @test Vector(model.graph.weights[9, 10:end]) == [0.45]
        end
    end
end

@testset "Many agents (interaction_freqency = 0.3)" begin
    for mode in keys(VARIABILITY_MODE)
        model = Model(
            Param(
                initial_N = 10,
                initial_k = 6,
                interaction_freqency = 0.3,
                rng = MersenneTwister(3),
                variability_mode = mode,
            ),
        )
        model.strategy_vec = [C, D, C, D, D, D, C, C, D, D]

        interaction!(model)

        @test diag(model.graph.weights) == fill(0.0, 10)
        @test model.graph.weights == transpose(model.graph.weights)

        if mode == POPULATION
            @test model.payoff_vec ≈ [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 1.1, 0.0]
            @test model.graph.weights[1, 8] == 0.55  # C-C
            @test model.graph.weights[8, 9] == 0.45  # D-D
        elseif mode == PAYOFF
            @test model.payoff_vec ≈ [-0.1, 0.0, -0.1, 0.0, 1.1, 0.0, 1.0, 1.0, 1.1, 0.0]
            @test model.graph.weights[7, 8] == 0.55  # C-C
            @test model.graph.weights[1, 9] == 0.45  # C-D
            @test model.graph.weights[3, 5] == 0.45  # C-D
        elseif mode == MUTATION
            @test model.payoff_vec ≈ [-0.1, 0.0, 0.0, 1.1, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
            @test model.graph.weights[7, 8] == 0.55  # C-C
            @test model.graph.weights[1, 4] == 0.45  # C-D
        end
    end
end