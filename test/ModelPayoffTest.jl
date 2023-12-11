using DataFrames: Not
using LinearAlgebra: Diagonal, diag
using Random
using StatsBase
using Test: @testset, @test

include("../src/Simulation.jl")
using .Simulation: ModelPayoff, Param, C, D, interaction!, death_and_birth!

#--------------------
# Model
#--------------------
@testset "Model" begin
    @testset "default" begin
        model = ModelPayoff(Param(rng = MersenneTwister(1)))

        # agents
        @test model.strategy_vec == fill(D, 1_000)
        @test model.payoff_vec == fill(0.0, 1_000)

        # graph_weights
        @test model.param.initial_graph_weight == 0.5
        @test model.graph_weights == fill(0.5, (1_000, 1_000)) - Diagonal(fill(0.5, 1_000))

        # payoff_table_vec
        @test [t[(C, C)] for t in model.payoff_table_vec] == fill((1.0, 1.0), model.param.generations)
        @test [t[(D, D)] for t in model.payoff_table_vec] == fill((0.0, 0.0), model.param.generations)
        @test [t[(C, D)][1] for t in model.payoff_table_vec] == fill(-0.1, model.param.generations)
        @test [t[(D, C)][2] for t in model.payoff_table_vec] == fill(-0.1, model.param.generations)
        @test [t[(C, D)][2] for t in model.payoff_table_vec] == [t[(D, C)][1] for t in model.payoff_table_vec]
        T_vec = [t[(C, D)][2] for t in model.payoff_table_vec]
        @test mean(T_vec) ≈ model.param.T atol = 0.01
        @test std(T_vec) == 0.1052127277387196
        @test maximum(T_vec) == 1.331774260169861
        @test minimum(T_vec) == 0.8964687911369331

        # misc
        @test model.payoff_table === nothing
        @test model.death_rate_vec === nothing
        @test model.param.interaction_freqency == 1.0
        @test model.param.relationship_volatility == 0.1
        @test model.param.reproduction_rate == 0.1
        @test model.param.δ == 0.01
        @test model.param.μ == 0.00
        @test model.param.σ == 0.1
        @test model.param.β == 0.1
        @test model.param.generations == 100
        @test isa(model.param.rng, MersenneTwister)
    end

    @testset "customized" begin
        param = Param(
            initial_N = 255,
            initial_graph_weight = 0.123,
            T = 1.3,
            S = 2.2,
            interaction_freqency = 0.111,
            relationship_volatility = 0.19,
            reproduction_rate = 0.56,
            δ = 0.23,
            μ = 0.45,
            σ = 0.2,
            β = 0.3,
            generations = 10_000,
            rng = MersenneTwister(1),
        )
        model = ModelPayoff(param)

        # agents
        @test model.strategy_vec == fill(D, 255)
        @test model.payoff_vec == fill(0.0, 255)

        # graph_weights
        @test model.param.initial_graph_weight == 0.123
        @test model.graph_weights == Float16.(fill(0.123, (255, 255)) - Diagonal(fill(0.123, 255)))

        # payoff_table_vec
        @test [t[(C, C)] for t in model.payoff_table_vec] == fill((1.0, 1.0), model.param.generations)
        @test [t[(D, D)] for t in model.payoff_table_vec] == fill((0.0, 0.0), model.param.generations)
        @test [t[(C, D)][1] for t in model.payoff_table_vec] == fill(2.2, model.param.generations)
        @test [t[(D, C)][2] for t in model.payoff_table_vec] == fill(2.2, model.param.generations)
        @test [t[(C, D)][2] for t in model.payoff_table_vec] == [t[(D, C)][1] for t in model.payoff_table_vec]
        T_vec = [t[(C, D)][2] for t in model.payoff_table_vec]
        @test mean(T_vec) ≈ 1.2920395218354517
        @test std(T_vec) ≈ 0.21086642707014025
        @test maximum(T_vec) ≈ 1.9538803147013142
        @test minimum(T_vec) ≈ 0.6

        # misc
        @test model.payoff_table === nothing
        @test model.death_rate_vec === nothing
        @test model.param.interaction_freqency == 0.111
        @test model.param.relationship_volatility == 0.19
        @test model.param.reproduction_rate == 0.56
        @test model.param.δ == 0.23
        @test model.param.μ == 0.45
        @test model.param.σ == 0.2
        @test model.param.β == 0.3
        @test model.param.generations == 10_000
    end
end

#--------------------
# interaction!
#--------------------
@testset "interaction!" begin
    @testset "C vs. C" begin
        model = ModelPayoff(Param(initial_N = 2, relationship_volatility = 0.2))
        model.strategy_vec[1] = C
        model.strategy_vec[2] = C
        interaction!(model, 1)

        @test model.payoff_vec[1] == 2.0
        @test model.payoff_vec[2] == 2.0
        @test model.graph_weights[1, 2] == Float16(0.5 * 1.2 * 1.2)
        @test model.graph_weights == transpose(model.graph_weights)

        # check weight limit
        model = ModelPayoff(Param(initial_N = 2, relationship_volatility = 0.9))
        model.strategy_vec[1] = C
        model.strategy_vec[2] = C
        interaction!(model, 1)
        @test model.graph_weights[1, 2] == Float16(1.0)
    end

    @testset "C vs. D" begin
        model = ModelPayoff(Param(initial_N = 2))
        model.strategy_vec[1] = C
        model.strategy_vec[2] = D
        interaction!(model, 1)

        @test model.payoff_vec[1] == -0.2
        @test model.payoff_vec[2] == 2.2
        @test model.graph_weights[1, 2] == Float16(0.5 * 0.9 * 0.9)
        @test model.graph_weights == transpose(model.graph_weights)
    end

    @testset "D vs. C" begin
        model = ModelPayoff(Param(initial_N = 2))
        model.strategy_vec[1] = D
        model.strategy_vec[2] = C
        interaction!(model, 1)

        @test model.payoff_vec[1] == 2.2
        @test model.payoff_vec[2] == -0.2
        @test model.graph_weights[1, 2] == Float16(0.5 * 0.9 * 0.9)
        @test model.graph_weights == transpose(model.graph_weights)
    end

    @testset "D vs. D" begin
        model = ModelPayoff(Param(initial_N = 2, relationship_volatility = 0.3))
        model.strategy_vec[1] = D
        model.strategy_vec[2] = D
        interaction!(model, 1)

        @test model.payoff_vec[1] == 0.0
        @test model.payoff_vec[2] == 0.0
        @test model.graph_weights[1, 2] == Float16(Float16(Float16(0.5) * 0.7) * 0.7)
        @test model.graph_weights == transpose(model.graph_weights)
    end

    @testset "Many agents" begin
        model = ModelPayoff(Param(initial_N = 10, rng = MersenneTwister(1)))
        model.strategy_vec = repeat([C, D], 5)
        interaction!(model, 1)

        @test model.payoff_vec ≈ [1.9, 1.1, 0.8, 0.0, 1.0, 0.0, 2.0, 1.1, 2.0, 1.1]

        @test diag(model.graph_weights) == fill(0.0, 10)
        @test model.graph_weights == transpose(model.graph_weights)
        @test model.graph_weights[1, 2:10] == Float16[0.5, 0.5, 0.5, 0.5, 0.5, 0.55, 0.5, 0.55, 0.45]
        @test model.graph_weights[2, 3:10] == Float16[0.45, 0.45, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        @test model.graph_weights[3, 4:10] == Float16[0.5, 0.5, 0.5, 0.5, 0.45, 0.55, 0.5]
        @test model.graph_weights[4, 5:10] == Float16[0.5, 0.45, 0.5, 0.45, 0.5, 0.5]
        @test model.graph_weights[5, 6:10] == Float16[0.5, 0.55, 0.5, 0.5, 0.5]
        @test model.graph_weights[6, 7:10] == Float16[0.5, 0.5, 0.5, 0.5]
        @test model.graph_weights[7, 8:10] == Float16[0.5, 0.5, 0.5]
        @test model.graph_weights[8, 9:10] == Float16[0.5, 0.5]
        @test model.graph_weights[9, 10:10] == Float16[0.5]
    end

    @testset "Many agents (interaction_freqency = 0.149)" begin
        model = ModelPayoff(Param(initial_N = 10, interaction_freqency = 0.149, rng = MersenneTwister(3)))
        model.strategy_vec = repeat([C, D], 5)
        interaction!(model, 1)

        @test model.payoff_vec ≈ [0.0, 0.0, 0.0, 0.0, -0.1, 1.1, 0.0, 0.0, 0.0, 0.0]

        expected_weights = fill(0.5, (10, 10))
        expected_weights -= Diagonal(expected_weights)
        expected_weights[5, 6] = expected_weights[6, 5] = 0.45
        @test model.graph_weights == Float16.(expected_weights)
    end
end

#--------------------
# death_and_birth!
#--------------------
# 1.0 - δ + δ * payoff
@testset "classic_fitness" begin
    δ = 1.0
    for π = -10.0:0.1:10.0
        @test Simulation.classic_fitness(π, δ) == π
    end
    δ = 0.1
    π_vec = [-10.0, -1.0, 0.0, 1.0, 2.0, 10.0]
    f_vec = [-0.1, 0.8, 0.9, 1.0, 1.1, 1.9]
    for (π, fitness) in zip(π_vec, f_vec)
        @test Simulation.classic_fitness(π, δ) ≈ fitness
    end
end

# 1.0 / (1.0 + exp(-δ * payoff))
@testset "sigmoid_fitness" begin
    δ = 0.5
    @test Simulation.sigmoid_fitness(1_000.0, δ) ≈ 1.0
    @test Simulation.sigmoid_fitness(1.0, δ) ≈ 0.622459331
    @test Simulation.sigmoid_fitness(0.0, δ) == 0.5
    @test Simulation.sigmoid_fitness(-1.0, δ) ≈ 0.37754067
    @test Simulation.sigmoid_fitness(-2_000.0, δ) ≈ 0.0
end

@testset "pick_deaths and pick_parents" begin
    @testset "pick_deaths" begin
        model = ModelPayoff(Param(initial_N = 10))
        Simulation.pick_deaths(model, 0)
    end

    @testset "uniform distribution" begin
        model = ModelPayoff(Param(initial_N = 100))

        # pick_deaths
        death_id_vec_vec = [Simulation.pick_deaths(model, 11) for _ = 1:10_000]
        death_id_vec = vcat(death_id_vec_vec...)
        @test length(death_id_vec) == 11 * 10_000

        death_id_freq = fit(Histogram, death_id_vec, 1:101).weights
        @test mean(death_id_freq) == 1100
        @test std(death_id_freq) < 150

        # pick_parents
        parent_id_vec_vec = [Simulation.pick_parents(model, 12) for _ = 1:10_000]
        parent_id_vec = vcat(parent_id_vec_vec...)
        @test length(parent_id_vec) == 12 * 10_000

        parent_id_freq = fit(Histogram, parent_id_vec, 1:101).weights
        @test mean(parent_id_freq) == 1200
        @test std(parent_id_freq) < 150
    end

    @testset "polarized situation" begin
        model = ModelPayoff(Param(initial_N = 100, δ = 1.0))
        model.payoff_vec[33] = 100
        model.payoff_vec[66] = -100

        # pick_deaths
        death_id_vec_vec = [Simulation.pick_deaths(model, 11) for _ = 1:10_000]
        death_id_vec = vcat(death_id_vec_vec...)
        @test length(death_id_vec) == 11 * 10_000

        death_id_freq = fit(Histogram, death_id_vec, 1:101).weights
        @test mean(death_id_freq) == 1100
        @test death_id_freq[33] < 100
        @test death_id_freq[66] > 1800

        # pick_parents
        parent_id_vec_vec = [Simulation.pick_parents(model, 12) for _ = 1:10_000]
        parent_id_vec = vcat(parent_id_vec_vec...)
        @test length(parent_id_vec) == 12 * 10_000

        parent_id_freq = fit(Histogram, parent_id_vec, 1:101).weights
        @test mean(parent_id_freq) == 1200
        @test parent_id_freq[33] > 1800
        @test parent_id_freq[66] < 100
    end
end

@testset "normalize_graph_weights!" begin
    # before
    model = ModelPayoff(Param(initial_N = 10))
    @test model.graph_weights == fill(0.5, (10, 10)) - Diagonal(fill(0.5, (10, 10)))
    @test sum(model.graph_weights) == 10 * 9 * 0.5 == 45
    model.graph_weights[1, 2:end] = fill(1.0, 9)

    # execute
    Simulation.normalize_graph_weights!(model.graph_weights, model.N, model.param.initial_graph_weight)

    # after
    @test diag(model.graph_weights) == fill(0.0, 10)
    @test model.graph_weights[1, 2:end] == Float16.(fill(0.909, 9))
    @test model.graph_weights[2, Not(2)] == Float16.(fill(0.4546, 9))
    @test model.graph_weights[end, Not(end)] == Float16.(fill(0.4546, 9))
    @test sum(model.graph_weights) == Float16(45.03)
end

function make_symmetric_matrix(N::Int)::Matrix{Float16}
    symmetric_matrix = repeat(collect(0.0:(1 / (N - 1)):1.0), 1, N)
    symmetric_matrix -= Diagonal(symmetric_matrix)
    symmetric_matrix = (symmetric_matrix + symmetric_matrix') / 2
    return Float16.(symmetric_matrix)
end

@testset "death_and_birth!" begin
    @testset "μ = 0.0" begin
        model = ModelPayoff(Param(initial_N = 100, reproduction_rate = 0.03, rng = MersenneTwister(1)))

        # before
        model.strategy_vec[[3, 26, 92 + 3]] .= C
        model.graph_weights = make_symmetric_matrix(100)

        # execution
        death_id_vec, parent_id_vec = death_and_birth!(model)
        @test death_id_vec == [61, 75, 91]
        @test parent_id_vec == [3, 26, 92]

        # after
        @test model.N == length(model.strategy_vec) == length(model.payoff_vec) == 100
        @test size(model.graph_weights) == (100, 100)
        @test model.strategy_vec[98:100] == [C, C, C]
        @test model.graph_weights == transpose(model.graph_weights)  # check symmetry
        @test diag(model.graph_weights) == fill(0.0, 100)  # diagonal is 0.0

        model.graph_weights ./= Float16(1.0205)  # for normalization
        _index = setdiff(1:100, [3, 98])
        @test model.graph_weights[3, 98] == model.graph_weights[98, 3] == 1.0
        @test model.graph_weights[3, _index] == model.graph_weights[98, _index]

        _index = setdiff(1:100, [26, 99])
        @test model.graph_weights[26, 99] == model.graph_weights[99, 26] == 1.0
        @test model.graph_weights[26, _index] == model.graph_weights[99, _index]

        _index = setdiff(1:100, [92, 100])
        @test model.graph_weights[92, 100] == model.graph_weights[100, 92] == 1.0
        @test model.graph_weights[92, _index] == model.graph_weights[100, _index]

        # before
        model.strategy_vec[[10 + 1, 24 + 1, 82 + 5]] .= D
        model.graph_weights = make_symmetric_matrix(100)

        # execution
        death_id_vec, parent_id_vec = death_and_birth!(model)
        @test death_id_vec == [11, 26, 38]
        @test parent_id_vec == [35, 69, 73]

        # after
        @test model.N == length(model.strategy_vec) == length(model.payoff_vec) == 100
        @test size(model.graph_weights) == (100, 100)
        @test model.strategy_vec[[98, 99, 100]] == [D, D, D]
        @test model.graph_weights == transpose(model.graph_weights)  # is symmetry
        @test diag(model.graph_weights) == fill(0.0, 100)  # diagonal is 0.0

        model.graph_weights ./= Float16(0.978)  # for normalization
        _index = setdiff(1:100, [35, 98])
        @test model.graph_weights[35, 98] == model.graph_weights[98, 35] == 1.0
        @test model.graph_weights[35, _index] == model.graph_weights[98, _index]

        _index = setdiff(1:100, [69, 99])
        @test model.graph_weights[69, 99] == model.graph_weights[99, 69] == 1.0
        @test model.graph_weights[69, _index] == model.graph_weights[99, _index]

        _index = setdiff(1:100, [73, 100])
        @test model.graph_weights[73, 100] == model.graph_weights[100, 73] == 1.0
        @test model.graph_weights[73, _index] == model.graph_weights[100, _index]
    end

    @testset "μ = 1.0" begin
        model = ModelPayoff(Param(initial_N = 100, reproduction_rate = 0.03, μ = 1.0, rng = MersenneTwister(1)))

        # before
        model.strategy_vec[[3, 26, 92 + 3]] = [C, D, C]
        model.graph_weights = make_symmetric_matrix(100)

        # execution
        death_id_vec, parent_id_vec = death_and_birth!(model)
        @test death_id_vec == [61, 75, 91]
        @test parent_id_vec == [3, 26, 92]

        # after
        @test model.N == length(model.strategy_vec) == length(model.payoff_vec) == 100
        @test size(model.graph_weights) == (100, 100)
        @test model.strategy_vec[98:100] == [D, C, D]
        @test model.graph_weights == transpose(model.graph_weights)  # check symmetry
        @test diag(model.graph_weights) == fill(0.0, 100)  # diagonal is 0.0

        model.graph_weights ./= Float16(1.0205)  # for normalization
        _index = [x for x in 1:100 if !(x in [3, 98])]
        @test model.graph_weights[3, 98] == model.graph_weights[98, 3] == 1.0
        @test model.graph_weights[3, _index] == model.graph_weights[3, _index]

        _index = [x for x in 1:100 if !(x in [26, 99])]
        @test model.graph_weights[26, 99] == model.graph_weights[99, 26] == 1.0
        @test model.graph_weights[26, _index] == model.graph_weights[99, _index]

        _index = [x for x in 1:100 if !(x in [92, 100])]
        @test model.graph_weights[92, 100] == model.graph_weights[100, 92] == 1.0
        @test model.graph_weights[92, _index] == model.graph_weights[100, _index]
    end
end
