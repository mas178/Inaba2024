using DataFrames: Not
using LinearAlgebra: Diagonal, diag
using Plots
using Random
using StatsBase
using Test: @testset, @test

include("../src/Simulation.jl")
using .Simulation

#--------------------
# Model
#--------------------
@testset "invert" begin
    @test Simulation.invert(C) == D
    @test Simulation.invert(D) == C
end

@testset "ar1_model" begin
    T = 100
    rng = MersenneTwister(1)
    for (β, σ) in Iterators.product(0.1:0.1:0.9, 0.1:0.1:0.9)
        @test mean([mean(Simulation.ar1_model(β, σ, T, 1.0, rng)) for _ = 1:1000]) ≈ 1.0 atol = 0.1
    end
    @test mean([
        std(Simulation.ar1_model(0.2, 0.1, T, 1.0, rng)) > std(Simulation.ar1_model(0.1, 0.1, T, 1.0, rng)) for
        _ = 1:1000
    ]) > 0.5
    @test std(Simulation.ar1_model(0.1, 0.2, T, 1.0, rng)) > std(Simulation.ar1_model(0.1, 0.1, T, 1.0, rng))
    @test std(Simulation.ar1_model(0.2, 0.2, T, 1.0, rng)) > std(Simulation.ar1_model(0.1, 0.1, T, 1.0, rng))
end

@testset "Model" begin
    @testset "default" begin
        model = Model(Param())

        # agents
        @test model.strategy_vec == fill(D, 1_000)
        @test model.payoff_vec == fill(0.0, 1_000)

        # graph_weights
        @test model.param.initial_graph_weight == 0.5
        @test model.graph_weights == fill(0.5, (1_000, 1_000)) - Diagonal(fill(0.5, 1_000))

        # misc
        @test model.payoff_table[(C, C)] == (1.0, 1.0)
        @test model.payoff_table[(C, D)] == (-0.1, 1.1)
        @test model.payoff_table[(D, C)] == (1.1, -0.1)
        @test model.payoff_table[(D, D)] == (0.0, 0.0)
        @test model.param.interaction_freqency == 1.0
        @test model.param.relationship_volatility == 0.1
        @test model.param.birth_rate == 0.1
        @test isa(model.env_severity_vec, Vector{Float64})
        @test length(model.env_severity_vec) == 11
        @test mean(model.env_severity_vec) ≈ 1.0 atol = 0.5
        @test model.param.δ == 0.01
        @test model.param.μ == 0.00
        @test model.param.generations == 100
        @test isa(model.param.rng, MersenneTwister)
    end

    @testset "customized" begin
        param = Param(
            initial_N = 255,
            initial_graph_weight = 0.123,
            T = 3.3,
            S = 2.2,
            interaction_freqency = 0.111,
            relationship_volatility = 0.19,
            birth_rate = 0.56,
            δ = 0.23,
            μ = 0.45,
            generations = 10_000,
        )
        model = Model(param)

        # agents
        @test model.strategy_vec == fill(D, 255)
        @test model.payoff_vec == fill(0.0, 255)

        # graph_weights
        @test model.param.initial_graph_weight == 0.123
        @test model.graph_weights == Float16.(fill(0.123, (255, 255)) - Diagonal(fill(0.123, 255)))

        # misc
        @test model.payoff_table[(C, C)] == (1.0, 1.0)
        @test model.payoff_table[(C, D)] == (2.2, 3.3)
        @test model.payoff_table[(D, C)] == (3.3, 2.2)
        @test model.payoff_table[(D, D)] == (0.0, 0.0)
        @test model.param.interaction_freqency == 0.111
        @test model.param.relationship_volatility == 0.19
        @test model.param.birth_rate == 0.56
        @test isa(model.env_severity_vec, Vector{Float64})
        @test length(model.env_severity_vec) == 1_001
        @test mean(model.env_severity_vec) ≈ 1.0 atol = 0.5
        @test model.param.δ == 0.23
        @test model.param.μ == 0.45
        @test model.param.generations == 10_000
    end
end

#--------------------
# interaction!
#--------------------
@testset "interaction!" begin
    @testset "C vs. C" begin
        model = Model(Param(initial_N = 2, relationship_volatility = 0.2))
        model.strategy_vec[1] = C
        model.strategy_vec[2] = C
        interaction!(model)

        @test model.payoff_vec[1] == 2.0
        @test model.payoff_vec[2] == 2.0
        @test model.graph_weights[1, 2] == Float16(0.5 * 1.2 * 1.2)
        @test model.graph_weights == transpose(model.graph_weights)

        # check weight limit
        model = Model(Param(initial_N = 2, relationship_volatility = 0.9))
        model.strategy_vec[1] = C
        model.strategy_vec[2] = C
        interaction!(model)
        @test model.graph_weights[1, 2] == Float16(1.0)
    end

    @testset "C vs. D" begin
        model = Model(Param(initial_N = 2))
        model.strategy_vec[1] = C
        model.strategy_vec[2] = D
        interaction!(model)

        @test model.payoff_vec[1] == -0.2
        @test model.payoff_vec[2] == 2.2
        @test model.graph_weights[1, 2] == Float16(0.5 * 0.9 * 0.9)
        @test model.graph_weights == transpose(model.graph_weights)
    end

    @testset "D vs. C" begin
        model = Model(Param(initial_N = 2))
        model.strategy_vec[1] = D
        model.strategy_vec[2] = C
        interaction!(model)

        @test model.payoff_vec[1] == 2.2
        @test model.payoff_vec[2] == -0.2
        @test model.graph_weights[1, 2] == Float16(0.5 * 0.9 * 0.9)
        @test model.graph_weights == transpose(model.graph_weights)
    end

    @testset "D vs. D" begin
        model = Model(Param(initial_N = 2, relationship_volatility = 0.3))
        model.strategy_vec[1] = D
        model.strategy_vec[2] = D
        interaction!(model)

        @test model.payoff_vec[1] == 0.0
        @test model.payoff_vec[2] == 0.0
        @test model.graph_weights[1, 2] == Float16(Float16(Float16(0.5) * 0.7) * 0.7)
        @test model.graph_weights == transpose(model.graph_weights)
    end

    @testset "Many agents" begin
        model = Model(Param(initial_N = 10, rng = MersenneTwister(1)))
        model.strategy_vec = repeat([C, D], 5)
        interaction!(model)

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
        model = Model(Param(initial_N = 10, interaction_freqency = 0.149, rng = MersenneTwister(3)))
        model.strategy_vec = repeat([C, D], 5)
        interaction!(model)

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
        model = Model(Param(initial_N = 10))
        Simulation.pick_deaths(model, 0)
    end

    @testset "uniform distribution" begin
        model = Model(Param(initial_N = 100))

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
        model = Model(Param(initial_N = 100, δ = 1.0))
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
    model = Model(Param(initial_N = 10))
    @test model.graph_weights == fill(0.5, (10, 10)) - Diagonal(fill(0.5, (10, 10)))
    @test sum(model.graph_weights) == 10 * 9 * 0.5 == 45
    model.graph_weights[1, 2:end] = fill(1.0, 9)

    # execute
    Simulation.normalize_graph_weights!(model)

    # after
    @test diag(model.graph_weights) == fill(0.0, 10)
    @test model.graph_weights[1, 2:end] == Float16.(fill(0.909, 9))
    @test model.graph_weights[2, Not(2)] == Float16.(fill(0.4546, 9))
    @test model.graph_weights[end, Not(end)] == Float16.(fill(0.4546, 9))
    @test sum(model.graph_weights) == Float16(45.03)
end

@testset "get_death_rate" begin
    model = Model(Param())
    expected_death_rate = model.env_severity_vec .* model.param.birth_rate

    for generation = 1:model.param.generations
        actual_death_rate = Simulation.get_death_rate(model, generation)

        if generation % 10 == 1
            @test actual_death_rate == expected_death_rate[ceil(Int, generation / 10)]
        else
            @test actual_death_rate == Simulation.get_death_rate(model, generation - 1)
        end
    end
end

function make_symmetric_matrix(N::Int)::Matrix{Float16}
    symmetric_matrix = repeat(collect(0.0:1/(N-1):1.0), 1, N)
    symmetric_matrix -= Diagonal(symmetric_matrix)
    symmetric_matrix = (symmetric_matrix + symmetric_matrix') / 2
    return Float16.(symmetric_matrix)
end

@testset "death_and_birth!" begin
    model = Model(Param())
    byte = @allocated death_and_birth!(model, 1)
    println("$(byte / 1024 / 1024) MB")

    @testset "μ = 0.0" begin
        model = Model(Param(initial_N = 100, birth_rate = 0.03, rng = MersenneTwister(1)))

        # before
        model.strategy_vec[[25 + 1, 50 + 1, 54 + 1]] .= C
        model.graph_weights = make_symmetric_matrix(100)

        # execution
        model.env_severity_vec[1] = 0.7
        death_id_vec, parent_id_vec = death_and_birth!(model, 1)
        @test death_id_vec == [18, 79]
        @test parent_id_vec == [25, 50, 54]

        # after
        @test model.N == length(model.strategy_vec) == length(model.payoff_vec) == 101
        @test size(model.graph_weights) == (101, 101)
        @test model.strategy_vec[99:101] == [C, C, C]
        @test model.graph_weights == transpose(model.graph_weights)  # check symmetry
        @test diag(model.graph_weights) == fill(0.0, 101)  # diagonal is 0.0

        model.graph_weights ./= Float16(1.002)  # for normalization
        _index = setdiff(1:101, [25, 99])
        @test model.graph_weights[25, 99] == model.graph_weights[99, 25] == 1.0
        @test model.graph_weights[25, _index] == model.graph_weights[99, _index]

        _index = setdiff(1:101, [50, 100])
        @test model.graph_weights[50, 100] == model.graph_weights[100, 50] == 1.0
        @test model.graph_weights[50, _index] == model.graph_weights[100, _index]

        _index = setdiff(1:101, [54, 101])
        @test model.graph_weights[54, 101] == model.graph_weights[101, 54] == 1.0
        @test model.graph_weights[54, _index] == model.graph_weights[101, _index]

        # before
        model.strategy_vec[[10 + 1, 24 + 1, 82 + 5]] .= D
        model.graph_weights = make_symmetric_matrix(101)

        # execution
        model.env_severity_vec[1] = 1.6
        death_id_vec, parent_id_vec = death_and_birth!(model, 1)
        @test death_id_vec == [6, 38, 59, 70, 78]
        @test parent_id_vec == [10, 24, 82]

        # after
        @test model.N == length(model.strategy_vec) == length(model.payoff_vec) == 99
        @test size(model.graph_weights) == (99, 99)
        @test model.strategy_vec[[97, 98, 99]] == [D, D, D]
        @test model.graph_weights == transpose(model.graph_weights)  # is symmetry
        @test diag(model.graph_weights) == fill(0.0, 99)  # diagonal is 0.0

        model.graph_weights ./= Float16(1.005)  # for normalization
        _index = setdiff(1:98, [10, 97])
        @test model.graph_weights[10, 97] == model.graph_weights[97, 10] == 1.0
        @test model.graph_weights[10, _index] == model.graph_weights[97, _index]

        _index = setdiff(1:98, [24, 98])
        @test model.graph_weights[24, 98] == model.graph_weights[98, 24] == 1.0
        @test model.graph_weights[24, _index] == model.graph_weights[98, _index]

        _index = setdiff(1:98, [82, 99])
        @test model.graph_weights[82, 99] == model.graph_weights[99, 82] == 1.0
        @test model.graph_weights[82, _index] == model.graph_weights[99, _index]
    end

    @testset "μ = 1.0" begin
        model = Model(Param(initial_N = 100, birth_rate = 0.03, μ = 1.0, rng = MersenneTwister(1)))

        # before
        model.strategy_vec[[25 + 1, 50 + 1, 54 + 1]] = [C, D, C]
        model.env_severity_vec[1] = 0.6
        model.graph_weights = make_symmetric_matrix(100)

        # execution
        death_id_vec, parent_id_vec = death_and_birth!(model, 1)
        @test death_id_vec == [18, 79]
        @test parent_id_vec == [25, 50, 54]

        # after
        @test model.N == length(model.strategy_vec) == length(model.payoff_vec) == 101
        @test size(model.graph_weights) == (101, 101)
        @test model.strategy_vec[99:101] == [D, C, D]
        @test model.graph_weights == transpose(model.graph_weights)  # check symmetry
        @test diag(model.graph_weights) == fill(0.0, 101)  # diagonal is 0.0

        model.graph_weights ./= Float16(1.002)  # for normalization
        _index = [x for x in 1:100 if !(x in [25, 99])]
        @test model.graph_weights[25, 99] == model.graph_weights[99, 25] == 1.0
        @test model.graph_weights[25, _index] == model.graph_weights[99, _index]

        _index = [x for x in 1:101 if !(x in [50, 100])]
        @test model.graph_weights[50, 100] == model.graph_weights[100, 50] == 1.0
        @test model.graph_weights[50, _index] == model.graph_weights[100, _index]

        _index = [x for x in 1:101 if !(x in [54, 101])]
        @test model.graph_weights[54, 101] == model.graph_weights[101, 54] == 1.0
        @test model.graph_weights[54, _index] == model.graph_weights[101, _index]
    end
end
