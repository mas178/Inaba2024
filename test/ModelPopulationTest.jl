using DataFrames: Not
using LinearAlgebra: Diagonal, diag
using Random
using StatsBase
using Test: @testset, @test

include("../src/Simulation.jl")
using .Simulation: Model, Param, POPULATION, C, D, interaction!, death_and_birth!, run

#--------------------
# Model
#--------------------
@testset "invert" begin
    @test Simulation.invert(C) == D
    @test Simulation.invert(D) == C
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

        # population
        @test model.param.initial_N == 1000
        @test model.N == 1000
        @test length(model.N_vec) == 101
        @test length(model.death_N_vec) == 100
        @test length(model.birth_N_vec) == 100
        @test mean(model.N_vec) ≈ 1000.0
        @test std(model.N_vec) ≈ 0.0

        # payoff_table
        for payoff_table in model.payoff_table_vec
            @test payoff_table[(C, C)] == (1.0, 1.0)
            @test payoff_table[(C, D)] == (-0.1, 1.1)
            @test payoff_table[(D, C)] == (1.1, -0.1)
            @test payoff_table[(D, D)] == (0.0, 0.0)
        end

        # misc
        @test model.param.generations == 100
        @test model.param.interaction_freqency == 1.0
        @test model.param.Δw == 0.1
        @test model.param.reproduction_rate == 0.1
        @test model.param.δ == 0.01
        @test model.param.μ == 0.00
        @test model.param.sigma == 0.1
        @test model.param.β == 0.1
        @test isa(model.param.rng, MersenneTwister)
        @test model.param.variability_mode == Simulation.POPULATION
    end

    @testset "customized" begin
        param = Param(
            initial_N = 255,
            initial_T = 3.3,
            S = 2.2,
            initial_graph_weight = 0.123,
            interaction_freqency = 0.111,
            Δw = 0.19,
            reproduction_rate = 0.56,
            δ = 0.23,
            μ = 0.45,
            β = 0.3,
            sigma = 50,
            generations = 10_000,
            variability_mode = POPULATION,
        )
        model = Model(param)

        # agents
        @test model.strategy_vec == fill(D, 255)
        @test model.payoff_vec == fill(0.0, 255)

        # graph_weights
        @test model.param.initial_graph_weight == 0.123
        @test model.graph_weights == Float16.(fill(0.123, (255, 255)) - Diagonal(fill(0.123, 255)))

        # population
        @test model.param.initial_N == 255
        @test model.N == 255
        @test mean(model.N_vec) ≈ 255 atol = 10
        @test 50 < std(model.N_vec) < 100

        # payoff_table
        for payoff_table in model.payoff_table_vec
            @test payoff_table[(C, C)] == (1.0, 1.0)
            @test payoff_table[(C, D)] == (2.2, 3.3)
            @test payoff_table[(D, C)] == (3.3, 2.2)
            @test payoff_table[(D, D)] == (0.0, 0.0)
        end

        # misc
        @test model.param.generations == 10_000
        @test model.param.interaction_freqency == 0.111
        @test model.param.Δw == 0.19
        @test model.param.reproduction_rate == 0.56
        @test model.param.δ == 0.23
        @test model.param.μ == 0.45
        @test model.param.sigma == 50
        @test model.param.β == 0.3
        @test model.param.variability_mode == Simulation.POPULATION
    end
end

#--------------------
# interaction!
#--------------------
@testset "interaction!" begin
    @testset "C vs. C" begin
        model = Model(Param(initial_N = 2, Δw = 0.2))
        model.strategy_vec[1] = C
        model.strategy_vec[2] = C
        interaction!(model)

        @test model.payoff_vec[1] == 2.0
        @test model.payoff_vec[2] == 2.0
        @test model.graph_weights[1, 2] == Float16(0.5 * 1.2 * 1.2)
        @test model.graph_weights == transpose(model.graph_weights)

        # check weight limit
        model = Model(Param(initial_N = 2, Δw = 0.9))
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
        model = Model(Param(initial_N = 2, Δw = 0.3))
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

        @test model.payoff_vec ≈ [2.0, 0.0, -0.1, 0.0, -0.2, 1.1, 1.0, 2.2, 0.9, 1.1]

        @test diag(model.graph_weights) == fill(0.0, 10)
        @test model.graph_weights == transpose(model.graph_weights)
        @test model.graph_weights[1, 2:10] == Float16[0.5, 0.5, 0.5, 0.5, 0.5, 0.55, 0.5, 0.55, 0.5]
        @test model.graph_weights[2, 3:10] == Float16[0.5, 0.5, 0.5, 0.45, 0.5, 0.5, 0.5, 0.45]
        @test model.graph_weights[3, 4:10] == Float16[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.45]
        @test model.graph_weights[4, 5:10] == Float16[0.5, 0.5, 0.5, 0.45, 0.5, 0.5]
        @test model.graph_weights[5, 6:10] == Float16[0.45, 0.5, 0.45, 0.5, 0.5]
        @test model.graph_weights[6, 7:10] == Float16[0.5, 0.5, 0.5, 0.45]
        @test model.graph_weights[7, 8:10] == Float16[0.5, 0.5, 0.5]
        @test model.graph_weights[8, 9:10] == Float16[0.45, 0.5]
        @test model.graph_weights[9, 10:10] == Float16[0.5]
    end

    @testset "Many agents (interaction_freqency = 0.3)" begin
        model = Model(Param(initial_N = 10, interaction_freqency = 0.3, rng = MersenneTwister(3)))
        model.strategy_vec = [C, D, C, D, D, D, C, C, D, D]
        interaction!(model)

        @test model.payoff_vec ≈ [-0.1, 0.0, 0.0, 0.0, 0.0, 1.1, 1.0, 1.0, 0.0, 0.0]

        expected_weights = fill(0.5, (10, 10))
        expected_weights -= Diagonal(expected_weights)
        expected_weights[1, 6] = expected_weights[6, 1] = 0.45  # C-D
        expected_weights[5, 6] = expected_weights[6, 5] = 0.45  # D-D
        expected_weights[7, 8] = expected_weights[8, 7] = 0.55  # C-C
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

@testset "pick_deaths and pick_parents" begin
    @testset "pick_deaths" begin
        model = Model(Param(initial_N = 10))
        model.death_N_vec[1] = 3
        @test length(Simulation.pick_deaths(model)) == 3
    end

    @testset "uniform distribution" begin
        model = Model(Param(initial_N = 100))

        # pick_deaths
        model.death_N_vec[1] = 11
        death_id_vec_vec = [Simulation.pick_deaths(model) for _ = 1:10_000]
        death_id_vec = vcat(death_id_vec_vec...)
        @test length(death_id_vec) == 11 * 10_000

        death_id_freq = fit(Histogram, death_id_vec, 1:101).weights
        @test mean(death_id_freq) == 1100
        @test std(death_id_freq) < 150

        # pick_parents
        model.birth_N_vec[1] = 12
        parent_id_vec_vec = [Simulation.pick_parents(model) for _ = 1:10_000]
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
        model.death_N_vec[1] = 11
        death_id_vec_vec = [Simulation.pick_deaths(model) for _ = 1:10_000]
        death_id_vec = vcat(death_id_vec_vec...)
        @test length(death_id_vec) == 11 * 10_000

        death_id_freq = fit(Histogram, death_id_vec, 1:101).weights
        @test mean(death_id_freq) == 1100
        @test death_id_freq[33] < 100
        @test death_id_freq[66] > 1800

        # pick_parents
        model.birth_N_vec[1] = 12
        parent_id_vec_vec = [Simulation.pick_parents(model) for _ = 1:10_000]
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
    Simulation.normalize_graph_weights!(model.graph_weights, model.N_vec[1], model.param.initial_graph_weight)

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
        model = Model(Param(initial_N = 100, reproduction_rate = 0.02, rng = MersenneTwister(1)))

        # before
        model.generation = 1
        model.death_N_vec[model.generation] = 2
        model.birth_N_vec[model.generation] = 3
        model.strategy_vec[[53 + 1, 78 + 2, 95 + 2]] .= C
        model.graph_weights = make_symmetric_matrix(100)

        # execution
        death_id_vec, parent_id_vec = death_and_birth!(model)
        @test death_id_vec == [16, 67]
        @test parent_id_vec == [53, 78, 95]

        # after
        @test model.N == length(model.strategy_vec) == length(model.payoff_vec) == 101
        @test size(model.graph_weights) == (101, 101)
        @test model.strategy_vec[99:101] == [C, C, C]
        @test model.graph_weights == transpose(model.graph_weights)  # check symmetry
        @test diag(model.graph_weights) == fill(0.0, 101)  # diagonal is 0.0

        model.graph_weights ./= Float16(0.9805)  # for normalization
        _index = setdiff(1:101, [53, 99])
        @test model.graph_weights[53, 99] == model.graph_weights[99, 53] == 1.0
        @test model.graph_weights[53, _index] == model.graph_weights[99, _index]

        _index = setdiff(1:101, [78, 100])
        @test model.graph_weights[78, 100] == model.graph_weights[100, 78] == 1.0
        @test model.graph_weights[78, _index] == model.graph_weights[100, _index]

        _index = setdiff(1:101, [95, 101])
        @test model.graph_weights[95, 101] == model.graph_weights[101, 95] == 1.0
        @test model.graph_weights[95, _index] == model.graph_weights[101, _index]

        # before
        model.generation = 2
        model.death_N_vec[model.generation] = 4
        model.birth_N_vec[model.generation] = 2
        model.strategy_vec[[55 + 1, 61 + 2]] .= D
        model.graph_weights = make_symmetric_matrix(101)

        # execution
        death_id_vec, parent_id_vec = death_and_birth!(model)
        @test death_id_vec == [24, 59, 79, 100]
        @test parent_id_vec == [55, 61]

        # after
        @test model.N == length(model.strategy_vec) == length(model.payoff_vec) == 99
        @test size(model.graph_weights) == (99, 99)
        @test model.strategy_vec[[98, 99]] == [D, D]
        @test model.graph_weights == transpose(model.graph_weights)  # is symmetry
        @test diag(model.graph_weights) == fill(0.0, 99)  # diagonal is 0.0

        model.graph_weights ./= Float16(1.008)  # for normalization
        _index = setdiff(1:98, [55, 98])
        @test model.graph_weights[55, 98] == model.graph_weights[98, 55] == 1.0
        @test model.graph_weights[55, _index] == model.graph_weights[98, _index]

        _index = setdiff(1:98, [61, 99])
        @test model.graph_weights[61, 99] == model.graph_weights[99, 61] == 1.0
        @test model.graph_weights[61, _index] == model.graph_weights[99, _index]
    end

    @testset "μ = 1.0" begin
        model = Model(Param(initial_N = 100, reproduction_rate = 0.03, μ = 1.0, rng = MersenneTwister(1)))

        # before
        model.generation = 1
        model.death_N_vec[model.generation] = 3
        model.birth_N_vec[model.generation] = 4
        model.strategy_vec[[22, 33, 40 + 1, 66 + 2]] = [C, D, C, D]
        model.graph_weights = make_symmetric_matrix(100)

        # execution
        death_id_vec, parent_id_vec = death_and_birth!(model)
        @test death_id_vec == [40, 53, 88]
        @test parent_id_vec == [22, 33, 40, 66]

        # after
        @test model.N == length(model.strategy_vec) == length(model.payoff_vec) == 101
        @test size(model.graph_weights) == (101, 101)
        @test model.strategy_vec[98:101] == [D, C, D, C]
        @test model.graph_weights == transpose(model.graph_weights)  # check symmetry
        @test diag(model.graph_weights) == fill(0.0, 101)  # diagonal is 0.0

        model.graph_weights ./= Float16(1.013)  # for normalization
        _index = [x for x in 1:100 if !(x in [22, 98])]
        @test model.graph_weights[22, 98] == model.graph_weights[98, 22] == 1.0
        @test model.graph_weights[22, _index] == model.graph_weights[98, _index]

        _index = [x for x in 1:100 if !(x in [33, 99])]
        @test model.graph_weights[33, 99] == model.graph_weights[99, 33] == 1.0
        @test model.graph_weights[33, _index] == model.graph_weights[99, _index]

        _index = [x for x in 1:101 if !(x in [40, 100])]
        @test model.graph_weights[40, 100] == model.graph_weights[100, 40] == 1.0
        @test model.graph_weights[40, _index] == model.graph_weights[100, _index]

        _index = [x for x in 1:101 if !(x in [66, 101])]
        @test model.graph_weights[66, 101] == model.graph_weights[101, 66] == 1.0
        @test model.graph_weights[66, _index] == model.graph_weights[101, _index]
    end
end

@testset "run_population" begin
    param = Param(
        initial_N = 100,
        initial_graph_weight = 0.2,
        δ = 1.0,
        β = 0.1,
        sigma = 10.0,
        μ = 0.123,
        generations = 100,
        rng = MersenneTwister(1),
    )

    df = run(param, log_level = 2)

    @test size(df) == (100, 48)

    # 1 〜 13
    @test df.initial_N == fill(100, 100)
    @test df.initial_T == fill(1.1, 100)
    @test df.S == fill(-0.1, 100)
    @test df.initial_graph_weight == fill(0.2, 100)
    @test df.interaction_freqency == fill(1.0, 100)
    @test df.Δw == fill(0.1, 100)
    @test df.reproduction_rate == fill(0.1, 100)
    @test df.δ == fill(1.0, 100)
    @test df.μ == fill(0.123, 100)
    @test df.β == fill(0.1, 100)
    @test df.sigma == fill(10.0, 100)
    @test df.generations == fill(100, 100)
    @test df.variability_mode == fill("POPULATION", 100)

    # 14 〜 18
    @test df.generation[1:50] == fill(0, 50)
    @test df.generation[51:100] == collect(51:100)
    @test df.N[1:50] == fill(0, 50)
    @test df.N[51:70] == [88, 111, 101, 108, 100, 96, 80, 92, 88, 118, 87, 93, 90, 85, 100, 94, 82, 101, 86, 94]
    @test df.N[71:90] == [95, 110, 113, 88, 83, 94, 112, 116, 123, 93, 111, 96, 97, 115, 109, 99, 95, 102, 104, 104]
    @test df.N[91:100] == [103, 105, 103, 115, 114, 100, 102, 92, 82, 113]
    @test df.T[1:50] == fill(0.0, 50)
    @test df.T[51:100] == fill(Float16(1.1), 50)
    @test df.cooperation_rate[90:100] ==
          Float16[0.519, 0.5435, 0.5522, 0.534, 0.5654, 0.588, 0.53, 0.539, 0.5435, 0.5244, 0.5664]
    @test df.payoff_μ[90:100] == Float16[1.011, 1.089, 1.009, 1.074, 0.9233, 1.101, 1.202, 1.091, 1.044, 1.131, 0.695]

    # 19 〜 23
    @test df.weak_k1[60:10:100] == Float16[29.75, 27.19, 31.0, 37.6, 46.4]
    @test df.weak_C1[60:10:100] == Float16[1.0, 1.0, 1.0, 0.9907, 0.9634]
    @test df.weak_comp1_count[60:10:100] == Float16[2.0, 2.0, 1.0, 2.0, 2.0]
    @test df.weak_comp1_size_μ[60:10:100] == Float16[0.1737, 0.25, 0.344, 0.3655, 0.4204]
    @test df.weak_comp1_size_max[60:10:100] == Float16[0.2966, 0.3618, 0.344, 0.423, 0.54]

    # 44 〜 48
    @test df.strong_k2[60:10:100] == Float16[34.0, 33.0, 13.664, 0.0, 0.0]
    @test df.strong_C2[60:10:100] == Float16[1.0, 1.0, 0.794, 0.0, 0.0]
    @test df.strong_comp2_count[60:10:100] == Float16[1.0, 1.0, 1.0, 0.0, 0.0]
    @test df.strong_comp2_size_μ[60:10:100] == Float16[0.2966, 0.3618, 0.258, 0.0, 0.0]
    @test df.strong_comp2_size_max[60:10:100] == Float16[0.2966, 0.3618, 0.258, 0.0, 0.0]
end
