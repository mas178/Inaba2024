using DataFrames: Not
using LinearAlgebra: Diagonal, diag
using Random
using StatsBase

using Graphs

using Test: @testset, @test

include("../src/Simulation.jl")
using .Simulation: Model, Param, C, D, pick_deaths, pick_parents, death!, birth!, VARIABILITY_MODE
include("../src/Network.jl")
using .Network: nv, update_weight!, rem_edge!, neighbors

@testset "invert" begin
    @test Simulation.invert(C) == D
    @test Simulation.invert(D) == C
end

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
    N = 100
    death_N = 10
    trial = 10_000

    @testset "uniform distribution" begin
        for mode in keys(VARIABILITY_MODE)
            model = Model(Param(initial_N = N, variability_mode = mode))
            model.generation = 11
            model.death_N_vec[model.generation] = death_N
            model.payoff_vec = fill(0.1, N)

            death_id_freq = fill(0, N)
            parent_id_freq = fill(0, N)
            for _ = 1:trial
                # pick_deaths
                for death_id in pick_deaths(model, model.param.rng)
                    death_id_freq[death_id] += 1
                end
                # pick_parents
                for parent_id in pick_parents(model, model.param.rng)
                    parent_id_freq[parent_id] += 1
                end
            end

            # pick_deaths
            @test mean(death_id_freq) == N * death_N == 1_000
            @test std(death_id_freq) ≈ 30 atol = 10

            # pick_parents
            @test mean(parent_id_freq) == N * death_N == 1_000
            @test std(parent_id_freq) ≈ 30 atol = 10
        end
    end

    @testset "Non-uniform distribution (δ = 1.0)" begin
        for mode in keys(VARIABILITY_MODE)
            model = Model(Param(initial_N = N, δ = 1.0, variability_mode = mode))
            model.generation = 21
            model.death_N_vec[model.generation] = death_N
            model.payoff_vec = fill(0.0, N)
            model.payoff_vec[1:10:N] = fill(-2.0, Int(N / 10))
            model.payoff_vec[2:10:N] = fill(-1.0, Int(N / 10))
            model.payoff_vec[4:10:N] = fill(1.0, Int(N / 10))
            model.payoff_vec[5:10:N] = fill(2.0, Int(N / 10))

            death_id_freq = fill(0, N)
            parent_id_freq = fill(0, N)
            for _ = 1:trial
                # pick_deaths
                for death_id in pick_deaths(model, model.param.rng)
                    death_id_freq[death_id] += 1
                end
                # pick_parents
                for parent_id in pick_parents(model, model.param.rng)
                    parent_id_freq[parent_id] += 1
                end
            end

            # pick_deaths
            @test mean(death_id_freq) == N * death_N == 1_000
            @test std(death_id_freq) ≈ 384 atol = 30
            ## Times the agents with -2.0 payoff are picked.
            @test mean(death_id_freq[1:10:N]) ≈ 1720 atol = 50
            ## Times the agents with -1.0 payoff are picked.
            @test mean(death_id_freq[2:10:N]) ≈ 1438 atol = 50
            ## Times the agents with 0.0 payoff are picked.
            @test mean(death_id_freq[3:10:N]) ≈ 1000 atol = 50
            ## Times the agents with 1.0 payoff are picked.
            @test mean(death_id_freq[4:10:N]) ≈ 563 atol = 50
            ## Times the agents with 2.0 payoff are picked.
            @test mean(death_id_freq[5:10:N]) ≈ 249 atol = 50

            # pick_parents
            @test mean(parent_id_freq) == N * death_N == 1_000
            @test std(parent_id_freq) ≈ 384 atol = 30
            ## Times the agents with -2.0 payoff are picked.
            @test mean(parent_id_freq[5:10:N]) ≈ 1720 atol = 50
            ## Times the agents with -1.0 payoff are picked.
            @test mean(parent_id_freq[4:10:N]) ≈ 1438 atol = 50
            ## Times the agents with 0.0 payoff are picked.
            @test mean(parent_id_freq[3:10:N]) ≈ 1000 atol = 50
            ## Times the agents with 1.0 payoff are picked.
            @test mean(parent_id_freq[2:10:N]) ≈ 563 atol = 50
            ## Times the agents with 2.0 payoff are picked.
            @test mean(parent_id_freq[1:10:N]) ≈ 249 atol = 50
        end
    end

    @testset "Non-uniform distribution (δ = 0.1)" begin
        for mode in keys(VARIABILITY_MODE)
            model = Model(Param(initial_N = N, δ = 0.1, variability_mode = mode))
            model.generation = 21
            model.death_N_vec[model.generation] = death_N
            model.payoff_vec = fill(0.0, N)
            model.payoff_vec[1:10:N] = fill(-2.0, Int(N / 10))
            model.payoff_vec[2:10:N] = fill(-1.0, Int(N / 10))
            model.payoff_vec[4:10:N] = fill(1.0, Int(N / 10))
            model.payoff_vec[5:10:N] = fill(2.0, Int(N / 10))

            death_id_freq = fill(0, N)
            parent_id_freq = fill(0, N)
            for _ = 1:trial
                # pick_deaths
                for death_id in pick_deaths(model, model.param.rng)
                    death_id_freq[death_id] += 1
                end
                # pick_parents
                for parent_id in pick_parents(model, model.param.rng)
                    parent_id_freq[parent_id] += 1
                end
            end

            # pick_deaths
            @test mean(death_id_freq) == N * death_N == 1_000
            @test std(death_id_freq) ≈ 54 atol = 10
            ## Times the agents with -2.0 payoff are picked.
            @test mean(death_id_freq[1:10:N]) ≈ 1100 atol = 50
            ## Times the agents with -1.0 payoff are picked.
            @test mean(death_id_freq[2:10:N]) ≈ 1050 atol = 50
            ## Times the agents with 0.0 payoff are picked.
            @test mean(death_id_freq[3:10:N]) ≈ 1000 atol = 50
            ## Times the agents with 1.0 payoff are picked.
            @test mean(death_id_freq[4:10:N]) ≈ 950 atol = 50
            ## Times the agents with 2.0 payoff are picked.
            @test mean(death_id_freq[5:10:N]) ≈ 900 atol = 50

            # pick_parents
            @test mean(parent_id_freq) == N * death_N == 1_000
            @test std(parent_id_freq) ≈ 54 atol = 10
            ## Times the agents with -2.0 payoff are picked.
            @test mean(parent_id_freq[5:10:N]) ≈ 1100 atol = 50
            ## Times the agents with -1.0 payoff are picked.
            @test mean(parent_id_freq[4:10:N]) ≈ 1050 atol = 50
            ## Times the agents with 0.0 payoff are picked.
            @test mean(parent_id_freq[3:10:N]) ≈ 1000 atol = 50
            ## Times the agents with 1.0 payoff are picked.
            @test mean(parent_id_freq[2:10:N]) ≈ 950 atol = 50
            ## Times the agents with 2.0 payoff are picked.
            @test mean(parent_id_freq[1:10:N]) ≈ 900 atol = 50
        end
    end

    @testset "Non-uniform distribution (δ = 0.0)" begin
        for mode in keys(VARIABILITY_MODE)
            model = Model(Param(initial_N = N, δ = 0.0, variability_mode = mode))
            model.generation = 21
            model.death_N_vec[model.generation] = death_N
            model.payoff_vec = fill(0.0, N)
            model.payoff_vec[1:10:N] = fill(-2.0, Int(N / 10))
            model.payoff_vec[2:10:N] = fill(-1.0, Int(N / 10))
            model.payoff_vec[4:10:N] = fill(1.0, Int(N / 10))
            model.payoff_vec[5:10:N] = fill(2.0, Int(N / 10))

            death_id_freq = fill(0, N)
            parent_id_freq = fill(0, N)
            for _ = 1:trial
                # pick_deaths
                for death_id in pick_deaths(model, model.param.rng)
                    death_id_freq[death_id] += 1
                end
                # pick_parents
                for parent_id in pick_parents(model, model.param.rng)
                    parent_id_freq[parent_id] += 1
                end
            end

            # pick_deaths
            @test mean(death_id_freq) == N * death_N == 1_000
            @test std(death_id_freq) ≈ 30 atol = 10
            ## Times the agents with -2.0 payoff are picked.
            @test mean(death_id_freq[1:10:N]) ≈ 1_000 atol = 50
            ## Times the agents with -1.0 payoff are picked.
            @test mean(death_id_freq[2:10:N]) ≈ 1_000 atol = 50
            ## Times the agents with 0.0 payoff are picked.
            @test mean(death_id_freq[3:10:N]) ≈ 1_000 atol = 50
            ## Times the agents with 1.0 payoff are picked.
            @test mean(death_id_freq[4:10:N]) ≈ 1_000 atol = 50
            ## Times the agents with 2.0 payoff are picked.
            @test mean(death_id_freq[5:10:N]) ≈ 1_000 atol = 50

            # pick_parents
            @test mean(parent_id_freq) == N * death_N == 1_000
            @test std(parent_id_freq) ≈ 30 atol = 10
            ## Times the agents with -2.0 payoff are picked.
            @test mean(parent_id_freq[5:10:N]) ≈ 1_000 atol = 50
            ## Times the agents with -1.0 payoff are picked.
            @test mean(parent_id_freq[4:10:N]) ≈ 1_000 atol = 50
            ## Times the agents with 0.0 payoff are picked.
            @test mean(parent_id_freq[3:10:N]) ≈ 1_000 atol = 50
            ## Times the agents with 1.0 payoff are picked.
            @test mean(parent_id_freq[2:10:N]) ≈ 1_000 atol = 50
            ## Times the agents with 2.0 payoff are picked.
            @test mean(parent_id_freq[1:10:N]) ≈ 1_000 atol = 50
        end
    end
end

@testset "death!" begin
    for mode in keys(VARIABILITY_MODE)
        model = Model(Param(initial_N = 100, initial_k = 10, variability_mode = mode))

        @testset "1回目" begin
            # before
            model.generation = 1
            model.death_N_vec[model.generation] = 2
            node_id = 11
            for i = 1:11
                if i < 6
                    update_weight!(model.weights, node_id, node_id + i - 6, Float16(i / 10))
                elseif i > 6
                    update_weight!(model.weights, node_id, node_id + i - 6, Float16((i - 1) / 10))
                end
            end
            neighbor_vec = neighbors(model.weights, node_id)
            @test neighbor_vec == [6, 7, 8, 9, 10, 12, 13, 14, 15, 16]
            @test model.weights[node_id, neighbor_vec] == Float16[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

            # execution
            death_id_vec = death!(model, MersenneTwister(1))

            # after
            @test death_id_vec == [9, 10]
            @test nv(model.weights) == length(model.strategy_vec) == length(model.payoff_vec) == 98
            @test model.weights == transpose(model.weights)  # check symmetry
            @test diag(model.weights) == fill(Float16(0.0), 98)  # diagonal is 0.0

            ## グラフ上のIDが正しくズレていることを確認
            node_id = 9
            neighbor_vec = neighbors(model.weights, node_id)
            @test neighbor_vec == [6, 7, 8, 10, 11, 12, 13, 14]
            @test model.weights[node_id, neighbor_vec] == Float16[0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 1.0]
        end

        @testset "2回目" begin
            # before
            model.generation = 21
            model.death_N_vec[model.generation] = 4
            node_id = 90
            for i = 1:11
                if i < 6
                    update_weight!(model.weights, node_id, node_id + i - 6, Float16(i / 10))
                elseif i > 6
                    update_weight!(model.weights, node_id, node_id + i - 6, Float16((i - 1) / 10))
                end
            end
            neighbor_vec = neighbors(model.weights, node_id)
            @test neighbor_vec == [85, 86, 87, 88, 89, 91, 92, 93, 94, 95]
            @test model.weights[node_id, neighbor_vec] == Float16[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

            # execution
            death_id_vec = death!(model, MersenneTwister(2))

            # after
            @test death_id_vec == [29, 56, 76, 87]
            @test nv(model.weights) == length(model.strategy_vec) == length(model.payoff_vec) == 94
            @test model.weights == transpose(model.weights)  # is symmetry
            @test diag(model.weights) == fill(0.0, 94)  # diagonal is 0.0

            ## グラフ上のIDが正しくズレていることを確認
            neighbor_vec = neighbors(model.weights, node_id)
            @test neighbor_vec == [1, 85, 86, 87, 88, 89, 91, 92, 93, 94]
            @test model.weights[node_id, neighbor_vec] == Float16[0.5, 0.5, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        end

        @testset "3回目: 孤立したノードが適切に再接続されていることを確認" begin
            # before
            model.generation = 41
            model.death_N_vec[model.generation] = 4

            # death!実行時に11が死ぬことを見越して、事前にそれ以外のノードとのコネクションを切っておく
            node_id = 10
            @test neighbors(model.weights, node_id) == [7, 8, 9, 11, 12, 13, 14, 15]
            neighbor_vec = copy(neighbors(model.weights, node_id))
            for neighbor in neighbor_vec
                if neighbor != 11
                    rem_edge!(model.weights, node_id, neighbor)
                end
            end
            @test neighbors(model.weights, node_id) == [11]

            # execution
            death_id_vec = death!(model, MersenneTwister(3))

            # after
            @test death_id_vec == [11, 27, 51, 77]
            @test nv(model.weights) == length(model.strategy_vec) == length(model.payoff_vec) == 90
            @test model.weights == transpose(model.weights)  # is symmetry
            @test diag(model.weights) == fill(0.0, 90)  # diagonal is 0.0

            ## 孤立したノードが適切に再接続されていることを確認
            @test neighbors(model.weights, node_id) == [38]
            @test model.weights[node_id, 38] == Float16(1.0)
        end
    end
end

@testset "birth!" begin
    for mode in keys(VARIABILITY_MODE)
        model = Model(Param(initial_N = 100, initial_k = 10, variability_mode = mode))
        for node_id = 1:100
            for neighbor_id in neighbors(model.weights, node_id)
                if node_id < neighbor_id
                    update_weight!(model.weights, node_id, neighbor_id, Float16(node_id * neighbor_id / 10_000))
                end
            end
        end

        @testset "μ_c = 0.0, μ_s = 0.0" begin
            # before
            model.generation = 1
            model.birth_N_vec[model.generation] = 3
            model.μ_c_vec[model.generation] = 0.0
            model.μ_s_vec[model.generation] = 0.0
            model.strategy_vec[[6, 61, 75]] .= C

            # execution
            parent_id_vec = birth!(model, MersenneTwister(1))
            @test parent_id_vec == [6, 61, 75]

            # after
            @test nv(model.weights) == length(model.strategy_vec) == length(model.payoff_vec) == 103
            @test model.strategy_vec[101:103] == [C, C, C]
            @test model.weights == transpose(model.weights)  # check symmetry
            @test diag(model.weights) == fill(0.0, 103)  # diagonal is 0.0

            _index = setdiff(1:103, [6, 101])
            @test model.weights[6, 101] == Float16(1.0)
            @test model.weights[6, _index] == model.weights[101, _index]

            _index = setdiff(1:103, [61, 102])
            @test model.weights[61, 102] == Float16(1.0)
            @test model.weights[61, _index] == model.weights[102, _index]

            _index = setdiff(1:103, [75, 103])
            @test model.weights[75, 103] == Float16(1.0)
            @test model.weights[75, _index] == model.weights[103, _index]
        end

        @testset "μ_c = 0.5, μ_s = 0.5" begin
            # before
            model.generation = 10
            model.birth_N_vec[model.generation] = 3
            model.μ_s_vec[model.generation] = 0.5
            model.μ_c_vec[model.generation] = 0.5
            model.strategy_vec[[6, 61, 75]] .= C

            # execution
            parent_id_vec = birth!(model, MersenneTwister(1))
            @test parent_id_vec == [6, 61, 75]

            # after
            @test nv(model.weights) == length(model.strategy_vec) == length(model.payoff_vec) == 106
            @test model.strategy_vec[104:106] == [D, C, D]
            @test model.weights == transpose(model.weights)  # check symmetry
            @test diag(model.weights) == fill(0.0, 106)  # diagonal is 0.0

            @test model.weights[6, 104] == Float16(1.0)
            @test neighbors(model.weights, 6) == [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 101, 104]
            @test neighbors(model.weights, 104) == [1, 5, 6, 8, 29, 32, 58, 81, 82, 86, 89, 101]
            @test Set(model.weights[6, :]) == Set(model.weights[104, :])

            @test model.weights[61, 105] == Float16(1.0)
            @test neighbors(model.weights, 61) == [56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 102, 105]
            @test neighbors(model.weights, 105) == [11, 23, 25, 57, 58, 61, 62, 63, 64, 66, 93, 94]
            @test Set(model.weights[61, :]) == Set(model.weights[105, :])

            @test model.weights[75, 106] == Float16(1.0)
            @test neighbors(model.weights, 75) == [70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 103, 106]
            @test neighbors(model.weights, 106) == [66, 71, 72, 74, 75, 78, 79, 80, 88, 91, 101, 103]
            @test Set(model.weights[75, :]) == Set(model.weights[106, :])
        end
    end
end
