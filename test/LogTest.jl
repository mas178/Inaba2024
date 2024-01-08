using Random: MersenneTwister
using StatsBase: mean, std

using Graphs
using SimpleWeightedGraphs

using Test: @testset, @test

include("../src/Simulation.jl")
include("../src/Network.jl")
using .Simulation: Param, Model, C, D, MUTATION, make_output_df, log!
using .Network: create_adjacency_matrix, rem_edge!, neighbors

@testset "log!" begin
    log_level = 2
    log_rate = 1.0
    log_skip = 1

    @testset "1:17" begin
        param = Param(
            initial_N = 10,
            initial_k = 4,
            initial_T = 1.1,
            S = 0.2,
            initial_w = 0.43,
            Δw = 0.2,
            interaction_freqency = 0.6,
            reproduction_rate = 0.15,
            δ = 0.9,
            initial_μ_s = 0.123,
            initial_μ_c = 0.234,
            β = 0.156,
            sigma = 10,
            generations = 11,
            variability_mode = MUTATION,
        )

        start_gen = floor(Int, param.generations * (1 - log_rate)) + 1
        log_generations = filter(x -> x % log_skip == 0, start_gen:(param.generations))
        log_row_n = 1

        output = make_output_df(param, log_level, length(log_generations))
        model = Model(param)

        for i = 1:11
            model.generation = i
            log!(output, model, log_level, log_row_n)
            log_row_n += 1
        end

        @test size(output) == (11, 50)
        @test output[:, 1] == fill(10, 11)
        @test output[:, 2] == fill(4, 11)
        @test output[:, 3] == fill(1.1, 11)
        @test output[:, 4] == fill(0.2, 11)
        @test output[:, 5] == fill(Float16(0.43), 11)
        @test output[:, 6] == fill(0.2, 11)
        @test output[:, 7] == fill(0.6, 11)
        @test output[:, 8] == fill(0.15, 11)
        @test output[:, 9] == fill(0.9, 11)
        @test output[:, 10] == fill(0.123, 11)
        @test output[:, 11] == fill(0.234, 11)
        @test output[:, 12] == fill(0.156, 11)
        @test output[:, 13] == fill(10, 11)
        @test output[:, 14] == fill(11, 11)
        @test output[:, 15] == fill("MUTATION", 11)
    end

    @testset "16:20" begin
        param = Param()
        start_gen = floor(Int, param.generations * (1 - log_rate)) + 1
        log_generations = filter(x -> x % log_skip == 0, start_gen:(param.generations))
        log_row_n = 1
        output = make_output_df(param, log_level, length(log_generations))
        model = Model(param)
        model.payoff_table_vec = [Dict((D, C) => (i / 10, 1.0)) for i = 1:100]

        for i = 1:100
            model.generation = i
            model.weights = create_adjacency_matrix(i * 10, 2, Float16(0.2))
            model.strategy_vec = [x <= i ? C : D for x = 1:1000]
            model.payoff_vec = [x <= i ? 0 : 1 for x = 1:1000]

            log!(output, model, log_level, log_row_n)
            log_row_n += 1
        end

        @test output[:, 16] == output[:, :generation] == collect(1:100)
        @test output[:, 17] == output[:, :N] == collect(1:100) .* 10
        @test output[:, 18] == output[:, :T] == Float16.(collect(1:100) ./ 10)
        @test output[:, 19] == output[:, :cooperation_rate] == Float16.(collect(1:100) ./ 1000)
        @test output[:, 20] == output[:, :payoff_μ] == Float16.(collect(999:-1:900) ./ 1000)
    end

    @testset "21:35" begin
        param = Param()
        output = make_output_df(param, log_level, 1)
        model = Model(param)

        @testset "全体が1つのコンポーネントである場合" begin
            model.strategy_vec = [fill(C, 5)..., fill(D, 5)...]

            for (strength, threashold) in [("weak", 0.26), ("medium", 0.51), ("strong", 0.76)]
                model.weights = create_adjacency_matrix(10, 4, Float16(threashold))

                log!(output, model, log_level, 1)

                @test output[1, "$(strength)_k1"] == 4
                @test output[1, "$(strength)_C1"] == 0.5
                @test output[1, "$(strength)_comp1_count"] == 1.0
                @test output[1, "$(strength)_comp1_size_μ"] == 1.0
                @test output[1, "$(strength)_comp1_size_max"] == 1.0
            end
        end

        @testset "繋がりが閾値以下の場合" begin
            model.strategy_vec = [fill(C, 5)..., fill(D, 5)...]

            for (strength, threashold) in [("weak", 0.25), ("medium", 0.5), ("strong", 0.75)]
                model.weights = create_adjacency_matrix(10, 4, Float16(threashold))

                log!(output, model, log_level, 1)

                @test output[1, "$(strength)_k1"] == 0
                @test output[1, "$(strength)_C1"] == 0
                @test output[1, "$(strength)_comp1_count"] == 0
                @test output[1, "$(strength)_comp1_size_μ"] == 0
                @test output[1, "$(strength)_comp1_size_max"] == 0
            end
        end

        @testset "協力者が50%未満の場合" begin
            model.strategy_vec = [fill(C, 4)..., fill(D, 6)...]

            for (strength, threashold) in [("weak", 0.26), ("medium", 0.51), ("strong", 0.76)]
                model.weights = create_adjacency_matrix(10, 4, Float16(threashold))

                log!(output, model, log_level, 1)

                @test output[1, "$(strength)_k1"] == 0
                @test output[1, "$(strength)_C1"] == 0
                @test output[1, "$(strength)_comp1_count"] == 0
                @test output[1, "$(strength)_comp1_size_μ"] == 0
                @test output[1, "$(strength)_comp1_size_max"] == 0
            end
        end

        @testset "3つのコンポーネントが存在する場合" begin
            model.strategy_vec .= C

            for (strength, threashold) in [("weak", 0.26), ("medium", 0.51), ("strong", 0.76)]
                model.weights = create_adjacency_matrix(10, 4, Float16(threashold))

                # {1, 2, 3} を切り離す
                rem_edge!(model.weights, 1, 9)
                rem_edge!(model.weights, 1, 10)
                rem_edge!(model.weights, 2, 10)
                rem_edge!(model.weights, 2, 4)
                rem_edge!(model.weights, 3, 4)
                rem_edge!(model.weights, 3, 5)

                # {7} を切り離す
                neighbord_vec = copy(neighbors(model.weights, 7))
                for neightbor_id in neighbord_vec
                    rem_edge!(model.weights, 7, neightbor_id)
                end

                log!(output, model, log_level, 1)

                @test output[1, "$(strength)_k1"] == Float16(2.223)
                @test output[1, "$(strength)_C1"] == Float16(0.852)
                @test output[1, "$(strength)_comp1_count"] == Float16(2.0)
                @test output[1, "$(strength)_comp1_size_μ"] == Float16(0.45)
                @test output[1, "$(strength)_comp1_size_max"] == Float16(0.6)
            end
        end
    end

    @testset "36:50" begin
        param = Param()
        output = make_output_df(param, log_level, 1)
        model = Model(param)

        @testset "全体が1つのコンポーネントである場合" begin
            model.strategy_vec = [fill(C, 5)..., fill(D, 5)...]

            for (strength, threashold) in [("weak", 0.26), ("medium", 0.51), ("strong", 0.76)]
                model.weights = create_adjacency_matrix(10, 4, Float16(threashold))

                log!(output, model, log_level, 1)

                @test output[1, "$(strength)_k2"] == 2
                @test output[1, "$(strength)_C2"] == 0
                @test output[1, "$(strength)_comp2_count"] == 1.0
                @test output[1, "$(strength)_comp2_size_μ"] == 1.0
                @test output[1, "$(strength)_comp2_size_max"] == 1.0
            end

            for (strength, threashold) in [("weak", 0.31), ("medium", 0.71)]
                model.weights = create_adjacency_matrix(10, 4, Float16(threashold))

                log!(output, model, log_level, 1)

                @test output[1, "$(strength)_k2"] == 4
                @test output[1, "$(strength)_C2"] == 0.5
                @test output[1, "$(strength)_comp2_count"] == 1.0
                @test output[1, "$(strength)_comp2_size_μ"] == 1.0
                @test output[1, "$(strength)_comp2_size_max"] == 1.0
            end
        end
    end
end
