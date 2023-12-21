using Graphs
using Test: @testset, @test
using Random: MersenneTwister
using StatsBase: mean, std

include("../src/Simulation.jl")
using .Simulation: Param, Model, C, D, make_output_df, unweighted_graph, convert_to_2nd_order_weights, log!, run

@testset "unweighted_graph" begin
    graph_weights = [
        0.00 0.50 0.51
        0.50 0.00 0.51
        0.51 0.51 0.00
    ]

    g = unweighted_graph(graph_weights, 0.5)
    @test nv(g) == 3
    @test ne(g) == 2
    @test has_edge(g, 1, 3)
    @test has_edge(g, 2, 3)

    g = unweighted_graph(graph_weights, 0.51)
    @test nv(g) == 3
    @test ne(g) == 0

    g = unweighted_graph(graph_weights, 0.49)
    @test nv(g) == 3
    @test ne(g) == 3
end

@testset "convert_to_2nd_order_weights" begin
    param = Param(initial_N = 4, initial_graph_weight = 0.35)
    model = Model(param)
    model.graph_weights = Float16[
        0.0 0.1 0.2 0.3
        0.1 0.0 0.4 0.5
        0.2 0.4 0.0 0.6
        0.3 0.5 0.6 0.0
    ]

    weights2 = convert_to_2nd_order_weights(model)

    factor = Float16(7.137) / Float16(4.2)

    @test weights2[1, 2] == weights2[2, 1] == Float16(0.1 + 0.2 * 0.4 + 0.3 * 0.5) / factor == Float16(0.1942)
    @test weights2[1, 3] == weights2[3, 1] == Float16(0.2 + 0.1 * 0.4 + 0.3 * 0.6) / factor == Float16(0.2471)
    @test weights2[1, 4] == weights2[4, 1] == Float16(0.3 + 0.1 * 0.5 + 0.2 * 0.6) / factor == Float16(0.2766)
    @test weights2[2, 3] ==
          weights2[3, 2] ==
          (Float16(0.4) + Float16(0.2 * 0.1) + Float16(0.6 * 0.5)) / factor ==
          Float16(0.4236)
    @test weights2[2, 4] == weights2[4, 2] == Float16(0.5 + 0.3 * 0.1 + 0.4 * 0.6) / factor == Float16(0.4531)
    @test weights2[3, 4] == weights2[4, 3] == Float16(0.6 + 0.2 * 0.3 + 0.4 * 0.5) / Float16(factor) == Float16(0.506)

    _weights = Float16.(rand(1000, 1000))
    param = Param(initial_N = 1000, initial_graph_weight = sum(Float64.(Float16.(rand(1000, 1000)))) / (1000 * 999))
    model = Model(param)
    model.graph_weights = _weights
    weights2 = convert_to_2nd_order_weights(model)
    @test all([0.48 < weights2[x, y] < 0.52 for x in 1000, y in 1000 if x != y])
    @test all([weights2[x, y] == 0.0 for x in 1000, y in 1000 if x == y])
    @test 0.48 < mean(Float64.(weights2)) < 0.52
    @test std(Float64.(weights2)) < 0.03
end

@testset "log!" begin
    log_level = 2
    skip = 1

    param = Param(initial_N = 10, initial_graph_weight = 1.0, generations = 11)
    output = make_output_df(param)
    model = Model(param)
    model.generation = 1
    model.N_vec[model.generation + 1] = 10

    log!(output, model, log_level, skip)

    # default
    @test size(output) == (11, 48)

    # all row, 1 〜 13 column
    for i = 1:11
        @test Vector(output[i, 1:13]) == [10, 1.1, -0.1, 1.0, 1.0, 0.1, 0.1, 0.01, 0.0, 0.1, 0.1, 11, "POPULATION"]
    end

    @test output[model.generation, 15] == model.N_vec[model.generation + 1]
    @test Vector(output[model.generation, 17:18]) == Float16[0.0, 0.0]
    @test Vector(output[model.generation, 19:48]) == fill(Float16(0.0), 30)

    # cooperation rate and average payoff
    model.generation = 2
    model.N_vec[model.generation + 1] = 10
    model.strategy_vec = repeat([C, D, D, D, D], 2)
    model.payoff_vec = 0.1:0.1:1.0
    log!(output, model, log_level, skip)
    @test output[model.generation, 15] == model.N_vec[model.generation + 1]
    @test Vector(output[model.generation, 17:18]) == Float16[0.2, 0.55]
    @test Vector(output[model.generation, 19:48]) == fill(Float16(0.0), 30)

    # average and std of weights
    model = Model(Param(initial_N = 3))
    model.generation = 3
    model.N_vec[model.generation + 1] = 3
    model.graph_weights = [0.0 1.0 2.0; 1.0 0.0 6.0; 2.0 6.0 0.0]
    log!(output, model, log_level, skip)
    @test output[model.generation, 15] == model.N_vec[model.generation + 1]
    @test Vector(output[model.generation, 17:18]) == [0.0, 0.0]

    # k1, C1, k2, C2
    ## regular graph
    N = 1_000
    model = Model(Param(initial_N = N))
    model.generation = 4
    model.N_vec[model.generation + 1] = N
    g = random_regular_graph(N, 8)
    model.graph_weights = Matrix(adjacency_matrix(g))
    model.strategy_vec = fill(C, N)
    log!(output, model, log_level, skip)
    @test output[model.generation, 15] == model.N_vec[model.generation + 1]
    @test Vector(output[model.generation, 19:20]) ≈ Float16[8.0, 0.005894]
    @test Vector(output[model.generation, 24:25]) ≈ Float16[8.0, 0.005894]
    @test Vector(output[model.generation, 29:30]) ≈ Float16[8.0, 0.005894]
    @test Vector(output[model.generation, 34:35]) ≈ Float16[62.62, 0.1716]
    @test Vector(output[model.generation, 39:40]) ≈ Float16[62.62, 0.1716]
    @test Vector(output[model.generation, 44:45]) ≈ Float16[62.62, 0.1716]

    ## scale free network (BA model)
    model.generation = 5
    model.N_vec[model.generation + 1] = N
    g = barabasi_albert(N, 2, seed = 1)
    model.graph_weights = Matrix(adjacency_matrix(g))
    log!(output, model, log_level, skip)
    @test output[model.generation, 15] == model.N_vec[model.generation + 1]
    @test Vector(output[model.generation, 19:20]) ≈ Float16[3.992, 0.02054]
    @test Vector(output[model.generation, 24:25]) ≈ Float16[3.992, 0.02054]
    @test Vector(output[model.generation, 29:30]) ≈ Float16[3.992, 0.02054]
    @test Vector(output[model.generation, 34:35]) ≈ Float16[43.44, 0.577]
    @test Vector(output[model.generation, 39:40]) ≈ Float16[43.44, 0.577]
    @test Vector(output[model.generation, 44:45]) ≈ Float16[43.44, 0.577]

    # divided graph
    N = 10
    model = Model(Param(initial_N = N))
    model.generation = 6
    model.N_vec[model.generation + 1] = N
    model.strategy_vec = fill(C, N)
    g = SimpleGraph(N)
    add_edge!(g, (1, 2))
    add_edge!(g, (1, 3))
    add_edge!(g, (1, 4))
    add_edge!(g, (2, 3))
    add_edge!(g, (3, 4))
    add_edge!(g, (5, 6))
    add_edge!(g, (6, 7))
    add_edge!(g, (8, 9))
    model.graph_weights = Matrix(adjacency_matrix(g))
    model.graph_weights[1, 2] = 0.6
    model.graph_weights[2, 1] = 0.6
    model.graph_weights[1, 4] = 0.4
    model.graph_weights[4, 1] = 0.4

    log!(output, model, log_level, skip)
    @test output[model.generation, 15] == N

    @test Vector(output[model.generation, 19:20]) ≈ Float16[2.0, 0.476]
    @test Vector(output[model.generation, 24:25]) ≈ Float16[1.714, 0.3333]
    @test Vector(output[model.generation, 29:30]) ≈ Float16[1.429, 0.0]
    @test Vector(output[model.generation, 34:35]) ≈ Float16[2.572, 1.0]
    @test Vector(output[model.generation, 39:40]) ≈ Float16[2.572, 1.0]
    @test Vector(output[model.generation, 44:45]) ≈ Float16[2.572, 1.0]

    # component (コンポーネント数, コンポーネントサイズの平均, 最大)
    @test Vector(output[model.generation, 21:23]) == Float16[2.0, mean([4, 3]) / 10, 4.0 / 10]
    @test Vector(output[model.generation, 26:28]) == Float16[2.0, mean([4, 3]) / 10, 4.0 / 10]
    @test Vector(output[model.generation, 31:33]) == Float16[2.0, mean([4, 3]) / 10, 4.0 / 10]
    @test Vector(output[model.generation, 36:38]) == Float16[2.0, mean([4, 3]) / 10, 4.0 / 10]

    for i = 1:6
        @test output[i, 14] == i
        @test output[i, 16] == Float16(1.1)
    end
end

@testset "log! log_level and log_rate" begin
    param = Param(initial_graph_weight = 0.2, δ = 1.0, μ = 0.123, rng = MersenneTwister(1))
    param_vec = [[getfield(param, f) for f in fieldnames(Param)[1:12]]..., "POPULATION"]

    output = run(param, log_level = 0, log_rate = 1.0)
    for i = 1:(param.generations)
        @test Vector(output[i, 1:13]) == param_vec
        @test Vector(output[i, 14:18]) ≠ fill(Float16(0.0), 15)
        @test Vector(output[i, 19:33]) == fill(Float16(0.0), 15)
        @test Vector(output[i, 34:48]) == fill(Float16(0.0), 15)
    end

    output = run(param, log_level = 1, log_rate = 0.7)
    for i = 1:(param.generations)
        @test Vector(output[i, 1:13]) == param_vec
        if i <= 30
            @test Vector(output[i, 14:48]) == fill(Float16(0.0), 35)
        else
            @test Vector(output[i, 14:18]) ≠ fill(Float16(0.0), 5)
            if i % 10 == 0
                @test Vector(output[i, 19:33]) ≠ fill(Float16(0.0), 15)
            else
                @test Vector(output[i, 19:33]) == fill(Float16(0.0), 15)
            end
            @test Vector(output[i, 34:48]) == fill(Float16(0.0), 15)
        end
    end

    output = run(param, log_level = 2, log_rate = 0.4)
    for i = 1:(param.generations)
        @test Vector(output[i, 1:13]) == param_vec
        if i <= 60
            @test Vector(output[i, 14:48]) == fill(Float16(0.0), 35)
        else
            @test Vector(output[i, 14:18]) ≠ fill(Float16(0.0), 5)
            if i % 10 == 0
                @test Vector(output[i, 19:33]) ≠ fill(Float16(0.0), 15)
                @test Vector(output[i, 34:48]) ≠ fill(Float16(0.0), 15)
            else
                @test Vector(output[i, 19:48]) == fill(Float16(0.0), 30)
            end
        end
    end
end
