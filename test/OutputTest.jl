using Graphs
using Test: @testset, @test
using StatsBase: mean, std

include("../src/Output.jl")
using .Output: make_output_df, unweighted_graph, convert_to_2nd_order_weights, log!
using .Output.Simulation: Param, ModelPopulation, C, D

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
    graph_weights = Float16[
        0.0 0.1 0.2 0.3
        0.1 0.0 0.4 0.5
        0.2 0.4 0.0 0.6
        0.3 0.5 0.6 0.0
    ]

    weights2 = convert_to_2nd_order_weights(graph_weights, 4, 0.35)

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

    graph_weights = Float16.(rand(1000, 1000))
    init_weight = sum(Float64.(graph_weights)) / (1000 * 999)
    weights2 = convert_to_2nd_order_weights(graph_weights, 1000, init_weight)
    @test all([0.48 < weights2[x, y] < 0.52 for x in 1000, y in 1000 if x != y])
    @test all([weights2[x, y] == 0.0 for x in 1000, y in 1000 if x == y])
    @test 0.48 < mean(Float64.(weights2)) < 0.52
    @test std(Float64.(weights2)) < 0.03
end

@testset "log!" begin
    param = Param(initial_N = 10, initial_graph_weight = 1.0, generations = 11)
    output = make_output_df(param)
    model = ModelPopulation(param)

    log!(output, 1, model, 2, 1)

    # default
    @test Vector(output[1, 1:11]) == Float64[10, 1.1, -0.1, 1.0, 1.0, 0.1, 0.1, 0.01, 0.0, 0.1, 0.1]
    @test Vector(output[1, 12:22]) == Float16[1.0, 10.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    @test Vector(output[1, 22:25]) == Float16[0.0, 0.0, 0.0, 0.0]

    # cooperation rate and average payoff
    model.strategy_vec = repeat([C, D, D, D, D], 2)
    model.payoff_vec = 0.1:0.1:1.0
    Output.log!(output, 2, model, 2, 1)
    @test Vector(output[2, 12:16]) == Float16[2, 10, 0.2, 0.55, 0.1]

    # average and std of weights
    model = ModelPopulation(Param(initial_N = 3))
    model.graph_weights = [0.0 1.0 2.0; 1.0 0.0 6.0; 2.0 6.0 0.0]
    Output.log!(output, 3, model, 2, 1)
    # @test Vector(output[3, 17:18]) == [mean(Float16[1.0, 2.0, 6.0]), std(Float16[1.0, 2.0, 6.0])]
    @test Vector(output[3, 17:18]) == [0.0, 0.0]

    # <k>, L, C
    ## regular graph
    model = ModelPopulation(Param(initial_N = 1_000))
    g = random_regular_graph(1_000, 8)
    model.graph_weights = Matrix(adjacency_matrix(g))
    model.strategy_vec = fill(C, 1_000)
    log!(output, 4, model, 2, 1)
    @test Vector(output[4, 19:21]) ≈ Float16[0.0, 0.006107, 0.1718]
    @test Vector(output[4, 30:32]) ≈ Float16[0.0, 0.006107, 0.1718]
    @test Vector(output[4, 41:43]) ≈ Float16[0.0, 0.006107, 0.1718]

    ## scale free network (BA model)
    g = barabasi_albert(1_000, 2, seed = 1)
    model.graph_weights = Matrix(adjacency_matrix(g))
    log!(output, 5, model, 2, 1)
    @test Vector(output[5, 19:21]) ≈ Float16[0.0, 0.02054, 0.577]
    @test Vector(output[5, 30:32]) ≈ Float16[0.0, 0.02054, 0.577]
    @test Vector(output[5, 41:43]) ≈ Float16[0.0, 0.02054, 0.577]

    # divided graph
    model = ModelPopulation(Param(initial_N = 10))
    model.strategy_vec = fill(C, 1_000)
    g = SimpleGraph(10)
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

    log!(output, 6, model, 2, 1)

    expected_k_weak = 0.0 # mean([3, 2, 3, 2, 1, 2, 1, 1, 1, 0])
    expected_k_medium = 0.0 # mean([2, 2, 3, 1, 1, 2, 1, 1, 1, 0])
    expected_k_strong = 0.0 # mean([1, 1, 3, 1, 1, 2, 1, 1, 1, 0])
    @test Vector(output[6, 19:21]) ≈ [expected_k_weak, 0.476, 1.0] atol = 0.01  # weak connection
    @test Vector(output[6, 30:32]) ≈ [expected_k_medium, 0.3333, 1.0] atol = 0.01  # medium connection
    @test Vector(output[6, 41:43]) ≈ [expected_k_strong, 0.0, 1.0] atol = 0.01  # strong connection

    # component (コンポーネント数, コンポーネントサイズの平均, 最大, 標準偏差)
    @test Vector(output[6, 22:25]) == Float16[2.0, mean([4, 3]) / 10, 4.0 / 10, 0.0]
    @test Vector(output[4, 22:25]) == Float16[1.0, 1.0, 1.0, 0.0]
    @test Vector(output[5, 22:25]) == Float16[1.0, 1.0, 1.0, 0.0]
    @test Vector(output[3, 22:25]) == Float16[0.0, 0.0, 0.0, 0.0]
end

@testset "log! level" begin
    param = Param(initial_N = 10, initial_graph_weight = 1.0, generations = 3)
    output = make_output_df(param)
    model = ModelPopulation(param)
    Output.log!(output, 1, model, 0, 1)
    Output.log!(output, 2, model, 1, 1)
    Output.log!(output, 3, model, 2, 1)

    @test collect(output[1, [1:11; 13:18]]) == collect(output[2, [1:11; 13:18]]) == collect(output[3, [1:11; 13:18]])
    @test output[1, 12] == 1
    @test output[2, 12] == 2
    @test output[3, 12] == 3

    @test collect(output[1, 19:end]) == fill(0.0, 33)
    @test collect(output[2, [21; 26:29; 32; 37:40; 43; 48:51]]) == fill(0.0, 15)
end
