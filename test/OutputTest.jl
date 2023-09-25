using Graphs
using Test: @testset, @test
using StatsBase: mean, std

include("../src/Output.jl")
using .Output: make_output_df, unweighted_graph, convert_to_2nd_order_weights, log!
using .Output.Simulation: Param, Model, C, D

@testset "make_output_df" begin
    df = make_output_df(Param(generations = 10_000))
    @test size(df) == (10_000, 57)

    byte = @allocated make_output_df(Param(generations = 10_000))
    println("$(byte / 1024 / 1024) MB")
end

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
    @test weights2[3, 4] ==
          weights2[4, 3] ==
          (Float16(0.6) + Float16(0.2 * 0.3) + Float16(0.4 * 0.5)) / factor ==
          Float16(0.5063)

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
    model = Model(param)

    # default
    byte = @allocated log!(output, 1, model, false, 1)
    println("$(byte / 1024 / 1024) MB")

    @test Vector(output[1, 1:11]) == Float64[10, 1.1, -0.1, 1.0, 1.0, 0.1, 0.1, 0.01, 0.0, 0.1, 0.1]
    @test Vector(output[1, 12:22]) == Float16[1, 10, 0.0, 0.0, 0.1, 1.0, 0.0, 9.0, 1.0, 1.0, 1.0]
    @test Vector(output[1, 22:26]) == Float16[1.0, 1.0, 1.0, 1.0, 0.0]

    # cooperation rate and average payoff
    model.strategy_vec = repeat([C, D, D, D, D], 2)
    model.payoff_vec = 0.1:0.1:1.0
    Output.log!(output, 2, model)
    @test Vector(output[2, 12:16]) == Float16[2, 10, 0.2, 0.55, 0.1]

    # average and std of weights
    model = Model(Param(initial_N = 3))
    model.graph_weights = [0.0 1.0 2.0; 1.0 0.0 6.0; 2.0 6.0 0.0]
    Output.log!(output, 3, model, false, 1)
    @test Vector(output[3, 17:18]) == [mean(Float16[1.0, 2.0, 6.0]), std(Float16[1.0, 2.0, 6.0])]

    # <k>, L, C
    ## regular graph
    model = Model(Param(initial_N = 1_000))
    g = random_regular_graph(1_000, 8)
    model.graph_weights = Matrix(adjacency_matrix(g))
    log!(output, 4, model, false, 1)
    @test Vector(output[4, 19:21]) ≈ [8.0, 3.6, 0.01] atol = 0.1

    ## scale free network (BA model)
    g = barabasi_albert(1_000, 2, seed = 1)
    model.graph_weights = Matrix(adjacency_matrix(g))
    log!(output, 5, model, false, 1)
    @test Vector(output[5, 19:21]) ≈ Float16[3.992, 4.038, 0.021] atol = 0.001

    # divided graph
    model = Model(Param(initial_N = 10))
    g = SimpleGraph(10)
    add_edge!(g, (1, 2))
    add_edge!(g, (2, 3))
    add_edge!(g, (3, 4))
    add_edge!(g, (5, 6))
    add_edge!(g, (6, 7))
    add_edge!(g, (8, 9))
    model.graph_weights = Matrix(adjacency_matrix(g))

    byte = @allocated log!(output, 6, model, false, 1)
    println("$(byte / 1024 / 1024) MB")

    expected_k = mean([1, 2, 2, 1, 1, 2, 1, 1, 1, 0])
    expected_L = mean([1, 2, 3, 1, 2, 1, 1, 2, 1, 1])
    @test Vector(output[6, 19:21]) ≈ [expected_k, expected_L, 0.0] atol = 0.1

    # component (コンポーネント数, コンポーネントサイズの平均, 最大, 最小, 標準偏差)
    @test Vector(output[6, 22:26]) == Float16[4.0, mean([4, 3, 2, 1]) / 10, 4.0 / 10, 1.0 / 10, std([4, 3, 2, 1]) / 10]
    @test Vector(output[4, 22:26]) == Float16[1.0, 1.0, 1.0, 1.0, 0.0]
    @test Vector(output[5, 22:26]) == Float16[1.0, 1.0, 1.0, 1.0, 0.0]
    @test Vector(output[3, 22:26]) == Float16[1.0, 1.0, 1.0, 1.0, 0.0]
end
