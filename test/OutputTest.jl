using Graphs: SimpleGraph, random_regular_graph, barabasi_albert, adjacency_matrix, add_edge!
using Test: @testset, @test
using StatsBase: mean, std

include("../src/Output.jl")
using .Output: make_output_df, log!
using .Output.Simulation: Param, Model, C, D

@testset "make_output_df" begin
    df = make_output_df(Param(generations = 10_000))
    @test size(df) == (10_000, 57)

    byte = @allocated make_output_df(Param(generations = 10_000))
    println("$(byte / 1024 / 1024) MB")
end

@testset "log!" begin
    param = Param(initial_N = 10, initial_graph_weight = 1.0, generations = 11)
    output = make_output_df(param)
    model = Model(param)

    # default
    byte = @allocated log!(output, 1, model, 1)
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
    Output.log!(output, 3, model, 1)
    @test Vector(output[3, 17:18]) == [mean(Float16[1.0, 2.0, 6.0]), std(Float16[1.0, 2.0, 6.0])]

    # <k>, L, C
    ## regular graph
    model = Model(Param(initial_N = 1_000))
    g = random_regular_graph(1_000, 8)
    model.graph_weights = Matrix(adjacency_matrix(g))
    log!(output, 4, model, 1)
    @test Vector(output[4, 19:21]) ≈ [8.0, 3.6, 0.01] atol = 0.1

    ## scale free network (BA model)
    g = barabasi_albert(1_000, 2, seed = 1)
    model.graph_weights = Matrix(adjacency_matrix(g))
    log!(output, 5, model, 1)
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

    byte = @allocated Output.log!(output, 6, model, 1)
    println("$(byte / 1024 / 1024) MB")

    expected_k = mean([1, 2, 2, 1, 1, 2, 1, 1, 1, 0])
    expected_L = mean([1, 2, 3, 1, 2, 1, 1, 2, 1, 1])
    @test Vector(output[6, 19:21]) ≈ [expected_k, expected_L, 0.0] atol = 0.1

    # component (コンポーネント数, コンポーネントサイズの平均, 最大, 最小, 標準偏差)
    @test Vector(output[6, 22:26]) == Float16[4.0, mean([4, 3, 2, 1])/10, 4.0/10, 1.0/10, std([4, 3, 2, 1])/10]
    @test Vector(output[4, 22:26]) == Float16[1.0, 1.0, 1.0, 1.0, 0.0]
    @test Vector(output[5, 22:26]) == Float16[1.0, 1.0, 1.0, 1.0, 0.0]
    @test Vector(output[3, 22:26]) == Float16[1.0, 1.0, 1.0, 1.0, 0.0]
end
