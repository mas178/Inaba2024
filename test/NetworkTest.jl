using Graphs
using SimpleWeightedGraphs
using LinearAlgebra: Diagonal, diag, dot
using StatsBase

using Test: @testset, @test, @test_throws

include("../src/Network.jl")
using .Network:
    create_adjacency_matrix,
    create_regular_weighted_graph,
    weighted_to_simple,
    weighted_to_2nd_order,
    rem_vertices!,
    rem_vertices_slow!,
    rem_vertices2

@testset "create_adjacency_matrix" begin
    @testset "ErrorException" begin
        create_adjacency_matrix(10, 2, 0.1)
        @test_throws ErrorException create_adjacency_matrix(11, 2, 0.1)
        @test_throws ErrorException create_adjacency_matrix(10, 3, 0.1)
        @test_throws ErrorException create_adjacency_matrix(11, 3, 0.1)
        @test_throws ErrorException create_adjacency_matrix(11, -1, 0.1)
        @test_throws ErrorException create_adjacency_matrix(11, 11, 0.1)
    end

    @testset "N = 10, k = 4" begin
        N = 10
        k = 4
        initial_weight = 0.3
        adjm = create_adjacency_matrix(N, k, initial_weight)
        g = SimpleWeightedGraph(adjm)
        @test nv(g) == N
        @test degree(g) == fill(k, N)
        @test neighbors(g, 1) == [2, 3, 9, 10]
        @test neighbors(g, 2) == [1, 3, 4, 10]
        @test neighbors(g, 3) == [1, 2, 4, 5]
        @test neighborhood(g, 1, 0.3) == [1, 2, 3, 9, 10]
        @test neighborhood(g, 1, 0.6) == [1, 2, 3, 9, 10, 4, 5, 7, 8]
        components = connected_components(g)
        @test length(components) == 1
        @test length(components[1]) == N

        @test get_weight(g, 1, 2) == initial_weight
        @test get_weight(g, 1, 3) == initial_weight
        @test get_weight(g, 1, 9) == initial_weight
        @test get_weight(g, 1, 10) == initial_weight

        rem_vertex!(g, 2)
        rem_vertex!(g, 5)
        @test vertices(g) == 1:8
        @test neighbors(g, 8) == [1, 6, 7]

        add_vertex!(g)
        @test vertices(g) == 1:9
        @test neighbors(g, 9) == []
    end

    N = 50_000
    k = 130
    initial_weight = 0.2

    @time begin
        @testset "N = 50_000, k = 130 (Float16)" begin
            adjm = create_adjacency_matrix(N, k, Float16(initial_weight))
            println("Memory Size (Float16): $(Base.summarysize(adjm) / (1024 * 1024)) MB")

            g = SimpleWeightedGraph(adjm)
            adjm = nothing
            @test nv(g) == N
            @test degree(g) == fill(k, N)
            components = connected_components(g)
            @test length(components) == 1
            @test length(components[1]) == N
            println("Memory Size (Float16): $(Base.summarysize(g) / (1024 * 1024)) MB")
        end
    end

    @time begin
        @testset "N = 50_000, k = 130 (Float64)" begin
            adjm = create_adjacency_matrix(N, k, Float64(initial_weight))
            println("Memory Size (Float64): $(Base.summarysize(adjm) / (1024 * 1024)) MB")

            g = SimpleWeightedGraph(adjm)
            adjm = nothing
            @test nv(g) == N
            @test degree(g) == fill(k, N)
            components = connected_components(g)
            @test length(components) == 1
            @test length(components[1]) == N
            println("Memory Size (Float64): $(Base.summarysize(g) / (1024 * 1024)) MB")
        end
    end

    # @time begin
    #     @testset "SP Matrix N = 50_000, k = 130 (Float16)" begin
    #         adjm = create_adjacency_sp_matrix(N, k, Float16(initial_weight))
    #         println("Memory Size (SP Float16): $(Base.summarysize(adjm) / (1024 * 1024)) MB")

    #         g = SimpleWeightedGraph(adjm)
    #         adjm = nothing
    #         @test nv(g) == N
    #         @test degree(g) == fill(k, N)
    #         components = connected_components(g)
    #         @test length(components) == 1
    #         @test length(components[1]) == N
    #         println("Memory Size (Float16): $(Base.summarysize(g) / (1024 * 1024)) MB")
    #     end
    # end

    # @time begin
    #     @testset "SP Matrix N = 50_000, k = 130 (Float64)" begin
    #         adjm = create_adjacency_sp_matrix(N, k, Float64(initial_weight))
    #         println("Memory Size (SP Float64): $(Base.summarysize(adjm) / (1024 * 1024)) MB")

    #         g = SimpleWeightedGraph(adjm)
    #         adjm = nothing
    #         @test nv(g) == N
    #         @test degree(g) == fill(k, N)
    #         components = connected_components(g)
    #         @test length(components) == 1
    #         @test length(components[1]) == N
    #         println("Memory Size (Float64): $(Base.summarysize(g) / (1024 * 1024)) MB")
    #     end
    # end
end

@testset "create_regular_weighted_graph" begin
    @testset "ErrorException" begin
        create_regular_weighted_graph(10, 2, 0.1)
        @test_throws ErrorException create_regular_weighted_graph(11, 2, 0.1)
        @test_throws ErrorException create_regular_weighted_graph(10, 3, 0.1)
        @test_throws ErrorException create_regular_weighted_graph(11, 3, 0.1)
        @test_throws ErrorException create_regular_weighted_graph(11, -1, 0.1)
        @test_throws ErrorException create_regular_weighted_graph(11, 11, 0.1)
    end

    @testset "N = 10, k = 4" begin
        N = 10
        k = 4
        initial_weight = 0.3
        g = create_regular_weighted_graph(N, k, initial_weight)
        @test nv(g) == N
        @test degree(g) == fill(k, N)
        @test neighbors(g, 1) == [2, 3, 9, 10]
        @test neighbors(g, 2) == [1, 3, 4, 10]
        @test neighbors(g, 3) == [1, 2, 4, 5]
        @test neighbors(g, 9) == [1, 7, 8, 10]
        @test neighbors(g, 10) == [1, 2, 8, 9]
        @test neighborhood(g, 1, 0.3) == [1, 2, 3, 9, 10]
        @test neighborhood(g, 1, 0.6) == [1, 2, 3, 9, 10, 4, 5, 7, 8]

        components = connected_components(g)
        @test length(components) == 1
        @test length(components[1]) == N
        @test get_weight(g, 1, 2) == initial_weight
        @test get_weight(g, 1, 3) == initial_weight
        @test get_weight(g, 1, 9) == initial_weight
        @test get_weight(g, 1, 10) == initial_weight
    end

    N = 50_000
    k = 130
    initial_weight = 0.2

    @time begin
        @testset "N = 50_000, k = 130 (Float16)" begin
            g = create_regular_weighted_graph(N, k, Float16(initial_weight))
            @test nv(g) == N
            @test degree(g) == fill(k, N)
            components = connected_components(g)
            @test length(components) == 1
            @test length(components[1]) == N
            println("Memory Size (Float16): $(Base.summarysize(g) / (1024 * 1024)) MB")
        end
    end

    @time begin
        @testset "N = 50_000, k = 130 (Float64)" begin
            g = create_regular_weighted_graph(N, k, Float64(initial_weight))
            @test nv(g) == N
            @test degree(g) == fill(k, N)
            components = connected_components(g)
            @test length(components) == 1
            @test length(components[1]) == N
            println("Memory Size (Float64): $(Base.summarysize(g) / (1024 * 1024)) MB")
        end
    end
end

g = create_regular_weighted_graph(50_000, 130, Float64(0.1))

@time begin
    @testset "N = 50_000, k = 130 (Float64)" begin
        N = 50_000
        k = 130
        initial_weight = 0.1
        copied_g = deepcopy(g)
        @test nv(copied_g) == N
        @test degree(copied_g) == fill(k, N)
        components = connected_components(g)
        @test length(components) == 1
        @test length(components[1]) == N
        println("Memory Size (Float64): $(Base.summarysize(copied_g) / (1024 * 1024)) MB")
    end
end

@testset "weighted_to_simple" begin
    weighted_g = SimpleWeightedGraph([
        0.00 0.50 0.51
        0.50 0.00 0.51
        0.51 0.51 0.00
    ])

    g = weighted_to_simple(weighted_g, 0.5)
    @test nv(g) == 3
    @test ne(g) == 2
    @test has_edge(g, 1, 3)
    @test has_edge(g, 2, 3)

    g = weighted_to_simple(weighted_g, 0.51)
    @test nv(g) == 3
    @test ne(g) == 0

    g = weighted_to_simple(weighted_g, 0.49)
    @test nv(g) == 3
    @test ne(g) == 3
    @test has_edge(g, 1, 2)
    @test has_edge(g, 2, 3)
    @test has_edge(g, 3, 1)
end

@testset "weighted_to_2nd_order" begin
    @testset "simple" begin
        adj_mat = [
            0.0 0.1 0.2 0.3
            0.1 0.0 0.4 0.5
            0.2 0.4 0.0 0.6
            0.3 0.5 0.6 0.0
        ]
        weighted_g = SimpleWeightedGraph(adj_mat)
        weighted2_g = weighted_to_2nd_order(weighted_g)
        weights = Graphs.weights(weighted2_g)
        factor = 0.86
        factor = maximum(adj_mat + adj_mat * adj_mat) / maximum(adj_mat)

        @test weights == transpose(weights)  # check symmetry
        @test diag(weights) == fill(0.0, 4)
        @test weights[1, 2] == (0.1 + 0.2 * 0.4 + 0.3 * 0.5) / factor == 0.2302325581395349
        @test weights[1, 3] == (0.2 + 0.1 * 0.4 + 0.3 * 0.6) / factor == 0.2930232558139535
        @test weights[1, 4] == (0.3 + 0.1 * 0.5 + 0.2 * 0.6) / factor == 0.327906976744186
        @test weights[2, 3] == (0.4 + 0.2 * 0.1 + 0.6 * 0.5) / factor == 0.5023255813953488
        @test weights[2, 4] == (0.5 + 0.3 * 0.1 + 0.4 * 0.6) / factor == 0.5372093023255814
        @test weights[3, 4] == 0.6
    end

    @testset "complex" begin
        adj_mat = rand(1000, 1000)
        adj_mat -= Diagonal(adj_mat)
        adj_mat = (adj_mat + adj_mat') / 2

        # calc normalization factor
        second_order_weights = adj_mat + adj_mat * adj_mat
        second_order_weights -= Diagonal(second_order_weights)
        factor = maximum(second_order_weights) / maximum(adj_mat)

        # execute
        weighted_g = SimpleWeightedGraph(adj_mat)
        weighted2_g = weighted_to_2nd_order(weighted_g)
        weights = Graphs.weights(weighted2_g)

        @test diag(weights) == fill(0.0, 1000)
        @test weights == transpose(weights)  # check symmetry
        @test all(0 .<= weights .<= 1)
        for x = 1:1000, y = 1:1000
            if x > y
                @test weights[x, y] ≈ (adj_mat[x, y] + dot(adj_mat[x, :], adj_mat[:, y])) / factor
            end
        end
    end
end

@testset "rem_vertices!" begin
    N = 100
    adj_mat = rand(Float16, N, N)
    adj_mat -= Diagonal(adj_mat)
    adj_mat = (adj_mat + adj_mat') / 2
    g1 = SimpleWeightedGraph(adj_mat)
    g2 = SimpleWeightedGraph(adj_mat)
    g3 = SimpleWeightedGraph(adj_mat)
    death_id_vec = collect(1:10:N)

    rem_vertices!(g1, death_id_vec)
    g2 = rem_vertices2(g2, death_id_vec)
    rem_vertices_slow!(g3, death_id_vec)

    @test g1.weights == g2.weights == g3.weights
end

@time begin
    @testset "rem_vertices" begin
        N = 1000
        adj_mat = rand(N, N)
        adj_mat -= Diagonal(adj_mat)
        adj_mat = (adj_mat + adj_mat') / 2
        g = SimpleWeightedGraph(adj_mat)
        death_id_vec = collect(1:100:N)

        rem_vertices!(g, death_id_vec)

        @test nv(g) == N - length(death_id_vec)
    end
end

@time begin
    @testset "rem_vertices2" begin
        N = 1000
        adj_mat = rand(N, N)
        adj_mat -= Diagonal(adj_mat)
        adj_mat = (adj_mat + adj_mat') / 2
        g = SimpleWeightedGraph(adj_mat)
        death_id_vec = collect(1:100:N)

        g = rem_vertices2(g, death_id_vec)

        @test nv(g) == N - length(death_id_vec)
    end
end

@time begin
    @testset "rem_vertices_slow!" begin
        N = 1000
        adj_mat = rand(N, N)
        adj_mat -= Diagonal(adj_mat)
        adj_mat = (adj_mat + adj_mat') / 2
        g = SimpleWeightedGraph(adj_mat)
        death_id_vec = collect(1:100:N)

        rem_vertices_slow!(g, death_id_vec)

        @test nv(g) == N - length(death_id_vec)
    end
end
