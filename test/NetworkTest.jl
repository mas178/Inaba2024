using Graphs
using SimpleWeightedGraphs
using LinearAlgebra: Diagonal, diag, dot
using StatsBase

using Test: @testset, @test, @test_throws

include("../src/Network.jl")
using .Network: create_adjacency_matrix, weights_to_network, convert_2nd_order, rem_vertices, normalize_weight!

# @testset "normalize_degree!" begin
#     @test false
# end

@testset "normalize_weight!" begin
    N = 2000
    k = 200
    initial_w = Float16(0.5)
    std_weight_sum = N * k * Float64(initial_w)

    weights = create_adjacency_matrix(N, k, initial_w)
    before_weights = copy(weights)
    weights .*= Float16(1.8)

    @test weights == before_weights .* Float16(1.8)

    normalize_weight!(weights, std_weight_sum)

    @test weights == before_weights
end

@testset "create_adjacency_matrix" begin
    @testset "ErrorException" begin
        create_adjacency_matrix(10, 2, Float16(0.1))
        @test_throws ErrorException create_adjacency_matrix(11, 2, Float16(0.1))
        @test_throws ErrorException create_adjacency_matrix(10, 3, Float16(0.1))
        @test_throws ErrorException create_adjacency_matrix(11, 3, Float16(0.1))
        @test_throws ErrorException create_adjacency_matrix(11, -1, Float16(0.1))
        @test_throws ErrorException create_adjacency_matrix(11, 11, Float16(0.1))
    end

    @testset "N = 10, k = 4" begin
        N = 10
        k = 4
        initial_weight = Float16(0.3)
        adjm = create_adjacency_matrix(N, k, initial_weight)
        g = SimpleWeightedGraph(adjm)
        @test nv(g) == N
        @test degree(g) == fill(k, N)
        @test neighbors(g, 1) == [2, 3, 9, 10]
        @test neighbors(g, 2) == [1, 3, 4, 10]
        @test neighbors(g, 3) == [1, 2, 4, 5]
        @test neighborhood(g, 1, Float16(0.3)) == [1, 2, 3, 9, 10]
        @test neighborhood(g, 1, Float16(0.6)) == [1, 2, 3, 9, 10, 4, 5, 7, 8]
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
end

# @testset "create_regular_weighted_graph" begin
#     @testset "ErrorException" begin
#         create_regular_weighted_graph(10, 2, 0.1)
#         @test_throws ErrorException create_regular_weighted_graph(11, 2, 0.1)
#         @test_throws ErrorException create_regular_weighted_graph(10, 3, 0.1)
#         @test_throws ErrorException create_regular_weighted_graph(11, 3, 0.1)
#         @test_throws ErrorException create_regular_weighted_graph(11, -1, 0.1)
#         @test_throws ErrorException create_regular_weighted_graph(11, 11, 0.1)
#     end

#     @testset "N = 10, k = 4" begin
#         N = 10
#         k = 4
#         initial_weight = 0.3
#         g = create_regular_weighted_graph(N, k, initial_weight)
#         @test nv(g) == N
#         @test degree(g) == fill(k, N)
#         @test neighbors(g, 1) == [2, 3, 9, 10]
#         @test neighbors(g, 2) == [1, 3, 4, 10]
#         @test neighbors(g, 3) == [1, 2, 4, 5]
#         @test neighbors(g, 9) == [1, 7, 8, 10]
#         @test neighbors(g, 10) == [1, 2, 8, 9]
#         @test neighborhood(g, 1, 0.3) == [1, 2, 3, 9, 10]
#         @test neighborhood(g, 1, 0.6) == [1, 2, 3, 9, 10, 4, 5, 7, 8]

#         components = connected_components(g)
#         @test length(components) == 1
#         @test length(components[1]) == N
#         @test get_weight(g, 1, 2) == initial_weight
#         @test get_weight(g, 1, 3) == initial_weight
#         @test get_weight(g, 1, 9) == initial_weight
#         @test get_weight(g, 1, 10) == initial_weight
#     end

#     N = 50_000
#     k = 130
#     initial_weight = 0.2

#     @time begin
#         @testset "N = 50_000, k = 130 (Float16)" begin
#             g = create_regular_weighted_graph(N, k, Float16(initial_weight))
#             @test nv(g) == N
#             @test degree(g) == fill(k, N)
#             components = connected_components(g)
#             @test length(components) == 1
#             @test length(components[1]) == N
#             println("Memory Size (Float16): $(Base.summarysize(g) / (1024 * 1024)) MB")
#         end
#     end

#     @time begin
#         @testset "N = 50_000, k = 130 (Float64)" begin
#             g = create_regular_weighted_graph(N, k, Float64(initial_weight))
#             @test nv(g) == N
#             @test degree(g) == fill(k, N)
#             components = connected_components(g)
#             @test length(components) == 1
#             @test length(components[1]) == N
#             println("Memory Size (Float64): $(Base.summarysize(g) / (1024 * 1024)) MB")
#         end
#     end
# end

# g = create_regular_weighted_graph(50_000, 130, Float64(0.1))

# @time begin
#     @testset "N = 50_000, k = 130 (Float64)" begin
#         N = 50_000
#         k = 130
#         initial_weight = 0.1
#         copied_g = deepcopy(g)
#         @test nv(copied_g) == N
#         @test degree(copied_g) == fill(k, N)
#         components = connected_components(g)
#         @test length(components) == 1
#         @test length(components[1]) == N
#         println("Memory Size (Float64): $(Base.summarysize(copied_g) / (1024 * 1024)) MB")
#     end
# end

@testset "weights_to_network" begin
    weights = Float16.([
        0.00 0.50 0.51
        0.50 0.00 0.51
        0.51 0.51 0.00
    ])

    g = weights_to_network(weights, 0.5)
    @test nv(g) == 3
    @test ne(g) == 2
    @test has_edge(g, 1, 3)
    @test has_edge(g, 2, 3)

    g = weights_to_network(weights, 0.51)
    @test nv(g) == 3
    @test ne(g) == 0

    g = weights_to_network(weights, 0.49)
    @test nv(g) == 3
    @test ne(g) == 3
    @test has_edge(g, 1, 2)
    @test has_edge(g, 2, 3)
    @test has_edge(g, 3, 1)
end

@testset "convert_2nd_order" begin
    @testset "simple" begin
        weights = Float16.([
            0.0 0.1 0.2 0.3
            0.1 0.0 0.4 0.5
            0.2 0.4 0.0 0.6
            0.3 0.5 0.6 0.0
        ])
        weights2 = convert_2nd_order(weights)
        factor = 0.86
        factor = maximum(weights + weights * weights) / maximum(weights)

        @test weights2 == transpose(weights2)  # check symmetry
        @test diag(weights2) == fill(0.0, 4)
        @test weights2[1, 2] == Float16((0.1 + 0.2 * 0.4 + 0.3 * 0.5) / factor) == Float16(0.2302)
        @test weights2[1, 3] == Float16((0.2 + 0.1 * 0.4 + 0.3 * 0.6) / factor) == Float16(0.293)
        @test weights2[1, 4] == Float16((0.3 + 0.1 * 0.5 + 0.2 * 0.6) / factor) == Float16(0.328)
        @test weights2[2, 3] ==
              (Float16(0.4) + Float16(0.2) * Float16(0.1) + Float16(0.6) * Float16(0.5)) / Float16(factor) ==
              Float16(0.502)
        @test weights2[2, 4] == Float16((0.5 + 0.3 * 0.1 + 0.4 * 0.6) / factor) == Float16(0.537)
        @test weights2[3, 4] == Float16(0.6)
    end

    @testset "complex" begin
        weights1 = rand(1000, 1000)
        weights1 -= Diagonal(weights1)
        weights1 = (weights1 + weights1') / 2

        # calc normalization factor
        weights2_expected = weights1 + weights1 * weights1
        weights2_expected -= Diagonal(weights2_expected)
        factor = maximum(weights2_expected) / maximum(weights1)

        # execute
        weighted2_actual = convert_2nd_order(Float16.(weights1))

        @test diag(weighted2_actual) == fill(0.0, 1000)
        @test weighted2_actual == transpose(weighted2_actual)  # check symmetry
        @test all(0 .<= weighted2_actual .<= 1)
        for x = 1:1000, y = 1:1000
            if x > y
                @test weighted2_actual[x, y] ≈ (weights1[x, y] + dot(weights1[x, :], weights1[:, y])) / factor
            end
        end
    end
end

@testset "rem_vertices" begin
    N = 100
    weights = rand(Float16, N, N)
    weights -= Diagonal(weights)
    weights = (weights + weights') / 2
    death_id_vec = collect(1:10:N)

    updated_weights = rem_vertices(weights, death_id_vec)

    @test size(updated_weights) == (90, 90)
    @test diag(updated_weights) == fill(Float16(0.0), 90)
    @test updated_weights == transpose(updated_weights)  # check symmetry
    @test updated_weights[1, 1:90] == weights[2, setdiff(1:100, death_id_vec)]
    @test updated_weights[11, 1:90] == weights[13, setdiff(1:100, death_id_vec)]
    @test updated_weights[90, 1:90] == weights[100, setdiff(1:100, death_id_vec)]
end

@time begin
    @testset "convert_2nd_order" begin
        weights = create_adjacency_matrix(2000, 500, Float16(0.1))
        convert_2nd_order(weights)
    end
end

# @time begin
#     @testset "rem_vertices2" begin
#         N = 1000
#         adj_mat = rand(N, N)
#         adj_mat -= Diagonal(adj_mat)
#         adj_mat = (adj_mat + adj_mat') / 2
#         g = SimpleWeightedGraph(adj_mat)
#         death_id_vec = collect(1:100:N)

#         g = rem_vertices2(g, death_id_vec)

#         @test nv(g) == N - length(death_id_vec)
#     end
# end

# @time begin
#     @testset "rem_vertices_slow!" begin
#         N = 1000
#         adj_mat = rand(N, N)
#         adj_mat -= Diagonal(adj_mat)
#         adj_mat = (adj_mat + adj_mat') / 2
#         g = SimpleWeightedGraph(adj_mat)
#         death_id_vec = collect(1:100:N)

#         rem_vertices_slow!(g, death_id_vec)

#         @test nv(g) == N - length(death_id_vec)
#     end
# end
