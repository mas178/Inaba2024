module Network

using Graphs
using LinearAlgebra: Diagonal
using SparseArrays

function nv(weights::Matrix{Float16})::Int
    return size(weights, 1)
end

function neighbors(weights::Matrix{Float16}, node::Int)::Vector{Int}
    return [neighbor_id for neighbor_id in 1:size(weights, 1) if weights[node, neighbor_id] > 0]
end

function rem_vertices(weights::Matrix{Float16}, rem_nodes::Vector{Int})::Matrix{Float16}
    n = size(weights, 1)
    keep = trues(n)
    keep[rem_nodes] .= false

    return weights[keep, keep]
end

function update_weight!(weights::Matrix{Float16}, x::Int, y::Int, weight::Float16)::Nothing
    weights[x, y] = weights[y, x] = weight
    return
end

function rem_edge!(weights::Matrix{Float16}, x::Int, y::Int)::Nothing
    update_weight!(weights, x, y, Float16(0.0))
    return
end

function check_N_k(N::Int, k::Int)::Nothing
    if k >= N || k < 0 || N % 2 == 1 || k % 2 == 1
        error("Invalid combination of N and k for a regular graph (N = $(N), k = $(k))")
    end
end

function create_adjacency_matrix(N::Int, k::Int, initial_weight::Float16)::Matrix{Float16}
    check_N_k(N, k)

    adjacency_matrix = zeros(Float16, N, N)
    neighbors = 1:(k ÷ 2)

    for i = 1:N
        # j = mod.(neighbors .+ i .- 1, N) .+ 1
        # adjacency_matrix[i, j] .= adjacency_matrix[j, i] .= initial_weight
        for offset in neighbors
            j = mod(i + offset - 1, N) + 1
            adjacency_matrix[i, j] = adjacency_matrix[j, i] = initial_weight
        end
    end

    return adjacency_matrix
end

# function create_regular_weighted_graph(N::Int, k::Int, initial_weight::Union{Float64,Float16})::SimpleWeightedGraph
#     return SimpleWeightedGraph(create_adjacency_matrix(N, k, initial_weight))
# end

weights_to_network(weights::Matrix{Float16}, θ::Float64)::SimpleGraph = SimpleGraph(weights .> θ)

"""
    weighted_to_2nd_order(g::Matrix{Float16})::Matrix{Float16}

Create a second-order weights from a given `Matrix{Float16}`.
"""
function convert_2nd_order(weights::Matrix{Float16})::Matrix{Float16}
    # calc 2nd order weights
    weights = sparse(weights)
    second_order_weights = weights + weights * weights
    second_order_weights -= Diagonal(second_order_weights)
    second_order_weights = Matrix(second_order_weights)

    # normalization
    second_order_weights ./= (maximum(second_order_weights) / maximum(weights))

    return second_order_weights
end

# function convert_2nd_order(weights::Matrix{Float16})::Matrix{Float16}
#     # calc 2nd order weights
#     second_order_weights = weights + weights * weights
#     second_order_weights -= Diagonal(second_order_weights)

#     # normalization
#     second_order_weights ./= (maximum(second_order_weights) / maximum(weights))

#     return second_order_weights
# end

# function add_vertices!(weights::Matrix{Float16}, birth_n:Int)::Nothing
#     n = size(weights, 1)
#     zeros_row = zeros(n, birth_n)
#     zeros_col = zeros(birth_n, n + birth_n)
#     new_weights = vcat(weights, zeros_row)
#     new_weights = hcat(new_weights, zeros_col)
#     weights .= new_weights
#     return
# end

# function rem_vertices!(g::SimpleWeightedGraph, rem_nodes::Vector{Int})::Nothing
#     n = nv(g)
#     keep_index = setdiff(1:n, rem_nodes)
#     g.weights = g.weights[keep_index, keep_index]
#     return
# end

# function rem_vertices2(g::SimpleWeightedGraph, rem_nodes::Vector{Int})::SimpleWeightedGraph
#     n = nv(g)
#     keep_nodes = setdiff(1:n, rem_nodes)
#     adjacency_matrix = zeros(Float16, length(keep_nodes), length(keep_nodes))

#     for (x_index, x) in enumerate(keep_nodes)
#         for (y_index, y) in enumerate(keep_nodes)
#             if has_edge(g, x, y)
#                 adjacency_matrix[x_index, y_index] = get_weight(g, x, y)
#             end
#         end
#     end

#     return SimpleWeightedGraph(adjacency_matrix)
# end

# function rem_vertices_slow!(g::SimpleWeightedGraph, rem_nodes::Vector{Int})::Nothing
#     for v in sort(rem_nodes, rev = true)
#         rem_vertex!(g, v)
#     end

#     return
# end
end
