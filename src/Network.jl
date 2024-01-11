module Network

using Graphs
using LinearAlgebra: Diagonal
using SparseArrays

nv(weights::Matrix{Float16})::Int = size(weights, 1)

function degree(weights::Matrix{Float16}, node::Int)::Int
    counter = 0
    @inbounds @simd for i = 1:size(weights, 2)
        if weights[node, i] > 0
            counter += 1
        end
    end
    return counter
end

degree(weights::Matrix{Float16})::Vector{Int} = [degree(weights, node) for node = 1:nv(weights)]

function neighbors(weights::Matrix{Float16}, node::Int)::Vector{Int}
    return filter(neighbor_id -> weights[node, neighbor_id] > 0, 1:size(weights, 1))
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

function normalize_degree!(weights::Matrix{Float16}, k::Int)::Nothing
    degree_cache = degree(weights)

    @inbounds for n = 1:nv(weights)
        if degree_cache[n] > k
            for neighbor_id in sort(neighbors(weights, n), by = i -> weights[n, i])
                if degree_cache[neighbor_id] > k
                    rem_edge!(weights, n, neighbor_id)
                    degree_cache[n] -= 1
                    degree_cache[neighbor_id] -= 1
                    degree_cache[n] <= k && break
                end
            end
        end
    end

    return
end

function normalize_weight!(weights::Matrix{Float16}, std_weight_sum::Float64)::Nothing
    factor = std_weight_sum / sum(Float64, weights)
    weights .*= Float16(factor)
    return
end

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
end
