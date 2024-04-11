module Network

using Graphs
using LinearAlgebra: Diagonal, diagm
using SparseArrays
using StatsBase: mean

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
    N = nv(weights)
    keep = setdiff(1:N, rem_nodes)

    return weights[keep, keep]
end

function add_vertices(weights::Matrix{Float16}, n::Int)::Matrix{Float16}
    N = size(weights, 1)
    new_weights = Matrix{Float16}(undef, N + n, N + n)

    new_weights[:, (N + 1):(N + n)] .= Float16(0.0)
    new_weights[(N + 1):(N + n), :] .= Float16(0.0)

    for x = 0:(N - 1)
        copyto!(new_weights, x * (N + n) + 1, weights, x * N + 1, N)
    end

    return new_weights
end

function update_weight!(weights::Matrix{Float16}, x::Int, y::Int, weight::Float16)::Nothing
    weights[x, y] = weights[y, x] = weight
    return
end

function rem_edge!(weights::Matrix{Float16}, x::Int, y::Int)::Nothing
    update_weight!(weights, x, y, Float16(0.0))
    return
end

function create_adjacency_matrix(N::Int, k::Int, initial_w::Float16)::Matrix{Float16}
    @assert N > k > 0 && (N % 2 == 0 || k % 2 == 0)

    adjacency_matrix = zeros(Float16, N, N)

    for y = 1:N
        for offset = 1:(k ÷ 2)
            x = mod(y + offset - 1, N) + 1
            adjacency_matrix[y, x] = initial_w

            x = mod(y - offset - 1, N) + 1
            adjacency_matrix[y, x] = initial_w
        end

        if k % 2 == 1
            x = mod(y + N ÷ 2 - 1, N) + 1
            adjacency_matrix[y, x] = initial_w
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

"""
    normalize_weight!(weights::Matrix{Float16}, initial_k::Int, initial_w::Float16)::Nothing

Keep average weight `initial_w`.
"""
function normalize_weight!(weights::Matrix{Float16}, initial_k::Int, initial_w::Float16)::Nothing
    N = nv(weights)
    mean_w = sum(Float64, weights) / N / min(initial_k, N)
    weights .*= (initial_w / Float16(mean_w))
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

"""
    local_wcc(weights::Matrix{Float16}, i::Int, N::Int, max_w::Float16)::Float64

calculate local weighted clustering coefficient.
"""
function local_wcc(weights::Matrix{Float16}, i::Int, max_w::Float16)::Float64
    max_w > 0 || return 0.0

    numerator = 0.0
    denominator = 0.0

    i_neighbors = neighbors(weights, i)

    for j in i_neighbors
        w_ij = weights[i, j]
        for h in i_neighbors
            if j < h
                w_ih = weights[i, h]
                harmonic_mean = (1 / w_ij + 1 / w_ih) / 2
                denominator += 2 / (harmonic_mean + 1 / max_w)

                w_jh = weights[j, h]
                if w_jh > 0
                    numerator += 2 / (harmonic_mean + 1 / w_jh)
                end
            end
        end
    end

    return denominator != 0 ? numerator / denominator : 0.0
end

function weighted_cc(weights::Matrix{Float16})::Vector{Float64}
    N = nv(weights)
    max_w = maximum(weights)

    # wcc_vec = fill(0.0, N)
    # 
    # Threads.@threads for i = 1:N
    #     wcc_vec[i] = local_wcc(weights, i, max_w)
    # end
    # 
    # return wcc_vec

    return [local_wcc(weights, i, max_w) for i = 1:N]
end

"""
    global_wcc(weights::Matrix{Float16}, i::Int, N::Int, max_w::Float16)::Float64

calculate global weighted clustering coefficient.
"""
function global_wcc(weights::Matrix{Float16})::Float64
    nv(weights) < 3 && return 0.0
    return mean(wcc(weights))
end

function weighted_degree(weights::Matrix{Float16})::Vector{Float64}
    return vec(sum(Float64, weights, dims = 1))
end

function weighted_mean_degree(weights::Matrix{Float16})::Float64
    nv(weights) < 2 && return 0.0
    return mean(weighted_degree(weights))
end
end
