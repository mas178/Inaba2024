module Network

using Graphs
using SimpleWeightedGraphs
using SparseArrays
using LinearAlgebra: Diagonal

function check_N_k(N::Int, k::Int)::Nothing
    if k >= N || k < 0 || N % 2 == 1 || k % 2 == 1
        error("Invalid combination of N and k for a regular graph (N = $(N), k = $(k))")
    end
end

function create_adjacency_matrix(N::Int, k::Int, initial_weight::Union{Float64,Float16})::Matrix{typeof(initial_weight)}
    check_N_k(N, k)

    adjacency_matrix = zeros(typeof(initial_weight), N, N)
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

function create_regular_weighted_graph(N::Int, k::Int, initial_weight::Union{Float64,Float16})::SimpleWeightedGraph
    return SimpleWeightedGraph(create_adjacency_matrix(N, k, initial_weight))
end

weighted_to_simple(g::SimpleWeightedGraph, θ::Float64)::SimpleGraph = SimpleGraph(weights(g) .> θ)

"""
    weighted_to_2nd_order(g::SimpleWeightedGraph)::SimpleWeightedGraph

Create a second-order weighted graph from a given `SimpleWeightedGraph`.

This function computes a second-order weighted adjacency matrix from the provided graph `g`. 
The second-order weighting is obtained by adding the square of the original adjacency matrix 
to itself, thereby capturing the two-step connections in the graph.

After computing the second-order weights, they are normalized by dividing each weight 
by the maximum weight in the matrix. This normalization brings all weights into the 
range between 0 and 1, making the weights of different edges comparable.

The resulting normalized second-order weighted matrix is used to create a new `SimpleWeightedGraph`.

# Arguments
- `g::SimpleWeightedGraph`: The original weighted graph from which the second-order 
  weighted graph is derived.

# Returns
`SimpleWeightedGraph`: A new graph where the weights represent the normalized second-order connections.
"""
function weighted_to_2nd_order(g::SimpleWeightedGraph)::SimpleWeightedGraph
    # calc 2nd order weights
    adjacency_matrix::SparseArrays.SparseMatrixCSC{Float64,Int64} = weights(g)
    second_order_weights = adjacency_matrix + adjacency_matrix * adjacency_matrix
    second_order_weights -= Diagonal(second_order_weights)

    # normalization
    second_order_weights ./= (maximum(second_order_weights) / maximum(adjacency_matrix))

    return SimpleWeightedGraph(second_order_weights)
end

function rem_vertices!(g::SimpleWeightedGraph, rem_nodes::Vector{Int})::Nothing
    n = nv(g)
    keep_index = setdiff(1:n, rem_nodes)
    g.weights = g.weights[keep_index, keep_index]
    return
end

function rem_vertices2(g::SimpleWeightedGraph, rem_nodes::Vector{Int})::SimpleWeightedGraph
    n = nv(g)
    keep_nodes = setdiff(1:n, rem_nodes)
    adjacency_matrix = zeros(Float16, length(keep_nodes), length(keep_nodes))

    for (x_index, x) in enumerate(keep_nodes)
        for (y_index, y) in enumerate(keep_nodes)
            if has_edge(g, x, y)
                adjacency_matrix[x_index, y_index] = get_weight(g, x, y)
            end
        end
    end

    return SimpleWeightedGraph(adjacency_matrix)
end

function rem_vertices_slow!(g::SimpleWeightedGraph, rem_nodes::Vector{Int})::Nothing
    for v in sort(rem_nodes, rev = true)
        rem_vertex!(g, v)
    end

    return
end
end
