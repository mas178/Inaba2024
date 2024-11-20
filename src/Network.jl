module Network

using Graphs
using StatsBase: countmap

using Plots
using GraphPlot

include("../src/ColorScheme.jl")
using .ColorScheme: DARK_RED, LIGHT_RED, GRAY, LIGHT_BLUE, DARK_BLUE, COLOR_GRAD

average_degree(g::SimpleGraph)::Float64 = 2 * ne(g) / nv(g)

function average_distance(g::SimpleGraph)::Float64
    n = nv(g)
    shortest_paths = floyd_warshall_shortest_paths(g).dists
    shortest_paths = [path for path in shortest_paths if 0 < path < n]
    total_distance = sum(shortest_paths)
    combinations = length(shortest_paths)
    
    return total_distance / combinations
end

function desc(g::SimpleGraph)::Nothing
    println("Average Degree (k̄):\t $(average_degree(g))")
    println("Average Distance (L):\t $(average_distance(g))")
    println("Average Clustering Coefficient (C):\t $(global_clustering_coefficient(g))")
end

function create_weighted_cycle_graph(N::Int, k::Int, w₀::Float64)::Matrix{Float64}
    @assert N > k > 1 && N % 2 == 0 && k % 2 == 0

    adjacency_matrix = zeros(Float16, N, N)

    for y = 1:N
        for offset = 1:(k ÷ 2)
            x = mod(y + offset - 1, N) + 1
            adjacency_matrix[y, x] = w₀

            x = mod(y - offset - 1, N) + 1
            adjacency_matrix[y, x] = w₀
        end
    end

    return adjacency_matrix
end

create_cycle_graph(N::Int, k::Int)::SimpleGraph = SimpleGraph(create_weighted_cycle_graph(N, k, 1.0))

degree_distribution(g::SimpleGraph)::Dict{Int, Int} = countmap(degree(g))

function plot_graph(g::SimpleGraph, nodefillc::Vector; circle::Bool = true)::Nothing
    if circle
        n = nv(g)
        θ = 2 * π / n  # 各ノードの間の角度
        x_coords = [cos(i * θ - π/2) for i in 0:n-1]
        y_coords = [sin(i * θ - π/2) for i in 0:n-1]

        # layout関数を定義してノードの座標を渡す
        layout = (g -> (x_coords, y_coords))

        gplot(g, nodelabel=1:n, layout=layout, nodelabelc=GRAY, nodefillc=nodefillc) |> display
    else
        gplot(g, nodelabel=1:n, nodelabelc=GRAY, nodefillc=nodefillc) |> display
    end

    return
end

function plot_graph(mat::Matrix{Float64}, nodefillc::Vector)::Nothing
    plot_graph(SimpleGraph(mat), nodefillc)
end

function plot_degree_distribution(g::SimpleGraph)::Nothing
    degree_count = sort(degree_distribution(g))
    xs = collect(keys(degree_count))
    ys = [degree_count[x] for x in xs]
    p1 = plot(xs, ys, seriestype=:scatter, legend=false,
        xlabel="Degree", ylabel="Frequency", 
        title="Degree Distribution")

    # # 対数スケールのプロット
    # p2 = plot(xs, ys, seriestype=:scatter, legend=false, xscale=:log10, yscale=:log10, 
    #     xlabel="Degree (log scale)", ylabel="Frequency (log scale)", 
    #     title="Degree Distribution (Log-Log Scale)")

    display(p1)
    # plot(p1, p2, layout=(1, 2), size=(1000, 400)) |> display
end

mat_nv(mat::Matrix{Float64})::Int = size(mat, 1)

mat_ne(mat::Matrix{Float64})::Int = Int(sum(mat .> 0.0) / 2)

mat_degree(mat::Matrix{Float64})::Vector{Int} = vec(sum(mat .> 0, dims=2))

function mat_update_weight!(mat::Matrix{Float64}, x::Int, y::Int, w::Float64)::Nothing
    mat[x, y] = mat[y, x] = w
    return
end

mat_non_neighbors(mat::Matrix{Float64}, x::Int)::Vector{Int} = filter(!=(x), findall(==(0.0), mat[x, :]))

end # end of module
