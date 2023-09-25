module Output

using DataFrames: DataFrame, select!, Not
using Graphs: SimpleGraph, degree, gdistances, local_clustering_coefficient, connected_components
using StatsBase: mean, std

include("../src/Simulation.jl")
using .Simulation

const WEAK_THRESHOLD = 0.25
const MEDIUM_THRESHOLD = 0.50
const STRONG_THRESHOLD = 0.75

function initialize_column!(df::DataFrame, columns::Vector{Symbol}, value = Float16(0))
    for column in columns
        df[!, column] .= value
    end
end

function make_output_df(param::Param)::DataFrame
    df = DataFrame([param])
    select!(df, Not([:generations, :rng]))
    df = repeat(df, param.generations)

    initialize_column!(df, [:generation, :N], Int16(0))
    initialize_column!(
        df,
        [
            :cooperation_rate,
            :payoff_μ,
            :death_rate,
            :weight_μ,
            :weight_σ,

            # weak connection (19〜31)
            :weak_k,
            :weak_L,
            :weak_C,
            :weak_component1_count,
            :weak_component1_size_μ,
            :weak_component1_size_max,
            :weak_component1_size_min,
            :weak_component1_size_σ,
            :weak_component2_count,
            :weak_component2_size_μ,
            :weak_component2_size_max,
            :weak_component2_size_min,
            :weak_component2_size_σ,

            # medium connection (32〜44)
            :medium_k,
            :medium_L,
            :medium_C,
            :medium_component1_count,
            :medium_component1_size_μ,
            :medium_component1_size_max,
            :medium_component1_size_min,
            :medium_component1_size_σ,
            :medium_component2_count,
            :medium_component2_size_μ,
            :medium_component2_size_max,
            :medium_component2_size_min,
            :medium_component2_size_σ,

            # strong connection (45〜57)
            :strong_k,
            :strong_L,
            :strong_C,
            :strong_component1_count,
            :strong_component1_size_μ,
            :strong_component1_size_max,
            :strong_component1_size_min,
            :strong_component1_size_σ,
            :strong_component2_count,
            :strong_component2_size_μ,
            :strong_component2_size_max,
            :strong_component2_size_min,
            :strong_component2_size_σ,
        ],
    )

    return df
end

unweighted_graph(graph_weights::Matrix, threshold::Float64)::SimpleGraph = SimpleGraph(graph_weights .> threshold)

function convert_to_2nd_order_weights(weights1::Matrix{Float16}, N::Int, initial_graph_weight::Float64)::Matrix{Float16}
    weights2 = fill(Float16(0.0), (N, N))

    @inbounds for x = 1:N
        x_weights = @views weights1[x, :]'
        @simd for y = (x + 1):N
            weights2[x, y] = weights1[x, y] + x_weights * weights1[:, y]
            weights2[y, x] = weights2[x, y]
        end
    end

    Simulation.normalize_graph_weights!(weights2, N, initial_graph_weight)

    return weights2
end

mean_k(g::SimpleGraph)::Float16 = Float16(mean(degree(g)))

function mean_L(g::SimpleGraph, N::Int)::Float16
    # _L_vec = collect(Iterators.flatten([gdistances(simple_g, i) for i = 1:model.N]))  # most costly
    L_vec = []

    @inbounds @simd for i = 1:N
        append!(L_vec, gdistances(g, i))
    end

    L_vec = filter(x -> 0 < x <= N, L_vec)
    L = length(L_vec) > 0 ? mean(L_vec) : 0.0

    return Float16(L)
end

mean_C(g::SimpleGraph, N::Int)::Float16 = Float16(mean(local_clustering_coefficient(g, 1:N)))

std_component_size(components::Vector{Int})::Float16 = Float16(length(components) > 1 ? std(components) : 0.0)

function log!(output::DataFrame, generation::Int, model::Model, detail::Bool = false, skip::Int = 10)::Nothing
    output[generation, 12:16] = [
        generation,
        model.N,                                       # population
        mean(model.strategy_vec .== C),                # 協力率
        mean(model.payoff_vec),                        # 平均ペイオフ
        Simulation.get_death_rate(model, generation),  # 死亡率
    ]

    (generation % skip != 0) && return

    # flatten graph_weights
    weight_vec = [Float64(model.graph_weights[i, j]) for i = 1:(model.N) for j in 1:(model.N) if i < j]  # costly

    # 1st order connection
    weak_connection_g = unweighted_graph(model.graph_weights, WEAK_THRESHOLD)  # costly
    medium_connection_g = unweighted_graph(model.graph_weights, MEDIUM_THRESHOLD)  # costly
    strong_connection_g = unweighted_graph(model.graph_weights, STRONG_THRESHOLD)  # costly

    # 2nd order connection
    if detail
        second_order_weights =
            convert_to_2nd_order_weights(model.graph_weights, model.N, model.param.initial_graph_weight)
        weak_connection2_g = unweighted_graph(second_order_weights, WEAK_THRESHOLD)  # costly
        medium_connection2_g = unweighted_graph(second_order_weights, MEDIUM_THRESHOLD)  # costly
        strong_connection2_g = unweighted_graph(second_order_weights, STRONG_THRESHOLD)  # costly
    end

    # 1st order components
    weak_components = [length(c) for c in connected_components(weak_connection_g)]
    medium_components = [length(c) for c in connected_components(medium_connection_g)]
    strong_components = [length(c) for c in connected_components(strong_connection_g)]

    # 2nd order components
    if detail
        weak_components2 = [length(c) for c in connected_components(weak_connection2_g)]
        medium_components2 = [length(c) for c in connected_components(medium_connection2_g)]
        strong_components2 = [length(c) for c in connected_components(strong_connection2_g)]
    end

    output[generation, 17:end] = [
        mean(weight_vec),                                 # 17 重みの平均
        std(weight_vec),                                  # 18 重みの標準偏差

        # weak connection
        mean_k(weak_connection_g),                        # 19 平均次数 <k>
        mean_L(weak_connection_g, model.N),               # 20 平均距離 (L)
        mean_C(weak_connection_g, model.N),               # 21 平均クラスタ係数 (C)
        length(weak_components),                          # 22 コンポーネント数
        mean(weak_components) / model.N,                  # 23 平均コンポーネントサイズ
        maximum(weak_components) / model.N,               # 24 最大コンポーネントサイズ
        minimum(weak_components) / model.N,               # 25 最小コンポーネントサイズ
        std_component_size(weak_components) / model.N,    # 26 コンポーネントサイズの標準偏差
        detail ? length(weak_components2) : 0.0,                         # 27 コンポーネント数
        detail ? mean(weak_components2) / model.N : 0.0,                 # 28 平均コンポーネントサイズ
        detail ? maximum(weak_components2) / model.N : 0.0,              # 29 最大コンポーネントサイズ
        detail ? minimum(weak_components2) / model.N : 0.0,              # 30 最小コンポーネントサイズ
        detail ? std_component_size(weak_components2) / model.N : 0.0,   # 31 コンポーネントサイズの標準偏差

        # medium connection
        mean_k(medium_connection_g),                      # 32 平均次数 <k>
        mean_L(medium_connection_g, model.N),             # 33 平均距離 (L)
        mean_C(medium_connection_g, model.N),             # 34 平均クラスタ係数 (C)
        length(medium_components),                        # 35 コンポーネント数
        mean(medium_components) / model.N,                # 36 平均コンポーネントサイズ
        maximum(medium_components) / model.N,             # 37 最大コンポーネントサイズ
        minimum(medium_components) / model.N,             # 38 最小コンポーネントサイズ
        std_component_size(medium_components) / model.N,  # 39 コンポーネントサイズの標準偏差
        detail ? length(medium_components2) : 0.0,                       # 40 コンポーネント数
        detail ? mean(medium_components2) / model.N : 0.0,               # 41 平均コンポーネントサイズ
        detail ? maximum(medium_components2) / model.N : 0.0,            # 42 最大コンポーネントサイズ
        detail ? minimum(medium_components2) / model.N : 0.0,            # 43 最小コンポーネントサイズ
        detail ? std_component_size(medium_components2) / model.N : 0.0, # 44 コンポーネントサイズの標準偏差

        # strong connection
        mean_k(strong_connection_g),                      # 45 平均次数 <k>
        mean_L(strong_connection_g, model.N),             # 46 平均距離 (L)
        mean_C(strong_connection_g, model.N),             # 47 平均クラスタ係数 (C)
        length(strong_components),                        # 48 コンポーネント数
        mean(strong_components) / model.N,                # 49 平均コンポーネントサイズ
        maximum(strong_components) / model.N,             # 50 最大コンポーネントサイズ
        minimum(strong_components) / model.N,             # 51 最小コンポーネントサイズ
        std_component_size(strong_components) / model.N,  # 52 コンポーネントサイズの標準偏差
        detail ? length(strong_components2) : 0.0,                       # 53 コンポーネント数
        detail ? mean(strong_components2) / model.N : 0.0,               # 54 平均コンポーネントサイズ
        detail ? maximum(strong_components2) / model.N : 0.0,            # 55 最大コンポーネントサイズ
        detail ? minimum(strong_components2) / model.N : 0.0,            # 56 最小コンポーネントサイズ
        detail ? std_component_size(strong_components2) / model.N : 0.0, # 57 コンポーネントサイズの標準偏差
    ]

    return
end
end  # end of module
