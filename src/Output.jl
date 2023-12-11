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

            # weak connection (19〜29)
            :weak_k,
            :weak_C1,
            :weak_C2,
            :weak_component1_count,
            :weak_component1_size_μ,
            :weak_component1_size_max,
            :weak_component1_size_σ,
            :weak_component2_count,
            :weak_component2_size_μ,
            :weak_component2_size_max,
            :weak_component2_size_σ,

            # medium connection (30〜40)
            :medium_k,
            :medium_C1,
            :medium_C2,
            :medium_component1_count,
            :medium_component1_size_μ,
            :medium_component1_size_max,
            :medium_component1_size_σ,
            :medium_component2_count,
            :medium_component2_size_μ,
            :medium_component2_size_max,
            :medium_component2_size_σ,

            # strong connection (41〜51)
            :strong_k,
            :strong_C1,
            :strong_C2,
            :strong_component1_count,
            :strong_component1_size_μ,
            :strong_component1_size_max,
            :strong_component1_size_σ,
            :strong_component2_count,
            :strong_component2_size_μ,
            :strong_component2_size_max,
            :strong_component2_size_σ,
        ],
    )

    return df
end

unweighted_graph(graph_weights::Matrix, threshold::Float64)::SimpleGraph = SimpleGraph(graph_weights .> threshold)

function convert_to_2nd_order_weights(weights1::Matrix{Float16}, N::Int, initial_graph_weight::Float64)::Matrix{Float16}
    weights2 = fill(0.0, (N, N))

    @inbounds for x = 1:N
        x_weights = @views weights1[x, :]'
        @simd for y = (x + 1):N
            # weights2[x, y] = weights1[x, y] + x_weights * weights1[:, y]
            sum = 0.0
            for i = 1:N
                sum += x_weights[i] * weights1[i, y]
            end
            weights2[x, y] = weights2[y, x] = weights1[x, y] + sum
        end
    end

    weights2 = Float16.(weights2)
    Simulation.normalize_graph_weights!(weights2, N, initial_graph_weight)

    return weights2
end

#==
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

function mean_C(g::SimpleGraph, model::Model)::Float16
    sum_C = 0.0
    count_C = 0

    for component in connected_components(g)
        if length(component) >= 3 && mean([model.strategy_vec[n] == C for n in component]) > 1 / 2
            for n in component
                sum_C += local_clustering_coefficient(g, n)
                count_C += 1
            end
        end
    end

    return Float16(count_C == 0 ? 0.0 : sum_C / count_C)
end
==#

function calc(g::SimpleGraph, model::Model)::Tuple{Float16,Float16,Float16,Float16}
    sum_C = 0.0
    count_C = 0
    component_count = 0
    sum_component_size = 0.0
    max_component_size = 0.0

    for component in connected_components(g)
        component_size = length(component)
        mean_strategy_C = mean(model.strategy_vec[n] == C for n in component)

        if component_size >= 3 && mean_strategy_C >= 0.5
            for n in component
                sum_C += local_clustering_coefficient(g, n)
                count_C += 1
            end

            component_count += 1
            sum_component_size += component_size / model.N
            max_component_size = max(max_component_size, component_size / model.N)
        end
    end

    # 平均クラスタ係数
    mean_C = Float16(count_C == 0 ? 0.0 : sum_C / count_C)

    # 平均コンポーネントサイズ
    mean_component_size = Float16(component_count == 0 ? 0.0 : sum_component_size / component_count)

    # 平均クラスタ係数, コンポーネント数, 平均コンポーネントサイズ, 最大コンポーネントサイズ
    return mean_C, component_count, mean_component_size, max_component_size
end

std_component_size(components::Vector{Int})::Float16 = Float16(length(components) > 1 ? std(components) : 0.0)

function log!(output::DataFrame, generation::Int, model::Model, level::Int = 0, skip::Int = 10)::Nothing
    if model.payoff_table_vec !== nothing
        output.T[generation] = model.payoff_table_vec[generation][(D, C)][1]
    end

    output[generation, 12:16] = [
        generation,
        model.N,                           # population
        mean(model.strategy_vec .== C),    # cooperation rate
        mean(model.payoff_vec),            # average payoff
        model.death_rate_vec === nothing ? model.param.reproduction_rate : model.death_rate_vec[generation],  # death rate
    ]

    level == 0 && return

    (generation % skip != 0) && return

    # flatten graph_weights
    # weight_vec = [Float64(model.graph_weights[i, j]) for i = 1:(model.N) for j in 1:(model.N) if i < j]  # costly

    # 1st order connection
    weak_connection_g = unweighted_graph(model.graph_weights, WEAK_THRESHOLD)  # costly
    medium_connection_g = unweighted_graph(model.graph_weights, MEDIUM_THRESHOLD)  # costly
    strong_connection_g = unweighted_graph(model.graph_weights, STRONG_THRESHOLD)  # costly

    # 1st order components
    # weak_components = [length(c) for c in connected_components(weak_connection_g)]
    # medium_components = [length(c) for c in connected_components(medium_connection_g)]
    # strong_components = [length(c) for c in connected_components(strong_connection_g)]

    output[generation, [20; 22:24; 31; 33:35; 42; 44:46]] = [
        calc(weak_connection_g, model)...,   # 20 平均クラスタ係数, 22 コンポーネント数, 23 平均コンポーネントサイズ, 24 最大コンポーネントサイズ
        calc(medium_connection_g, model)..., # 31 平均クラスタ係数, 33 コンポーネント数, 34 平均コンポーネントサイズ, 35 最大コンポーネントサイズ
        calc(strong_connection_g, model)..., # 42 平均クラスタ係数, 44 コンポーネント数, 45 平均コンポーネントサイズ, 46 最大コンポーネントサイズ
    ]

    level <= 1 && return

    # 2nd order connection
    second_order_weights = convert_to_2nd_order_weights(model.graph_weights, model.N, model.param.initial_graph_weight)
    weak_connection2_g = unweighted_graph(second_order_weights, WEAK_THRESHOLD)  # costly
    medium_connection2_g = unweighted_graph(second_order_weights, MEDIUM_THRESHOLD)  # costly
    strong_connection2_g = unweighted_graph(second_order_weights, STRONG_THRESHOLD)  # costly

    # 2nd order components
    # weak_components2 = [length(c) for c in connected_components(weak_connection2_g)]
    # medium_components2 = [length(c) for c in connected_components(medium_connection2_g)]
    # strong_components2 = [length(c) for c in connected_components(strong_connection2_g)]

    output[generation, [21; 26:28; 32; 37:39; 43; 48:50]] = [
        calc(weak_connection2_g, model)...,    # 21 平均クラスタ係数, 26 コンポーネント数, 27 平均コンポーネントサイズ, 28 最大コンポーネントサイズ
        calc(medium_connection2_g, model)...,  # 32 平均クラスタ係数, 37 コンポーネント数, 38 平均コンポーネントサイズ, 39 最大コンポーネントサイズ
        calc(strong_connection2_g, model)...,  # 43 平均クラスタ係数, 48 コンポーネント数, 49 平均コンポーネントサイズ, 50 最大コンポーネントサイズ
    ]

    return
end
end  # end of module
