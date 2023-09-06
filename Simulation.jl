module Simulation

using CSV: write
using DataFrames: DataFrame, nrow, dropmissing
using Dates
using Graphs: SimpleGraph, degree, gdistances, local_clustering_coefficient, connected_components
using LinearAlgebra
using Plots: Plot, plot!, plot, twinx, PlotMeasures
using Random: MersenneTwister
using StatsBase: mean, std, sample, Weights

export Model,
    Strategy,
    C,
    D,
    invert,
    interaction!,
    classic_fitness,
    sigmoid_fitness,
    pick_parent_and_child,
    death_and_reproduction!,
    make_output_df,
    log!,
    run,
    run_all,
    plot_output_df

@enum Strategy C D

invert(s::Strategy)::Strategy = (s == C ? D : C)

mutable struct Model
    agents::DataFrame
    graph_weights::Matrix{Float16}
    initial_graph_weight::Float64
    payoff_table::Dict{Tuple{Strategy,Strategy},Tuple{Float64,Float64}}
    interaction_freqency::Float64
    relationship_volatility::Float64  # relationship volatility on interaction
    reproduction_rate::Float64        # Ratio of agents performing death and reproduction
    δ::Float64                        # strength of selection
    μ::Float64                        # mutation rate
    rng::MersenneTwister              # random seed

    function Model(;
        N::Int = 1_000,      # size of population
        initial_graph_weight::Float64 = 0.5,
        T::Float64 = 1.1,    # Temptation payoff
        S::Float64 = -0.1,   # Sucker's payoff
        interaction_freqency::Float64 = 1.0,
        relationship_volatility::Float64 = 0.1,
        reproduction_rate = 0.1,
        δ::Float64 = 0.01,
        μ::Float64 = 0.00,
    )
        agents = DataFrame(strategy = fill(D, N), payoff = fill(0.0, N))
        graph_weights = Float16.((fill(1.0, (N, N)) - Matrix(I, N, N)) * initial_graph_weight)
        new(
            agents,
            graph_weights,
            initial_graph_weight,
            Dict((C, C) => (1.0, 1.0), (C, D) => (S, T), (D, C) => (T, S), (D, D) => (0.0, 0.0)),
            interaction_freqency,
            relationship_volatility,
            reproduction_rate,
            δ,
            μ,
            MersenneTwister(),
        )
    end
end

"""
    interaction!(model::Model)::Nothing

1. Each agent becomes a focal agent.
2. Each focal agent picks an opponent based on the edge weight.
3. The agents then play a game and adjust their payoffs according to the payoff table.

    |   |  C  |  D  |
    |:-:|:---:|:---:|
    | C |R = 1|  S  |
    | D |  T  |P = 0|

    - Snowdrift:          T > R > S > P
    - Stag Hunt:          R > T > P > S
    - Prisoner's Dilemma: T > R > P > S

4. Following the game, the weight of the edge between them is updated:

    - If C vs. C, the weight is increased by v (0.0 < v < 1.0).
    - If D vs. D, the weight is decreased by v.
    - Otherwise, the weight remains unchanged.
"""
function interaction!(model::Model)::Nothing
    N = nrow(model.agents)

    # Reset payoff
    model.agents.payoff .= 0.0
    _neighbors = nothing
    _weights = nothing

    @inbounds @simd for focal_id in sample(model.rng, 1:N, Int(round(N * model.interaction_freqency)))
        # Pick an opponent
        _neighbors = deleteat!(collect(1:N), focal_id)
        _weights = model.graph_weights[focal_id, _neighbors]
        opponent_id = sample(model.rng, _neighbors, Weights(_weights))

        strategy_pair = (model.agents[focal_id, :strategy], model.agents[opponent_id, :strategy])

        # Update payoff
        focal_payoff, opponent_payoff = model.payoff_table[strategy_pair]
        model.agents[focal_id, :payoff] += focal_payoff
        model.agents[opponent_id, :payoff] += opponent_payoff

        # Update relationship
        new_weight = if strategy_pair == (C, C)
            min((1.0 + model.relationship_volatility) * model.graph_weights[focal_id, opponent_id], 1.0)
        else
            (1.0 - model.relationship_volatility) * model.graph_weights[focal_id, opponent_id]
        end
        model.graph_weights[focal_id, opponent_id] = new_weight
        model.graph_weights[opponent_id, focal_id] = new_weight
    end
end

classic_fitness(payoff::Float64, δ::Float64)::Float64 = 1.0 - δ + δ * payoff

sigmoid_fitness(payoff::Float64, δ::Float64)::Float64 = 1.0 / (1.0 + exp(-δ * payoff))

function pick_parent_and_child(model::Model)::Tuple{Vector{Int},Vector{Int}}
    N = nrow(model.agents)
    n = round(Int, N * model.reproduction_rate)

    fitness_vec = [sigmoid_fitness(payoff, model.δ) for payoff in model.agents.payoff]

    parent_id_vec = sample(model.rng, 1:N, Weights(fitness_vec), n, replace = false)

    neg_fitness_vec = [sigmoid_fitness(-payoff, model.δ) for payoff in model.agents.payoff]
    deleteat!(neg_fitness_vec, sort(parent_id_vec))

    child_id_vec = sample(model.rng, setdiff(1:N, parent_id_vec), Weights(neg_fitness_vec), n, replace = false)

    return parent_id_vec, child_id_vec
end

function normalize_graph_weights!(model::Model)::Nothing
    N = nrow(model.agents)
    initial_graph_weight_sum = N * (N - 1) * model.initial_graph_weight

    # graph_weight_sum = sum(Float64.(model.graph_weights))
    graph_weight_sum = 0.0
    @inbounds @simd for weight in model.graph_weights
        graph_weight_sum += Float64(weight)
    end

    model.graph_weights ./= (graph_weight_sum / initial_graph_weight_sum)

    return
end

"""
    death_and_reproduction!(model::Model)::Tuple{Vector{Int},Vector{Int}}

- Some agents die based on fitness.
- Some agents reproduce based on fitness.
- Child copies the parent's strategy.
- Child copies the parent's relationship.
- Strategy is mutated based on mutation rate.
"""
function death_and_reproduction!(model::Model)::Tuple{Vector{Int},Vector{Int}}
    parent_id_vec, child_id_vec = pick_parent_and_child(model)
    n = length(parent_id_vec)
    mutation_vec = rand(model.rng, n) .< model.μ

    for i = 1:n
        # strategy
        parent_strategy = model.agents.strategy[parent_id_vec[i]]
        model.agents.strategy[child_id_vec[i]] = mutation_vec[i] ? invert(parent_strategy) : parent_strategy

        # relationship
        model.graph_weights[child_id_vec[i], :] = model.graph_weights[parent_id_vec[i], :]
        model.graph_weights[:, child_id_vec[i]] = model.graph_weights[:, parent_id_vec[i]]
        model.graph_weights[parent_id_vec[i], child_id_vec[i]] = 1.0
        model.graph_weights[child_id_vec[i], parent_id_vec[i]] = 1.0
        model.graph_weights[child_id_vec[i], child_id_vec[i]] = 0.0
    end

    normalize_graph_weights!(model)

    return parent_id_vec, child_id_vec
end

function make_output_df(generations)::DataFrame
    int16_vec = Vector{Union{Int16,Missing}}(undef, generations)
    float16_vec = Vector{Union{Float16,Missing}}(undef, generations)

    return DataFrame(
        N = int16_vec,
        initial_graph_weight = float16_vec,
        T = float16_vec,
        S = float16_vec,
        interaction_freqency = float16_vec,
        relationship_volatility = float16_vec,
        reproduction_rate = float16_vec,
        δ = float16_vec,
        μ = float16_vec,
        generation = int16_vec,
        cooperation_rate = float16_vec,
        payoff_μ = float16_vec,
        weight_μ = float16_vec,
        weight_σ = float16_vec,
        k = float16_vec,
        L = float16_vec,
        C = float16_vec,
        component_count = int16_vec,
        component_size_μ = float16_vec,
        component_size_max = int16_vec,
        component_size_min = int16_vec,
        component_size_σ = float16_vec,
    )
end

unweighted_graph(graph_weights::Matrix, threshold::Float64)::SimpleGraph = SimpleGraph(graph_weights .> threshold)

function log!(output::DataFrame, generation::Int, model::Model, skip::Int = 10)::Nothing
    N = nrow(model.agents)
    T, S = model.payoff_table[(D, C)]

    # 重みベクトル
    weight_vec = [Float64(model.graph_weights[i, j]) for i = 1:N for j in 1:N if i < j]  # costly
    weight_μ = mean(weight_vec)
    weight_σ = std(weight_vec)
    weight_vec = nothing

    simple_g = unweighted_graph(model.graph_weights, 0.5)  # costly

    # 平均次数 (<k>)
    _k = mean(degree(simple_g))

    # 平均距離 (L)
    _L = missing
    if generation % skip == 0
        _L_vec = collect(Iterators.flatten([gdistances(simple_g, i) for i = 1:N]))  # most costly
        _L_vec = filter(x -> 0 < x <= N, _L_vec)
        _L = length(_L_vec) > 0 ? mean(_L_vec) : 0.0
        _L_vec = nothing
    end

    # 平均クラスタ係数 (C)
    _C = mean(local_clustering_coefficient(simple_g, 1:N))  # costly

    # コンポーネント
    _components = [length(_component) for _component in connected_components(simple_g)]
    std_components = std(_components)
    isnan(std_components) && (std_components = 0.0)
    simple_g = nothing

    output[generation, :] = [
        N,
        model.initial_graph_weight,
        T,
        S,
        model.interaction_freqency,
        model.relationship_volatility,
        model.reproduction_rate,
        model.δ,
        model.μ,
        generation,
        count(==(C), model.agents.strategy) / N,  # 協力率
        sum(model.agents.payoff) / N,             # 平均ペイオフ
        Float16(weight_μ),                        # 重みの平均
        Float16(weight_σ),                        # 重みの標準偏差
        _k,                                       # 平均次数 <k>
        _L,                                       # 平均距離 (L)
        _C,                                       # 平均クラスタ係数 (C)
        # コンポーネント
        length(_components),   # コンポーネント数
        mean(_components),     # 平均コンポーネントサイズ
        maximum(_components),  # 最大コンポーネントサイズ
        minimum(_components),  # 最小コンポーネントサイズ
        std_components,        # コンポーネントサイズの標準偏差
    ]

    return
end

function plot_output_df(df::DataFrame)::Plot
    p1 = plot(xl = "Generation", title = "Cooperation")
    plot!(df[:, :cooperation_rate], label = "Cooperation Rate")
    plot!(df[:, :payoff_μ], label = "Payoff (μ)")

    p2 = plot(xl = "Generation", title = "Network Attributes")
    plot!(df[:, :weight_μ], label = "Weight (μ)")
    plot!(df[:, :weight_σ], label = "Weight (σ)")
    filtered_df = dropmissing(df, :L)[:, [:generation, :L]]
    plot!(filtered_df[:, :generation], filtered_df[:, :L], label = "L")
    plot!(df[:, :generation], df[:, :C], label = "C")
    plot!(twinx(), df[:, :k], label = "<k>", line = :dash)

    p3 = plot(xl = "Generation", title = "Component Attributes")
    plot!(df[:, :component_count], label = "Count")
    plot!(df[:, :component_size_μ], label = "Size (μ)")
    plot!(df[:, :component_size_max], label = "Size (Max)")
    plot!(df[:, :component_size_min], label = "Size (Min)")
    plot!(df[:, :component_size_σ], label = "Size (σ)")

    params = join(["$(k) = $(v)" for (k, v) in pairs(df[1, 1:8])], ", ")

    return plot(
        p1,
        p2,
        p3,
        layout = (1, 3),
        size = (1200, 400),
        bottom_margin = 6 * PlotMeasures.mm,
        suptitle = params,
        plot_titlefontsize = 10,
    )
end

now_str(format::String)::String = Dates.format(now(), format)

function run(
    N::Int = 1_000,      # size of population
    T::Float64 = 1.1,
    S::Float64 = -0.1,
    interaction_freqency::Float64 = 1.0,
    initial_graph_weight::Float64 = 0.5,
    relationship_volatility::Float64 = 0.1,
    δ::Float64 = 0.01,   # strength of selection
    μ::Float64 = 0.01,   # mutation rate
    generations::Int = 100,
)::DataFrame
    model = Model(
        N = N,
        T = T,
        S = S,
        interaction_freqency = interaction_freqency,
        initial_graph_weight = initial_graph_weight,
        relationship_volatility = relationship_volatility,
        δ = δ,
        μ = μ,
    )

    model.agents.strategy = rand(model.rng, [C, D], N)

    output_df = make_output_df(generations)

    for generation = 1:generations
        interaction!(model)
        death_and_reproduction!(model)
        log!(output_df, generation, model)
    end

    return output_df
end

function run_all(;
    N_vec::Vector{Int},      # size of population
    T_vec::Vector{Float64},
    S_vec::Vector{Float64},
    interaction_freqency_vec::Vector{Float64},
    initial_graph_weight_vec::Vector{Float64},
    relationship_volatility_vec::Vector{Float64},
    δ_vec::Vector{Float64},   # strength of selection
    μ_vec::Vector{Float64},   # mutation rate
    generations_vec::Vector{Int},
)::Nothing
    println("$(now_str("HH:MM:SS")) Start")

    dir_name = "output/$(now_str("yyyymmdd_HHMMSS"))/"
    mkdir(dir_name)
    println(dir_name)

    params = Iterators.product(
        N_vec,
        T_vec,
        S_vec,
        interaction_freqency_vec,
        initial_graph_weight_vec,
        relationship_volatility_vec,
        δ_vec,
        μ_vec,
        generations_vec,
    )

    Threads.@threads for param in params
        println("$(now_str("HH:MM:SS")) $(Threads.threadid()) $(params[param_num])")
        write("$(dir_name)$(param_num).csv", run(param...))
        GC.gc()
    end

    println("$(now_str("HH:MM:SS")) End")
end

end  # end of module
