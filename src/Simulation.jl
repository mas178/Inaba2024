module Simulation

using DataFrames: DataFrame, nrow, dropmissing, select!, Not
using Graphs: SimpleGraph, degree, gdistances, local_clustering_coefficient, connected_components
using LinearAlgebra
using Random: MersenneTwister
using StatsBase: mean, std, sample, Weights

export Model, Param, Strategy, C, D, interaction!, death_and_birth!, log!, run_all, plot_output_df

@enum Strategy C D

invert(s::Strategy)::Strategy = (s == C ? D : C)

"""
    ar1_model(β::Float64, σ::Float64, μ::Float64 = 1.0, T::Int = 100)::Vector{Float64}

AR(1) Model to create environmental volatility (`env_severity_vec`).
"""
function ar1_model(
    β::Float64,        # 自己回帰の係数。|β| < 1 の場合に平均回帰性を持つ。μ = α / (1 - β)
    σ::Float64,        # std of white noise
    T::Int,            # time steps
    μ::Float64 = 1.0,  # expected average value
)::Vector{Float64}
    α = μ * (1 - β)    # 定数項。μから逆算する。
    x = Vector{Float64}(undef, T + 1)
    x[1] = μ

    for t = 1:T
        ϵ = σ * randn()  # white noise
        x[t+1] = clamp(α + β * x[t] + ϵ, 0.0, 2.0)
    end

    return x
end

@kwdef struct Param
    initial_N::Int = 1_000
    T::Float64 = 1.1                         # Temptation payoff
    S::Float64 = -0.1                        # Sucker's payoff
    initial_graph_weight::Float64 = 0.5
    interaction_freqency::Float64 = 1.0
    relationship_volatility::Float64 = 0.1
    birth_rate::Float64 = 0.1
    δ::Float64 = 0.01                        # strength of selection
    μ::Float64 = 0.0                         # mutation rate
    β::Float64 = 0.1                         # environmental volatility (自己回帰の係数)
    σ::Float64 = 0.1                         # environmental volatility (std of white noise)
    generations::Int = 100                   # generations (Time steps)
    rng::MersenneTwister = MersenneTwister() # random seed
end

mutable struct Model
    N::Int                             # agents' population
    strategy_vec::Vector{Strategy}     # agents' strategy
    payoff_vec::Vector{Float64}        # agents' payoff
    graph_weights::Matrix{Float16}     # agents' relationship
    payoff_table::Dict{Tuple{Strategy,Strategy},Tuple{Float64,Float64}}
    env_severity_vec::Vector{Float64}  # deathe rate = birth rate * env_severity_vec
    param::Param

    function Model(p::Param)
        new(
            p.initial_N,
            fill(D, p.initial_N),
            fill(0.0, p.initial_N),
            (fill(1.0, (p.initial_N, p.initial_N)) - Matrix(I, p.initial_N, p.initial_N)) * p.initial_graph_weight,
            Dict((C, C) => (1.0, 1.0), (C, D) => (p.S, p.T), (D, C) => (p.T, p.S), (D, D) => (0.0, 0.0)),
            ar1_model(p.β, p.σ, round(Int, p.generations / 10)),
            p,
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
    - Otherwise, the weight is decreased by v.
"""
function interaction!(model::Model)::Nothing
    # Reset payoff
    model.payoff_vec .= 0.0

    n_interaction = Int(round(model.N * model.param.interaction_freqency))
    focal_id_vec = sample(model.param.rng, 1:model.N, n_interaction, replace = false)
    all_ids = collect(1:model.N)
    temp_weights = Vector{Float16}(undef, model.N - 1)

    @inbounds for focal_id in focal_id_vec
        # Pick an opponent
        neighbors = deleteat!(copy(all_ids), focal_id)
        @views temp_weights .= model.graph_weights[focal_id, neighbors]
        opponent_id = sample(model.param.rng, neighbors, Weights(temp_weights))

        strategy_pair = (model.strategy_vec[focal_id], model.strategy_vec[opponent_id])

        # Update payoff
        focal_payoff, opponent_payoff = model.payoff_table[strategy_pair]
        model.payoff_vec[focal_id] += focal_payoff
        model.payoff_vec[opponent_id] += opponent_payoff

        # Update relationship
        new_weight = if strategy_pair == (C, C)
            min((1.0 + model.param.relationship_volatility) * model.graph_weights[focal_id, opponent_id], 1.0)
        else
            (1.0 - model.param.relationship_volatility) * model.graph_weights[focal_id, opponent_id]
        end
        model.graph_weights[focal_id, opponent_id] = new_weight
        model.graph_weights[opponent_id, focal_id] = new_weight
    end
end

classic_fitness(payoff::Float64, δ::Float64)::Float64 = 1.0 - δ + δ * payoff

sigmoid_fitness(payoff::Float64, δ::Float64)::Float64 = 1.0 / (1.0 + exp(-δ * payoff))

function pick_deaths(model::Model, n::Int)::Vector{Int}
    neg_fitness_vec = sigmoid_fitness.(-model.payoff_vec, model.param.δ)
    death_id_vec = sample(model.param.rng, 1:model.N, Weights(neg_fitness_vec), n, replace = false)

    return sort(death_id_vec)
end

function pick_parents(model::Model, n::Int)::Vector{Int}
    fitness_vec = sigmoid_fitness.(model.payoff_vec, model.param.δ)
    parent_id_vec = sample(model.param.rng, 1:model.N, Weights(fitness_vec), n, replace = false)

    return sort(parent_id_vec)
end

function normalize_graph_weights!(model::Model)::Nothing
    initial_graph_weight_sum = model.N * (model.N - 1) * model.param.initial_graph_weight

    # graph_weight_sum = sum(Float64.(model.graph_weights))
    factor = 0.0
    @inbounds @simd for weight in model.graph_weights
        factor += Float64(weight)
    end

    # model.graph_weights ./= (graph_weight_sum / initial_graph_weight_sum)
    factor = Float16(initial_graph_weight_sum / factor)
    @inbounds @simd for i in eachindex(model.graph_weights)
        model.graph_weights[i] *= factor
    end

    return
end

get_death_rate(model::Model, generation::Int)::Float64 =
    model.env_severity_vec[ceil(Int, generation / 10)] * model.param.birth_rate

"""
    death_and_reproduction!(model::Model)::Tuple{Vector{Int},Vector{Int}}

- Some agents die based on fitness.
- Some agents reproduce based on fitness.
- Child copies the parent's strategy.
- Child copies the parent's relationship.
- Strategy is mutated based on mutation rate.
"""
function death_and_birth!(model::Model, generation::Int)::Tuple{Vector{Int},Vector{Int}}
    # number of deaths and births (should be calculated before model.N changed).
    n_deaths = round(Int, model.N * get_death_rate(model, generation))
    n_births = round(Int, model.N * model.param.birth_rate)

    # death
    if model.N - n_deaths > model.param.initial_N / 2
        death_id_vec = pick_deaths(model, n_deaths)
        survived_index = deleteat!(collect(1:model.N), death_id_vec)
        deleteat!(model.strategy_vec, death_id_vec)
        deleteat!(model.payoff_vec, death_id_vec)
        model.N -= n_deaths

        # model.graph_weights = model.graph_weights[survived_index, survived_index]
        _new_weights = fill(Float16(0.0), model.N, model.N)
        @inbounds for x = 1:model.N
            @simd for y = 1:model.N
                _new_weights[x, y] = model.graph_weights[survived_index[x], survived_index[y]]
            end
        end
        model.graph_weights = _new_weights
    else
        death_id_vec = []
    end

    # birth
    if model.N + n_births < model.param.initial_N * 2
        parent_id_vec = pick_parents(model, n_births)
        mutation_vec = rand(model.param.rng, n_births) .< model.param.μ

        for i = 1:n_births
            # strategy
            parent_strategy = model.strategy_vec[parent_id_vec[i]]
            parent_strategy = mutation_vec[i] ? invert(parent_strategy) : parent_strategy
            push!(model.strategy_vec, parent_strategy)

            # payoff
            push!(model.payoff_vec, 0.0)
        end

        # relationship
        # model.graph_weights = vcat(model.graph_weights, fill(Float16(0.0), (n_births, model.N)))
        # model.N += n_births
        # model.graph_weights = hcat(model.graph_weights, fill(Float16(0.0), (model.N, n_births)))
        # model.graph_weights[(model.N-n_births+1):end, :] = model.graph_weights[parent_id_vec, :]
        # model.graph_weights[:, (model.N-n_births+1):end] = model.graph_weights[:, parent_id_vec]
        model.N += n_births
        _new_weights = fill(Float16(0.0), (model.N, model.N))
        @inbounds for x = 1:model.N-n_births
            @simd for y = 1:model.N-n_births
                _new_weights[x, y] = model.graph_weights[x, y]
            end
        end
        @inbounds @simd for i = 1:n_births
            _new_weights[model.N-n_births+i, :] = _new_weights[parent_id_vec[i], :]
            _new_weights[:, model.N-n_births+i] = _new_weights[:, parent_id_vec[i]]
            _new_weights[parent_id_vec[i], model.N-n_births+i] = 1.0
            _new_weights[model.N-n_births+i, parent_id_vec[i]] = 1.0
            _new_weights[model.N-n_births+i, model.N-n_births+i] = 0.0
        end
        model.graph_weights = _new_weights

        normalize_graph_weights!(model)
    else
        parent_id_vec = []
    end

    return death_id_vec, parent_id_vec
end

function make_output_df(param::Param)::DataFrame
    df = DataFrame([param])
    select!(df, Not([:generations, :rng]))
    df = repeat(df, param.generations)

    df.generation .= Int16(0)
    df.N .= Int16(0)
    df.cooperation_rate .= Float16(0)
    df.payoff_μ .= Float16(0)
    df.death_rate .= Float16(0)
    df.weight_μ .= Float16(0)
    df.weight_σ .= Float16(0)
    df.k .= Float16(0)
    df.L .= Float16(0)
    df.C .= Float16(0)
    df.component_count .= Float16(0)
    df.component_size_μ .= Float16(0)
    df.component_size_max .= Float16(0)
    df.component_size_min .= Float16(0)
    df.component_size_σ .= Float16(0)
    df.strong_k .= Float16(0)
    df.strong_L .= Float16(0)
    df.strong_C .= Float16(0)
    df.strong_component_count .= Float16(0)
    df.strong_component_size_μ .= Float16(0)
    df.strong_component_size_max .= Float16(0)
    df.strong_component_size_min .= Float16(0)
    df.strong_component_size_σ .= Float16(0)

    return df
end

unweighted_graph(graph_weights::Matrix, threshold::Float64)::SimpleGraph = SimpleGraph(graph_weights .> threshold)

function log!(output::DataFrame, generation::Int, model::Model, skip::Int = 10)::Nothing
    output[generation, 12:16] = [
        generation,
        model.N,                            # population
        mean(model.strategy_vec .== C),     # 協力率
        mean(model.payoff_vec),             # 平均ペイオフ
        get_death_rate(model, generation),  # 死亡率
    ]

    (generation % skip != 0) && return

    # 重みベクトル
    weight_vec = [Float64(model.graph_weights[i, j]) for i = 1:model.N for j in 1:model.N if i < j]  # costly
    weight_μ = mean(weight_vec)
    weight_σ = std(weight_vec)

    weak_connection_g = unweighted_graph(model.graph_weights, 0.5)  # costly
    strong_connection_g = unweighted_graph(model.graph_weights, 0.75)  # costly

    # 平均次数 (<k>)
    weak_k = mean(degree(weak_connection_g))
    strong_k = mean(degree(strong_connection_g))

    # 平均距離 (L)
    # _L_vec = collect(Iterators.flatten([gdistances(simple_g, i) for i = 1:model.N]))  # most costly
    L_vec = []
    @inbounds @simd for i = 1:model.N
        append!(L_vec, gdistances(weak_connection_g, i))
    end
    L_vec = filter(x -> 0 < x <= model.N, L_vec)
    weak_L = length(L_vec) > 0 ? mean(L_vec) : 0.0

    L_vec = []
    @inbounds @simd for i = 1:model.N
        append!(L_vec, gdistances(strong_connection_g, i))
    end
    L_vec = filter(x -> 0 < x <= model.N, L_vec)
    strong_L = length(L_vec) > 0 ? mean(L_vec) : 0.0

    # 平均クラスタ係数 (C)
    weak_C = mean(local_clustering_coefficient(weak_connection_g, 1:model.N))  # costly
    strong_C = mean(local_clustering_coefficient(strong_connection_g, 1:model.N))  # costly

    # コンポーネント
    weak_components = [length(c) for c in connected_components(weak_connection_g)]
    weak_std_components = std(weak_components)
    isnan(weak_std_components) && (weak_std_components = 0.0)

    strong_components = [length(c) for c in connected_components(strong_connection_g)]
    strong_std_components = std(strong_components)
    isnan(strong_std_components) && (strong_std_components = 0.0)

    output[generation, 17:end] = [
        weight_μ,  # 17 重みの平均
        weight_σ,  # 18 重みの標準偏差
        weak_k,    # 19 平均次数 <k>
        weak_L,    # 20 平均距離 (L)
        weak_C,    # 21 平均クラスタ係数 (C)
        length(weak_components),               # 22 コンポーネント数
        mean(weak_components) / model.N,       # 23 平均コンポーネントサイズ
        maximum(weak_components) / model.N,    # 24 最大コンポーネントサイズ
        minimum(weak_components) / model.N,    # 25 最小コンポーネントサイズ
        weak_std_components / model.N,         # 26 コンポーネントサイズの標準偏差
        strong_k,  # 27 平均次数 <k>
        strong_L,  # 28 平均距離 (L)
        strong_C,  # 29 平均クラスタ係数 (C)
        length(strong_components),             # 30 コンポーネント数
        mean(strong_components) / model.N,     # 31 平均コンポーネントサイズ
        maximum(strong_components) / model.N,  # 32 最大コンポーネントサイズ
        minimum(strong_components) / model.N,  # 33 最小コンポーネントサイズ
        strong_std_components / model.N,       # 34 コンポーネントサイズの標準偏差
    ]

    return
end

function run(param::Param)::DataFrame
    model = Model(param)
    model.strategy_vec = rand(model.param.rng, [C, D], model.param.initial_N)
    output_df = make_output_df(param)

    for generation = 1:model.param.generations
        interaction!(model)
        death_and_birth!(model, generation)
        log!(output_df, generation, model)
    end

    return output_df
end

end  # end of module
