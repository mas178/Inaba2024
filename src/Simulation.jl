module Simulation

using DataFrames: DataFrame, nrow, dropmissing, select!, Not
using Graphs: SimpleGraph, degree, gdistances, local_clustering_coefficient, connected_components
using LinearAlgebra
using Random: MersenneTwister
using StatsBase: mean, std, sample, Weights

export Model, Param, C, D, POPULATION, PAYOFF, interaction!, death_and_birth!, run

@enum VariabilityMode POPULATION PAYOFF
const VARIABILITY_MODE = Dict(POPULATION => "POPULATION", PAYOFF => "PAYOFF")

@enum Strategy C D
const WEAK_THRESHOLD = 0.25
const MEDIUM_THRESHOLD = 0.50
const STRONG_THRESHOLD = 0.75

invert(s::Strategy)::Strategy = (s == C ? D : C)

"""
    ar1(β::Float64, σ::Float64, μ::Float64, T::Int, rng::MersenneTwister)::Vector{Float64}

AR(1) Model to create environmental volatility.
"""
function ar1(
    β::Float64,        # 自己回帰の係数。|β| < 1 の場合に平均回帰性を持つ。μ = α / (1 - β)
    sigma::Float64,        # std of white noise
    μ::Float64,        # expected average value
    T::Int,            # time steps
    rng::MersenneTwister,
)::Vector{Float64}
    x = Vector{Float64}(undef, T)
    x[1] = μ
    alpha = μ * (1 - β)    # 定数項。μから逆算する。
    for t = 2:T
        x[t] = alpha + β * x[t - 1] + sigma * randn(rng)
    end

    return x
end

@kwdef struct Param
    initial_N::Int = 1_000
    initial_T::Float64 = 1.1                 # Temptation payoff
    S::Float64 = -0.1                        # Sucker's payoff
    initial_graph_weight::Float64 = 0.5
    interaction_freqency::Float64 = 1.0
    Δw::Float64 = 0.1
    reproduction_rate::Float64 = 0.1
    δ::Float64 = 0.01                        # strength of selection
    μ::Float64 = 0.0                         # mutation rate
    β::Float64 = 0.1                         # environmental volatility (自己回帰の係数)
    sigma::Float64 = 0.1                     # environmental volatility (std of white noise)
    generations::Int = 100                   # generations (Time steps)
    rng::MersenneTwister = MersenneTwister() # random seed
    variability_mode::VariabilityMode = POPULATION
end

function get_N_vec(p::Param)::Vector{Int}
    p.variability_mode == POPULATION || return fill(p.initial_N, p.generations + 1)

    N_vec = ar1(p.β, p.sigma, Float64(p.initial_N), p.generations + 1, p.rng)

    for i = 2:(p.generations + 1)
        N_vec[i] = clamp(N_vec[i], p.initial_N / 10, p.initial_N * 2)
        N_vec[i] = clamp(N_vec[i], N_vec[i - 1] / 2 + 1, N_vec[i - 1] * 2 - 1)
    end

    return round.(Int, N_vec)
end

function get_death_birth_N_vec(p::Param, N_vec::Vector{Int})::Tuple{Vector{Int},Vector{Int}}
    base_N = round(Int, p.initial_N * p.reproduction_rate)
    death_N_vec = fill(base_N, p.generations)
    birth_N_vec = fill(base_N, p.generations)

    p.variability_mode == POPULATION || return death_N_vec, birth_N_vec

    for i = 1:(p.generations)
        ΔN = N_vec[i + 1] - N_vec[i]
        base_n = round(Int, N_vec[i] * p.reproduction_rate)

        if ΔN ≥ 0  # population increasing
            death_N_vec[i] = base_n
            birth_N_vec[i] = base_n + ΔN

            if death_N_vec[i] > N_vec[i] || birth_N_vec[i] > N_vec[i] - death_N_vec[i]
                death_N_vec[i] = 0
                birth_N_vec[i] = ΔN
            end
        else  # population decreasing
            birth_N_vec[i] = base_n
            death_N_vec[i] = base_n - ΔN

            if death_N_vec[i] > N_vec[i] || birth_N_vec[i] > N_vec[i] - death_N_vec[i]
                birth_N_vec[i] = 0
                death_N_vec[i] = -ΔN
            end
        end
    end

    return death_N_vec, birth_N_vec
end

function get_payoff_table_vec(p::Param)::Vector{Dict}
    if p.variability_mode == PAYOFF
        T_vec = ar1(p.β, p.sigma, p.initial_T, p.generations, p.rng)
        T_vec = [clamp(T, 0.0, 2.0) for T in T_vec]
        return [Dict((C, C) => (1.0, 1.0), (C, D) => (p.S, T), (D, C) => (T, p.S), (D, D) => (0.0, 0.0)) for T in T_vec]
    else
        payoff_table =
            Dict((C, C) => (1.0, 1.0), (C, D) => (p.S, p.initial_T), (D, C) => (p.initial_T, p.S), (D, D) => (0.0, 0.0))
        return fill(payoff_table, p.generations)
    end
end

mutable struct Model
    # model's parameters
    ## constant value
    param::Param
    ## temporal value
    generation::Int
    N::Int  # agents' population
    ## time series values
    N_vec::Vector{Int}
    interaction_N_vec::Vector{Int}
    death_N_vec::Vector{Int}
    birth_N_vec::Vector{Int}
    payoff_table_vec::Vector{Dict{Tuple{Strategy,Strategy},Tuple{Float64,Float64}}}

    # agent's parameters
    strategy_vec::Vector{Strategy}     # agents' strategy
    payoff_vec::Vector{Float64}        # agents' payoff
    graph_weights::Matrix{Float16}     # agents' relationship

    function Model(p::Param)
        N_vec = get_N_vec(p)
        (death_N_vec, birth_N_vec) = get_death_birth_N_vec(p, N_vec)

        new(
            ## constant value
            p,
            ## temporal value
            1,
            p.initial_N,
            ## time series values
            N_vec,
            round.(Int, N_vec .* p.interaction_freqency),
            death_N_vec,
            birth_N_vec,
            get_payoff_table_vec(p),
            # agent's parameters
            fill(D, p.initial_N),
            fill(0.0, p.initial_N),
            (fill(1.0, (p.initial_N, p.initial_N)) - Matrix(I, p.initial_N, p.initial_N)) * p.initial_graph_weight,
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
    payoff_table = model.payoff_table_vec[model.generation]

    focal_id_vec = sample(model.param.rng, 1:(model.N), model.interaction_N_vec[model.generation], replace = false)
    all_ids = collect(1:(model.N))
    temp_weights = Vector{Float16}(undef, model.N - 1)
    update_weights = fill(Float16(1.0), (model.N, model.N))

    @inbounds for focal_id in focal_id_vec
        # Pick an opponent
        neighbors = deleteat!(copy(all_ids), focal_id)
        @views temp_weights .= model.graph_weights[focal_id, neighbors]
        opponent_id = sample(model.param.rng, neighbors, Weights(temp_weights))

        strategy_pair = (model.strategy_vec[focal_id], model.strategy_vec[opponent_id])

        # Update payoff
        focal_payoff, opponent_payoff = payoff_table[strategy_pair]
        model.payoff_vec[focal_id] += focal_payoff
        model.payoff_vec[opponent_id] += opponent_payoff

        # for updating relationships
        increment = model.param.Δw * (strategy_pair == (C, C) ? 1 : -1)
        update_weights[focal_id, opponent_id] = update_weights[opponent_id, focal_id] *= (1.0 + increment)
    end

    # Update relationships
    model.graph_weights = min.(model.graph_weights .* update_weights, 1.0)

    return
end

classic_fitness(payoff::Float64, δ::Float64)::Float64 = 1.0 - δ + δ * payoff

sigmoid_fitness(payoff::Float64, δ::Float64)::Float64 = 1.0 / (1.0 + exp(-δ * payoff))

function pick_deaths(model::Model)::Vector{Int}
    death_N = model.death_N_vec[model.generation]
    neg_fitness_vec = sigmoid_fitness.(-model.payoff_vec, model.param.δ)
    death_id_vec = sample(model.param.rng, 1:(model.N), Weights(neg_fitness_vec), death_N, replace = false)

    return sort(death_id_vec)
end

function pick_parents(model::Model)::Vector{Int}
    birth_N = model.birth_N_vec[model.generation]
    fitness_vec = sigmoid_fitness.(model.payoff_vec, model.param.δ)
    parent_id_vec = sample(model.param.rng, 1:(model.N), Weights(fitness_vec), birth_N, replace = false)

    return sort(parent_id_vec)
end

function normalize_graph_weights!(graph_weights::Matrix{Float16}, N::Int, initial_graph_weight::Float64)::Nothing
    initial_graph_weight_sum = N * (N - 1) * initial_graph_weight

    # factor = sum(Float64.(graph_weights))
    factor = 0.0
    @inbounds @simd for weight in graph_weights
        factor += Float64(weight)
    end

    # graph_weights ./= (factor / initial_graph_weight_sum)
    factor = Float16(initial_graph_weight_sum / factor)
    @inbounds @simd for i in eachindex(graph_weights)
        graph_weights[i] *= factor
    end

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
function death_and_birth!(model::Model)::Tuple{Vector{Int},Vector{Int}}
    death_id_vec = []
    parent_id_vec = []

    # death
    death_id_vec = pick_deaths(model)
    survived_index = deleteat!(collect(1:(model.N)), death_id_vec)
    deleteat!(model.strategy_vec, death_id_vec)
    deleteat!(model.payoff_vec, death_id_vec)
    model.N -= model.death_N_vec[model.generation]

    # model.graph_weights = model.graph_weights[survived_index, survived_index]
    _new_weights = fill(Float16(0.0), model.N, model.N)
    @inbounds for x = 1:(model.N)
        @simd for y = 1:(model.N)
            _new_weights[x, y] = model.graph_weights[survived_index[x], survived_index[y]]
        end
    end
    model.graph_weights = _new_weights

    # birth
    parent_id_vec = pick_parents(model)
    birth_N = model.birth_N_vec[model.generation]
    mutation_vec = rand(model.param.rng, birth_N) .< model.param.μ

    for i = 1:birth_N
        # strategy
        parent_strategy = model.strategy_vec[parent_id_vec[i]]
        parent_strategy = mutation_vec[i] ? invert(parent_strategy) : parent_strategy
        push!(model.strategy_vec, parent_strategy)

        # payoff
        push!(model.payoff_vec, 0.0)
    end

    model.N += birth_N
    old_n = model.N - birth_N

    # relationship
    # model.graph_weights = vcat(model.graph_weights, fill(Float16(0.0), (n_births, model.N)))
    # model.N += n_births
    # model.graph_weights = hcat(model.graph_weights, fill(Float16(0.0), (model.N, n_births)))
    # model.graph_weights[(model.N-n_births+1):end, :] = model.graph_weights[parent_id_vec, :]
    # model.graph_weights[:, (model.N-n_births+1):end] = model.graph_weights[:, parent_id_vec]
    _new_weights = fill(Float16(0.0), (model.N, model.N))
    @inbounds for x = 1:old_n
        @simd for y = 1:old_n
            _new_weights[x, y] = model.graph_weights[x, y]
        end
    end
    @inbounds @simd for i = 1:birth_N
        _new_weights[old_n + i, :] = _new_weights[parent_id_vec[i], :]
        _new_weights[:, old_n + i] = _new_weights[:, parent_id_vec[i]]
        _new_weights[parent_id_vec[i], old_n + i] = 1.0
        _new_weights[old_n + i, parent_id_vec[i]] = 1.0
        _new_weights[old_n + i, old_n + i] = 0.0
    end
    model.graph_weights = _new_weights

    normalize_graph_weights!(model.graph_weights, model.N, model.param.initial_graph_weight)

    return death_id_vec, parent_id_vec
end

function initialize_column!(df::DataFrame, columns::Vector{Symbol}, value = Float16(0))
    for column in columns
        df[!, column] .= value
    end
end

function make_output_df(param::Param)::DataFrame
    # 1 〜 13
    df = DataFrame([param])
    select!(df, Not([:rng]))
    df = repeat(df, param.generations)
    df.variability_mode = fill(VARIABILITY_MODE[param.variability_mode], param.generations)
    # 14 〜 15
    initialize_column!(df, [:generation, :N], Int16(0))
    # 16 〜 18
    initialize_column!(df, [:T, :cooperation_rate, :payoff_μ])
    # 19 〜 23
    initialize_column!(df, [:weak_k1, :weak_C1, :weak_comp1_count, :weak_comp1_size_μ, :weak_comp1_size_max])
    # 24 〜 28
    initialize_column!(df, [:medium_k1, :medium_C1, :medium_comp1_count, :medium_comp1_size_μ, :medium_comp1_size_max])
    # 29 〜 33
    initialize_column!(df, [:strong_k1, :strong_C1, :strong_comp1_count, :strong_comp1_size_μ, :strong_comp1_size_max])
    # 34 〜 38
    initialize_column!(df, [:weak_k2, :weak_C2, :weak_comp2_count, :weak_comp2_size_μ, :weak_comp2_size_max])
    # 39 〜 43
    initialize_column!(df, [:medium_k2, :medium_C2, :medium_comp2_count, :medium_comp2_size_μ, :medium_comp2_size_max])
    # 44 〜 48
    initialize_column!(df, [:strong_k2, :strong_C2, :strong_comp2_count, :strong_comp2_size_μ, :strong_comp2_size_max])

    return df
end

unweighted_graph(graph_weights::Matrix, threshold::Float64)::SimpleGraph = SimpleGraph(graph_weights .> threshold)

function convert_to_2nd_order_weights(model::Model)::Matrix{Float16}
    weights2 = fill(0.0, (model.N, model.N))

    @inbounds for x = 1:(model.N)
        x_weights = @views model.graph_weights[x, :]'
        @simd for y = (x + 1):(model.N)
            # weights2[x, y] = model.graph_weights[x, y] + x_weights * model.graph_weights[:, y]
            sum = 0.0
            for i = 1:(model.N)
                sum += x_weights[i] * model.graph_weights[i, y]
            end
            weights2[x, y] = weights2[y, x] = model.graph_weights[x, y] + sum
        end
    end

    weights2 = Float16.(weights2)
    normalize_graph_weights!(weights2, model.N, model.param.initial_graph_weight)

    return weights2
end

function calc(g::SimpleGraph, model::Model)::Tuple{Float16,Float16,Float16,Float16,Float16}
    sum_C = 0.0
    sum_k = 0.0
    count = 0
    component_count = 0
    sum_component_size = 0.0
    max_component_size = 0.0
    n = length(model.strategy_vec)

    for component in connected_components(g)
        component_size = length(component)
        mean_strategy_C = mean(model.strategy_vec[x] == C for x in component)

        if component_size >= 3 && mean_strategy_C >= 0.5
            for x in component
                sum_k += degree(g, x)
                sum_C += local_clustering_coefficient(g, x)
                count += 1
            end

            component_count += 1
            sum_component_size += component_size / n
            max_component_size = max(max_component_size, component_size / n)
        end
    end

    # 平均次数
    mean_k = Float16(count == 0 ? 0.0 : sum_k / count)

    # 平均クラスタ係数
    mean_C = Float16(count == 0 ? 0.0 : sum_C / count)

    # 平均コンポーネントサイズ
    mean_component_size = Float16(component_count == 0 ? 0.0 : sum_component_size / component_count)

    # 平均次数, 平均クラスタ係数, コンポーネント数, 平均コンポーネントサイズ, 最大コンポーネントサイズ
    return mean_k, mean_C, component_count, mean_component_size, max_component_size
end

std_component_size(components::Vector{Int})::Float16 = Float16(length(components) > 1 ? std(components) : 0.0)

function log!(output::DataFrame, model::Model, level::Int = 0, skip::Int = 10)::Nothing
    output[model.generation, 14:18] = [
        model.generation,
        model.N,       # population
        model.payoff_table_vec[model.generation][(D, C)][1], # T
        mean(model.strategy_vec .== C),    # cooperation rate
        mean(model.payoff_vec),            # average payoff
    ]

    level == 0 && return

    (model.generation % skip != 0) && return

    # 1st order connection
    weak_connection_g = unweighted_graph(model.graph_weights, WEAK_THRESHOLD)  # costly
    medium_connection_g = unweighted_graph(model.graph_weights, MEDIUM_THRESHOLD)  # costly
    strong_connection_g = unweighted_graph(model.graph_weights, STRONG_THRESHOLD)  # costly

    # 平均次数, 平均クラスタ係数, コンポーネント数, 平均コンポーネントサイズ, 最大コンポーネントサイズ
    output[model.generation, 19:23] = calc(weak_connection_g, model)
    output[model.generation, 24:28] = calc(medium_connection_g, model)
    output[model.generation, 29:33] = calc(strong_connection_g, model)

    level <= 1 && return

    # 2nd order connection
    second_order_weights = convert_to_2nd_order_weights(model)
    weak_connection2_g = unweighted_graph(second_order_weights, WEAK_THRESHOLD)  # costly
    medium_connection2_g = unweighted_graph(second_order_weights, MEDIUM_THRESHOLD)  # costly
    strong_connection2_g = unweighted_graph(second_order_weights, STRONG_THRESHOLD)  # costly

    # 平均次数, 平均クラスタ係数, コンポーネント数, 平均コンポーネントサイズ, 最大コンポーネントサイズ
    output[model.generation, 34:38] = calc(weak_connection2_g, model)
    output[model.generation, 39:43] = calc(medium_connection2_g, model)
    output[model.generation, 44:48] = calc(strong_connection2_g, model)

    return
end

function run(param::Param; log_level::Int = 0, log_rate::Float64 = 0.5)::DataFrame
    model = Model(param)
    output_df = make_output_df(param)

    for generation = 1:(param.generations)
        model.generation = generation
        model.N = model.N_vec[generation]
        model.payoff_vec .= 0.0

        interaction!(model)
        death_and_birth!(model)

        if generation > param.generations * (1 - log_rate)
            log!(output_df, model, log_level)
        end
    end

    return output_df
end

end  # end of module
