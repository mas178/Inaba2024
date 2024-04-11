module Simulation

using Random: MersenneTwister
using StatsBase: mean, std, sample, Weights
using Graphs
using DataFrames: DataFrame, select!, Not

include("../src/Network.jl")
using .Network:
    nv,
    add_vertices,
    rem_vertices,
    update_weight!,
    rem_edge!,
    create_adjacency_matrix,
    weights_to_network,
    convert_2nd_order,
    normalize_degree!,
    normalize_weight!

@enum VariabilityMode POPULATION PAYOFF STRATEGY_MUTATION RELATIONSHIP_MUTATION
const VARIABILITY_MODE = Dict(
    POPULATION => "POPULATION",
    PAYOFF => "PAYOFF",
    STRATEGY_MUTATION => "STRATEGY_MUTATION",
    RELATIONSHIP_MUTATION => "RELATIONSHIP_MUTATION",
)

@enum Strategy C D

const WEAK_THRESHOLD = 0.25
const MEDIUM_THRESHOLD = 0.50
const STRONG_THRESHOLD = 0.75

"""
    ar1(β::Float64, σ::Float64, μ::Float64, T::Int, rng::MersenneTwister)::Vector{Float64}

AR(1) Model to create environmental volatility.
"""
function ar1(
    β::Float64,        # 自己回帰の係数。|β| < 1 の場合に平均回帰性を持つ。μ = α / (1 - β)
    σ::Float64,        # std of white noise
    τ::Int,            # time scale
    μ::Float64,        # expected average value
    T::Int,            # time steps
    rng::MersenneTwister,
)::Vector{Float64}
    @assert T % τ == 0

    t_max = T ÷ τ
    x = Vector{Float64}(undef, t_max)
    x[1] = μ
    alpha = μ * (1 - β)    # 定数項。μから逆算する。
    for t = 2:t_max
        x[t] = alpha + β * x[t - 1] + σ * randn(rng)
    end

    return repeat(x, inner = τ)[1:T]
end

ev_determined(σ::Float64, τ::Int, μ::Float64, T::Int)::Vector{Float64} = [t % 2τ < τ ? -1 : 1 for t = 1:T] .* σ .+ μ

@kwdef struct Param
    initial_N::Int = 1_000
    initial_k::Int = 10
    initial_T::Float64 = 1.1    # Temptation payoff
    S::Float64 = -0.1           # Sucker's payoff
    initial_w::Float16 = Float16(0.5)
    Δw::Float64 = 0.1
    reproduction_rate::Float64 = 0.1
    δ::Float64 = 0.01           # strength of selection
    initial_μ_s::Float64 = 0.0  # mutation rate of strategy
    initial_μ_r::Float64 = 0.0  # mutation rate of relationship
    β::Float64 = 0.1            # environmental volatility (自己回帰係数)
    σ::Float64 = 0.1            # environmental volatility (std of white noise)
    τ::Int = 10                 # environmental volatility (time scale)
    generations::Int = 100      # generations (Time steps)
    variability_mode::VariabilityMode = POPULATION
    rng::MersenneTwister = MersenneTwister() # random seed
end

function get_N_vec(p::Param, variable::Bool)::Vector{Int}
    variable || return fill(p.initial_N, p.generations + 1)

    N_vec = ar1(p.β, p.σ, p.τ, Float64(p.initial_N), p.generations, p.rng)
    push!(N_vec, N_vec[end])

    for i = 2:(p.generations + 1)
        lower_bound = max(p.initial_N / 10, N_vec[i - 1] / 2 + 1)
        upper_bound = min(p.initial_N * 2, N_vec[i - 1] * 2 - 1)
        N_vec[i] = clamp(N_vec[i], lower_bound, upper_bound)
    end

    return round.(Int, N_vec)
end

function get_death_birth_N_vec(p::Param, N_vec::Vector{Int}, variable::Bool)::Tuple{Vector{Int},Vector{Int}}
    base_N = round(Int, p.initial_N * p.reproduction_rate)
    death_N_vec = fill(base_N, p.generations)
    birth_N_vec = fill(base_N, p.generations)

    variable || return death_N_vec, birth_N_vec

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

function get_payoff_table_vec(p::Param, variable::Bool)::Vector{Dict}
    if variable
        T_vec = ar1(p.β, p.σ, p.τ, p.initial_T, p.generations, p.rng)
        T_vec = clamp.(T_vec, p.initial_T - (2.0 - p.initial_T), 2.0)
        return [Dict((C, C) => (1.0, 1.0), (C, D) => (p.S, T), (D, C) => (T, p.S), (D, D) => (0.0, 0.0)) for T in T_vec]
    else
        payoff_table =
            Dict((C, C) => (1.0, 1.0), (C, D) => (p.S, p.initial_T), (D, C) => (p.initial_T, p.S), (D, D) => (0.0, 0.0))
        return fill(payoff_table, p.generations)
    end
end

function get_μ_vec(p::Param, initial_μ::Float64, variable::Bool)::Vector{Float16}
    variable || return fill(Float16(initial_μ), p.generations)

    μ_vec = ar1(p.β, p.σ, p.τ, initial_μ, p.generations, p.rng)
    μ_vec = clamp.(μ_vec, 0.0, initial_μ * 2)
    return Float16.(μ_vec)
end

mutable struct Model
    param::Param
    generation::Int

    # environmental variables
    death_N_vec::Vector{Int}
    birth_N_vec::Vector{Int}
    payoff_table_vec::Vector{Dict{Tuple{Strategy,Strategy},Tuple{Float64,Float64}}}
    μ_s_vec::Vector{Float16}
    μ_r_vec::Vector{Float16}

    # agent's parameters
    strategy_vec::Vector{Strategy}     # agents' strategy
    payoff_vec::Vector{Float64}        # agents' payoff
    weights::Matrix{Float16}

    function Model(param::Param)
        N_vec = get_N_vec(param, param.variability_mode == POPULATION)
        (death_N_vec, birth_N_vec) = get_death_birth_N_vec(param, N_vec, param.variability_mode == POPULATION)

        new(
            param,
            1,  # generation

            # environmental variables
            death_N_vec,
            birth_N_vec,
            get_payoff_table_vec(param, param.variability_mode == PAYOFF),
            get_μ_vec(param, param.initial_μ_s, param.variability_mode == STRATEGY_MUTATION),
            get_μ_vec(param, param.initial_μ_r, param.variability_mode == RELATIONSHIP_MUTATION),

            # agent's parameters
            fill(D, param.initial_N),
            fill(0.0, param.initial_N),
            create_adjacency_matrix(param.initial_N, param.initial_k, param.initial_w),
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
function interaction!(model::Model, rng::MersenneTwister)::Nothing
    N = nv(model.weights)
    payoff_table = model.payoff_table_vec[model.generation]

    weights_vec = Weights.(eachcol(model.weights))
    opponent_id_vec = sample.(Ref(rng), weights_vec)

    for (focal_id, opponent_id) in zip(1:N, opponent_id_vec)
        # strategy
        strategy_pair = (model.strategy_vec[focal_id], model.strategy_vec[opponent_id])

        # payoff
        focal_payoff, opponent_payoff = payoff_table[strategy_pair]
        model.payoff_vec[focal_id] += focal_payoff
        model.payoff_vec[opponent_id] += opponent_payoff

        # graph
        new_weight =
            model.weights[focal_id, opponent_id] * (1.0 + model.param.Δw * (strategy_pair == (C, C) ? +1.0 : -1.0))
        update_weight!(model.weights, focal_id, opponent_id, Float16(new_weight))
    end

    return
end

classic_fitness(payoff::Float64, δ::Float64)::Float64 = 1.0 - δ + δ * payoff

sigmoid_fitness(payoff::Float64, δ::Float64)::Float64 = 1.0 / (1.0 + exp(-δ * payoff))

function pick_deaths(model::Model, rng::MersenneTwister)::Vector{Int}
    N = nv(model.weights)
    death_N = model.death_N_vec[model.generation]
    neg_fitness_vec = sigmoid_fitness.(-model.payoff_vec, model.param.δ)
    death_id_vec = sample(rng, 1:N, Weights(neg_fitness_vec), death_N, replace = false)

    return sort(death_id_vec)
end

function pick_parents(model::Model, rng::MersenneTwister)::Vector{Int}
    N = nv(model.weights)
    birth_N = model.birth_N_vec[model.generation]
    fitness_vec = sigmoid_fitness.(model.payoff_vec, model.param.δ)
    parent_id_vec = sample(rng, 1:N, Weights(fitness_vec), birth_N, replace = false)

    return sort(parent_id_vec)
end

"""
    death!(model::Model)::Vector{Int}

- Some agents die based on fitness.
"""
function death!(model::Model, rng::MersenneTwister)::Vector{Int}
    death_id_vec = pick_deaths(model, rng)
    deleteat!(model.strategy_vec, death_id_vec)
    deleteat!(model.payoff_vec, death_id_vec)
    model.weights = rem_vertices(model.weights, death_id_vec)

    # connect orphan nodes to a random node if they exist.
    N = nv(model.weights)
    orphan_vec = findall(==(0), vec(sum(model.weights, dims = 1)))

    for orphan in orphan_vec
        adoptive_parent = rand(rng, setdiff(1:N, [orphan]))
        update_weight!(model.weights, orphan, adoptive_parent, Float16(1.0))
    end

    return death_id_vec
end

invert(s::Strategy)::Strategy = (s == C ? D : C)

"""
    birth!(model::Model)::Vector{Int}

- Some agents reproduce based on fitness.
- Child copies the parent's strategy.
- Child copies the parent's relationship.
- Strategy is mutated based on mutation rate.
"""
function birth!(model::Model, rng::MersenneTwister)::Vector{Int}
    parent_id_vec = pick_parents(model, rng)
    birth_N = length(parent_id_vec)
    N = nv(model.weights)
    child_id_vec = collect((N + 1):(N + birth_N))

    # strategy
    mutate_vec = rand(rng, birth_N) .< model.μ_s_vec[model.generation]
    strategy_vec = model.strategy_vec[parent_id_vec]
    strategy_vec = [mutate ? invert(strategy) : strategy for (strategy, mutate) in zip(strategy_vec, mutate_vec)]
    append!(model.strategy_vec, strategy_vec)

    # payoff
    append!(model.payoff_vec, zeros(Float64, birth_N))

    # graph
    model.weights = add_vertices(model.weights, birth_N)

    ## inherit parent's connection
    @inbounds for (parent_id, child_id) in zip(parent_id_vec, child_id_vec)
        model.weights[child_id, :] = model.weights[:, child_id] = model.weights[parent_id, :]
        update_weight!(model.weights, parent_id, child_id, Float16(1.0))
        update_weight!(model.weights, child_id, child_id, Float16(0.0))
    end

    ## mutation
    N = nv(model.weights)
    for (parent_id, child_id) in zip(parent_id_vec, child_id_vec)
        neighbor_vec =
            [neighbor_id for neighbor_id in 1:N if neighbor_id != parent_id && model.weights[child_id, neighbor_id] > 0]
        for neighbor_id in neighbor_vec
            if rand(rng) < model.μ_r_vec[model.generation]
                alien_id = child_id
                while alien_id in [child_id, neighbor_vec...]
                    alien_id = rand(rng, 1:N)
                end
                update_weight!(model.weights, child_id, alien_id, model.weights[child_id, neighbor_id])
                rem_edge!(model.weights, child_id, neighbor_id)
            end
        end
    end

    return parent_id_vec
end

function initialize_column!(df::DataFrame, columns::Vector{Symbol}, value = Float16(0))
    for column in columns
        df[!, column] .= value
    end
end

function make_output_df(param::Param, log_level::Int, log_count::Int)::DataFrame
    # 1 〜 15
    df = DataFrame([param])
    select!(df, Not([:rng]))
    df = repeat(df, log_count)
    df.variability_mode .= VARIABILITY_MODE[param.variability_mode]
    # 16 〜 17
    initialize_column!(df, [:generation, :N], Int16(0))
    # 18 〜 20
    initialize_column!(df, [:T, :cooperation_rate, :payoff_μ])

    log_level == 0 && return df

    # 21 〜 25
    initialize_column!(df, [:weak_k1, :weak_C1, :weak_comp1_count, :weak_comp1_size_μ, :weak_comp1_size_max])
    # 26 〜 30
    initialize_column!(df, [:medium_k1, :medium_C1, :medium_comp1_count, :medium_comp1_size_μ, :medium_comp1_size_max])
    # 31 〜 35
    initialize_column!(df, [:strong_k1, :strong_C1, :strong_comp1_count, :strong_comp1_size_μ, :strong_comp1_size_max])

    log_level <= 1 && return df

    # 36 〜 40
    initialize_column!(df, [:weak_k2, :weak_C2, :weak_comp2_count, :weak_comp2_size_μ, :weak_comp2_size_max])
    # 41 〜 45
    initialize_column!(df, [:medium_k2, :medium_C2, :medium_comp2_count, :medium_comp2_size_μ, :medium_comp2_size_max])
    # 46 〜 50
    initialize_column!(df, [:strong_k2, :strong_C2, :strong_comp2_count, :strong_comp2_size_μ, :strong_comp2_size_max])

    return df
end

function calc(g::SimpleGraph, model::Model)::Tuple{Float16,Float16,Float16,Float16,Float16}
    sum_C = 0.0
    sum_k = 0.0
    count = 0
    component_count = 0
    sum_component_size = 0.0
    max_component_size = 0.0
    n = Graphs.nv(g)

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

function log!(output::DataFrame, model::Model, log_level::Int, log_row_n::Int)::Nothing
    output[log_row_n, 16:20] = [
        model.generation,
        nv(model.weights),       # population
        model.payoff_table_vec[model.generation][(D, C)][1], # T
        mean(model.strategy_vec .== C),    # cooperation rate
        mean(model.payoff_vec),            # average payoff
    ]

    log_level == 0 && return

    # 1st order connection
    weak_connection_g = weights_to_network(model.weights, WEAK_THRESHOLD)  # costly
    medium_connection_g = weights_to_network(model.weights, MEDIUM_THRESHOLD)  # costly
    strong_connection_g = weights_to_network(model.weights, STRONG_THRESHOLD)  # costly

    # 平均次数, 平均クラスタ係数, コンポーネント数, 平均コンポーネントサイズ, 最大コンポーネントサイズ
    output[log_row_n, 21:25] = calc(weak_connection_g, model)
    output[log_row_n, 26:30] = calc(medium_connection_g, model)
    output[log_row_n, 31:35] = calc(strong_connection_g, model)

    log_level <= 1 && return

    # 2nd order connection
    second_order_weighted = convert_2nd_order(model.weights)
    weak_connection2_g = weights_to_network(second_order_weighted, WEAK_THRESHOLD)  # costly
    medium_connection2_g = weights_to_network(second_order_weighted, MEDIUM_THRESHOLD)  # costly
    strong_connection2_g = weights_to_network(second_order_weighted, STRONG_THRESHOLD)  # costly

    # 平均次数, 平均クラスタ係数, コンポーネント数, 平均コンポーネントサイズ, 最大コンポーネントサイズ
    output[log_row_n, 36:40] = calc(weak_connection2_g, model)
    output[log_row_n, 41:45] = calc(medium_connection2_g, model)
    output[log_row_n, 46:50] = calc(strong_connection2_g, model)

    return
end

function fast_run(param::Param; log_rate::Float64 = 0.2, log_skip::Int = 10)::Float64
    model = Model(param)
    start_gen = floor(Int, model.param.generations * (1 - log_rate)) + 1
    log_generations = filter(x -> x % log_skip == 0, start_gen:(model.param.generations))
    log_row_n = 1

    c_rate_vec = fill(0.0, length(log_generations))

    for generation = 1:(model.param.generations)
        model.generation = generation
        model.payoff_vec .= 0.0

        interaction!(model, model.param.rng)
        death!(model, model.param.rng)
        birth!(model, model.param.rng)

        # normalize_degree!(model.weights, model.param.initial_k)
        normalize_weight!(model.weights, model.param.initial_k, model.param.initial_w)

        if generation ∈ log_generations
            c_rate_vec[log_row_n] = mean(model.strategy_vec .== C)
            log_row_n += 1
        end
    end

    return round(mean(c_rate_vec), digits = 5)
end

function run(model::Model; log_level::Int = 0, log_rate::Float64 = 0.5, log_skip::Int = 10)::DataFrame
    start_gen = floor(Int, model.param.generations * (1 - log_rate)) + 1
    log_generations = filter(x -> x % log_skip == 0, start_gen:(model.param.generations))
    log_row_n = 1

    output_df = make_output_df(model.param, log_level, length(log_generations))

    for generation = 1:(model.param.generations)
        model.generation = generation
        model.payoff_vec .= 0.0

        interaction!(model, model.param.rng)
        death!(model, model.param.rng)
        birth!(model, model.param.rng)

        # normalize_degree!(model.weights, model.param.initial_k)
        normalize_weight!(model.weights, model.param.initial_k, model.param.initial_w)

        if generation ∈ log_generations
            log!(output_df, model, log_level, log_row_n)
            log_row_n += 1
        end
    end

    return output_df
end

function run(param::Param; log_level::Int = 0, log_rate::Float64 = 0.5, log_skip::Int = 10)::DataFrame
    return run(Model(param), log_level = log_level, log_rate = log_rate, log_skip = log_skip)
end

end  # end of module
