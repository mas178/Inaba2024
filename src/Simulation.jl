module Simulation

using Random: MersenneTwister
using StatsBase: mean, std, sample, Weights

include("../src/Network.jl")
using .Network

export Param, Model, C, D, cooperation_rate, run_one_step!, run

@enum Strategy C D

const PayoffTable = Dict{Tuple{Strategy, Strategy}, Tuple{Float64, Float64}}

const RunOutput = Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}

@kwdef struct Param
    k₁::Int = 10
    w₁::Float16 = Float16(0.5)
    Δw::Float64 = 0.3
    reproduction_rate::Float64 = 0.05
    δ::Float64 = 1.0     # strength of selection
    μ::Float64 = 0.01    # mutation rate of strategy
    t_max::Int = 100     # Time steps
    trial_max::Int = 10  # Trial count
    rng::MersenneTwister = MersenneTwister() # random seed
end

mutable struct Model
    t::Int
    param::Param

    # environmental variables
    N_vec::Vector{Int}
    payoff_table_vec::Vector{PayoffTable}

    # agent's parameters
    strategy_vec::Vector{Strategy}  # agents' strategy
    payoff_vec::Vector{Float64}     # agents' payoff
    weights::Matrix{Float16}        # agents' relationship

    function Model(param::Param, N_vec::Vector{Int}, payoff_table_vec::Vector{PayoffTable})
        N₁ = N_vec[1]

        new(
            1,  # Time step
            param,
            N_vec,
            payoff_table_vec,

            # agent's parameters
            rand(param.rng, [C, D], N₁),
            fill(0.0, N₁),
            create_adjacency_matrix(N₁, param.k₁, param.w₁),
        )
    end
end

function interaction!(model::Model, rng::MersenneTwister)::Nothing
    N = model.N_vec[model.t]
    payoff_table = model.payoff_table_vec[model.t]

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
        update_weight!(model.weights, focal_id, opponent_id, Float16(clamp(new_weight, 0.0, 1.0)))
    end

    return
end

sigmoid_fitness(payoff::Float64, δ::Float64)::Float64 = 1.0 / (1.0 + exp(-δ * payoff))

function pick_agents(model::Model, n::Int, method::Symbol, rng::MersenneTwister)::Vector{Int}
    fitness_vec = if method == :payoff_positive
        sigmoid_fitness.(model.payoff_vec, model.param.δ)
    elseif method == :payoff_negative
        sigmoid_fitness.(-model.payoff_vec, model.param.δ)
    elseif method == :degree
        degree(model.weights)
    elseif method == :weight
        vec(sum(model.weights, dims = 1))
    else
        error("Invalid method: $method. Use :payoff_positive, :payoff_negative, :degree, or :weight.")
    end

    N = length(fitness_vec)
    count(fitness_vec .> 0) <= n && return filter(x -> fitness_vec[x] > 0, 1:N)
    agent_id_vec = sample(rng, 1:N, Weights(fitness_vec), n, replace = false)

    return sort(agent_id_vec)
end

invert(s::Strategy)::Strategy = (s == C ? D : C)

function get_n_death_birth(model::Model)::Tuple{Int, Int}
    n_death = n_birth = round(Int, model.N_vec[model.t] * model.param.reproduction_rate)
    n_growth = model.N_vec[model.t + 1] - model.N_vec[model.t]

    if n_growth > 0
        n_birth += n_growth
    elseif n_growth < 0
        n_death += abs(n_growth)
    end

    return n_death, n_birth
end

function death!(model::Model, n_death::Int, rng::MersenneTwister)::Vector{Int}
    death_id_vec = pick_agents(model, n_death, :payoff_negative, rng)

    deleteat!(model.strategy_vec, death_id_vec)
    deleteat!(model.payoff_vec, death_id_vec)
    model.weights = rem_vertices(model.weights, death_id_vec)

    return death_id_vec
end

function birth!(model::Model, n_birth::Int, rng::MersenneTwister)::Vector{Int}
    parent_id_vec = pick_agents(model, n_birth, :payoff_positive, rng)

    append!(model.strategy_vec, fill(D, n_birth))
    append!(model.payoff_vec, zeros(Float64, n_birth))
    model.weights = add_vertices(model.weights, n_birth)

    return parent_id_vec
end

function imitate_relationship_conformist_bias!(model::Model, rng::MersenneTwister)::Nothing
    N = nv(model.weights)

    # calc edge shortage
    as_is_edge_count = ne(model.weights)
    to_be_edge_count = round(Int, N * model.param.k₁ / 2)
    edge_shortage = to_be_edge_count - as_is_edge_count
    edge_shortage <= 0 && return

    # pick solitary nodes
    orphan_id_vec = findall(==(0), vec(sum(model.weights, dims = 1)))
    length(orphan_id_vec) <= 0 && return

    # k = max(round(Int, edge_shortage / length(orphan_id_vec)), model.param.k₁)
    k = round(Int, edge_shortage / length(orphan_id_vec))

    for orphan_id in orphan_id_vec
        # for popular_id in pick_agents(model, k, :degree, rng)
        for popular_id in pick_agents(model, k, :weight, rng)
            if popular_id != orphan_id
                update_weight!(model.weights, popular_id, orphan_id, model.param.w₁)
            end
        end
    end

    return
end

function imitate_strategy_conformist_bias!(model::Model, n_birth::Int, rng::MersenneTwister)::Nothing
    N = nv(model.weights)

    for i in (N - n_birth + 1):N
        model.strategy_vec[i] = local_cooperation_rate(model, i) > 0.5 ? C : D

        # mutation
        if rand(rng) < model.param.μ
            model.strategy_vec[i] = invert(model.strategy_vec[i])
        end
    end

    return
end

function imitate_strategy_payoff_bias!(model::Model, parent_id_vec::Vector{Int}, rng::MersenneTwister)::Nothing
    # parent_id_vec がペイオフバイアスで選択されていることが前提となっている。
    N = nv(model.weights)
    n_birth = length(parent_id_vec)
    child_id_vec = collect((N - n_birth + 1):N)

    for (parent_id, child_id) in zip(parent_id_vec, child_id_vec)
        model.strategy_vec[child_id] = model.strategy_vec[parent_id]

        # mutation
        if rand(rng) < model.param.μ
            model.strategy_vec[child_id] = invert(model.strategy_vec[child_id])
        end
    end

    return
end

function run_one_step!(model::Model, t::Int)::Nothing
    model.t = t
    model.payoff_vec .= 0.0

    interaction!(model, model.param.rng)
    n_death, n_birth = get_n_death_birth(model)
    death!(model, n_death, model.param.rng)
    parent_id_vec = birth!(model, n_birth, model.param.rng)
    imitate_relationship_conformist_bias!(model, model.param.rng)
    imitate_strategy_conformist_bias!(model, n_birth, model.param.rng)
    # imitate_strategy_payoff_bias!(model, parent_id_vec, model.param.rng)
end

cooperation_rate(model::Model)::Float64 = sum(model.strategy_vec .== C) / length(model.strategy_vec)

function local_cooperation_rate(model::Model, node_id::Int)::Float64
    neighbor_id_vec = neighbors(model.weights, node_id)
    n_neighbor = length(neighbor_id_vec)
    n_neighbor == 0 && return 0.0

    n_neighbor_C = count(model.strategy_vec[neighbor_id_vec] .== C)

    return n_neighbor_C / n_neighbor
end

function local_cooperation_rate(model::Model)::Float64
    cooperation_rate(model) == 0.0 && return 0.0

    N = nv(model.weights)
    C_index_vec = filter(node_id -> model.strategy_vec[node_id] == C, 1:N)
    local_cooperation_rate_vec = [local_cooperation_rate(model, node_id) for node_id in C_index_vec]

    return mean(local_cooperation_rate_vec)
end

function run(param::Param, N_vec::Vector{Int}, payoff_table_vec::Vector{PayoffTable})::RunOutput
    @assert param.t_max + 1 == length(N_vec)
    @assert param.t_max == length(payoff_table_vec)

    cooperation_rate_matrix = fill(0.0, param.t_max, param.trial_max)
    mean_degree_matrix = fill(0.0, param.t_max, param.trial_max)
    std_degree_matrix = fill(0.0, param.t_max, param.trial_max)
    local_cooperation_rate_matrix = fill(0.0, param.t_max, param.trial_max)

    for trial in 1:param.trial_max
        model = Model(param, N_vec, payoff_table_vec)

        for t = 1:(model.param.t_max)
            run_one_step!(model, t)

            cooperation_rate_matrix[t, trial] = cooperation_rate(model)
            degree_vec = degree(model.weights)
            mean_degree_matrix[t, trial] = mean(degree_vec)
            std_degree_matrix[t, trial] = std(degree_vec)
            local_cooperation_rate_matrix[t, trial] = local_cooperation_rate(model)
        end
    end

    return cooperation_rate_matrix, mean_degree_matrix, std_degree_matrix, local_cooperation_rate_matrix
end

function run(param::Param, N_vec::Vector{Int}, T::Float64, S::Float64)::RunOutput
    payoff_table = Dict((C, C) => (1.0, 1.0), (C, D) => (S, T), (D, C) => (T, S), (D, D) => (0.0, 0.0))
    payoff_table_vec = fill(payoff_table, param.t_max)

    return run(param, N_vec, payoff_table_vec)
end

end  # end of module
