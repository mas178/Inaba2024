module Simulation
"""
この様なモデルの構築を目指す。
- 人口が大きい状態が維持されると、複数の小集団が維持される。
- 人口が小さい状態が維持されると、一つの小集団が維持される。
- 人口が小さい状態から大きい状態に変化すると、一つの大集団がしばらく維持されるが、やがて崩壊する。

そのために、「友達には協力し、敵には協力しない」という相互作用ロジックを試してみる。
"""

using Random: MersenneTwister
using StatsBase: sample, Weights
using LinearAlgebra: I

include("../src/Network.jl")
using .Network

export Param, Model, C, D, cooperation_rate, run_one_step!, run

@enum Strategy C D

const PayoffTable = Dict{Tuple{Strategy, Strategy}, Tuple{Float64, Float64}}

@kwdef struct Param
    Δw::Float64 = 0.1
    db_rate::Float64 = 0.05 # standard death / birth rate.
    δ::Float64 = 1.0        # strength of selection
    t_max::Int = 100        # Time steps
    trial_max::Int = 10     # Trial count
end

mutable struct Model
    t::Int
    p::Param

    # environmental variables
    N_vec::Vector{Int}
    payoff_table_vec::Vector{PayoffTable}
    rng::MersenneTwister

    # agent's parameters
    payoff_vec::Vector{Float64}     # agents' payoff
    weights::Matrix{Float16}        # agents' relationship

    function Model(p::Param, N_vec::Vector{Int}, payoff_table_vec::Vector{PayoffTable}, rng::MersenneTwister = MersenneTwister())
        N = N_vec[1]
        w₀ = 0.5

        new(
            1,  # Time step
            p,
            N_vec,
            payoff_table_vec,
            rng,

            # agent's parameters
            fill(0.0, N),
            Float16.((ones(N, N) - I) * w₀)
        )
    end
end

function interaction!(model::Model)::Nothing
    N = model.N_vec[model.t]
    payoff_table = model.payoff_table_vec[model.t]

    weights_vec = Weights.(eachcol(model.weights))
    opponent_id_vec = sample.(Ref(model.rng), weights_vec)

    for (focal_id, opponent_id) in zip(1:N, opponent_id_vec)
        # strategy
        weight = model.weights[focal_id, opponent_id]
        focal_strategy = weight > rand(model.rng) ? C : D
        opponent_strategy = weight > rand(model.rng) ? C : D
        strategy_pair = (focal_strategy, opponent_strategy)

        # payoff
        focal_payoff, opponent_payoff = payoff_table[strategy_pair]
        model.payoff_vec[focal_id] += focal_payoff
        model.payoff_vec[opponent_id] += opponent_payoff

        # graph
        new_weight =
            model.weights[focal_id, opponent_id] * (1.0 + model.p.Δw * (strategy_pair == (C, C) ? +1 : -1))
        update_weight!(model.weights, focal_id, opponent_id, Float16(clamp(new_weight, 0.0, 1.0)))
    end

    return
end

sigmoid_fitness(payoff::Float64, δ::Float64)::Float64 = 1.0 / (1.0 + exp(-δ * payoff))

function pick_agents(model::Model, n::Int, method::Symbol)::Vector{Int}
    fitness_vec = if method == :payoff_positive
        sigmoid_fitness.(model.payoff_vec, model.p.δ)
    elseif method == :payoff_negative
        sigmoid_fitness.(-model.payoff_vec, model.p.δ)
    else
        error("Invalid method: $method.")
    end

    N = length(fitness_vec)
    count(fitness_vec .> 0) <= n && return filter(x -> fitness_vec[x] > 0, 1:N)
    agent_id_vec = sample(model.rng, 1:N, Weights(fitness_vec), n, replace = false)

    return sort(agent_id_vec)
end

invert(s::Strategy)::Strategy = (s == C ? D : C)

function get_n_death_birth(model::Model)::Tuple{Int, Int}
    n_death = n_birth = round(Int, model.N_vec[model.t] * model.p.db_rate)
    n_growth = model.N_vec[model.t + 1] - model.N_vec[model.t]

    if n_growth > 0
        n_birth += n_growth
    elseif n_growth < 0
        n_death += abs(n_growth)
    end

    return n_death, n_birth
end

function death!(model::Model, n_death::Int)::Vector{Int}
    death_id_vec = pick_agents(model, n_death, :payoff_negative)
    deleteat!(model.payoff_vec, death_id_vec)
    model.weights = rem_vertices(model.weights, death_id_vec)

    return death_id_vec
end

function birth!(model::Model, n_birth::Int)::Vector{Int}
    parent_id_vec = pick_agents(model, n_birth, :payoff_positive)
    append!(model.payoff_vec, zeros(Float64, n_birth))
    model.weights = add_vertices(model.weights, n_birth)

    return parent_id_vec
end

function imitate_relationship_kin_bias!(model::Model, parent_id_vec::Vector{Int})::Nothing
    N = nv(model.weights)
    n_birth = length(parent_id_vec)
    child_id_vec = collect((N - n_birth + 1):N)

    for (parent_id, child_id) in zip(parent_id_vec, child_id_vec)
        model.weights[child_id, :] = model.weights[parent_id, :]
        model.weights[:, child_id] = model.weights[:, parent_id]
        update_weight!(model.weights, child_id, parent_id, Float16(1))
        update_weight!(model.weights, child_id, child_id, Float16(0))
    end

    return
end

function normalize_weight!(model::Model)::Nothing
    N = nv(model.weights)
    to_be_weights_sum = N * (N - 1) * 0.5
    as_is_weights_sum = sum(Float64.(model.weights))
    model.weights ./= Float16(as_is_weights_sum / to_be_weights_sum)

    return
end

function run_one_step!(model::Model, t::Int)::Nothing
    model.t = t
    model.payoff_vec .= 0.0

    interaction!(model)
    n_death, n_birth = get_n_death_birth(model)
    death!(model, n_death)
    parent_id_vec = birth!(model, n_birth)
    imitate_relationship_kin_bias!(model, parent_id_vec)
    normalize_weight!(model)
end

function run(p::Param, N_vec::Vector{Int}, payoff_table_vec::Vector{PayoffTable})::Matrix{Matrix{Int}}
    @assert p.t_max + 1 == length(N_vec)
    @assert p.t_max == length(payoff_table_vec)

    weights_matrix = fill(fill(0, 1, 1), p.t_max, p.trial_max)

    for trial in 1:p.trial_max
        model = Model(p, N_vec, payoff_table_vec)

        for t = 1:(model.p.t_max)
            run_one_step!(model, t)
            weights_matrix[t, trial] = (model.weights .> 0.5)
        end
    end

    return weights_matrix
end

function run(p::Param, N_vec::Vector{Int}, T::Float64, S::Float64)::Matrix{Matrix{Int}}
    payoff_table = Dict((C, C) => (1.0, 1.0), (C, D) => (S, T), (D, C) => (T, S), (D, D) => (0.0, 0.0))
    payoff_table_vec = fill(payoff_table, p.t_max)

    return run(p, N_vec, payoff_table_vec)
end

end  # end of module
