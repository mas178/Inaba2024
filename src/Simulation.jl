module Simulation

using Graphs
using Parameters
using Random: MersenneTwister
using StatsBase: shuffle, Weights, sample, mean, std

include("../src/EnvironmentalVariability.jl")
using .EnvironmentalVariability: ar1

include("../src/Network.jl")
using .Network: create_weighted_cycle_graph, mat_ne, mat_update_weight!, mat_degree, average_distance

@enum Strategy C D

@with_kw struct Param
    # Spatial Structure
    N::Int = 100       # population
    k₀::Int = 4        # initial degree
    w₀::Float64 = 1.0  # initial weight

    # Interaction
    C_rate₀::Float64 = 0.5     # initial cooperator(C)'s freqtrialscy
    b::Float64 = 1.25          # benefit to defect
    relationship_increment_factor::Float64 = 1.1

    # Environmental Variability
    peak_node_resource::Float64  = 1.0
    resource_decrement_factor::Float64 = 0.02
    peak_node_variability::Int = 1   # Movable range of prime node (EVMode: PEAK_NODE, BOTH)
    resource_limit_μ::Float64 = 0.5  # expected value of resource limit
    resource_limit_β::Float64 = 0.1  # Autoregressive coefficient of resource limit
    resource_limit_σ::Float64 = 0.1  # SD of white noise of resource limit

    # Imitation
    δ::Float64 = 0.99  # selection pressure
    μ::Float64 = 0.01  # mutation rate

    # Misc
    generations::Int = 2_000  # time steps
    trials::Int = 10          # trial count
end

mutable struct Model
    # model's parameters
    p::Param
    t::Int          # time step
    peak_node::Int  # node which has the greatest resource
    resource_limit_vec::Vector{Float64}

    # agent's parameters
    strategy_vec::Vector{Strategy}  # agents' strategy
    resource_vec::Vector{Float64}   # agents' resource
    relation_mat::Matrix{Float64}

    function Model(p::Param, rng::MersenneTwister)
        resource_limit_vec = ar1(p.resource_limit_μ, p.resource_limit_β, p.resource_limit_σ, p.generations, rng)

        C_count = round(Int, p.N * p.C_rate₀)
        strategy_vec = shuffle(rng, [fill(C, C_count); fill(D, p.N - C_count)])

        resource_vec = fill(0.0, p.N)

        relation_mat = create_weighted_cycle_graph(p.N, p.k₀, p.w₀)

        return new(p, 1, 1, resource_limit_vec, strategy_vec, resource_vec, relation_mat)
    end
end

function run!(m::Model, rng::MersenneTwister, log_level::Symbol = :default)::Union{Tuple, Vector}
    if log_level == :full
        model_vec = []
    elseif log_level == :C_rate_only
        C_rate_vec = fill(0.0, m.p.generations)
    else
        C_rate_vec = fill(0.0, m.p.generations)
        average_resource_vec = fill(0.0, m.p.generations)
        death_count_vec = fill(0, m.p.generations)
        peak_node_vec = fill(0, m.p.generations)
        strategy_mat = fill(C, m.p.generations, m.p.N)
        degree_mat = fill(0, m.p.generations, m.p.N)
        average_distance_vec = fill(0.0, m.p.generations)
        clustering_coefficient_vec = fill(0.0, m.p.generations)
        average_C_neighbour_rate_vec = fill(0.0, m.p.generations)
    end

    for generation in 1:m.p.generations
        m.t = generation
        resource_allocation!(m, rng)
        weighted_interaction!(m, rng)
        death_id_vec, _ = weighted_death_birth!(m, rng)

        # Log
        if log_level == :full
            push!(model_vec, deepcopy(m))
        elseif log_level == :C_rate_only
            C_rate_vec[generation] = C_rate(m)
        else
            C_rate_vec[generation] = C_rate(m)
            average_resource_vec[generation] = average_resource(m)
            death_count_vec[generation] = length(death_id_vec)
            peak_node_vec[generation] = m.peak_node
            strategy_mat[generation, :] .= m.strategy_vec
            degree_mat[generation, :] .= mat_degree(m.relation_mat)
            
            g = SimpleGraph(m.relation_mat)
            average_distance_vec[generation] = average_distance(g)
            clustering_coefficient_vec[generation] = global_clustering_coefficient(g)
            average_C_neighbour_rate_vec[generation] = average_C_neighbour_rate(m.relation_mat, m.strategy_vec)
        end
    end

    if log_level == :full
        return model_vec
    elseif log_level == :C_rate_only
        return C_rate_vec
    else
        return C_rate_vec, average_resource_vec, death_count_vec, peak_node_vec, strategy_mat, degree_mat, average_distance_vec, clustering_coefficient_vec, average_C_neighbour_rate_vec
    end
end

function run_simple!(m::Model, rng::MersenneTwister)::Vector{Float64}
    C_rate_vec = []

    for generation in 1:m.p.generations
        m.t = generation
        resource_allocation!(m, rng)
        # weighted_interaction!(m, rng)
        weighted_death_birth!(m, rng)
        # simple_strategy_update!(m, rng)
        push!(C_rate_vec, C_rate(m))
    end

    return C_rate_vec
end

function average_C_neighbour_rate(relation_mat::Matrix{Float64}, strategy_vec::Vector{Strategy})::Float64
    N = length(strategy_vec)
    degree_vec = mat_degree(relation_mat)
    
    sum_C_neighbour_rate = 0.0
    C_count = 0

    for i in 1:N
        if strategy_vec[i] == C && degree_vec[i] > 0
            C_neighbour_rate = sum(relation_mat[i, strategy_vec .== C] .> 0) / degree_vec[i]
            @assert 0 ≤ C_neighbour_rate ≤ 1
            sum_C_neighbour_rate += C_neighbour_rate
            C_count += 1
        end
    end

    return C_count == 0 ? 0.0 : sum_C_neighbour_rate / C_count
end

function resource_allocation!(m::Model, rng::MersenneTwister)::Nothing
    if m.p.peak_node_variability < 0
        m.peak_node = rand(rng, 1:m.p.N)
    elseif m.p.peak_node_variability == 0
        # peak_node is not moved
    else
        m.peak_node = mod(m.peak_node + rand(rng, -m.p.peak_node_variability:m.p.peak_node_variability) - 1, m.p.N) + 1
    end
    
    distance_vec = [abs(i - m.peak_node) for i in 1:m.p.N]
    distance_vec = [distance > m.p.N ÷ 2 ? m.p.N - distance : distance for distance in distance_vec]
    m.resource_vec = [m.p.peak_node_resource - (m.p.resource_decrement_factor * distance) for distance in distance_vec]
    m.resource_vec = [max(r, 0) for r in m.resource_vec]

    return
end

function weighted_interaction!(m::Model, rng::MersenneTwister)::Vector{Tuple{Int, Int}}
    id_pair_vec = Vector{Tuple{Int, Int}}()

    for focal_id in shuffle(rng, 1:m.p.N)
        # focal と他のエージェントとの関係ベクトル
        relation_vec = m.relation_mat[focal_id, :]
        sum(relation_vec) == 0 && continue

        # opponent を選択 (関係が深いエージェントが選ばれやすい)
        opponent_id = sample(rng, 1:m.p.N, Weights(relation_vec))
        # opponent_id = rand(rng, filter(!=(focal_id), 1:m.p.N))
        push!(id_pair_vec, (focal_id, opponent_id))

        # focal と opponent が拠出可能なリソース
        focal_resource, opponent_resource = [contributable_resource(r, m.resource_limit_vec[m.t]) for r in m.resource_vec[[focal_id, opponent_id]]]

        # Pair-wise PGG
        strategy_pair = m.strategy_vec[[focal_id, opponent_id]]
        if strategy_pair == [C, C]
            pool_resource = (focal_resource + opponent_resource) * m.p.b
            m.resource_vec[focal_id] += pool_resource / 2 - focal_resource
            m.resource_vec[opponent_id] += pool_resource / 2 - opponent_resource
        elseif strategy_pair == [C, D]
            pool_resource = focal_resource * m.p.b
            m.resource_vec[focal_id] += pool_resource / 2 - focal_resource
            m.resource_vec[opponent_id] += pool_resource / 2
        elseif strategy_pair == [D, C]
            pool_resource = opponent_resource * m.p.b
            m.resource_vec[focal_id] += pool_resource / 2
            m.resource_vec[opponent_id] += pool_resource / 2 - opponent_resource
        end

        # Update relationship
        if strategy_pair == [C, C]
            m.relation_mat[focal_id, opponent_id] *= m.p.relationship_increment_factor
        else
            m.relation_mat[focal_id, opponent_id] /= m.p.relationship_increment_factor
        end
        m.relation_mat[focal_id, opponent_id] = max(m.relation_mat[focal_id, opponent_id], 1.0)
        m.relation_mat[opponent_id, focal_id] = m.relation_mat[focal_id, opponent_id]
    end

    return id_pair_vec
end

function contributable_resource(resource::Float64, resource_limit::Float64)::Float64
    # 全リソースを拠出するパターン
    # return resource
    # 余剰リソースを拠出するパターン
    return max(resource - resource_limit, 0)
    # リソースリミットを超えている場合は余剰リソースを拠出し、リソースリミットを下回っている場合は全リソースを拠出するパターン
    # return resource - (resource > resource_limit ? resource_limit : 0)
end

function simple_strategy_update!(m::Model, rng::MersenneTwister)::Vector{Int}
    death_id_vec, parent_id_vec = get_death_parent_id_vec(m, rng)
    length(death_id_vec) == 0 && return []

    parent_strategy_vec = m.strategy_vec[parent_id_vec]
    m.strategy_vec[death_id_vec] .= [rand(rng) > m.p.μ ? s : mutate(s) for s in parent_strategy_vec]

    mutated_id_vec = death_id_vec[m.strategy_vec[death_id_vec] .!= parent_strategy_vec]

    return mutated_id_vec
end

function weighted_death_birth!(m::Model, rng::MersenneTwister)::Tuple{Vector{Int}, Vector{Int}, Int, Int}
    death_id_vec, parent_id_vec = get_death_parent_id_vec(m, rng)
    death_count = length(death_id_vec)
    if death_count == 0
        return death_id_vec, parent_id_vec, 0, 0
    end

    # Strategy
    parent_strategy_vec = m.strategy_vec[parent_id_vec]
    prev_C_count = sum(m.strategy_vec .== C)
    prev_mutation_C_count = sum(parent_strategy_vec .== C)
    parent_strategy_vec = [rand(rng) > m.p.μ ? s : mutate(s) for s in parent_strategy_vec]
    m.strategy_vec[death_id_vec] .= parent_strategy_vec
    mutation_C_count = sum(parent_strategy_vec .== C) - prev_mutation_C_count
    increase_C_count = sum(m.strategy_vec .== C) - prev_C_count

    # Graph
    ## Delete edges
    m.relation_mat[death_id_vec, :] .= 0.0
    m.relation_mat[:, death_id_vec] .= 0.0

    ## Add edges
    add_count = Int(m.p.N * m.p.k₀ / 2 - mat_ne(m.relation_mat))

    ### initialize non_neighbors_dict and candidate_id_dict
    all_nodes = collect(1:m.p.N)
    parent_id_set = Set(parent_id_vec)
    non_neighbors_dict = Dict{Int, Vector{Int}}()
    candidates_dict = Dict{Int, Vector{Int}}()
    for death_id in death_id_vec
        non_neighbors_dict[death_id] = filter(!=(death_id), all_nodes)
        candidates_dict[death_id] = filter(in(parent_id_set), non_neighbors_dict[death_id])
    end

    while add_count > 0
        for death_id in death_id_vec
            # parent_id の候補から parent_id をランダムに選択
            if isempty(candidates_dict[death_id])
                candidates_dict[death_id] = non_neighbors_dict[death_id]
            end
            parent_id = rand(rng, candidates_dict[death_id])

            # update relationship graph
            mat_update_weight!(m.relation_mat, parent_id, death_id, m.p.w₀)

            # update non_neighbors_dict and candidates_dict
            non_neighbors_dict[death_id] = filter(!=(parent_id), non_neighbors_dict[death_id])
            candidates_dict[death_id] = filter(!=(parent_id), candidates_dict[death_id])
            if parent_id in death_id_vec
                non_neighbors_dict[parent_id] = filter(!=(death_id), non_neighbors_dict[parent_id])
                candidates_dict[parent_id] = filter(!=(death_id), candidates_dict[parent_id])
            end

            add_count -= 1
            add_count == 0 && break
        end
    end

    @assert m.p.N * m.p.k₀ / 2 == mat_ne(m.relation_mat) "$(m.p.N * m.p.k₀ / 2) == $(mat_ne(m.relation_mat))"

    return death_id_vec, parent_id_vec, increase_C_count, mutation_C_count
end

function get_death_parent_id_vec(m::Model, rng::MersenneTwister)::Tuple{Vector{Int}, Vector{Int}}
    death_id_vec = findall(<(m.resource_limit_vec[m.t]), m.resource_vec)
    death_count = length(death_id_vec)
    parent_id_vec = Int[]

    if death_count > 0
        parent_weights = regularize_resource_vec(m.resource_vec, true)
        parent_id_vec = sample(rng, 1:m.p.N, parent_weights, death_count, replace = false)
    end

    return shuffle(rng, death_id_vec), shuffle(rng, parent_id_vec)
end

function regularize_resource_vec(resource_vec::Vector{Float64}, positive::Bool)::Weights
    weights = positive ? resource_vec .- minimum(resource_vec) : maximum(resource_vec) .- resource_vec
    weights .+= 0.0001
    weights ./= maximum(weights)

    return Weights(weights)
end

mutate(strategy::Strategy)::Strategy = (strategy == C ? D : C)
# mutate(strategy::Strategy)::Strategy = rand([C, D])

C_rate(m::Model)::Float64 = mean(m.strategy_vec .== C)

average_resource(m::Model)::Float64 = mean(m.resource_vec)

end  # end of module
