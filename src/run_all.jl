# cd ~/Dropbox/workspace/inaba2024
# nohup julia src/run_all.jl > run_all.log &
# nohup julia --threads 8 src/run_all.jl > run_all.log &
using Pkg
Pkg.activate("../Inaba2024")

using CSV: write
using Dates
using Distributed: @distributed, addprocs

include("./Simulation.jl")
using .Simulation

@kwdef struct ParamOptions
    initial_N_vec = [1_000]
    T_vec = 0.0:0.2:2.0
    S_vec = -1.0:0.2:1.0
    initial_graph_weight_vec = [0.2]
    interaction_freqency_vec = [1.0]
    relationship_volatility_vec = [0.1]
    δ_vec = [1.0]
    μ_vec = [0.01]
    β_σ_vec = [(0.4, 0.4)]  # [(0.0, 0.0), (0.1, 0.1), (0.4, 0.4)]
    generations_vec = [10_000]
end

function to_vector(params::ParamOptions)::Vector{Param}
    result = [
        Param(
            initial_N = initial_N,
            T = T,
            S = S,
            interaction_freqency = interaction_freqency,
            initial_graph_weight = initial_graph_weight,
            relationship_volatility = relationship_volatility,
            δ = δ,
            μ = μ,
            β = β_σ[1],
            σ = β_σ[2],
            generations = generations,
        ) for initial_N in params.initial_N_vec, T in params.T_vec, S in params.S_vec,
        initial_graph_weight in params.initial_graph_weight_vec,
        interaction_freqency in params.interaction_freqency_vec,
        relationship_volatility in params.relationship_volatility_vec, δ in params.δ_vec, μ in params.μ_vec,
        β_σ in params.β_σ_vec, generations in params.generations_vec
    ]

    return reshape(result, :)
end

const PARAM_OPTIONS = to_vector(ParamOptions())
const DIR_NAME = "output/$(Dates.format(now(), "yyyymmdd_HHMMSS"))"

mkdir(DIR_NAME)

# addprocs(8)
# @distributed for i in eachindex(PARAM_OPTIONS)
#     write("$(DIR_NAME)/$(i).csv", Simulation.run(PARAM_OPTIONS[i]))
# end

Threads.@threads for i in eachindex(PARAM_OPTIONS)
    write("$(DIR_NAME)/$(i).csv", Simulation.run(PARAM_OPTIONS[i]))
end
