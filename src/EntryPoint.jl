module EntryPoint

using DataFrames: DataFrame

include("./Output.jl")
using .Output: Output
using .Output.Simulation: Model, C, D, interaction!, death_and_birth!

function run(param::Output.Simulation.Param)::DataFrame
    model = Model(param)
    model.strategy_vec = rand(model.param.rng, [C, D], model.param.initial_N)
    output_df = Output.make_output_df(param)

    for generation = 1:model.param.generations
        interaction!(model)
        death_and_birth!(model, generation)
        Output.log!(output_df, generation, model)
    end

    return output_df
end

@kwdef struct ParamOptions
    initial_N_vec = [1_000]
    T_vec = [0.9, 1.1]   # 0.0:0.1:2.0, [0.9, 1.1]
    S_vec = [-0.1, 0.1]  # -1.0:0.1:1.0, [-0.1, 0.1]
    initial_graph_weight_vec = [0.2]
    interaction_freqency_vec = [1.0]
    relationship_volatility_vec = [0.1]
    δ_vec = [1.0]
    μ_vec = [0.01]
    β_σ_vec = vec([(β, σ) for β = 0.0:0.05:1.0, σ = 0.0:0.05:1.0]) # [(0.0, 0.0), (0.1, 0.1), (0.4, 0.4)], vec([(β, σ) for β = 0.0:0.1:1.0, σ = 0.0:0.1:1.0])
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

end  # end of module

# using Pkg
# Pkg.activate("../Inaba2024")
# using CSV: write
# using Dates

# const PARAM_OPTIONS = to_vector(ParamOptions())
# const DIR_NAME = "output/$(Dates.format(now(), "yyyymmdd_HHMMSS"))"

# mkdir(DIR_NAME)

# Threads.@threads for i in eachindex(PARAM_OPTIONS)
#     df, _ = run(PARAM_OPTIONS[i])
#     write("$(DIR_NAME)/$(i).csv", df)
# end
