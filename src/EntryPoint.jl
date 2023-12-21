module EntryPoint

using DataFrames: DataFrame
using Random: MersenneTwister

include("./Simulation.jl")
using .Simulation: Param, POPULATION, PAYOFF, run

@kwdef struct ParamOptions
    initial_N_vec = [1_000]
    initial_T_vec = [1.15]   # 0.0:0.1:2.0, [0.9, 1.1]
    S_vec = [-0.1]  # -1.0:0.1:1.0, [-0.1, 0.1]
    initial_graph_weight_vec = [0.2]
    interaction_freqency_vec = [1.0]
    Δw_vec = [0.1]
    reproduction_rate_vec = [0.05]
    δ_vec = [1.0]
    μ_vec = [0.01]
    β_sigma_vec = vec([(β, sigma) for β = 0.0:0.1:1.0, sigma = 0.0:10.0:100.0])
    generations_vec = [10_000]
    variability_mode = POPULATION
    trials = 1
end

function to_vector(params::ParamOptions)::Vector{Param}
    result = []
    seed_counter = rand(UInt)
    for _ = 1:(params.trials)
        for initial_N in params.initial_N_vec,
            initial_T in params.initial_T_vec,
            S in params.S_vec,
            initial_graph_weight in params.initial_graph_weight_vec,
            interaction_freqency in params.interaction_freqency_vec,
            Δw in params.Δw_vec,
            reproduction_rate in params.reproduction_rate_vec,
            δ in params.δ_vec,
            μ in params.μ_vec,
            β_sigma in params.β_sigma_vec,
            generations in params.generations_vec

            push!(
                result,
                Param(
                    initial_N = initial_N,
                    initial_T = initial_T,
                    S = S,
                    interaction_freqency = interaction_freqency,
                    initial_graph_weight = initial_graph_weight,
                    Δw = Δw,
                    reproduction_rate = reproduction_rate,
                    δ = δ,
                    μ = μ,
                    β = β_sigma[1],
                    sigma = β_sigma[2],
                    generations = generations,
                    rng = MersenneTwister(seed_counter),
                    variability_mode = params.variability_mode,
                ),
            )
            seed_counter += 1
        end
    end

    return result
end

end  # end of module

if abspath(PROGRAM_FILE) == @__FILE__
    using CSV: write
    using Dates
    using .EntryPoint: Simulation, ParamOptions, to_vector

    const PARAM_OPTIONS = to_vector(ParamOptions())
    const DIR_NAME = "output/$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
    const LOG_LEVEL = 1

    mkdir(DIR_NAME)

    Threads.@threads for i in eachindex(PARAM_OPTIONS)
        write("$(DIR_NAME)/$(i).csv", Simulation.run(PARAM_OPTIONS[i], log_level = LOG_LEVEL))
    end
end
