module EntryPoint

using DataFrames: DataFrame
using Random: MersenneTwister

include("./Simulation.jl")
using .Simulation: Param, VARIABILITY_MODE, POPULATION, PAYOFF, STRATEGY_MUTATION, RELATIONSHIP_MUTATION, run

@kwdef struct ParamOptions
    initial_N_vec = [1000]
    initial_k_vec = [999]
    initial_T_vec = 0.0:0.1:2.0
    S_vec = -1.0:0.1:1.0
    initial_w_vec = [0.2]
    Δw_vec = [0.05]
    reproduction_rate_vec = [0.05]
    δ_vec = [1.0]
    initial_μ_s_vec = [0.01]
    initial_μ_r_vec = [0.01]
    β_σ_vec = [(0.0, 0.0)]
    # β_σ_vec = [(0.8, 200.0)]
    # β_σ_vec = [(0.8, 0.5)]
    generations_vec = [10_000]
    variability_mode = POPULATION
    # variability_mode = PAYOFF
    τ_vec = [10]
    trials = 20
end

# @kwdef struct ParamOptions
#     initial_N_vec = [1000]
#     initial_k_vec = [999]
#     initial_T_vec = [1.15]
#     S_vec = [-0.15]
#     initial_w_vec = [0.2]
#     Δw_vec = [0.05]
#     reproduction_rate_vec = [0.05]
#     δ_vec = [1.0]
#     initial_μ_s_vec = [0.01]
#     initial_μ_r_vec = [0.01]
#     generations_vec = [10_000]
#     #------------------
#     # variability_mode
#     #------------------
#     variability_mode = POPULATION
#     β_σ_vec = vec([(β, σ) for β = 0.0:0.1:0.9, σ = 0.0:20.0:200.0])  # vec([(β, σ) for β = 0.0:0.1:1.0, σ = 0.0:100.0:1000.0])
#     # variability_mode = PAYOFF
#     # β_σ_vec = vec([(β, σ) for β = 0.0:0.1:0.9, σ = 0.0:0.05:0.5])  # vec([(β, σ) for β = 0.0:0.1:1.0, σ = 0.0:100.0:1000.0])
#     # variability_mode = STRATEGY_MUTATION
#     # variability_mode = RELATIONSHIP_MUTATION
#     # β_σ_vec = vec([(β, σ) for β = 0.0:0.1:0.9, σ = 0.0:0.001:0.01])  # vec([(β, σ) for β = 0.0:0.1:1.0, σ = 0.0:100.0:1000.0])
#     τ_vec = [10]
#     trials = 100
# end

function to_vector(params::ParamOptions)::Vector{Param}
    result = []
    seed_counter = rand(UInt)
    for _ = 1:(params.trials)
        for initial_N in params.initial_N_vec,
            initial_k in params.initial_k_vec,
            initial_T in params.initial_T_vec,
            S in params.S_vec,
            initial_w in params.initial_w_vec,
            Δw in params.Δw_vec,
            reproduction_rate in params.reproduction_rate_vec,
            δ in params.δ_vec,
            initial_μ_s in params.initial_μ_s_vec,
            initial_μ_r in params.initial_μ_r_vec,
            β_σ in params.β_σ_vec,
            τ in params.τ_vec,
            generations in params.generations_vec

            push!(
                result,
                Param(
                    initial_N = initial_N,
                    initial_k = initial_k,
                    initial_T = initial_T,
                    S = S,
                    initial_w = initial_w,
                    Δw = Δw,
                    reproduction_rate = reproduction_rate,
                    δ = δ,
                    initial_μ_s = initial_μ_s,
                    initial_μ_r = initial_μ_r,
                    β = β_σ[1],
                    σ = β_σ[2],
                    τ = τ,
                    generations = generations,
                    variability_mode = params.variability_mode,
                    rng = MersenneTwister(seed_counter),
                ),
            )
            seed_counter += 1
        end
    end

    return result
end

function to_string(param::Param)::String
    fields = [
        param.initial_N,
        param.initial_k,
        param.initial_T,
        param.S,
        param.initial_w,
        param.Δw,
        param.reproduction_rate,
        param.δ,
        param.initial_μ_s,
        param.initial_μ_r,
        param.β,
        param.σ,
        param.τ,
        param.generations,
        VARIABILITY_MODE[param.variability_mode],
    ]
    return join(fields, ",")
end

end  # end of module

# if abspath(PROGRAM_FILE) == @__FILE__
#     using CSV: write
#     using Dates
#     using Random: shuffle
#     using .EntryPoint: Simulation, ParamOptions, to_vector

#     const PARAM_OPTIONS = shuffle(to_vector(ParamOptions()))
#     const DIR_NAME = "output/$(Dates.format(now(), "yyyymmdd_HHMMSS"))"
#     const LOG_LEVEL = 0
#     const LOG_RATE = 0.5
#     const LOG_SKIP = 10

#     mkdir(DIR_NAME)

#     Threads.@threads for i in eachindex(PARAM_OPTIONS)
#         write(
#             "$(DIR_NAME)/$(i).csv",
#             Simulation.run(PARAM_OPTIONS[i], log_level = LOG_LEVEL, log_rate = LOG_RATE, log_skip = LOG_SKIP),
#         )
#     end
# end

if abspath(PROGRAM_FILE) == @__FILE__
    using Dates
    using Random: shuffle
    using .EntryPoint: Simulation, ParamOptions, to_vector, to_string

    const PARAM_OPTIONS = to_vector(ParamOptions())
    const FILE_NAME = "output/$(Dates.format(now(), "yyyymmdd_HHMMSS")).csv"

    fieldnames_str = join(fieldnames(Simulation.Param)[1:(end - 1)], ",")
    write(FILE_NAME, fieldnames_str * ",cooperation_rate\n")

    lk = Threads.ReentrantLock()

    Threads.@threads for param in PARAM_OPTIONS
        result = Simulation.fast_run(param)
        lock(lk) do
            open(FILE_NAME, "a") do io
                write(io, "$(to_string(param)),$(result)\n")
                flush(io)
            end
        end
    end
end
