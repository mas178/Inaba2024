using DataFrames: nrow
using Random: MersenneTwister
using StatsBase
using Test: @testset, @test

include("../src/EntryPoint.jl")

using .EntryPoint: ParamOptions, to_vector, Simulation

@testset "ParamOptions and to_vector (POPULATION)" begin
    params = ParamOptions(
        initial_N_vec = [100, 200, 300],
        initial_T_vec = [0.1, 0.2],
        S_vec = [0.3, 0.35, 0.4],
        initial_graph_weight_vec = [0.21, 0.6],
        interaction_freqency_vec = [0.31, 0.8],
        Δw_vec = [0.1, 0.11, 0.12],
        reproduction_rate_vec = [0.05],
        δ_vec = [0.1, 1.0],
        μ_vec = [0.01],
        β_sigma_vec = [(0.1, 10.0), (0.3, 20.0), (0.5, 30.0)],
        generations_vec = [10, 20],
        variability_mode = Simulation.POPULATION,
        trials = 3,
    )

    params = to_vector(params)
    @test typeof(params) == Vector{Simulation.Param}
    @test length(params) == 7776 == 3 * 2 * 3 * 2 * 2 * 3 * 2 * 1 * 3 * 2 * 3

    @test params[1].initial_N == params[2].initial_N == params[3].initial_N == 100
    @test params[1].initial_T == params[2].initial_T == params[3].initial_T == 0.1
    @test params[1].S == params[2].S == params[3].S == 0.3
    @test params[1].initial_graph_weight == params[2].initial_graph_weight == 0.21
    @test params[1].interaction_freqency == params[2].interaction_freqency == 0.31
    @test params[1].Δw == params[2].Δw == 0.1
    @test params[1].δ == params[2].δ == params[3].δ == 0.1
    @test params[1].μ == params[2].μ == params[3].μ == 0.01
    @test params[1].β == params[2].β == 0.1
    @test params[3].β == params[4].β == 0.3
    @test params[1].sigma == params[2].sigma == 10.0
    @test params[3].sigma == params[4].sigma == 20.0
    @test params[1].generations == params[3].generations == params[5].generations == 10
    @test params[2].generations == params[4].generations == params[6].generations == 20

    @test params[2592].initial_N == 300
    @test params[2592].initial_T == 0.2
    @test params[2592].S == 0.4
    @test params[2592].initial_graph_weight == 0.6
    @test params[2592].interaction_freqency == 0.8
    @test params[2592].Δw == 0.12
    @test params[2592].δ == 1.0
    @test params[2592].μ == 0.01
    @test params[2592].β == 0.5
    @test params[2592].sigma == 30.0
    @test params[2592].generations == 20

    for i = 1:2592
        a = params[i]
        b = params[i + 2592]
        c = params[i + 2592 * 2]

        @test a.initial_N == b.initial_N == c.initial_N
        @test a.initial_T == b.initial_T == c.initial_T
        @test a.S == b.S == c.S
        @test a.initial_graph_weight == b.initial_graph_weight == c.initial_graph_weight
        @test a.interaction_freqency == b.interaction_freqency == c.interaction_freqency
        @test a.Δw == b.Δw == c.Δw
        @test a.δ == b.δ == c.δ
        @test a.μ == b.μ == c.μ
        @test a.β == b.β == c.β
        @test a.sigma == b.sigma == c.sigma
        @test a.generations == b.generations == c.generations
    end

    @test [param.variability_mode for param in params] == fill(Simulation.POPULATION, length(params))

    # 乱数ジェネレータの一意性を確認
    rngs = [p.rng for p in params]
    @test length(rngs) == length(unique(rngs))

    # 実行
    for i in eachindex(params[1:10])
        Simulation.run(params[i], log_level = 2)
    end
end

@testset "ParamOptions and to_vector (PAYOFF)" begin
    params = ParamOptions(
        initial_N_vec = [100, 200],
        initial_T_vec = [0.1, 0.2],
        S_vec = [0.3, 0.35, 0.4],
        initial_graph_weight_vec = [0.5, 0.6],
        interaction_freqency_vec = [0.7, 0.8],
        Δw_vec = [0.1, 0.11, 0.12],
        δ_vec = [0.1, 1.0],
        μ_vec = [0.01],
        β_sigma_vec = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)],
        generations_vec = [10, 20],
        variability_mode = Simulation.PAYOFF,
        trials = 2,
    )

    params = to_vector(params)

    @test [param.variability_mode for param in params] == fill(Simulation.PAYOFF, length(params))

    # 実行
    for i in eachindex(params[1:10])
        Simulation.run(params[i], log_level = 2)
    end
end
