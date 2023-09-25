using DataFrames: nrow
using Random: MersenneTwister
using Test: @testset, @test

include("../src/EntryPoint.jl")

using .EntryPoint: run, ParamOptions, to_vector
using .EntryPoint.Output.Simulation: Param

@testset "run" begin
    param = Param(
        initial_N = 100,
        initial_graph_weight = 0.2,
        δ = 1.0,
        β = 0.1,
        σ = 0.1,
        generations = 100,
        rng = MersenneTwister(1),
    )

    df = run(param)

    @test size(df) == (100, 57)
    @test df.initial_N == fill(100, 100)
    @test df.T == fill(1.1, 100)
    @test df.S == fill(-0.1, 100)
    @test df.initial_graph_weight == fill(0.2, 100)
    @test df.generation == collect(1:100)
    @test df.N[1:70] == [fill(100, 30)..., fill(101, 20)..., 102, 103, fill(104, 8)..., 103, fill(102, 9)...]
    @test df.N[71:end] == [99, 97, 95, 93, 91, 89, 87, 85, 83, 81, 83, 84, 85, 86, 88, 90, 92, 94, 96, fill(98, 11)...]
    @test df.strong_component2_size_σ == fill(0, 100)
end

@testset "ParamOptions and to_vector" begin
    params = ParamOptions(
        initial_N_vec = [100, 200, 300],
        T_vec = [0.1, 0.2],
        S_vec = [0.3, 0.35, 0.4],
        initial_graph_weight_vec = [0.5, 0.6],
        interaction_freqency_vec = [0.7, 0.8],
        relationship_volatility_vec = [0.1, 0.11, 0.12],
        δ_vec = [0.1, 1.0],
        μ_vec = [0.01],
        β_σ_vec = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)],
        generations_vec = [10, 20],
    )

    params = to_vector(params)
    @test typeof(params) == Vector{Param}
    @test length(params) == 2592 == 3 * 2 * 3 * 2 * 2 * 3 * 2 * 1 * 3 * 2

    @test params[1].initial_N == 100
    @test params[2].initial_N == 200
    @test params[3].initial_N == 300
    @test params[1].T == params[2].T == 0.1
    @test params[1].S == params[2].S == 0.3
    @test params[1].initial_graph_weight == params[2].initial_graph_weight == 0.5
    @test params[1].interaction_freqency == params[2].interaction_freqency == 0.7
    @test params[1].relationship_volatility == 0.1
    @test params[1].δ == params[2].δ == 0.1
    @test params[1].μ == params[2].μ == 0.01
    @test params[1].β == params[2].β == 0.1
    @test params[1].σ == params[2].σ == 0.2
    @test params[1].generations == params[2].generations == 10

    @test params[2592].initial_N == 300
    @test params[2592].T == 0.2
    @test params[2592].S == 0.4
    @test params[2592].initial_graph_weight == 0.6
    @test params[2592].interaction_freqency == 0.8
    @test params[2592].relationship_volatility == 0.12
    @test params[2592].δ == 1.0
    @test params[2592].μ == 0.01
    @test params[2592].β == 0.5
    @test params[2592].σ == 0.6
    @test params[2592].generations == 20
end
