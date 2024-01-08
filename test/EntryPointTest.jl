using DataFrames: nrow
using Random: MersenneTwister
using StatsBase
using Test: @testset, @test

include("../src/EntryPoint.jl")

using .EntryPoint: ParamOptions, to_vector, Simulation

@testset "ParamOptions and to_vector" begin
    for mode in keys(Simulation.VARIABILITY_MODE)
        params = ParamOptions(
            initial_N_vec = [100, 200, 300],
            initial_k_vec = [10, 20],
            initial_T_vec = [0.1, 0.2],
            S_vec = [0.3, 0.35, 0.4],
            initial_w_vec = [0.21, 0.6],
            Δw_vec = [0.1, 0.11, 0.12],
            interaction_freqency_vec = [0.31, 0.8],
            reproduction_rate_vec = [0.05, 0.1],
            δ_vec = [0.1, 1.0],
            initial_μ_s_vec = [0.01, 0.0],
            initial_μ_c_vec = [0.02, 0.1],
            β_sigma_vec = [(0.1, 10.0), (0.3, 20.0), (0.5, 30.0)],
            generations_vec = [10, 20],
            variability_mode = mode,
            trials = 3,
        )

        params = to_vector(params)
        @test typeof(params) == Vector{Simulation.Param}
        @test length(params) == 124_416 == 3 * 2 * 2 * 3 * 2 * 2 * 3 * 2 * 2 * 2 * 2 * 3 * 2 * 3

        @test params[1].initial_N == 100
        @test params[1].initial_k == 10
        @test params[1].initial_T == 0.1
        @test params[1].S == 0.3
        @test params[1].initial_w == Float16(0.21)
        @test params[1].Δw == 0.1
        @test params[1].interaction_freqency == 0.31
        @test params[1].reproduction_rate == 0.05
        @test params[1].δ == 0.1
        @test params[1].initial_μ_s == 0.01
        @test params[1].initial_μ_c == 0.02
        @test params[1].β == 0.1
        @test params[1].sigma == 10.0
        @test params[1].generations == 10
        @test params[1].variability_mode == mode

        @test params[124_416].initial_N == 300
        @test params[124_416].initial_k == 20
        @test params[124_416].initial_T == 0.2
        @test params[124_416].S == 0.4
        @test params[124_416].initial_w == Float16(0.6)
        @test params[124_416].Δw == 0.12
        @test params[124_416].interaction_freqency == 0.8
        @test params[124_416].reproduction_rate == 0.1
        @test params[124_416].δ == 1.0
        @test params[124_416].initial_μ_s == 0.0
        @test params[124_416].initial_μ_c == 0.1
        @test params[124_416].β == 0.5
        @test params[124_416].sigma == 30.0
        @test params[124_416].generations == 20
        @test params[124_416].variability_mode == mode

        # 乱数ジェネレータの一意性を確認
        rngs = [p.rng for p in params]
        @test length(rngs) == length(unique(rngs))

        # 実行
        for i in eachindex(params[1:10_000:124_416])
            Simulation.run(params[i], log_level = 2, log_skip = 10)
        end
    end
end
