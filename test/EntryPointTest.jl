using DataFrames: nrow
using Random: MersenneTwister
using Test: @testset, @test

include("../src/EntryPoint.jl")

using .EntryPoint: run
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
