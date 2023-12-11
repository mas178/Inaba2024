using Random
using StatsBase: mean
using Test: @testset, @test

include("../src/SimPlot.jl")
include("../src/Output.jl")
using .SimPlot: calc_mean
using .Output: make_output_df
using .Output.Simulation: Param

@testset "calc_mean" begin
    df = make_output_df(Param())
    @test size(df) == (100, 51)

    Random.seed!(1)
    df[:, 14:end] .= rand(Float16, size(df, 1), size(df, 2) - 13)

    mean_df = calc_mean(df)
    @test size(mean_df) == (1, 32)

    @test mean_df.cooperation_rate[1] == mean(df.cooperation_rate[51:100])
end
