using Random
using StatsBase: mean
using Test: @testset, @test

include("../src/SimPlot.jl")
include("../src/Simulation.jl")
using .SimPlot: calc_mean
using .Simulation: Param, make_output_df

@testset "calc_mean" begin
    df = make_output_df(Param())
    @test size(df) == (100, 48)

    Random.seed!(1)
    df[:, 16:end] .= rand(Float16, size(df, 1), size(df, 2) - 15)

    mean_df = calc_mean(df)
    @test size(mean_df) == (1, 32)

    @test mean_df.cooperation_rate[1] == mean(df.cooperation_rate[51:100])
end
