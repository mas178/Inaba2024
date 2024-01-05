using DataFrames: Not
using LinearAlgebra: Diagonal, diag
using Random: MersenneTwister
using StatsBase
using Test: @testset, @test

include("../src/Simulation.jl")
using .Simulation: Model, Param, POPULATION, PAYOFF, MUTATION, run

@testset "run" begin
    common_params = Dict(
        :initial_N => 100,
        :initial_k => 10,
        :initial_T => 1.15,
        :S => -0.1,
        :initial_w => 0.2,
        :Δw => 0.1,
        :interaction_freqency => 1.0,
        :reproduction_rate => 0.05,
        :δ => 1.0,
        :initial_μ_s => 0.123,
        :initial_μ_c => 0.234,
        :β => 0.1,
        :sigma => 0.1,
        :generations => 100,
        :rng => MersenneTwister(1),
    )

    population_params = copy(common_params)
    population_params[:variability_mode] = POPULATION
    population_params[:sigma] = 10

    payoff_params = copy(common_params)
    payoff_params[:variability_mode] = PAYOFF
    payoff_params[:sigma] = 0.1

    mutation_params = copy(common_params)
    mutation_params[:variability_mode] = MUTATION
    mutation_params[:sigma] = 0.1

    for params in [population_params, payoff_params, mutation_params]
        param = Param(; params...)

        df = run(param, log_level = 2, log_rate = 0.5, log_skip = 10)

        @test size(df) == (5, 50)

        # 1 〜 15
        @test df.initial_N == fill(100, 5)
        @test df.initial_T == fill(1.15, 5)
        @test df.S == fill(-0.1, 5)
        @test df.initial_w == fill(0.2, 5)
        @test df.Δw == fill(0.1, 5)
        @test df.interaction_freqency == fill(1.0, 5)
        @test df.reproduction_rate == fill(0.05, 5)
        @test df.δ == fill(1.0, 5)
        @test df.initial_μ_s == fill(0.123, 5)
        @test df.initial_μ_c == fill(0.234, 5)
        @test df.β == fill(0.1, 5)
        @test df.generations == fill(100, 5)
        if param.variability_mode == POPULATION
            @test df.sigma == fill(10, 5)
            @test df.variability_mode == fill("POPULATION", 5)
        elseif param.variability_mode == PAYOFF
            @test df.sigma == fill(0.1, 5)
            @test df.variability_mode == fill("PAYOFF", 5)
        elseif param.variability_mode == MUTATION
            @test df.sigma == fill(0.1, 5)
            @test df.variability_mode == fill("MUTATION", 5)
        end

        # 16 〜 17
        @test df.generation == [60, 70, 80, 90, 100]

        if param.variability_mode == POPULATION
            @test df.N == [118, 94, 93, 104, 113]
        else
            @test df.N == [100, 100, 100, 100, 100]
        end

        # 18 〜 20
        if param.variability_mode == PAYOFF
            @test df.T == Float16[1.006, 1.2295, 1.108, 1.04, 1.111]
        else
            @test df.T == fill(Float16(1.15), 5)
        end

        if param.variability_mode == POPULATION
            @test df.cooperation_rate == Float16[0.4407, 0.553, 0.5913, 0.6636, 0.6196]
            @test df.payoff_μ == Float16[0.647, 1.048, 1.296, 1.424, 0.9473]
        elseif param.variability_mode == PAYOFF
            @test df.cooperation_rate == Float16[0.53, 0.6, 0.62, 0.66, 0.65]
            @test df.payoff_μ == Float16[1.135, 1.381, 1.307, 1.411, 1.256]
        elseif param.variability_mode == MUTATION
            @test df.cooperation_rate == Float16[0.23, 0.32, 0.36, 0.4, 0.47]
            @test df.payoff_μ == Float16[0.4395, 0.6963, 0.7764, 0.855, 1.08]
        end

        if param.variability_mode == POPULATION
            # 21 〜 25
            @test df.weak_k1 == Float16[0.0, 10.32, 13.55, 16.66, 21.86]
            @test df.weak_C1 == Float16[0.0, 0.3635, 0.329, 0.334, 0.3306]
            @test df.weak_comp1_count == Float16[0.0, 1.0, 1.0, 1.0, 1.0]
            @test df.weak_comp1_size_μ == Float16[0.0, 1.0, 1.0, 1.0, 1.0]
            @test df.weak_comp1_size_max == Float16[0.0, 1.0, 1.0, 1.0, 1.0]

            # 46 〜 50
            @test df.strong_k2 == Float16[1.75, 1.333, 2.0, 1.714, 2.445]
            @test df.strong_C2 == Float16[0.2917, 0.0, 0.5835, 0.0, 0.685]
            @test df.strong_comp2_count == Float16[2.0, 1.0, 1.0, 1.0, 2.0]
            @test df.strong_comp2_size_μ == Float16[0.0339, 0.03192, 0.043, 0.0673, 0.03983]
            @test df.strong_comp2_size_max == Float16[0.0339, 0.03192, 0.043, 0.0673, 0.04425]
        elseif param.variability_mode == PAYOFF
            # 21 〜 25
            @test df.weak_k1 == Float16[14.18, 15.555, 19.22, 21.66, 23.95]
            @test df.weak_C1 == Float16[0.4036, 0.4062, 0.3606, 0.3628, 0.3582]
            @test df.weak_comp1_count == Float16[1.0, 1.0, 1.0, 1.0, 1.0]
            @test df.weak_comp1_size_μ == Float16[0.99, 0.99, 1.0, 1.0, 1.0]
            @test df.weak_comp1_size_max == Float16[0.99, 0.99, 1.0, 1.0, 1.0]

            # 46 〜 50
            @test df.strong_k2 == Float16[2.223, 4.266, 2.0, 1.75, 1.5]
            @test df.strong_C2 == Float16[0.2328, 0.5225, 0.4333, 0.2917, 0.0]
            @test df.strong_comp2_count == Float16[1.0, 1.0, 1.0, 2.0, 1.0]
            @test df.strong_comp2_size_μ == Float16[0.09, 0.15, 0.05, 0.04, 0.04]
            @test df.strong_comp2_size_max == Float16[0.09, 0.15, 0.05, 0.05, 0.04]
        elseif param.variability_mode == MUTATION
            # 21 〜 25
            @test df.weak_k1 == Float16[2.8, 0.0, 0.0, 0.0, 0.0]
            @test df.weak_C1 == Float16[0.7, 0.0, 0.0, 0.0, 0.0]
            @test df.weak_comp1_count == Float16[1.0, 0.0, 0.0, 0.0, 0.0]
            @test df.weak_comp1_size_μ == Float16[0.05, 0.0, 0.0, 0.0, 0.0]
            @test df.weak_comp1_size_max == Float16[0.05, 0.0, 0.0, 0.0, 0.0]

            # 46 〜 50
            @test df.strong_k2 == Float16[2.572, 3.334, 2.666, 3.6, 1.714]
            @test df.strong_C2 == Float16[1.0, 0.6943, 0.757, 0.641, 0.0]
            @test df.strong_comp2_count == Float16[2.0, 1.0, 1.0, 1.0, 1.0]
            @test df.strong_comp2_size_μ == Float16[0.035, 0.06, 0.09, 0.15, 0.07]
            @test df.strong_comp2_size_max == Float16[0.04, 0.06, 0.09, 0.15, 0.07]
        end
    end
end

@testset "run (log_level, log_rate, log_skip)" begin
    param = Param(initial_μ_s = 0.1, initial_μ_c = 0.1, rng = MersenneTwister(123))

    df_0_100 = run(param, log_level = 0, log_rate = 1.0, log_skip = 3)
    df_1_070 = run(param, log_level = 1, log_rate = 0.7, log_skip = 4)
    df_2_040 = run(param, log_level = 2, log_rate = 0.4, log_skip = 7)

    @test size(df_0_100) == (33, 20)
    @test size(df_1_070) == (18, 35)
    @test size(df_2_040) == (6, 50)

    @test df_0_100.generation == 3:3:99
    @test df_1_070.generation == 32:4:100
    @test df_2_040.generation == 63:7:98

    # 1:20
    @test all(x -> all(x .!= 0), eachcol(df_0_100[:, 1:20]))
    @test all(x -> all(x .!= 0), eachcol(df_1_070[:, 1:20]))
    @test all(x -> all(x .!= 0), eachcol(df_2_040[:, 1:20]))

    # 21:35
    @test any(x -> any(x .!= 0), eachcol(df_1_070[:, 21:35]))
    @test any(x -> any(x .!= 0), eachcol(df_2_040[:, 21:35]))

    # 36:50
    @test all(x -> all(x .!= 0), eachcol(df_2_040[:, 36:50]))
end
