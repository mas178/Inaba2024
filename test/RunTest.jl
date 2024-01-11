using DataFrames: Not
using LinearAlgebra: Diagonal, diag
using Random: MersenneTwister
using StatsBase
using Test: @testset, @test

include("../src/Simulation.jl")
using .Simulation: Model, Param, POPULATION, PAYOFF, STRATEGY_MUTATION, RELATIONSHIP_MUTATION, run

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
        :initial_μ_r => 0.234,
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

    s_mutation_params = copy(common_params)
    s_mutation_params[:variability_mode] = STRATEGY_MUTATION
    s_mutation_params[:sigma] = 0.1

    r_mutation_params = copy(common_params)
    r_mutation_params[:variability_mode] = RELATIONSHIP_MUTATION
    r_mutation_params[:sigma] = 0.1

    for params in [population_params, payoff_params, s_mutation_params, r_mutation_params]
        param = Param(; params...)

        df = run(param, log_level = 2, log_rate = 0.5, log_skip = 10)

        @test size(df) == (5, 50)

        # 1 〜 15
        @test df.initial_N == fill(100, 5)
        @test df.initial_T == fill(1.15, 5)
        @test df.S == fill(-0.1, 5)
        @test df.initial_w == fill(Float16(0.2), 5)
        @test df.Δw == fill(0.1, 5)
        @test df.interaction_freqency == fill(1.0, 5)
        @test df.reproduction_rate == fill(0.05, 5)
        @test df.δ == fill(1.0, 5)
        @test df.initial_μ_s == fill(0.123, 5)
        @test df.initial_μ_r == fill(0.234, 5)
        @test df.β == fill(0.1, 5)
        @test df.generations == fill(100, 5)
        if param.variability_mode == POPULATION
            @test df.sigma == fill(10, 5)
            @test df.variability_mode == fill("POPULATION", 5)
        elseif param.variability_mode == PAYOFF
            @test df.sigma == fill(0.1, 5)
            @test df.variability_mode == fill("PAYOFF", 5)
        elseif param.variability_mode == STRATEGY_MUTATION
            @test df.sigma == fill(0.1, 5)
            @test df.variability_mode == fill("STRATEGY_MUTATION", 5)
        elseif param.variability_mode == RELATIONSHIP_MUTATION
            @test df.sigma == fill(0.1, 5)
            @test df.variability_mode == fill("RELATIONSHIP_MUTATION", 5)
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
            @test df.T == Float16[1.229, 1.109, 1.322, 1.075, 1.383]
        else
            @test df.T == fill(Float16(1.15), 5)
        end

        if param.variability_mode == POPULATION
            @test df.cooperation_rate == Float16[0.839, 0.8296, 0.839, 0.885, 0.77]
            @test df.payoff_μ == Float16[1.279, 1.432, 1.742, 1.766, 1.186]
        elseif param.variability_mode == PAYOFF
            @test df.cooperation_rate == Float16[0.37, 0.48, 0.61, 0.64, 0.66]
            @test df.payoff_μ == Float16[0.8604, 1.055, 1.363, 1.374, 1.472]
        elseif param.variability_mode == STRATEGY_MUTATION
            @test df.cooperation_rate == Float16[0.39, 0.44, 0.57, 0.66, 0.71]
            @test df.payoff_μ == Float16[0.811, 0.9756, 1.249, 1.425, 1.493]
        elseif param.variability_mode == RELATIONSHIP_MUTATION
            @test df.cooperation_rate == Float16[0.46, 0.46, 0.52, 0.53, 0.59]
            @test df.payoff_μ == Float16[1.048, 1.005, 1.106, 1.121, 1.202]
        end

        if param.variability_mode == POPULATION
            # 21 〜 25
            @test df.weak_k1 == Float16[3.264, 4.273, 3.396, 3.674, 3.572]
            @test df.weak_C1 == Float16[0.511, 0.4631, 0.4802, 0.5366, 0.4707]
            @test df.weak_comp1_count == Float16[5.0, 3.0, 4.0, 6.0, 3.0]
            @test df.weak_comp1_size_μ == Float16[0.161, 0.2874, 0.1963, 0.1475, 0.289]
            @test df.weak_comp1_size_max == Float16[0.6187, 0.7764, 0.6772, 0.702, 0.761]

            # 46 〜 50
            @test df.strong_k2 == Float16[0.0, 0.0, 2.688, 1.556, 0.0]
            @test df.strong_C2 == Float16[0.0, 0.0, 0.8853, 0.3333, 0.0]
            @test df.strong_comp2_count == Float16[0.0, 0.0, 8.0, 3.0, 0.0]
            @test df.strong_comp2_size_μ == Float16[0.0, 0.0, 0.043, 0.02884, 0.0]
            @test df.strong_comp2_size_max == Float16[0.0, 0.0, 0.05377, 0.02884, 0.0]
        elseif param.variability_mode == PAYOFF
            # 21 〜 25
            @test df.weak_k1 == Float16[3.898, 4.0, 4.32, 4.207, 3.44]
            @test df.weak_C1 == Float16[0.324, 0.3467, 0.3328, 0.494, 0.4895]
            @test df.weak_comp1_count == Float16[3.0, 2.0, 1.0, 2.0, 1.0]
            @test df.weak_comp1_size_μ == Float16[0.1967, 0.315, 0.75, 0.34, 0.75]
            @test df.weak_comp1_size_max == Float16[0.5, 0.53, 0.75, 0.65, 0.75]

            # 46 〜 50
            @test df.strong_k2 == Float16[2.0, 0.0, 2.0, 2.572, 2.4]
            @test df.strong_C2 == Float16[0.526, 0.0, 0.5, 1.0, 0.533]
            @test df.strong_comp2_count == Float16[3.0, 0.0, 4.0, 2.0, 1.0]
            @test df.strong_comp2_size_μ == Float16[0.04333, 0.0, 0.04, 0.035, 0.05]
            @test df.strong_comp2_size_max == Float16[0.05, 0.0, 0.04, 0.04, 0.05]
        elseif param.variability_mode == STRATEGY_MUTATION
            # 21 〜 25
            @test df.weak_k1 == Float16[3.059, 3.18, 4.906, 4.668, 3.766]
            @test df.weak_C1 == Float16[0.5596, 0.413, 0.388, 0.4783, 0.4956]
            @test df.weak_comp1_count == Float16[4.0, 4.0, 1.0, 2.0, 2.0]
            @test df.weak_comp1_size_μ == Float16[0.085, 0.1525, 0.64, 0.345, 0.34]
            @test df.weak_comp1_size_max == Float16[0.16, 0.46, 0.64, 0.66, 0.65]

            # 46 〜 50
            @test df.strong_k2 == Float16[2.8, 3.334, 4.0, 2.0, 1.5]
            @test df.strong_C2 == Float16[0.7, 0.9165, 0.7334, 1.0, 0.0]
            @test df.strong_comp2_count == Float16[1.0, 1.0, 1.0, 3.0, 1.0]
            @test df.strong_comp2_size_μ == Float16[0.05, 0.09, 0.07, 0.03, 0.04]
            @test df.strong_comp2_size_max == Float16[0.05, 0.09, 0.07, 0.03, 0.04]
        elseif param.variability_mode == RELATIONSHIP_MUTATION
            # 21 〜 25
            @test df.weak_k1 == Float16[4.832, 4.656, 3.705, 3.508, 3.62]
            @test df.weak_C1 == Float16[0.3206, 0.419, 0.4846, 0.3975, 0.5156]
            @test df.weak_comp1_count == Float16[1.0, 1.0, 1.0, 2.0, 4.0]
            @test df.weak_comp1_size_μ == Float16[0.65, 0.64, 0.68, 0.325, 0.1575]
            @test df.weak_comp1_size_max == Float16[0.65, 0.64, 0.68, 0.61, 0.29]

            # 46 〜 50
            @test df.strong_k2 == Float16[2.125, 2.25, 2.0, 3.666, 3.0]
            @test df.strong_C2 == Float16[0.5522, 0.7188, 0.5835, 0.7666, 0.625]
            @test df.strong_comp2_count == Float16[4.0, 4.0, 1.0, 1.0, 2.0]
            @test df.strong_comp2_size_μ == Float16[0.04, 0.04, 0.04, 0.06, 0.04]
            @test df.strong_comp2_size_max == Float16[0.05, 0.05, 0.04, 0.06, 0.05]
        end
    end
end

@testset "run (log_level, log_rate, log_skip)" begin
    param = Param(initial_μ_s = 0.1, initial_μ_r = 0.1, rng = MersenneTwister(123))

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
