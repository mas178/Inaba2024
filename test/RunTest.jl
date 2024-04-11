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
        :reproduction_rate => 0.05,
        :δ => 1.0,
        :initial_μ_s => 0.123,
        :initial_μ_r => 0.234,
        :β => 0.1,
        :σ => 0.1,
        :τ => 1,
        :generations => 100,
        :rng => MersenneTwister(1),
    )

    population_params = copy(common_params)
    population_params[:variability_mode] = POPULATION
    population_params[:σ] = 10

    payoff_params = copy(common_params)
    payoff_params[:variability_mode] = PAYOFF
    payoff_params[:σ] = 0.1

    s_mutation_params = copy(common_params)
    s_mutation_params[:variability_mode] = STRATEGY_MUTATION
    s_mutation_params[:σ] = 0.1

    r_mutation_params = copy(common_params)
    r_mutation_params[:variability_mode] = RELATIONSHIP_MUTATION
    r_mutation_params[:σ] = 0.1

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
        @test df.reproduction_rate == fill(0.05, 5)
        @test df.δ == fill(1.0, 5)
        @test df.initial_μ_s == fill(0.123, 5)
        @test df.initial_μ_r == fill(0.234, 5)
        @test df.β == fill(0.1, 5)
        @test df.generations == fill(100, 5)
        if param.variability_mode == POPULATION
            @test df.σ == fill(10, 5)
            @test df.variability_mode == fill("POPULATION", 5)
        elseif param.variability_mode == PAYOFF
            @test df.σ == fill(0.1, 5)
            @test df.variability_mode == fill("PAYOFF", 5)
        elseif param.variability_mode == STRATEGY_MUTATION
            @test df.σ == fill(0.1, 5)
            @test df.variability_mode == fill("STRATEGY_MUTATION", 5)
        elseif param.variability_mode == RELATIONSHIP_MUTATION
            @test df.σ == fill(0.1, 5)
            @test df.variability_mode == fill("RELATIONSHIP_MUTATION", 5)
        end

        # 16 〜 17
        @test df.generation == [60, 70, 80, 90, 100]

        if param.variability_mode == POPULATION
            @test df.N == [118, 94, 93, 104, 82]
        else
            @test df.N == [100, 100, 100, 100, 100]
        end

        # 18 〜 20
        if param.variability_mode == PAYOFF
            @test df.T == Float16[1.084, 1.218, 1.348, 1.162, 1.181]
        else
            @test df.T == fill(Float16(1.15), 5)
        end

        if param.variability_mode == POPULATION
            @test df.cooperation_rate == Float16[0.729, 0.8403, 0.8604, 0.8076, 0.8535]
            @test df.payoff_μ == Float16[1.147, 1.564, 1.889, 1.625, 1.682]
        elseif param.variability_mode == PAYOFF
            @test df.cooperation_rate == Float16[0.32, 0.35, 0.44, 0.56, 0.72]
            @test df.payoff_μ == Float16[0.7456, 0.8657, 1.067, 1.237, 1.56]
        elseif param.variability_mode == STRATEGY_MUTATION
            @test df.cooperation_rate == Float16[0.4, 0.46, 0.57, 0.67, 0.77]
            @test df.payoff_μ == Float16[0.987, 1.008, 1.234, 1.4, 1.544]
        elseif param.variability_mode == RELATIONSHIP_MUTATION
            @test df.cooperation_rate == Float16[0.64, 0.67, 0.69, 0.79, 0.81]
            @test df.payoff_μ == Float16[1.377, 1.373, 1.484, 1.629, 1.648]
        end

        if param.variability_mode == POPULATION
            # 21 〜 25
            @test df.weak_k1 == Float16[4.086, 3.15, 3.2, 3.482, 3.227]
            @test df.weak_C1 == Float16[0.3508, 0.6465, 0.3088, 0.519, 0.4502]
            @test df.weak_comp1_count == Float16[2.0, 10.0, 2.0, 4.0, 3.0]
            @test df.weak_comp1_size_μ == Float16[0.4026, 0.0925, 0.4302, 0.214, 0.252]
            @test df.weak_comp1_size_max == Float16[0.78, 0.4893, 0.8174, 0.721, 0.4634]

            # 46 〜 50
            @test df.strong_k2 == Float16[0.0, 2.354, 2.889, 3.0, 2.572]
            @test df.strong_C2 == Float16[0.0, 0.8237, 0.8706, 0.8643, 0.9683]
            @test df.strong_comp2_count == Float16[0.0, 5.0, 2.0, 7.0, 6.0]
            @test df.strong_comp2_size_μ == Float16[0.0, 0.03616, 0.0484, 0.0357, 0.0427]
            @test df.strong_comp2_size_max == Float16[0.0, 0.04254, 0.05377, 0.0673, 0.06097]
        elseif param.variability_mode == PAYOFF
            # 21 〜 25
            @test df.weak_k1 == Float16[3.137, 2.791, 2.4, 1.81, 3.24]
            @test df.weak_C1 == Float16[0.3594, 0.2983, 0.4229, 0.389, 0.6606]
            @test df.weak_comp1_count == Float16[1.0, 3.0, 7.0, 11.0, 8.0]
            @test df.weak_comp1_size_μ == Float16[0.44, 0.1433, 0.05, 0.03818, 0.08875]
            @test df.weak_comp1_size_max == Float16[0.44, 0.33, 0.08, 0.06, 0.45]

            # 46 〜 50
            @test df.strong_k2 == Float16[4.0, 2.5, 3.334, 3.75, 3.08]
            @test df.strong_C2 == Float16[0.5425, 0.8335, 0.8335, 0.592, 0.8984]
            @test df.strong_comp2_count == Float16[1.0, 1.0, 1.0, 1.0, 13.0]
            @test df.strong_comp2_size_μ == Float16[0.16, 0.04, 0.06, 0.08, 0.03845]
            @test df.strong_comp2_size_max == Float16[0.16, 0.04, 0.06, 0.08, 0.08]
        elseif param.variability_mode == STRATEGY_MUTATION
            # 21 〜 25
            @test df.weak_k1 == Float16[3.031, 3.246, 3.36, 3.047, 3.227]
            @test df.weak_C1 == Float16[0.388, 0.284, 0.311, 0.3718, 0.4717]
            @test df.weak_comp1_count == Float16[1.0, 4.0, 1.0, 2.0, 3.0]
            @test df.weak_comp1_size_μ == Float16[0.33, 0.1324, 0.75, 0.315, 0.25]
            @test df.weak_comp1_size_max == Float16[0.33, 0.31, 0.75, 0.55, 0.62]

            # 46 〜 50
            @test df.strong_k2 == Float16[4.08, 3.0, 2.611, 3.926, 2.445]
            @test df.strong_C2 == Float16[0.5728, 0.6665, 0.732, 0.7173, 0.852]
            @test df.strong_comp2_count == Float16[1.0, 3.0, 8.0, 4.0, 2.0]
            @test df.strong_comp2_size_μ == Float16[0.25, 0.04666, 0.045, 0.135, 0.045]
            @test df.strong_comp2_size_max == Float16[0.25, 0.05, 0.08, 0.36, 0.06]
        elseif param.variability_mode == RELATIONSHIP_MUTATION
            # 21 〜 25
            @test df.weak_k1 == Float16[2.883, 2.94, 2.666, 2.418, 2.924]
            @test df.weak_C1 == Float16[0.3262, 0.3513, 0.563, 0.4387, 0.4893]
            @test df.weak_comp1_count == Float16[8.0, 4.0, 9.0, 9.0, 6.0]
            @test df.weak_comp1_size_μ == Float16[0.085, 0.165, 0.06335, 0.07446, 0.10834]
            @test df.weak_comp1_size_max == Float16[0.23, 0.45, 0.15, 0.4, 0.36]

            # 46 〜 50
            @test df.strong_k2 == Float16[3.25, 1.818, 1.883, 2.0, 3.0]
            @test df.strong_C2 == Float16[1.0, 0.5454, 0.51, 0.7617, 1.0]
            @test df.strong_comp2_count == Float16[2.0, 3.0, 5.0, 2.0, 1.0]
            @test df.strong_comp2_size_μ == Float16[0.04, 0.03665, 0.034, 0.035, 0.04]
            @test df.strong_comp2_size_max == Float16[0.05, 0.05, 0.04, 0.04, 0.04]
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
    @test all(x -> any(x .!= 0), eachcol(df_2_040[:, 36:50]))
end
