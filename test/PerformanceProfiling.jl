using Profile
using ProfileView

using Graphs
using SimpleWeightedGraphs

include("../src/Simulation.jl")
include("../src/Network.jl")

using .Simulation: Simulation
using .Network: create_regular_weighted_graph

#==
# interaction! のパフォーマンスチェック
N = 50_000
k = 100
initial_w = 0.2
focal_id = 1
g = create_regular_weighted_graph(N, k, initial_w)
opponent_vec = neighbors(g, focal_id)

function interaction_test()
    for _ in 1:1000
        # 1: best
        weights = get_weight.(Ref(g), Ref(1), opponent_vec)

        # 2: bad
        weights = [g.weights[1, opponent] for opponent in opponent_vec]

        3: worst
        weights = Vector{Float64}(undef, length(opponent_vec))
        for (i, opponent) in enumerate(opponent_vec)
            weights[i] = get_weight(g, 1, opponent)
        end
    end
end

Profile.clear()
interaction_test()
@profview interaction_test()

# birth! のパフォーマンスチェック
function birth_test(N::Int = 50_000, child_id::Int = 1)::Nothing
    for _ in 1:100
        # 1: bad
        options = collect(1:N)
        deleteat!(options, child_id)
        chosen = rand(options)

        # 2: best
        chosen = child_id
        while chosen == child_id
            chosen = rand(1:N)
        end

        # 3: worst
        options = setdiff(1:N, [child_id])
        chosen = rand(options)
    end
end

Profile.clear()
birth_test()
@profview birth_test()
==#

Profile.clear()
param = Simulation.Param(
    initial_N = 2000,
    initial_k = 100,
    initial_w = 0.2,
    δ = 1.0,
    β = 0.1,
    sigma = 20.0,
    generations = 200,
    variability_mode = Simulation.POPULATION,
)
Simulation.run(param, log_level = 2)  # 初回実行時のオーバーヘッドを避けることで、プロファイリングの正確性を向上させる
@profview Simulation.run(param, log_level = 2)

Profile.clear()
param = Simulation.Param(
    initial_N = 2000,
    initial_k = 100,
    initial_w = 0.2,
    δ = 1.0,
    β = 0.1,
    sigma = 0.1,
    generations = 200,
    variability_mode = Simulation.PAYOFF,
)
Simulation.run(param, log_level = 2)
@profview Simulation.run(param, log_level = 2)

Profile.clear()
param = Simulation.Param(
    initial_N = 2000,
    initial_k = 100,
    initial_w = 0.2,
    δ = 1.0,
    initial_μ_s = 0.1,
    initial_μ_c = 0.1,
    β = 0.1,
    sigma = 0.1,
    generations = 200,
    variability_mode = Simulation.MUTATION,
)
Simulation.run(param, log_level = 2)
@profview Simulation.run(param, log_level = 2)
