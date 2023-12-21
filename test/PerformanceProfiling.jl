using Profile
using ProfileView

include("../src/Simulation.jl")
using .Simulation: Simulation

Profile.clear()
param = Simulation.Param(
    initial_N = 200,
    initial_graph_weight = 0.2,
    δ = 1.0,
    β = 0.3,
    sigma = 20.0,
    generations = 200,
    variability_mode = Simulation.POPULATION,
)
Simulation.run(param, log_level = 2)  # 初回実行時のオーバーヘッドを避けることで、プロファイリングの正確性を向上させる
@profview Simulation.run(param, log_level = 2)

Profile.clear()
param = Simulation.Param(
    initial_N = 200,
    initial_graph_weight = 0.2,
    δ = 1.0,
    β = 0.3,
    sigma = 20.0,
    generations = 200,
    variability_mode = Simulation.PAYOFF,
)
Simulation.run(param, log_level = 2)  # 初回実行時のオーバーヘッドを避けることで、プロファイリングの正確性を向上させる
@profview Simulation.run(param, log_level = 2)
