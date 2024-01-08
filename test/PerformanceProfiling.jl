using Profile
using ProfileView

using Graphs

include("../src/Simulation.jl")
using .Simulation: Simulation

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
