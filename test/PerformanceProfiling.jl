using Profile
using ProfileView

include("../src/EntryPoint.jl")
using .EntryPoint: run
using .EntryPoint.Output.Simulation: Param

Profile.clear()
param = Param(initial_N = 200, initial_graph_weight = 0.2, δ = 1.0, β = 0.01, σ = 0.01, generations = 200)
run(param)  # 初回実行時のオーバーヘッドを避けることで、プロファイリングの正確性を向上させる
@profview run(param)
