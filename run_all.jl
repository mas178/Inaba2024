# cd ~/Dropbox/workspace/inaba2024
# nohup julia --threads 10 run_all.jl > run_all.log &
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate("../Inaba2024")
    include("./Simulation.jl")
    using .Simulation
    run_all(
         N_vec = [1_000],
         interaction_freqency_vec = [1.0],
         initial_graph_weight_vec = [0.2, 0.4],
         relationship_volatility_vec = [0.1],
         T_vec = [0.8, 1.2],
         S_vec = [-0.2, 0.2],
         δ_vec = [0.01, 0.1, 1.0],
         μ_vec = [0.01],
         generations_vec = [10_000],
    )
end
