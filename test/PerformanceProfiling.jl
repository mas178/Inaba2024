using Profile
using ProfileView

using Graphs

include("../src/Simulation.jl")
using .Simulation: Simulation

Profile.clear()
param = Simulation.Param(
    initial_N = 2000,
    initial_k = 200,
    initial_w = 0.2,
    δ = 1.0,
    β = 0.1,
    sigma = 20.0,
    generations = 200,
    variability_mode = Simulation.POPULATION,
)
Simulation.run(param, log_level = 2)  # 初回実行時のオーバーヘッドを避けることで、プロファイリングの正確性を向上させる
@profview Simulation.run(param, log_level = 2)

# Profile.clear()
# param = Simulation.Param(
#     initial_N = 2000,
#     initial_k = 100,
#     initial_w = 0.2,
#     δ = 1.0,
#     β = 0.1,
#     sigma = 0.1,
#     generations = 200,
#     variability_mode = Simulation.PAYOFF,
# )
# Simulation.run(param, log_level = 2)
# @profview Simulation.run(param, log_level = 2)

# Profile.clear()
# param = Simulation.Param(
#     initial_N = 2000,
#     initial_k = 100,
#     initial_w = 0.2,
#     δ = 1.0,
#     initial_μ_s = 0.1,
#     initial_μ_c = 0.1,
#     β = 0.1,
#     sigma = 0.1,
#     generations = 200,
#     variability_mode = Simulation.MUTATION,
# )
# Simulation.run(param, log_level = 2)
# @profview Simulation.run(param, log_level = 2)

#==
function degree1(weights::Matrix{Float16}, node::Int)::Int
    return count(>(Float16(0.0)), weights[node, :])
end

function degree2(weights::SparseMatrixCSC{Float16}, node::Int)::Int
    return nnz(weights[node, :])
end

function degree3(weights::Matrix{Float16}, node::Int)::Int
    counter = 0
    for i in 1:size(weights, 2)
        if weights[node, i] > 0
            counter += 1
        end
    end
    return counter
end

function degree4(weights::Matrix{Float16}, node::Int)::Int
    counter = 0
    @inbounds @simd for i in 1:size(weights, 2)
        if weights[node, i] > 0
            counter += 1
        end
    end
    return counter
end

function degree5(weights::Matrix{Float16}, node::Int)::Int
    return sum(weights[node, :] .> Float16(0))
end

function test_degree(weights::Matrix{Float16}, sparse_weights::SparseMatrixCSC{Float16})::Nothing
    for node in 1:10000
        degree1(weights, node)
        degree2(sparse_weights, node)
        degree3(weights, node)
        degree4(weights, node)
        # degree5(weights, node)
    end

    return
end

Profile.clear()
weights = Simulation.Network.create_adjacency_matrix(10000, 1000, Float16(0.5))
sparse_weights = sparse(weights)
Profile.init(n=10000000, delay=0.00000001)
test_degree(weights, sparse_weights) # 初回実行時のオーバーヘッドを避けることで、プロファイリングの正確性を向上させる
@profview test_degree(weights, sparse_weights)
==#
