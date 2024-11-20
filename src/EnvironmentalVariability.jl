module EnvironmentalVariability

using Random: MersenneTwister
using StatsBase

"""
AR(1) Model to create environmental variability.
"""
function ar1(
    μ::Float64,         # expected average value
    β::Float64,         # Autoregressive coefficient. When |β| < 1, it exhibits mean reversion.
    σ::Float64,         # standard deviation (σ) of white noise
    generations::Int,   # time steps
    rng::MersenneTwister = MersenneTwister()
)::Vector{Float64}
    x = fill(μ, generations)
    alpha = μ * (1 - β) # μ = alpha / (1 - β)
    for t = 2:generations
        noise = σ * randn(rng)
        x_temp = alpha + β * x[t - 1] + noise
        x[t] = clamp(x_temp, 0.0, 1.0) # ensure that the value remains within the [0, 1] range
    end

    return x
end

end # end of module