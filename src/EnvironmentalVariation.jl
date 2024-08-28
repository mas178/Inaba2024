module EnvironmentalVariation

using Random: MersenneTwister
using StatsBase: std, describe

export ar1, two_level_var, prune, smoothen, desc

"""
AR(1) Model to create environmental volatility.
"""
function ar1(
    μ::T,               # expected average value
    σ::T,               # std of white noise
    β::Float64,         # 自己回帰の係数。|β| < 1 の場合に平均回帰性を持つ。μ = α / (1 - β)
    τ::Int,             # time scale
    t_max::Int,         # time steps
    rng::MersenneTwister = MersenneTwister()
) where {T<:Union{Int64, Float64}}
    t_temp = ceil(Int, t_max / τ)
    x = Vector{T}(undef, t_temp)
    x[1] = μ
    alpha = μ * (1 - β)  # 定数項。μから逆算する。
    for t = 2:t_temp
        noise = σ * randn(rng)
        x_temp = alpha + β * x[t - 1] + noise
        if typeof(μ) == Int
            x_temp = round(Int64, x_temp)
        end
        x[t] = x_temp
    end

    return repeat(x, inner = τ)[1:t_max]::Vector{T}
end

"""
low_level, low_level_span, high_level, high_level_span, time steps:`t_max`の2準位の波`Vector{T}`を返す。
"""
function two_level_var(low_level::T, low_level_span::Int, high_level::T, high_level_span::Int, t_max::Int) where {T<:Union{Int, Float64}}
    @assert (low_level_span > 0 || high_level_span > 0) "low_level_span = $(low_level_span), high_level_span = $(high_level_span)"

    one_wave = [fill(low_level, low_level_span); fill(high_level, high_level_span)]
    wave_count = ceil(Int, t_max / (low_level_span + high_level_span))
    
    return repeat(one_wave, wave_count)[1:t_max]::Vector{T}
end

function prune(var_vec::Vector{T}, ceiling::T, floor::T) where {T<:Union{Int, Float64}}
    t_max = length(var_vec)

    for t in 1:t_max
        var_vec[t] = min(var_vec[t], ceiling)
        var_vec[t] = max(var_vec[t], floor)
    end

    return var_vec::Vector{T}
end

function smoothen(var_vec::Vector{T}, var_rate::Float64) where {T<:Union{Int, Float64}}
    t_max = length(var_vec)
    is_Int = typeof(var_vec[1]) == Int

    for t in 1:(t_max - 1)
        if var_vec[t + 1] / var_vec[t] > var_rate
            temp = var_vec[t] * var_rate
            var_vec[t + 1] = is_Int ? round(Int, temp) : temp
        elseif var_vec[t + 1] / var_vec[t] < 1 / var_rate
            temp = var_vec[t] / var_rate
            var_vec[t + 1] = is_Int ? round(Int, temp) : temp
        end
    end

    return var_vec::Vector{T}
end

function desc(var_vec::Vector{T}) where {T<:Union{Int, Float64}}
    describe(var_vec)
    println("Std:            $(round(std(var_vec), digits = 6))")
    return
end

end # end of module