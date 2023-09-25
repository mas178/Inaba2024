module SimPlot

using CSV
using DataFrames: DataFrame, SubDataFrame, nrow
using Glob: glob
using Plots: Plot, plot!, plot, heatmap, twinx, PlotMeasures
using StatsBase: mean

function csv_to_df(dir_name_vec::Vector{String})::Vector{DataFrame}
    csv_file_names = []

    for dir_name in dir_name_vec
        append!(csv_file_names, glob("*.csv", "../output/$(dir_name)"))
    end

    return [CSV.File(csv_file_name) |> DataFrame for csv_file_name in csv_file_names]
end

function plot_network_attributes(df::DataFrame, prefix::String, skip::Int)::Plot
    df = df[df.generation .% skip .== 0, :]
    p = plot(xl = "Generation", title = "Network Attributes ($(prefix))")
    plot!(df.generation, df.weight_μ, ribbon = df.weight_σ, fillalpha = 0.5, label = "Weight")
    plot!(df.generation, df[:, Symbol("$(prefix)_L")], label = "L")
    plot!(df.generation, df[:, Symbol("$(prefix)_C")], label = "C")
    plot!(twinx(), df.generation, df[:, Symbol("$(prefix)_k")], label = "<k>", line = :dash)

    return p
end

function plot_component_attributes(df::DataFrame, prefix::String, weight_order::Int, skip::Int)::Plot
    df = df[df.generation .% skip .== 0, :]
    p = plot(xl = "Generation", title = "Component Attributes ($(prefix))")
    component_size_μ = Symbol("$(prefix)_component$(weight_order)_size_μ")
    component_size_min = Symbol("$(prefix)_component$(weight_order)_size_min")
    component_size_max = Symbol("$(prefix)_component$(weight_order)_size_max")
    component_size_σ = Symbol("$(prefix)_component$(weight_order)_size_σ")
    component_count = Symbol("$(prefix)_component$(weight_order)_count")
    plot!(
        df.generation,
        df[:, component_size_μ],
        ribbon = (df[:, component_size_μ] - df[:, component_size_min], df[:, component_size_σ]),
        fillalpha = 0.5,
        label = "Size",
    )
    plot!(df.generation, df[:, component_size_max], label = "Size (Max)")
    plot!(twinx(), df.generation, df[:, component_count], label = "Count", line = :dash, yscale = :log10)

    return p
end

function plot_output_df(df::DataFrame, skip::Int = 10)::Plot
    p1 = plot(xl = "Generation", title = "Population")
    plot!(df.generation, df.birth_rate, label = "Birth Rate")
    plot!(df.generation, df.death_rate, label = "Death Rate")
    plot!(twinx(), df.generation, df.N, label = "Population", line = :dash)

    p2 = plot(xl = "Generation", title = "Cooperation")
    plot!(df.cooperation_rate, label = "Cooperation Rate")
    plot!(df.payoff_μ, label = "Payoff (μ)")

    params1 = join(["$(k) = $(v)" for (k, v) in pairs(df[1, [2, 3, 10, 11]])], ", ")
    params2 = join(["$(k) = $(v)" for (k, v) in pairs(df[1, 4:9])], ", ")

    return plot(
        p1,
        p2,
        plot_network_attributes(df, "weak", skip),
        plot_component_attributes(df, "weak", 1, skip),
        plot_network_attributes(df, "medium", skip),
        plot_component_attributes(df, "medium", 1, skip),
        plot_network_attributes(df, "strong", skip),
        plot_component_attributes(df, "strong", 1, skip),
        layout = (4, 2),
        size = (800, 1600),
        bottom_margin = 6 * PlotMeasures.mm,
        suptitle = "$(params1)\n$(params2)",
        plot_titlefontsize = 10,
    )
end

function calc_mean(df::DataFrame)::DataFrame
    df = df[df.generation .% 10 .== 0, :]

    generations = nrow(df)
    _df = DataFrame(df[1, 1:11])
    start = round(Int, generations * 0.1)
    _df.cooperation_rate .= mean(df.cooperation_rate[start:end])
    _df.component_count .= mean(df.component_count[start:end])
    _df.component_size_μ .= mean(df.component_size_μ[start:end])
    _df.component_size_max .= mean(df.component_size_max[start:end])
    _df.strong_component_count .= mean(df.strong_component_count[start:end])
    _df.strong_component_size_μ .= mean(df.strong_component_size_μ[start:end])
    _df.strong_component_size_max .= mean(df.strong_component_size_max[start:end])

    return _df
end

function make_mean_df(df_vec::Vector{DataFrame})::DataFrame
    mean_df = vcat([calc_mean(df) for df in df_vec]...)

    return sort(mean_df, names(mean_df)[4:11])
end

function get_value(
    df::DataFrame,
    x::Float64,
    y::Float64,
    x_symbol::Symbol,
    y_symbol::Symbol,
    value_symbol::Symbol,
)::Union{Float64,Missing}
    values = df[df[:, x_symbol] .== x .&& df[:, y_symbol] .== y, value_symbol]
    if length(values) > 0
        mean(values)
    else
        missing
    end
end

function plot_cooperation_heatmap(df::SubDataFrame)::Plot
    # using Makie
    # using WGLMakie
    # fig = Figure()
    # ax = Axis(fig[1, 1], xlabel = "T", ylabel = "S")
    # ax.limits = (0, 2, -1, 1)
    # hm = WGLMakie.heatmap!(ax, df.T, df.S, df.cooperation_rate)  # , interpolate = true
    # cbar = Colorbar(fig[1, 2], hm, label = "cooperation rate")
    # return fig

    title1 = join(["$(k) = $(v)" for (k, v) in pairs(df[1, 4:6])], ", ")
    title2 = join(["$(k) = $(v)" for (k, v) in pairs(df[1, 7:11])], ", ")

    df = df[:, [:T, :S, :cooperation_rate]]
    T = sort(unique(df.T))
    S = sort(unique(df.S))

    mat = reshape([get_value(df, t, s, :T, :S, :cooperation_rate) for s in S, t in T], length(S), length(T))

    p = heatmap(
        T,
        S,
        mat,
        xlabel = "T",
        ylabel = "S",
        xlims = (-0.05, 2.05),
        ylims = (-1.05, 1.05),
        title = "$(title1)\n$(title2)",
        titlefontsize = 9,
    )
    plot!([-0.05, 2.05], [0, 0], color = :gray, lw = 2, legend = false)
    plot!([1, 1], [-1.05, 1.05], color = :gray, lw = 2, legend = false)
    plot!([2.05, 2.05], [-1.05, 1.05], color = :black, lw = 0.5, legend = false)
    plot!([-0.05, 2.05], [1.05, 1.05], color = :black, lw = 0.5, legend = false)

    return p
end

end  # end of module
