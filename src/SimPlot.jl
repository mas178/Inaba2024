module SimPlot

using DataFrames: DataFrame, SubDataFrame, nrow
using Plots: Plot, plot!, plot, heatmap, twinx, PlotMeasures
using StatsBase: mean

function plot_output_df(df::DataFrame, skip::Int = 10)::Plot
    p1 = plot(xl = "Generation", title = "Population")
    plot!(df.generation, df.birth_rate, label = "Birth Rate")
    plot!(df.generation, df.death_rate, label = "Death Rate")
    plot!(twinx(), df.generation, df.N, label = "Population", line = :dash)

    p2 = plot(xl = "Generation", title = "Cooperation")
    plot!(df.cooperation_rate, label = "Cooperation Rate")
    plot!(df.payoff_μ, label = "Payoff (μ)")

    df = df[df.generation.%skip.==0, :]

    p3 = plot(xl = "Generation", title = "Network Attributes (Weak)")
    plot!(df.generation, df.weight_μ, ribbon = df.weight_σ, fillalpha = 0.5, label = "Weight")
    plot!(df.generation, df.L, label = "L")
    plot!(df.generation, df.C, label = "C")
    plot!(twinx(), df.generation, df.k, label = "<k>", line = :dash)

    p4 = plot(xl = "Generation", title = "Network Attributes (Strong)")
    plot!(df.generation, df.weight_μ, ribbon = df.weight_σ, fillalpha = 0.5, label = "Weight")
    plot!(df.generation, df.strong_L, label = "L")
    plot!(df.generation, df.strong_C, label = "C")
    plot!(twinx(), df.generation, df.strong_k, label = "<k>", line = :dash)

    p5 = plot(xl = "Generation", title = "Component Attributes (Weak)")
    plot!(
        df.generation,
        df.component_size_μ,
        ribbon = (df.component_size_μ - df.component_size_min, df.component_size_σ),
        fillalpha = 0.5,
        label = "Size",
    )
    plot!(df.generation, df.component_size_max, label = "Size (Max)")
    plot!(twinx(), df.generation, df.component_count, label = "Count", line = :dash, yscale = :log10)

    p6 = plot(xl = "Generation", title = "Component Attributes (Strong)")
    plot!(
        df.generation,
        df.strong_component_size_μ,
        ribbon = (df.strong_component_size_μ - df.strong_component_size_min, df.strong_component_size_σ),
        fillalpha = 0.5,
        label = "Size",
    )
    plot!(df.generation, df.strong_component_size_max, label = "Size (Max)")
    plot!(twinx(), df.generation, df.strong_component_count, label = "Count", line = :dash, yscale = :log10)

    params1 = join(["$(k) = $(v)" for (k, v) in pairs(df[1, [2, 3, 10, 11]])], ", ")
    params2 = join(["$(k) = $(v)" for (k, v) in pairs(df[1, 4:9])], ", ")

    return plot(
        p1,
        p2,
        p3,
        p5,
        p4,
        p6,
        layout = (3, 2),
        size = (800, 1200),
        bottom_margin = 6 * PlotMeasures.mm,
        suptitle = "$(params1)\n$(params2)",
        plot_titlefontsize = 10,
    )
end

function calc_cooperation_rate(df::DataFrame)::DataFrame
    generations = nrow(df)
    _df = DataFrame(df[1, 1:11])
    _df.cooperation_rate .= mean(df.cooperation_rate[round(Int, generations * 0.1):end])

    return _df
end

function make_cooperation_df(df_vec::Vector{DataFrame})::DataFrame
    cooperation_df = vcat([calc_cooperation_rate(df) for df in df_vec]...)

    return sort(cooperation_df, names(cooperation_df)[4:11])
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

    df = sort(df, [:T, :S])[:, [:T, :S, :cooperation_rate]]
    T = unique(df.T)
    S = unique(df.S)

    mat = reshape([df[df.T.==t.&&df.S.==s, :cooperation_rate][1] for s in S, t in T], length(S), length(T))

    return heatmap(T, S, mat, xlabel = "T", ylabel = "S", title = "$(title1)\n$(title2)", titlefontsize = 9)
end

end  # end of module
