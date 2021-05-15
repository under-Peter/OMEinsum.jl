using BenchmarkTools, Plots

function plot_benchmark()
    res0 = BenchmarkTools.load(joinpath(@__DIR__, "OMEinsum_master.json"))[]
    res1 = BenchmarkTools.load(joinpath(@__DIR__, "OMEinsum_refactor.json"))[]
    t0 = Float64[]
    t1 = Float64[]
    xl = String[]
    for k in keys(res0)
        push!(t0, minimum(res0[k]["Float64"]["large"].times))
        push!(t1, minimum(res1[k]["Float64"]["large"].times))
        push!(xl, k)
    end
    xs = 1:length(res0)
    plot(xs, t0, marker = (:hexagon, 10, 0.6, :green, stroke(3, 0.2, :black, :dot)), yscale=:log10, lw=0, label="master")
    for i in xs
        annotate!(i+0.3, t1[i], text(xl[i], :black, :left, 12))
    end
    plot!(xs, t1, marker = (:circle, 10, 0.6, :red, stroke(3, 0.2, :black, :dot)), lw=0, label="refactor")
    #Plots.xticks!(xs, xl)
end

function plot_compile()
    pl = []
    for dynamic in [true, false]
        for uniqueindex in [true]
            times = readdlm(joinpath(@__DIR__, "compiletime_8_$(dynamic)_$(uniqueindex).dat"))
            push!(pl, plot(1:1000, times, label=(dynamic ? "dynamic" : "static") * " ($(round(sum(times); sigdigits=2)))"))
        end
    end
    plt = plot(pl..., xlabel="iteration", ylabel="time", layout=(2,1))
end
#plot_benchmark()
plot_compile()
