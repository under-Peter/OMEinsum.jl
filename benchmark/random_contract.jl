using OMEinsum, StatsBase
using DelimitedFiles

function random_contract(D, uniqueindex, dynamic)
    Ds = [rand(1:D) for _=1:3]
    T = Ds[1] + Ds[2]
    ixs = Tuple.((sample(1:T, Ds[1]; replace=!uniqueindex), sample(1:T, Ds[2]; replace=!uniqueindex)))
    allinds = ixs[1] ∪ ixs[2]
    iy = Tuple(sample(allinds, min(Ds[3], length(allinds)); replace=!uniqueindex))
    xs = (randn(fill(2, Ds[1])...), randn(fill(2, Ds[2])...))
    if dynamic
        OMEinsum.dynamic_einsum(ixs, xs, iy)
    else
        EinCode{ixs,iy}()(xs...)
    end
end

function benchmark_compiletime(; D, nrepeat, dynamic, uniqueindex)
    times = zeros(nrepeat)
    for i=1:100
        push!(times, @elapsed random_contract(8, true, true))
    end
    output_file=join(@__DIR__, "benchmark", "compiletime_$(D)_$(dynamic)_$(uniqueindex).dat")
    writedlm(output_file, times)
    return times
end

benchmark_compiletime(D=8, nrepeat=1000, dynamic=true, uniqueindex=true)

#using Profile
#@profile for i=1:10 random_contract(8, true, true) end