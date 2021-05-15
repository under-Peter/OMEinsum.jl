using OMEinsum, StatsBase
using DelimitedFiles

function random_contract(D, dynamic, uniqueindex)
    Ds = [rand(1:D) for _=1:3]
    T = Ds[1] + Ds[2]
    ixs = Tuple.((sample(1:T, Ds[1]; replace=!uniqueindex), sample(1:T, Ds[2]; replace=!uniqueindex)))
    allinds = ixs[1] ∪ ixs[2]
    iy = sample(allinds, min(Ds[3], length(allinds)); replace=!uniqueindex)
    iy = Tuple(iy ∪ setdiff(allinds, iy))  # no dangling legs
    xs = (randn(fill(2, Ds[1])...), randn(fill(2, Ds[2])...))
    if dynamic
        OMEinsum.dynamic_einsum(ixs, xs, iy)
    else
        EinCode{ixs,iy}()(xs...)
    end
end

function benchmark_compiletime(; D, nrepeat, dynamic, uniqueindex)
    times = zeros(nrepeat)
    for i=1:nrepeat
        @show i
        times[i] = @elapsed random_contract(D, dynamic, uniqueindex)
    end
    output_file=joinpath(@__DIR__, "compiletime_$(D)_$(dynamic)_$(uniqueindex).dat")
    writedlm(output_file, times)
    return times
end

benchmark_compiletime(D=8, nrepeat=1000, dynamic=false, uniqueindex=true)

#using Profile
#@profile for i=1:10 random_contract(8, true, true) end