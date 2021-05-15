using OMEinsum, StatsBase
using DelimitedFiles

function random_contract(D, dynamic, uniqueindex, nodangling)
    Ds = [rand(1:D) for _=1:3]
    T = Ds[1] + Ds[2]
    ixs = Tuple.((sample(1:T, Ds[1]; replace=!uniqueindex), sample(1:T, Ds[2]; replace=!uniqueindex)))
    allinds = ixs[1] ∪ ixs[2]
    iy = sample(allinds, min(Ds[3], length(allinds)); replace=!uniqueindex)
    if nodangling
        iy = Tuple(iy ∪ setdiff(allinds, iy))  # no dangling legs
    else
        iy = Tuple(iy)
    end
    xs = (randn(fill(2, Ds[1])...), randn(fill(2, Ds[2])...))
    if dynamic
        OMEinsum.dynamic_einsum(ixs, xs, iy)
    else
        EinCode{ixs,iy}()(xs...)
    end
end

function benchmark_compiletime(; D, nrepeat, dynamic, uniqueindex, nodangling, write=false)
    times = zeros(nrepeat)
    for i=1:nrepeat
        @show i
        times[i] = @elapsed random_contract(D, dynamic, uniqueindex, nodangling)
    end
    if write
        output_file=joinpath(@__DIR__, "compiletime_$(D)_$(dynamic)_$(uniqueindex).dat")
        writedlm(output_file, times)
    end
    return times
end

times = benchmark_compiletime(D=8, nrepeat=1000, dynamic=true, uniqueindex=true, nodangling=true, write=false)

#using Profile
#@profile for i=1:10 random_contract(8, true, true, true) end