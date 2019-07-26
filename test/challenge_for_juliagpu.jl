# `]add OMEinsum#master`
# `]add TupleTools`
using OMEinsum, CuArrays

using CUDAnative, TupleTools
using OMEinsum: cudiv, index_map, map_prod
using Base.Cartesian

function OMEinsum.loop!(locs_xs::NTuple{N,Any}, xs::NTuple{N, CuArray}, locs_y, y::CuArray{T}, outer_ci::CartesianIndices, inner_ci::CartesianIndices) where {N, T}
    function loop_kernel(locs_xs, xs, locs_y, y, outer_ci, inner_ci)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        i > length(outer_ci) && return nothing
        @inbounds ind_y = outer_ci[i]
        iy = index_map(ind_y, locs_y)
        for ind_x in inner_ci
            ind_xy = CartesianIndex(TupleTools.vcat(ind_y.I, ind_x.I))
            @inbounds y[iy] += map_prod(T, xs, ind_xy, locs_xs)
        end
        nothing
    end
    X, Y = cudiv(length(outer_ci))
    @cuda threads=X blocks=Y loop_kernel(locs_xs, xs, locs_y, y, outer_ci, inner_ci)
    y
end

"""get an item from each tensor, and return the product of them"""
@inline @generated function OMEinsum.map_prod(::Type{T}, xs::Tuple, ind::CartesianIndex, locs_xs::NTuple{N,Any}) where {N, T}
    quote
        p = one(T)
        @nexprs $N i -> @inbounds p *= xs[i][CartesianIndex(TupleTools.getindices(ind.I, locs_xs[i]))]
    end
end

using Test, BenchmarkTools
a = randn(Float32, 100, 50)
ca = CuArray(a)
@test maximum(abs.(Array(ein"ij,ik,il->jkl"(ca, ca, ca)) - ein"ij,ik,il->jkl"(a, a, a))) < 1e-4

# task: accelerate the following code by a factor of 50 (close to pytorch performance then).
@benchmark (CuArrays.@sync ein"ij,ik,il->jkl"($ca, $ca, $ca)) seconds=1
