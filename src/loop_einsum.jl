using TupleTools, Base.Cartesian
export loop_einsum, loop_einsum!, EinCode

struct EinCode{ixs, iy} end
EinCode(ixs::NTuple{N, NTuple{M, T} where M},iy::NTuple{<:Any,T}) where {N, T} = EinCode{ixs, iy}()

"""
    loop_einsum(::EinCode, xs, size_dict)

The brute-force looping einsum, `xs` is a tuple of input tensors.
"""
function loop_einsum(code::EinCode{ixs, iy},
                xs::NTuple{N, AbstractArray{<:Any,M} where M},
                size_dict) where {N,T, ixs, iy}
    TO = promote_type(map(eltype,xs)...)
    out = zeros(TO, getindex.(Ref(size_dict), iy))
    loop_einsum!(code, xs, out, size_dict)
end

"""
    loop_einsum!(::EinCode, xs, y, size_dict)

The inplace brute-force looping einsum, `y` is the output tensor.
"""
@generated function loop_einsum!(::EinCode{ixs, iy},
                xs::NTuple{N, AbstractArray{<:Any,M} where M},
                y::AbstractArray{T,L}, size_dict) where {N,L,T,IT <: Union{AbstractChar,Integer}, ixs, iy}
    inner_indices, outer_indices, locs_xs, locs_y = indices_and_locs(ixs, iy)

    quote
        # find size for each leg
        outer_sizes = getindex.(Ref(size_dict), $outer_indices)
        inner_sizes = getindex.(Ref(size_dict), $inner_indices)

        # cartesian indices for outer and inner legs
        outer_ci = CartesianIndices((outer_sizes...,))
        inner_ci = CartesianIndices((inner_sizes...,))

        x_indexers = EinIndexer.(size.(xs), $locs_xs)
        y_indexer = EinIndexer(size(y), $locs_y)

        loop!(x_indexers, xs, y_indexer, y, outer_ci, inner_ci)
    end
end

"""indiex tensors, and return the product of elements"""
@inline @generated function map_prod(::Type{T}, xs::Tuple, ind, indexers::NTuple{N,Any}) where {N, T}
    quote
        p = one(T)
        @nexprs $N i -> @inbounds p *= xs[i][subindex(indexers[i], ind)]
    end
end

"""
loop and accumulate products to y, the CPU version.
"""
function loop!(x_indexers::NTuple{N,Any}, xs::NTuple{N, AbstractArray}, y_indexer, y::AbstractArray{T}, outer_ci::CartesianIndices, inner_ci::CartesianIndices) where {N, T}
    for ind_y in outer_ci
        iy = subindex(y_indexer,ind_y)
        for ind_x in inner_ci
            ind_xy = TupleTools.vcat(ind_x.I, ind_y.I)
            @inbounds y[iy] += map_prod(T, xs, ind_xy, x_indexers)
        end
    end
    y
end

"""take an index subset from `ind`"""
index_map(ind::CartesianIndex, locs::Tuple) = CartesianIndex(TupleTools.getindices(ind.I, locs))
index_map(ind::Tuple, locs::Tuple) = CartesianIndex(TupleTools.getindices(ind, locs))

export EinIndexer
struct EinIndexer{N}
    locs::NTuple{N,Int}
    cumsize::NTuple{N,Int}
    function EinIndexer(size::NTuple{N,Int}, locs::NTuple{N,Int}) where N
        N==0 && return new{0}((), ())
        new{N}(locs, (1,cumprod(size[1:end-1])...))
    end
end

subindex(indexer::EinIndexer, ind::CartesianIndex) = subindex(indexer, ind.I)
subindex(indexer::EinIndexer{0}, ind::Tuple) = 1
function subindex(indexer::EinIndexer{N}, ind::Tuple) where N
    @inbounds res = ind[indexer.locs[1]]
    for i = 2:N
        @inbounds res += (ind[indexer.locs[i]]-1) * indexer.cumsize[i]
    end
    return res
end

using Test
@testset "indexer" begin
    si = EinIndexer((), ())
    @test subindex(si, (1,2,3)) == 1
    si = EinIndexer((7,6), (3,2))
    a = randn(7,6)
    @test a[subindex(si, (4,5,2))] == a[2,5]
end

# get inner indices, outer indices,
# locations of input indices in total indices
# and locations of output indices in outer indices.
function indices_and_locs(ixs, iy)
    # outer legs and inner legs
    outer_indices = tunique(iy)
    inner_indices = tsetdiff(TupleTools.vcat(ixs...), outer_indices)

    # for indexing tensors (leg binding)
    indices = (inner_indices...,outer_indices...)
    locs_xs = map(ixs) do ix
        map(i->findfirst(isequal(i), indices), ix)
    end
    locs_y = map(i->findfirst(isequal(i), outer_indices), iy)
    return inner_indices, outer_indices, locs_xs, locs_y
end
