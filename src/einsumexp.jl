using TupleTools, Base.Cartesian
export einsumexp, einsumexp!, EinCode

struct EinCode{ixs, iy} end
EinCode(ixs::NTuple{N, NTuple{M, T} where M},iy::NTuple{<:Any,T}) where {N, T} = EinCode{ixs, iy}()

"""
    einsumexp(::EinCode, xs, size_dict)

The brute-force looping einsum, `xs` is a tuple of input tensors.
"""
function einsumexp(code::EinCode{ixs, iy},
                xs::NTuple{N, AbstractArray{<:Any,M} where M},
                size_dict) where {N,T, ixs, iy}
    TO = promote_type(map(eltype,xs)...)
    out = zeros(TO, getindex.(Ref(size_dict), iy))
    einsumexp!(code, xs, out, size_dict)
end

"""
    einsumexp!(::EinCode, xs, y, size_dict)

The inplace brute-force looping einsum, `y` is the output tensor.
"""
@generated function einsumexp!(::EinCode{ixs, iy},
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

        loop!($locs_xs, xs, $locs_y, y, outer_ci, inner_ci)
    end
end

"""indiex tensors, and return the product of elements"""
@inline @generated function map_prod(::Type{T}, xs::Tuple, ind, locs_xs::NTuple{N,Any}) where {N, T}
    quote
        p = one(T)
        @nexprs $N i -> @inbounds p *= xs[i][index_map(ind, locs_xs[i])]
    end
end

"""
loop and accumulate products to y, the CPU version.
"""
function loop!(locs_xs::NTuple{N,Any}, xs::NTuple{N, AbstractArray}, locs_y, y::AbstractArray{T}, outer_ci::CartesianIndices, inner_ci::CartesianIndices) where {N, T}
    @simd for i in outer_ci
        @inbounds ind_y = outer_ci[i]
        iy = index_map(ind_y, locs_y)
        for ind_x in inner_ci
            ind_xy = CartesianIndex(TupleTools.vcat(ind_y.I, ind_x.I))
            @inbounds y[iy] += map_prod(T, xs, ind_xy, locs_xs)
        end
    end
    y
end

"""take an index subset from `ind`"""
index_map(ind::CartesianIndex, locs::Tuple) = CartesianIndex(TupleTools.getindices(ind.I, locs))
index_map(ind::Tuple, locs::Tuple) = CartesianIndex(TupleTools.getindices(ind, locs))

# get inner indices, outer indices,
# locations of input indices in total indices
# and locations of output indices in outer indices.
function indices_and_locs(ixs, iy)
    # outer legs and inner legs
    outer_indices = tunique(iy)
    inner_indices = tsetdiff(TupleTools.vcat(ixs...), outer_indices)

    # for indexing tensors (leg binding)
    indices = (outer_indices..., inner_indices...)
    locs_xs = map(ixs) do ix
        map(i->findfirst(isequal(i), indices), ix)
    end
    locs_y = map(i->findfirst(isequal(i), outer_indices), iy)
    return inner_indices, outer_indices, locs_xs, locs_y
end
