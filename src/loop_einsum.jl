using TupleTools, Base.Cartesian
export loop_einsum, loop_einsum!

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

        x_indexers = einindexer.(size.(xs), $locs_xs)
        y_indexer = einindexer(size(y), $locs_y)

        loop!(x_indexers, xs, y_indexer, y, outer_ci, inner_ci)
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
            @inbounds y[iy] += map_prod(xs, ind_xy, x_indexers)
        end
    end
    y
end
