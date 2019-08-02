using TupleTools, Base.Cartesian
export loop_einsum, loop_einsum!

"""
    loop_einsum(::EinCode, xs, size_dict)

The brute-force looping einsum, `xs` is a tuple of input tensors.
"""
function loop_einsum(code::EinCode{ixs, iy}, xs::NTuple{N, AbstractArray{<:Any,M} where M},
                size_dict) where {N,T, ixs, iy}
    size = getindex.(Ref(size_dict), iy)
    loop_einsum!(code, xs, get_output_array(xs, size), size_dict)
end

function loop_einsum!(code::EinCode{ixs, iy},
                xs::NTuple{N, AbstractArray{<:Any,M} where M},
                y::AbstractArray{T,L}, size_dict) where {N,L,T,IT <: Union{AbstractChar,Integer}, ixs, iy}
    A = einarray(code, xs, size_dict)
    reduce_einarray(A, y)
end

function reduce_einarray(A::EinArray, y)
    @inbounds for ind_y in A.OCIS
        iy = subindex(A.y_indexer,ind_y)
        for ind_x in A.ICIS
            y[iy] += map_prod(A.xs, TupleTools.vcat(ind_x.I,ind_y.I), A.x_indexers)
        end
    end
    y
end

function get_output_array(xs::NTuple{N, AbstractArray{<:Any,M} where M}, size_dict)
    zeros(promote_type(map(eltype,xs)...), size)
end
