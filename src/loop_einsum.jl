using TupleTools, Base.Cartesian
export loop_einsum, loop_einsum!, allow_loops

"""
    loop_einsum(::EinCode, xs, size_dict)

evaluates the eincode specified by `EinCode` and the tensors `xs` by looping
over all possible indices and calculating the contributions ot the result.
Scales exponentially in the number of distinct index-labels.
"""
function loop_einsum(code::EinCode{ixs, iy}, xs::NTuple{N, AbstractArray{<:Any,M} where M},
                size_dict) where {N,T, ixs, iy}
    size = getindex.(Ref(size_dict), iy)
    loop_einsum!(code, xs, get_output_array(xs, size), size_dict)
end

"""
    loop_einsum!(::EinCode, xs, y, size_dict)

inplace-version of `loop_einsum`, saving the result in a preallocated tensor
of correct size `y`.
"""
function loop_einsum!(code::EinCode{ixs, iy},
                xs::NTuple{N, AbstractArray{<:Any,M} where M},
                y::AbstractArray{T,L}, size_dict) where {N,L,T,IT <: Union{AbstractChar,Integer}, ixs, iy}
    ALLOW_LOOPS[] || @error "using `loop_einsum` to evaluate" code size.(xs) size(y)
    A = einarray(code, xs, size_dict)
    reduce_einarray!(A, y)
end

function reduce_einarray!(A::EinArray, y)
    @inbounds for ind_y in A.OCIS
        iy = subindex(A.y_indexer,ind_y)
        for ind_x in A.ICIS
            y[iy] += map_prod(A.xs, TupleTools.vcat(ind_x.I,ind_y.I), A.x_indexers)
        end
    end
    y
end

function get_output_array(xs::NTuple{N, AbstractArray{<:Any,M} where M}, size) where N
    zeros(promote_type(map(eltype,xs)...), size)
end

const ALLOW_LOOPS = Ref(true)

"""
    allow_loops(flag::Bool)

Setting this to `false` will cause OMEinsum to log an error if it falls back to
`loop_einsum` evaluation, instead of calling specialised kernels. The default is `true`.
"""
function allow_loops(flag::Bool)
    ALLOW_LOOPS[] = flag
end
