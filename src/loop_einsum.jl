"""
    loop_einsum(::EinCode, xs, size_dict)

evaluates the eincode specified by `EinCode` and the tensors `xs` by looping
over all possible indices and calculating the contributions ot the result.
Scales exponentially in the number of distinct index-labels.
"""
function loop_einsum(code::EinCode, xs::NTuple{N, AbstractArray{<:Any,M} where M},
                size_dict) where {N}
    iy = getiy(code)
    size = getindex.(Ref(size_dict), iy)
    loop_einsum!(code, xs, get_output_array(xs, size; has_repeated_indices=!allunique(iy)), size_dict)
end

"""
    loop_einsum!(::EinCode, xs, y, size_dict)

inplace-version of `loop_einsum`, saving the result in a preallocated tensor
of correct size `y`.
"""
function loop_einsum!(code::EinCode,
                xs::NTuple{N, AbstractArray{<:Any,M} where M},
                y::AbstractArray{T,L}, size_dict) where {N,L,T}
    ALLOW_LOOPS[] || error("using `loop_einsum` is forbidden: code: $code")
    A = einarray(Val((Tuple.(getixs(code))...,)), Val((getiy(code)...,)), xs, size_dict)
    reduce_einarray!(A, y)
end

function reduce_einarray!(A::EinArray{T}, y) where T
    @inbounds for ind_y in A.OCIS
        iy = subindex(A.y_indexer,ind_y)
        yi = zero(T)
        for ind_x in A.ICIS
            ind = TupleTools.vcat(ind_x.I,ind_y.I)
            yi += map_prod(A.xs, ind, A.x_indexers)
        end
        y[iy] = yi
    end
    y
end

# speed up the get output array for the case when the inputs have the same type.
get_output_eltype(::NTuple{N, AbstractArray{T,M} where M}) where {T,N} = T
get_output_eltype(xs::NTuple{N, AbstractArray{<:Any,M} where M}) where N = promote_type(map(eltype,xs)...)
function get_output_array(xs::NTuple{N, AbstractArray{<:Any,M} where M}, size; has_repeated_indices=true) where N
    T = get_output_eltype(xs)
    if has_repeated_indices
        zeros(T, size...)::Array{T}
    else
        Array{T}(undef, size...)::Array{T}
    end
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
