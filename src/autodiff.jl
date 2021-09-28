using ChainRulesCore

@doc raw"
    einsum_grad(ixs, xs, iy, size_dict, cdy, i)

return the gradient of the result of evaluating the `EinCode` w.r.t
the `i`th tensor in `xs`. `cdy` is the result of applying the `EinCode`
to the `xs`.

# example
```jldoctest; setup = :(using OMEinsum)
julia> using OMEinsum: einsum_grad, get_size_dict

julia> a, b = rand(2,2), rand(2,2);

julia> c = einsum(EinCode((('i','j'),('j','k')), ('i','k')), (a,b));

julia> sd = get_size_dict((('i','j'),('j','k')), (a,b));

julia> einsum_grad((('i','j'),('j','k')), (a,b), ('i','k'), sd, c, 1) ≈ c * transpose(b)
true
```
"
function einsum_grad(ixs, xs, iy, size_dict, cdy, i)
    nixs = TupleTools.insertat(ixs, i, (iy,))
    nxs  = TupleTools.insertat( xs, i, (cdy,))
    niy = ixs[i]
    y = einsum(EinCode(nixs, niy), nxs, size_dict)
    try
        conj!(y)
    catch e
        y = conj(y)
    end
    typeof(y) == typeof(xs[i]) && return y
    xs[i] isa Array{<:Real} && return convert(typeof(xs[i]), real(y))
    convert(typeof(xs[i]), y)
end

function ChainRulesCore.rrule(::typeof(einsum), code::EinCode, xs::NTuple{N,T where T}, size_dict) where {N}
    y = einsum(code, xs, size_dict)
    function einsum_pullback(dy)
        dxs = ChainRulesCore.@thunk ntuple(i -> einsum_grad(getixs(code), xs, getiy(code), size_dict, map(conj, dy), i), N)
        return (NoTangent(), NoTangent(), dxs, NoTangent())
    end
    einsum_pullback(::NoTangent) = (NoTangent(), NoTangent(), NoTangent(), NoTangent())
    return y, einsum_pullback
end

function dynamic_einsum_grad(ixs, xs, iy, size_dict, cdy, i)
    nixs = TupleTools.insertat(ixs, i, (iy,))
    nxs  = TupleTools.insertat( xs, i, (cdy,))
    niy = ixs[i]
    y = dynamic_einsum(nixs, nxs, niy, size_dict)
    try
        conj!(y)
    catch e
        y = conj(y)
    end
    typeof(y) == typeof(xs[i]) && return y
    xs[i] isa Array{<:Real} && return convert(typeof(xs[i]), real(y))
    convert(typeof(xs[i]), y)
end

function ChainRulesCore.rrule(::typeof(dynamic_einsum), ixs, xs::NTuple{N,T where T}, iy, size_dict) where {N}
    y = dynamic_einsum(ixs, xs, iy, size_dict)
    function einsum_pullback(dy)
        dxs = ChainRulesCore.@thunk ntuple(i -> einsum_grad(ixs, xs, iy, size_dict, map(conj, dy), i), N)
        return (NoTangent(), NoTangent(), dxs, NoTangent(), NoTangent())
    end
    einsum_pullback(::NoTangent) = (NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())
    return y, einsum_pullback
end
