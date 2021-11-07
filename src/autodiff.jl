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
function einsum_grad(ixs, @nospecialize(xs), iy, size_dict, cdy, i)
    nixs = _insertat(ixs, i, iy)
    nxs  = _insertat( xs, i, cdy)
    niy = ixs[i]
    y = einsum(DynamicEinCode(nixs, niy), nxs, size_dict)
    y = conj(y)  # do not use `conj!` to help computing Hessians.
    typeof(y) == typeof(xs[i]) && return y
    xs[i] isa Array{<:Real} && return convert(typeof(xs[i]), real(y))
    convert(typeof(xs[i]), y)
end

function ChainRulesCore.rrule(::typeof(einsum), code::EinCode, @nospecialize(xs), size_dict)
    y = einsum(code, xs, size_dict)
    function einsum_pullback(dy)
        dxs = ChainRulesCore.@thunk ntuple(i -> einsum_grad(getixs(code), xs, getiy(code), size_dict, map(conj, dy), i), length(xs))
        return (NoTangent(), NoTangent(), dxs, NoTangent())
    end
    einsum_pullback(::NoTangent) = (NoTangent(), NoTangent(), NoTangent(), NoTangent())
    return y, einsum_pullback
end

function ChainRulesCore.rrule(::typeof(_safe_set), lst, i, x)
    y = _safe_set(lst, i, x)
    function set_pullback(dy)
        return (NoTangent(), dy, NoTangent(), dy[i])
    end
    set_pullback(::NoTangent) = (NoTangent(), NoTangent(), NoTangent(), NoTangent())
    return y, set_pullback
end

@non_differentiable get_size_dict!(::Any, ::Any, ::Any)
@non_differentiable DynamicEinCode(::Any, ::Any)
@non_differentiable DynamicEinCode(::Any)
