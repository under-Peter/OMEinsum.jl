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
function einsum_grad(ixs, @nospecialize(xs), iy, size_dict, dy, i)
    nixs = _insertat(ixs, i, iy)
    nxs  = _insertat( xs, i, conj(dy))
    niy = ixs[i]
    y = einsum(DynamicEinCode(nixs, niy), nxs, size_dict)
    return ChainRulesCore.ProjectTo(xs[i])(conj(y))  # do not use `conj!` because we want to support Hessians.
end
function einsum_backward_rule(eins, @nospecialize(xs), @nospecialize(y), size_dict, @nospecialize(dy))
    return ntuple(i -> einsum_grad(getixs(eins), xs, getiy(eins), size_dict, dy, i), length(xs))
end

function ChainRulesCore.rrule(::typeof(einsum), code::EinCode, @nospecialize(xs), size_dict)
    y = einsum(code, xs, size_dict)
    function einsum_pullback(dy)
        dy = convert(typeof(y), unthunk(dy))  # for filled array/cuarray et al.
        dxs = ntuple(i -> (ChainRulesCore.@thunk einsum_grad(getixs(code), xs, getiy(code), size_dict, dy, i)), length(xs))
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
@non_differentiable getixsv(::Any)

echo(x; tag="echo") = x
function ChainRulesCore.rrule(::typeof(echo), x; tag="echo")
    @info "$tag: $x"
    x, function (dy)
        @info "$tag (back): x̄ = $dy"
        return (NoTangent(), dy)
    end
end

macro echo(var)
    name = QuoteNode(var)
    esc(:($var = $echo($var; tag="$($name)")))
end

function ChainRulesCore.rrule(::Type{CacheTree}, content::AbstractArray{T}, siblings) where T
    y = CacheTree(content, siblings)
    function cachetree_pullback(dy)
        dy = unthunk(dy)
        # we need to convert the siblings to a tuple, because the `Zygote.accum` will not work with a vector.
        return (NoTangent(), dy.content, dy.siblings isa Vector ? (dy.siblings...,) : dy.siblings)
    end
    cachetree_pullback(::NoTangent) = (NoTangent(), NoTangent(), NoTangent())
    return y, cachetree_pullback
end