include("EinRule.jl")

# TODO: fix the docstring
@doc raw"
    einsum(::EinCode{ixs, iy}, out, size_dict) where {ixs, iy}

return the tensor that results from contracting the tensors `xs` according
to their indices `ixs`, where all indices that do not appear in the output are
summed over.
The result is permuted according to `out`.

- `ixs` - tuple of tuple of integers that label all indices of a tensor.
       Indices that appear twice (in different tensors) are summed over

- `xs` - tuple of tensors

- `out` - tuple of integers that should correspond to remaining indices in `ixs` after contractions.


# example
```jldoctest; setup = :(using OMEinsum)
julia> a = rand(2,2);

julia> b = rand(2,2);

julia> einsum(EinCode((('i','j'),('j','k')),('i','k')), (a, b)) ≈ a * b
true

julia> einsum(EinCode((('i','j'),('j','k')),('k','i')), (a, b)) ≈ permutedims(a * b, (2,1))
true
```
"
@generated function einsum(code::EinCode{ixs, iy}, xs, size_dict) where {ixs, iy}
    rule = match_rule(ixs, iy)
    :(einsum($rule, code, xs, size_dict))
end

function einsum(::Tr, ::EinCode, xs, size_dict)
    asarray(tr(xs[1]), xs[1])  # should be dispatched to tensortrace too.
end

using TensorOperations

function einsum(::PTrace, ::EinCode{ixs,iy}, xs, size_dict) where {ixs, iy}
    tensortrace(xs[1], ixs[1], iy)
end

function einsum(::Hadamard, ::EinCode{ixs, iy}, xs, size_dict) where {ixs, iy}
    broadcast(*, xs...)
end

@generated function einsum(::PairWise, ::EinCode{ixs, iy}, xs::NTuple{NT,AbstractArray{T} where T<:Union{Complex, Real}}, size_dict) where {ixs, iy, NT}
    if NT > 1
        body = Expr(:call, :*, (:(xs[$i][$(Symbol.(ixs[i])...)]) for i in 1:NT)...)
    else
        body = :(xs[1][$(Symbol.(ixs[1])...)])
    end
    :(@tensoropt res[$(Symbol.(iy)...)] := $body)
end

function einsum(sm::Sum, code::EinCode{ixs, iy}, xs, size_dict) where {ixs, iy}
    dims = (findall(i -> i ∉ iy, ixs[1])...,)
    (ix1,) = ixs
    ix1f = filter!(i -> i in iy, collect(ix1))
    perm = map(i -> findfirst(==(i), ix1f), iy)
    permutedims(dropdims(sum(xs[1], dims=dims), dims=dims), perm)
end

@generated function einsum(sm::MatMul, code::EinCode{ixs, iy}, xs, size_dict) where {ixs, iy}
    ix1, ix2 = ixs
    l = ifelse(ix1[1] in ix2, ix1[1], ix1[2])
    if ix1[2] == l && ix2[1] == l
        if iy == (ix1[1], ix2[2])
            #"ij,jk -> ik"
            return :(xs[1] * xs[2])
        else
            #"ij,jk -> ki"
            return :(permutedims(xs[1] * xs[2]))
        end
    elseif ix1[1] == l && ix2[1] == l
        if iy == (ix1[2], ix2[2])
            #"ji,jk -> ik"
            return :(transpose(xs[1]) * xs[2])
        else
            #"ji,jk -> ki"
            return :(transpose(xs[2]) * xs[1])
        end
    elseif ix1[2] == l && ix2[2] == l
        if iy == (ix1[1], ix2[1])
            #"ij,kj -> ik"
            return :(xs[1] * transpose(xs[2]))
        else
            #"ij,kj -> ki"
            return :(xs[2] * transpose(xs[1]))
        end
    else #ix1[1] == l && ix2[2] == l
        if iy == (ix1[2], ix2[1])
            #"ji,kj -> ik"
            return :(permutedims(xs[2] * xs[1]))
        else
            #"ji,kj -> ki"
            return :(xs[2] * xs[1])
        end
    end
end

function einsum(::Permutedims, code::EinCode{ixs, iy}, xs, size_dict) where {ixs, iy}
    (ix,) = ixs
    (x,) = xs
    perm = map(i -> findfirst(==(i), ix), iy)
    return permutedims(x, perm)
end

function einsum(::Identity, ::EinCode{ixs, iy}, xs, size_dict) where {ixs, iy}
    xs[1]
end

# the fallback
function einsum(::DefaultRule, code::EinCode{ixs, iy}, xs, size_dict) where {ixs, iy}
    loop_einsum(code, xs, size_dict)
end

function einsum(::PairWise, code::EinCode{ixs, iy}, xs::NTuple{NT, Any}, size_dict) where {ixs, iy, NT}
    loop_einsum(code, xs, size_dict)
end

function einsum(::PTrace, code::EinCode{ixs, iy}, xs::NTuple{NT, Any}, size_dict) where {ixs, iy, NT}
    loop_einsum(code, xs, size_dict)
end
