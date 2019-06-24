include("EinsumOp.jl")

# TODO: fix the docstring
@doc raw"
    einsum(::EinCode{ixs, iy}, out, size_dict) where {ixs, iy}

return the tensor that results from contracting the tensors `xs` according
to their indices `ixs`, where all indices that do not appear in the output are
summed over. The indices are contracted in the order implied by their numerical value,
smaller first.
The result is permuted according to `out`.

- `ixs` - tuple of tuple of integers that label all indices of a tensor.
       Indices that appear twice (in different tensors) are summed over

- `xs` - tuple of tensors

- `out` - tuple of integers that should correspond to remaining indices in `ixs` after contractions.


# example
```jldoctest; setup = :(using OMEinsum)
julia> a = rand(2,2);

julia> b = rand(2,2);

julia> einsum(((1,2),(2,3)), (a, b), (1,3)) ≈ a * b
true

julia> einsum(((1,2),(2,3)), (a, b), (3,1)) ≈ permutedims(a * b, (2,1))
true
```
"
@generated function einsum(code::EinCode{ixs, iy}, xs, size_dict) where {ixs, iy}
    check_tensor_order(ixs, xs)
    rule = match_rule(ixs, iy)
    :(einsum($rule, code, xs, size_dict))
end

function einsum(::Trace, ::EinCode, xs, size_dict)
    asarray(tr(xs[1]))  # should be dispatched to tensortrace too.
end

@generated function einsum(::PairWise, ::EinCode{ixs, iy}, xs::NTuple{NT,Any}, size_dict) where {ixs, iy, NT}
    if NT > 1
        body = Expr(:call, :*, (:(xs[$i][$(Symbol.(ixs[i])...)]) for i in 1:NT)...)
    else
        body = :(xs[1][$(Symbol.(ixs[1])...)])
    end
    :(@tensoropt res[$(Symbol.(iy)...)] := $body)
end

# the fallback
function einsum(sm::Sum, code::EinCode, xs, size_dict)
    dropdims(sum(xs[1], dims=sm.dims), dims=sm.dims)
end

# the fallback
function einsum(::Fallback, code::EinCode{ixs, iy}, xs, size_dict) where {ixs, iy}
    einsumexp(code, xs, size_dict)
end
