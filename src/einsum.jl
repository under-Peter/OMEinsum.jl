include("EinRule.jl")

# TODO: fix the docstring
@doc raw"
    einsum(::EinCode{ixs, iy}, out, y_shape) where {ixs, iy}

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
@generated function einsum(::EinCode{ixs, iy}, xs, y_shape) where {ixs, iy}
    rule = EinRule(code_to_rule(ixs, iy))
    :(einsum($rule, xs, y_shape))
end

function einsum(::Trace, ::EinCode, xs, y_shape)
    asarray(tr(xs[1]))
end

function outindsfrominput(ixs)
    allixs = vcat(collect.(ixs)...)
    iy = sort!(filter!(x -> count(==(x), TupleTools.vcat(ixs...)) == 1, allixs))
    return tuple(iy...)
end

@generated function einsum(::EinRule{:PairWise}, ::EinCode{ixs, iy}, xs, y_shape) where {ixs, iy}
    out_indices = outindsfrominput(ixs)
    ex = :(res[])
end

function einsum(::EinRule{:General}, code::EinCode{ixs, iy}, xs, y_shape) where {ixs, iy}
    einsumexp(code, xs, y_shape)
end
