@doc raw"
    einsumopt(::EinCode{ixs, iy}, xs) where {ixs, iy}

returns the result of the einsum operation implied by `ixs`, `iy` but
evaluated in the optimal order according to `meinsumcost`.
"
@generated function einsumopt(::EinCode{ixs, iy}, xs) where {ixs, iy}
    quote
        size_dict = get_size_dict(ixs, xs)
        ops = optimalorder(ixs, xs, iy)  # should be static if `xs` is not included
        evaluateall(ixs, xs, ops, iy)
    end
end
