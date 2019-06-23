einsumopt(ixs, xs) = einsumopt(ixs, xs, outindsfrominput(ixs))
@doc raw"
    meinsumopt(ixs, xs, iy)
returns the result of the einsum operation implied by `ixs`, `iy` but
evaluated in the optimal order according to `meinsumcost`.
"
function einsumopt(ixs, xs, iy)
    checkargs(ixs, xs, iy)
    ops = optimalorder(ixs, xs, iy)
    evaluateall(ixs, xs, ops, iy)
end
