asarray(x::Number) = fill(x, ())
asarray(x::AbstractArray) = x

tsetdiff(t::Tuple, b) = setdiff!(collect(t), b)
tunique(t::Tuple) = unique!(collect(t))

"""
    nopermute(ix,iy)
check that all values in `iy` that are also in `ix` have the same relative order,
e.g. `nopermute((1,2,3),(1,2))` is true while `nopermute((1,2,3),(2,1))` is false
"""
function nopermute(ix::NTuple, iy::NTuple)
    i, j, jold = 1, 1, 0
    # find each element of iy in ix and check that the order is the same
    for i in 1:length(iy)
        j = findfirst(==(iy[i]), ix)
        (j === nothing || j <= jold) && return false
        jold = j
    end
    return true
end

"""
    allunique(ix)
return true if all elements of `ix` appear only once in `ix`
"""
allunique(ix::NTuple) = all(i -> count(==(i), ix) == 1, ix)
