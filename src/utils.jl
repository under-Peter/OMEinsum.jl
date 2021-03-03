export asarray

"""
    asarray(x::Number[, parent::AbstractArray]) -> AbstactArray

Return a 0-dimensional array with item `x`, otherwise, do nothing.
If a `parent` is supplied, it will try to match the parent array type.
"""
asarray(x::Number) = fill(x, ())
asarray(x::Number, arr::Array) = fill(x, ())
asarray(x::AbstractArray, args...) = x

tsetdiff(t::Tuple, b) = setdiff!(collect(t), b)
tunique(t::Tuple) = unique!(collect(t))

"""
    nopermute(ix,iy)

check that all values in `iy` that are also in `ix` have the same relative order,

# example

```jldoctest; setup = :(using OMEinsum)
julia> using OMEinsum: nopermute

julia> nopermute((1,2,3),(1,2))
true

julia> nopermute((1,2,3),(2,1))
false
```
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
    allunique(ix::Tuple)

return true if all elements of `ix` appear only once in `ix`.

# example

```jldoctest; setup = :(using OMEinsum)
julia> using OMEinsum: allunique

julia> allunique((1,2,3,4))
true

julia> allunique((1,2,3,1))
false
```
"""
allunique(ix::NTuple) = all(i -> count(==(i), ix) == 1, ix)

function conditioned_permutedims(A::AbstractArray{T,N}, perm, ind=()) where {T,N}
    if any(i-> (@inbounds perm[i]!=i), 1:N)
        @debug "conditioned_permutedims" size(A) Tuple(perm) Tuple(ind)
        return tensorpermute(A, perm)
    else
        return A
    end
end

function _conditioned_permutedims(@nospecialize(A), @nospecialize(perm), @nospecialize(ind))
    if any(i-> (@inbounds perm[i]!=i), 1:length(perm))
        @debug "conditioned_permutedims" size(A) Tuple(perm)
        return permutedims(A, perm)
    else
        return A
    end
end

function align_eltypes(xs::AbstractArray...)
    T = promote_type(eltype.(xs)...)
    return map(x->eltype(x)==T ? x : T.(x), xs)
end

function align_eltypes(xs::AbstractArray{T}...) where T
    xs
end

"""
    tensorpermute(A, perm)

Like `permutedims(A, perm)`, but calls the faster `TensorOperations.tensorcopy` when possible.
"""
function tensorpermute(A::StridedArray{T,N}, perm) where {T,N}
    TensorOperations.tensorcopy(A, ntuple(identity,N), perm)
end
tensorpermute(A::AbstractArray, perm) = permutedims(A, perm)
