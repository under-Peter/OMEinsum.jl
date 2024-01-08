macro addmul!(ex)
    esc(addmul_impl(ex, false))
end
macro flatten_addmul!(ex)
    esc(addmul_impl(ex, true))
end

function addmul_impl(ex::Expr, flatten::Bool)
    @assert ex.head === :call && length(ex.args) == 3
    dotadd, ay, bxs = ex.args
    @assert dotadd == :+
    @assert ay.head === :call && length(ay.args) == 3
    dotmul, a, y = ay.args
    @assert dotmul == :*
    @assert bxs.head === :call
    dotmul2, b, xs... = bxs.args
    @assert dotmul2 == :*
    @assert length(xs) > 0

    added = :(Ref($b))
    for x in xs
        added = :($added .* $(flatten ? :(vec($x)) : x))
    end
    vy = flatten ? :(vec($y)) : y
    quote
        if iszero($b)   # no need to multiply
            $lmul!($a, $vy)
        elseif iszero($a)  # empty y
            $vy .= $added
        elseif isone($a)
            $vy .+= $added
        else  # a != 1, a != 0, b != 0
            $vy .= Ref($a) .* $vy .+ $added
        end
        $y
    end
end

"""
    asarray(x[, parent::AbstractArray]) -> AbstactArray

Return a 0-dimensional array with item `x`, otherwise, do nothing.
If a `parent` is supplied, it will try to match the parent array type.
"""
asarray(x) = fill(x, ())
asarray(x::AbstractArray) = x
asarray(x, arr::AbstractArray) = fill(x, ())
asarray(x::AbstractArray, y::Array) = x
asscalar(x) = x
asscalar(x::AbstractArray) = x[]
_collect(x) = collect(x)
_collect(x::Vector) = x
_collect(::Type{T}, x::Vector{T}) where T = x
_collect(::Type{T}, x) where T = collect(T, x)
_insertat(lst::Tuple, i, item) = TupleTools.insertat(lst, i, (item,))
_insertat(lst::AbstractVector, i, item) = (lst=copy(lst); lst[i]=item; lst)

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
allunique(ix) = all(i -> count(==(i), ix) == 1, ix)
_unique(::Type{T}, x::NTuple{N,T}) where {N,T} = unique!(collect(T, x))
_unique(::Type{T}, x::Vector{T}) where T = unique(x)

function align_eltypes(xs::AbstractArray...)
    T = promote_type(eltype.(xs)...)
    return map(x->eltype(x)==T ? x : T.(x), xs)
end

function align_eltypes(xs::AbstractArray{T}...) where T
    xs
end

"""
    tensorpermute(A, perm)

`permutedims(A, perm)` with grouped dimensions.
"""
function tensorpermute!(C::AbstractArray{T, N}, A::AbstractArray{T,N}, perm, sx, sy) where {T, N}
    @assert N == length(perm) && all(p->1<=p<=N, perm)
    N == 0 && return copy(A)
    # group `perm`s
    newshape_slots = fill(-1, N)
    dk = 1  # the size of dimension-batch
    @inbounds begin
        permk = perm[1]
        newperm = [permk]
        newshape_slots[permk] = size(A, permk)
    end
    @inbounds for i=2:N
        permi = perm[i]
        if permi == permk + dk  # same group
            newshape_slots[permk] *= size(A, permi)
            dk += 1
        else
            permk = permi
            newshape_slots[permk] = size(A, permi)
            push!(newperm, permk)
            dk = 1
        end
    end
    newshape = filter(!=(-1), newshape_slots)
    newperm = sortperm(sortperm(newperm))
    A_ = reshape(A, newshape...)
    permed_shape = ntuple(i->size(A_, @inbounds newperm[i]), ndims(A_))
    if iszero(sy)
        permutedims!(reshape(C, permed_shape), A_, newperm)
        !iszero(sx) && lmul!(sx, C)
        return C
    else
        return @flatten_addmul! sy * C + sx * permutedims(A_, newperm)
    end
end

# new interface for GPU support!
# function _batched_gemm!(C1::Char, C2::Char, alpha, A::StridedArray{T,3}, B::StridedArray{T2,3}, beta, C::StridedArray{T3,3}) where {T<:BlasFloat, T2<:BlasFloat, T3<:BlasFloat}
#     batched_gemm!(C1, C2, alpha, A, B, beta, C)
# end
function _batched_gemm!(C1::Char, C2::Char, alpha, A::AbstractArray{T,3}, B::AbstractArray{T2,3}, beta, C::AbstractArray{T3,3}) where {T<:BlasFloat, T2<:BlasFloat,T3<:BlasFloat}
    # NOTE: convert alpha and beta to T3, since booleans are not supported by BatchedRoutines
    #batched_gemm!(C1, C2, T3(alpha), Array(A), Array(B), T3(beta), C)
    batched_gemm!(C1, C2, T3(alpha), A, B, T3(beta), C)
end
function _batched_gemm!(C1::Char, C2::Char, alpha, A::AbstractArray{T,3}, B::AbstractArray{T2,3}, beta, C::AbstractArray{T3,3}) where {T, T2,T3}
    @assert size(A, 3) == size(B, 3) == size(C, 3) "batch dimension mismatch, got $(size(A,3)), $(size(B,3)) and $(size(C,3))"
    @assert C1 === 'N' || C1 === 'T'
    @assert C2 === 'N' || C2 === 'T'
    for l = 1:size(A, 3)
        a = C1 === 'T' ? transpose(view(A,:,:,l)) : view(A,:,:,l)
        b = C2 === 'T' ? transpose(view(B,:,:,l)) : view(B,:,:,l)
        mul!(view(C,:,:,l), a, b, alpha, beta)
    end
    return C
end

# macro addmul!(a, y, b, xs...)
#     added = :(Ref(b))
#     for x in xs
#         added = :($added .* $x)
#     end
#     yeval = gensym("y")
#     quote
#         $yeval = $y
#         if iszero($b)   # no need to multiply
#             $lmul!($a, $yeval)
#         elseif iszero($a)  # empty y
#             $yeval .= $added
#         elseif isone($a)
#             $yeval .+= $added
#         else  # a != 1, a != 0, b != 0
#             $yeval .= Ref($a) .* $yeval .+ $added
#         end
#         $yeval
#     end |> esc
# end