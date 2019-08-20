export @ein_str, @ein, IndexSize
"""
    ein"ij,jk -> ik"(A,B)
    einsum(ein"ij,jk -> ik", (A,B))

String macro interface which understands `numpy.einsum`'s notation.
"""
macro ein_str(s::AbstractString)
    s = replace(s, " " => "")
    m = match(r"([\(\)a-z,α-ω]+)->([a-zα-ω]*)", s)
    m == nothing && throw(ArgumentError("invalid einsum specification $s"))
    sixs, siy = m.captures
    if '(' in sixs
        return parse_nested(sixs, collect(siy))
    else
        iy  = Tuple(siy)
        ixs = Tuple(Tuple(ix) for ix in split(sixs,','))
        return EinCode(ixs, iy)
    end
end

function (code::EinCode)(xs...; size_info=nothing) where {T, N}
    size_dict = get_size_dict(getixs(code), xs)
    if !(size_info isa Nothing)
        size_dict += size_info
    end
    einsum(code, xs, size_dict)
end

einsum(code::EinCode{ixs, iy}, xs) where {ixs, iy} = einsum(code, xs, nothing)

function einsum(code::EinCode{ixs, iy}, xs, ::Nothing) where {ixs, iy}
    einsum(code, xs, get_size_dict(ixs, xs))
end

"""get the dictionary of `index=>size`, error if there are conflicts"""
function get_size_dict(ixs::NTuple{N, NTuple{M, T} where M}, xs::NTuple{X, AbstractArray}) where {N,T,X}
    # check size of input tuples
    N != X && throw(ArgumentError("$X tensors labelled by $N indices"))

    # check tensor orders
    foreach(ixs, xs) do ix, x
        length(ix) == ndims(x) || throw(
            ArgumentError("indices $ix invalid for tensor with ndims = $(ndims(x))"))
    end
    sd = getindexsize(ixs, xs)
    check_dimensions(sd)
    return sd
end

struct IndexSize{N,T}
    k::NTuple{N,T}
    v::NTuple{N,Int}
end

IndexSize(sizes::Pair...) = IndexSize(first.(sizes), last.(sizes))
Base.:+(x::IndexSize, y::IndexSize) = IndexSize((x.k..., y.k...), (x.v..., y.v...))

function getindexsize(ixs, xs)
    k = TupleTools.flatten(ixs)
    v = TupleTools.flatten(map(size,xs))
    IndexSize(k,v)
end

Base.getindex(inds::IndexSize{N,T},i::T) where {N,T} = inds.v[findfirst(==(i), inds.k)]

function check_dimensions(inds::IndexSize)
    for (c,i) in enumerate(inds.k)
        j = findnext(==(i), inds.k, c+1)
        j != nothing && inds.v[c] != inds.v[j] && return throw(
            DimensionMismatch("index $i has incosistent sizes"))
    end
    return true
end

using MacroTools
"""
    @ein A[i,k] := B[i,j] * C[j,k]     # A = B * C

Macro interface similar to that of other packages.

You may use numbers in place of letters for dummy indices, as in `@tensor`,
and need not name the output array. Thus `A = @ein [1,2] := B[1,ξ] * C[ξ,2]`
is equivalent to the above. This can also be written `A = ein"ij,jk -> ik"(B,C)`
using the numpy-style string macro.
"""
macro ein(exs...)
    _ein_macro(exs...)
end


primefix!(ind) = map!(i -> @capture(i, (j_)') ? Symbol(j, '′') : i, ind, ind)

function _ein_macro(ex; einsum=:einsum)
    @capture(ex, (left_ := right_)) || throw(ArgumentError("expected A[] := B[]... "))
    @capture(left, Z_[leftind__] | [leftind__] ) || throw(
        ArgumentError("can't understand LHS, expected A[i,j] etc."))
    if Z===nothing
        @gensym Z
    end
    primefix!(leftind)

    rightind, rightpairs = [], []
    @capture(right, *(factors__)) || (factors = Any[right])
    for fact in factors
        @capture(fact, A_[Aind__]) || return _nested_ein_macro(ex)
        primefix!(Aind)
        append!(rightind, Aind)
        push!(rightpairs, (A, Aind) )
    end
    unique!(rightind)
    isempty(setdiff(leftind, rightind)) || throw(
        ArgumentError("some indices appear only on the left"))

    lefttuple = Tuple(indexin(leftind, rightind))
    righttuples = [ Tuple(indexin(ind, rightind)) for (A, ind) in rightpairs ]
    rightnames = [ esc(A) for (A, ind) in rightpairs ]

    return :( $(esc(Z)) = $einsum( EinCode{($(righttuples...),), $lefttuple}(), ($(rightnames...),)) )
end
