export @ein_str, @ein, IndexSize
"""
    ein"ij,jk -> ik"(A,B)

String macro interface which understands `numpy.einsum`'s notation.
Translates strings into `EinCode`-structs that can be called to evaluate
an `einsum`.
To control evaluation order, use parentheses - instead of an `EinCode`,
a `NestedEinsumStable` is returned which evaluates the expression
according to parens.
The valid character ranges for index-labels are `a-z` and `α-ω`.

# example

```jldoctest; setup = :(using OMEinsum)
julia> a, b, c = rand(10,10), rand(10,10), rand(10,1);

julia> ein"ij,jk,kl -> il"(a,b,c) ≈ ein"(ij,jk),kl -> il"(a,b,c) ≈ a * b * c
true
```
"""
macro ein_str(s::AbstractString)
    s = replace(s, " " => "")
    m = match(r"([\(\)a-z,α-ω]*)->([a-zα-ω]*)", s)
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

@doc raw"
    get_size_dict(ixs, xs)

return the `IndexSize` struct that is used to get the size of an index-label
in the einsum-specification with input-indices `ixs` and tensors `xs` after
consistency within `ixs` and between `ixs` and `xs` has been verified.
"
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

@doc raw"
    IndexSize{N,T}(k::NTuple{N,T},v::NTuple{N,Int})

struct to hold the size of indices specified by their labels.
Note that while a dict would work, for the small sizes we usually
have, a tuple of keys and values is much faster to construct
and competitive for lookup.
"
struct IndexSize{N,T}
    indices::NTuple{N,T}
    sizes::NTuple{N,Int}
end

IndexSize(sizes::Pair...) = IndexSize(first.(sizes), last.(sizes))
IndexSize(indices::Tuple{}, sizes::Tuple{}) = IndexSize{0,Any}(indices, sizes)

Base.:+(x::IndexSize, y::IndexSize) = IndexSize((x.indices..., y.indices...), (x.sizes..., y.sizes...))

function getindexsize(ixs, xs)
    indices = TupleTools.flatten(ixs)
    sizes = TupleTools.flatten(map(size,xs))
    IndexSize(indices,sizes)
end

Base.getindex(inds::IndexSize{N,T},i::T) where {N,T} = inds.sizes[findfirst(==(i), inds.indices)]

@doc raw"
    check_dimensions(inds::IndexSize)
    
check whether all non-unique indexlabels point to the same
dimension - otherwise throw an error.
"
function check_dimensions(inds::IndexSize)
    for (c,i) in enumerate(inds.indices)
        j = findnext(==(i), inds.indices, c+1)
        j != nothing && inds.sizes[c] != inds.sizes[j] && return throw(
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

# example

```jldoctest; setup = :(using OMEinsum)
julia> a, b = rand(2,2), rand(2,2);

julia> @ein c[i,k] := a[i,j] * b[j,k];

julia> c ≈ a * b
true

julia> c ≈ ein"ij,jk -> ik"(a,b)
true
```
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
