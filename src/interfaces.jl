"""
    ein"ij,jk -> ik"(A,B)

String macro interface which understands `numpy.einsum`'s notation.
Translates strings into `StaticEinCode`-structs that can be called to evaluate
an `einsum`.
To control evaluation order, use parentheses - instead of an `EinCode`,
a `NestedEinsum` is returned which evaluates the expression
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
    ein(s)
end

"""
    optein"ij,jk,kl -> ik"(A, B, C)

String macro interface that similar to [`@ein_str`](@ref), with optimized contraction order (dimensions are assumed to be uniform).
"""
macro optein_str(s::AbstractString)
    code = ein(s)
    optimize_code(code, uniformsize(code, 2), TreeSA(; ntrials=1, niters=10))
end

function ein(s::AbstractString)
    s = replace(replace(s, "\n" => ""), " "=>"")
    m = match(r"([\(\)a-z,α-ω]*)->([a-zα-ω]*)", s)
    m === nothing && throw(ArgumentError("invalid einsum specification $s"))
    sixs, siy = m.captures
    if '(' in sixs
        return parse_nested(sixs, collect(siy))
    else
        iy  = Tuple(siy)
        ixs = Tuple(Tuple(ix) for ix in split(sixs,','))
        return StaticEinCode{Char, ixs, iy}()
    end
end

function (code::DynamicEinCode{LT})(@nospecialize(xs...); size_info=nothing) where LT
    size_dict = get_size_dict!(getixs(code), xs, size_info===nothing ? Dict{LT,Int}() : copy(size_info))
    einsum(code, xs, size_dict)
end

function (code::StaticEinCode{LT})(xs...; size_info=nothing) where LT
    size_dict = get_size_dict!(getixs(code), xs, size_info===nothing ? Dict{LT,Int}() : copy(size_info))
    einsum(code, xs, size_dict)
end

# 2us overheads if @nospecialize
@doc raw"
    get_size_dict!(ixs, xs, size_info)

return a dictionary that is used to get the size of an index-label
in the einsum-specification with input-indices `ixs` and tensors `xs` after
consistency within `ixs` and between `ixs` and `xs` has been verified.
"
@inline function get_size_dict!(ixs, xs, size_info::Dict{LT}) where LT
    if length(ixs) == 1
        get_size_dict_unary!(ixs[1], size(xs[1]), size_info)
    else
        get_size_dict_!(ixs, [collect(Int, size(x)) for x in xs], size_info)
    end
end

function get_size_dict_!(ixs, sizes::AbstractVector, size_info::Dict{LT}) where LT
    # check size of input tuples
    length(sizes)<1 && error("empty input tensors")
    length(ixs) != length(sizes) && throw(ArgumentError("$(length(sizes)) tensors labelled by $(length(ixs)) indices"))
    # check tensor orders
    @inbounds for i=1:length(sizes)
        ix, s = ixs[i], sizes[i]
        length(ix) == length(s) || throw(
            ArgumentError("indices $ix invalid for tensor with ndims = $(length(s))"))
        for j = 1:length(ix)
            k = ix[j]
            if haskey(size_info, k)
                s[j] == size_info[k] || throw(DimensionMismatch("$k = $(size_info[k]) or $(s[j]))?"))
            else
                size_info[k] = s[j]
            end
        end
    end
    return size_info
end
# to speed up unary operations
function get_size_dict_unary!(ix, s, size_info::Dict{LT}) where LT
    @inbounds for j = 1:length(ix)
        k = ix[j]
        if haskey(size_info, k)
            s[j] == size_info[k] || throw(DimensionMismatch("$k = $(size_info[k]) or $(s[j]))?"))
        else
            size_info[k] = s[j]
        end
    end
    return size_info
end

@inline function get_size_dict(ixs, xs, size_info=nothing)
    LT = foldl((a, b) -> promote_type(a, eltype(b)), ixs; init=Union{})
    return get_size_dict!(ixs, xs, size_info===nothing ? Dict{LT,Int}() : size_info)
end
@inline function get_size_dict(ixs::AbstractVector{<:AbstractVector{LT}}, xs, size_info=nothing) where LT
    return get_size_dict!(ixs, xs, size_info===nothing ? Dict{LT,Int}() : size_info)
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

function _ein_macro(ex; einsum = :einsum)
    @capture(ex, (left_ := right_)) || throw(ArgumentError("expected @ein A[] := B[]..."))

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

    return :( $(esc(Z)) = $einsum( EinCode(($(righttuples...),), $lefttuple), ($(rightnames...),)) )
end

"""
    @ein! A[i,k] := B[i,j] * C[j,k]     # A = B * C
    @ein! A[i,k] += B[i,j] * C[j,k]     # A += B * C
    @ein! A[i,k] -= B[i,j] * C[j,k]     # A -= B * C

Macro interface similar to that of other packages.

Inplace version of `@ein`. 

# example

```jldoctest; setup = :(using OMEinsum)
julia> a, b, c, d = rand(2,2), rand(2,2), rand(2,2), zeros(2,2);

julia> cc = copy(c);

julia> @ein! d[i,k] := a[i,j] * b[j,k];

julia> d ≈ a * b
true

julia> d ≈ ein"ij,jk -> ik"(a,b)
true

julia> @ein! c[i,k] += a[i,j] * b[j,k];

julia> c ≈ cc + a * b
true

julia> @ein! c[i,k] -= a[i,j] * b[j,k];

julia> c ≈ cc
true
```
"""
macro ein!(exs...)
    _ein_macro!(exs...)
end

function _ein_macro!(ex; einsum = :einsum!)
    if @capture(ex, (left_ := right_))
        sx, sy = 1, 0
    elseif @capture(ex, (left_ += right_))
        sx, sy = 1, 1
    elseif @capture(ex, (left_ -= right_))
        sx, sy = -1, 1
    else
        throw(ArgumentError("expected @ein! A[] := B[]..., @ein! A[] += B[]..., or @ein! A[] -= B[]..."))
    end

    @capture(left, Z_[leftind__] | [leftind__] ) || throw(
        ArgumentError("can't understand LHS, expected A[i,j] etc."))
    if Z===nothing
        throw(ArgumentError("LHS is needed for inplace einsum, expected A[i,j] etc."))
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

    return :( $(esc(Z)) = $einsum( EinCode(($(righttuples...),), $lefttuple), ($(rightnames...),), $(esc(Z)), $sx, $sy) )
end