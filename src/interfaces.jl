export @ein_str, @ein, ein
"""
    ein"ij,jk -> ik"(A,B)

String macro interface which understands `numpy.einsum`'s notation.
Translates strings into `EinCode`-structs that can be called to evaluate
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
        return EinCode(ixs, iy)
    end
end

function (code::EinCode)(xs...; size_info=nothing)
    size_dict = get_size_dict(getixs(code), xs, size_info)
    einsum(code, xs, size_dict)
end

@doc raw"
    get_size_dict(ixs, xs, size_info=nothing)

return a dictionary that is used to get the size of an index-label
in the einsum-specification with input-indices `ixs` and tensors `xs` after
consistency within `ixs` and between `ixs` and `xs` has been verified.
"
function get_size_dict(@nospecialize(ixs), @nospecialize(xs), size_info=nothing)
    # check size of input tuples
    length(xs)<1 && error("empty input tensors")
    d = size_info === nothing ? Dict{promote_type(eltype.(ixs)...),Int}() : size_info
    length(ixs) != length(xs) && throw(ArgumentError("$(length(xs)) tensors labelled by $(length(ixs)) indices"))

    # check tensor orders
    for i=1:length(xs)
        ix, x = ixs[i], xs[i]
        length(ix) == ndims(x) || throw(
            ArgumentError("indices $ix invalid for tensor with ndims = $(ndims(x))"))
        for j = 1:length(ix)
            k = ix[j]
            if haskey(d, k)
                @assert size(x, j) == d[k] || throw(DimensionMismatch("$k = $(d[k]) or $(size(x,j))?"))
            else
                d[k] = size(x, j)
            end
        end
    end
    return d
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

@doc raw"
    einsum(::EinCode{ixs, iy}, xs, size_dict) where {ixs, iy}

return the tensor that results from contracting the tensors `xs` according
to their indices `ixs`, where all indices that do not appear in the output `iy` are
summed over.
The result is permuted according to `out`.

- `ixs` - tuple of tuples of index-labels of the input-tensors `xs`

- `iy` - tuple of index-labels of the output-tensor

- `xs` - tuple of tensors

- `size_dict` - a dictionary that maps index-labels to their sizes

# example

```jldoctest; setup = :(using OMEinsum)
julia> a, b = rand(2,2), rand(2,2);

julia> einsum(EinCode((('i','j'),('j','k')),('i','k')), (a, b)) ≈ a * b
true

julia> einsum(EinCode((('i','j'),('j','k')),('k','i')), (a, b)) ≈ permutedims(a * b, (2,1))
true
```
"
@generated function einsum(code::EinCode{ixs, iy}, xs, size_dict::Dict{LT}) where {ixs, iy, LT}
    rule = match_rule(ixs, iy)
    if length(ixs) == 1
        :(einsum($rule, ixs[1], iy, xs[1], size_dict))
    else
        :(einsum($rule, ixs, iy, xs, size_dict))
    end
end

function einsum(code::EinCode{ixs, iy}, xs) where {ixs, iy}
    einsum(code, xs, get_size_dict(ixs, xs))
end

function dynamic_einsum(@nospecialize(ixs), @nospecialize(xs), @nospecialize(iy); size_info=nothing)
    size_dict = get_size_dict(ixs, xs, size_info)
    dynamic_einsum(ixs, xs, iy, size_dict)
end

function dynamic_einsum(@nospecialize(ixs), @nospecialize(xs), @nospecialize(iy), size_dict)
    rule = match_rule(ixs, iy)
    if length(ixs) == 1
        einsum(rule, ixs[1], iy, xs[1], size_dict)
    else
        einsum(rule, ixs, iy, (xs...,), size_dict)
    end
end

dynamic_einsum(@nospecialize(code::EinCode{ixs, iy}), @nospecialize(xs); kwargs...) where {ixs, iy} = dynamic_einsum(ixs, xs, iy; kwargs...)

# the fallback
function einsum(::DefaultRule, ixs, iy, xs, size_dict)
    @debug "DefaultRule loop_einsum" ixs => iy size.(xs)
    loop_einsum(EinCode{ixs, iy}(), (xs...,), size_dict)
end
