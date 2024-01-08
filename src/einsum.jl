## non-inplace einsum
@doc raw"
    einsum(code::EinCode, xs, size_dict)
    einsum(rule, ixs, iy, xs, size_dict)

return the tensor that results from contracting the tensors `xs` according
to their indices `ixs` (`getixs(code)`), where all indices that do not appear in the output `iy` (`getiy(code)`) are
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
function einsum(code::EinCode, @nospecialize(xs::Tuple), size_dict::Dict = get_size_dict!(getixs(code), xs, Dict{labeltype(code),Int}()))
    y = get_output_array(xs, map(y->size_dict[y],getiyv(code)); fillzero=false)
    einsum!(code, xs, y, true, false, size_dict)
end

# inplace einsum, EinCode as the input
function einsum!(code::EinCode, @nospecialize(xs::Tuple), @nospecialize(y), sx, sy, size_dict::Dict)
    einsum!(getixs(code), getiy(code), xs, y, sx, sy, size_dict)
end
# inplace einsum, the fallback
function einsum!(ixs, iy, @nospecialize(xs::Tuple), @nospecialize(y), sx, sy, size_dict::Dict)
    @debug "fallback to loop_einsum" ixs => iy size.(xs)
    loop_einsum!(ixs, iy, (xs...,), y, sx, sy, size_dict)
end

# for unary operations
# overhead ~ 2.3us
# @benchmark OMEinsum.einsum(DefaultRule(), $((('a', 'a', 'b'),)), $(('c', 'b','a')), (x,), $(Dict('a'=>1, 'b'=>1, 'c'=>1))) setup=(x=randn(1,1,1))
function einsum!(ixs, iy, @nospecialize(xs::NTuple{1, Any}), @nospecialize(y), sx, sy, size_dict::Dict{LT}) where LT
    ix, x = ixs[1], xs[1]
    @debug "compiling unary" ix => iy size(x)
    ix_unique = _unique(LT, ix)
    iy_unique = _unique(LT, iy)
    iy_a = filter(i->i ∈ ix, iy_unique)
    do_diag = length(ix_unique) != length(ix)
    do_duplicate = length(iy_unique) != length(iy)
    do_repeat = length(iy_a) != length(iy_unique)

    # diag
    if do_diag
        x_unique = similar(x, ([size_dict[l] for l in ix_unique]...,))
        unary_einsum!(Diag(), ix, (ix_unique...,), x, x_unique, true, false)
    else
        x_unique = x
    end

    # sum/permute
    if length(ix_unique) != length(iy_a)
        y_a = similar(x, ([size_dict[l] for l in iy_a]...,))
        unary_einsum!(Sum(), (ix_unique...,), (iy_a...,), x_unique, y_a, true, false)
    elseif ix_unique != iy_a
        y_a = similar(x, ([size_dict[l] for l in iy_a]...,))
        unary_einsum!(Permutedims(), (ix_unique...,), (iy_a...,), x_unique, y_a, true, false)
    else
        y_a = x_unique
    end
    # repeat indices
    # TODO: fix, should copy to y
    if do_repeat
        y_unique = similar(y, ([size_dict[l] for l in iy_unique]...,))
        unary_einsum!(Repeat(), (iy_a...,), (iy_unique...,), y_a, y_unique, true, false)
    else
        y_unique = y_a
    end
    # duplicate dimensions
    if do_duplicate
        return unary_einsum!(Duplicate(), (iy_unique...,), iy, y_unique, y, sx, sy)
    else
        return @flatten_addmul! sy * y + sx * y_unique
    end
end

# there are too many combination in the binary case, so nospecialize
function einsum!(ixs, iy, @nospecialize(xs::NTuple{2, Any}), @nospecialize(y), sx, sy, size_dict::Dict{LT}) where LT
    @debug "compiling binary" ixs => iy size.(xs)
    ix1, ix2 = ixs
    x1, x2 = xs
    c1, c2, cy, s1, s2, s3, i1, i2, iyb = analyze_binary(_collect(LT,ix1), _collect(LT,ix2), _collect(LT,iy), size_dict)
    rule = SimpleBinaryRule{(i1...,), (i2...,), (iyb...,)}()
    xs1 = similar(x1, ([size_dict[l] for l in c1]...,))
    xs2 = similar(x2, ([size_dict[l] for l in c2]...,))
    einsum!((_collect(LT,ix1),), c1, (x1,), xs1, true, false, size_dict)
    einsum!((_collect(LT,ix2),), c2, (x2,), xs2, true, false, size_dict)
    x1_ = reshape(xs1, s1...)
    x2_ = reshape(xs2, s2...)
    @debug rule size.((x1_, x2_))
    y_ = similar(y, (s3...,))
    y_ = reshape(binary_einsum!(rule, x1_, x2_, y_, true, false), [size_dict[x] for x in cy]...)
    return einsum!((cy,), _collect(LT,iy), (y_,), y, sx, sy, size_dict)
end

"""
Get the expected labels.
"""
function analyze_binary(ix1::Vector{T}, ix2::Vector{T}, iy::Vector{T}, size_dict::Dict{T,Int}) where T
    ix_inner, ix1_outer, ix2_outer, batch = _analyze_binary_input(ix1, ix2, iy)
    c1 = vcat(ix1_outer, ix_inner, batch)
    c2 = vcat(ix_inner, ix2_outer, batch)
    cy = vcat(ix1_outer, ix2_outer, batch)
    si = prod(map(x->size_dict[x], ix1_outer))
    sj = prod(map(x->size_dict[x], ix_inner))
    sk = prod(map(x->size_dict[x], ix2_outer))
    sl = prod(map(x->size_dict[x], batch))
    has_i = !isempty(ix1_outer)
    has_j = !isempty(ix_inner)
    has_k = !isempty(ix2_outer)
    has_l = !isempty(batch)
    i1 = Char[]
    i2 = Char[]
    iyb = Char[]
    s1 = Int[]
    s2 = Int[]
    s3 = Int[]
    if has_i
        push!(i1, 'i')
        push!(iyb, 'i')
        push!(s1, si)
        push!(s3, si)
    end
    if has_j
        push!(i1, 'j')
        push!(i2, 'j')
        push!(s1, sj)
        push!(s2, sj)
    end
    if has_k
        push!(i2, 'k')
        push!(iyb, 'k')
        push!(s2, sk)
        push!(s3, sk)
    end
    if has_l
        push!(i1, 'l')
        push!(i2, 'l')
        push!(iyb, 'l')
        push!(s1, sl)
        push!(s2, sl)
        push!(s3, sl)
    end
    return c1, c2, cy, s1, s2, s3, i1, i2, iyb
end

function _analyze_binary_input(ix1::Vector{T}, ix2::Vector{T}, iy::Vector{T}) where T
    ix1_batch = T[]
    ix1_inner = T[]
    ix1_outer = T[]
    for l1 in ix1
        if l1 ∈ ix2
            if l1 ∈ iy  # batch
                l1 ∉ ix1_batch && push!(ix1_batch, l1)
            else        # inner
                l1 ∉ ix1_inner && push!(ix1_inner, l1)
            end
        elseif l1 ∈ iy  # outer
            l1 ∉ ix1_outer && push!(ix1_outer, l1)
        else
            # dangling
        end
    end
    ix2_outer = T[]     # outer dimension of x2
    for l2 in ix2
        if l2 ∉ ix1 && l2 ∈ iy && l2 ∉ ix2_outer
            push!(ix2_outer, l2)
        end
    end
    ix1_inner, ix1_outer, ix2_outer, ix1_batch
end