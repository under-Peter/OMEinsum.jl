## non-inplace einsum
@doc raw"
    einsum(code::EinCode, xs, size_dict)

Return the tensor that results from contracting the tensors `xs` according to the contraction code `code`.

### Arguments
- `code`: The einsum notation, which can be an instance of [`EinCode`](@ref), [`NestedEinsum`](@ref), or [`SlicedEinsum`](@ref).
- `xs` - the input tensors
- `size_dict` - a dictionary that maps index-labels to their sizes

### Examples

```jldoctest; setup = :(using OMEinsum)
julia> a, b = rand(2,2), rand(2,2);

julia> einsum(EinCode((('i','j'),('j','k')),('i','k')), (a, b)) ≈ a * b
true

julia> einsum(EinCode((('i','j'),('j','k')),('k','i')), (a, b)) ≈ permutedims(a * b, (2,1))
true
```
"
function einsum(code::AbstractEinsum, @nospecialize(xs::Tuple), size_dict::Dict=get_size_dict!(getixs(code), xs, Dict{labeltype(code),Int}()))
    y = get_output_array(xs, map(y -> size_dict[y], getiyv(code)), false)
    einsum!(code, xs, y, true, false, size_dict)
end

# inplace einsum, EinCode as the input
"""
    einsum!(code::EinCode, xs, y, sx, sy, size_dict)

Inplace version of `einsum`. The result is stored in `y`.

### Arguments
- `code`: The einsum notation, which can be an instance of [`EinCode`](@ref), [`NestedEinsum`](@ref), or [`SlicedEinsum`](@ref).
- `xs`: The input tensors.
- `y`: The output tensor.
- `sx`: Scale `x` by `sx`.
- `sy`: Scale `y` by `sy`.
- `size_dict`: A dictionary that maps index-labels to their sizes.
"""
function einsum!(code::EinCode, @nospecialize(xs::Tuple), @nospecialize(y), sx, sy, size_dict::Dict=get_size_dict(getixs(code), xs))
    einsum!(getixs(code), getiy(code), xs, y, sx, sy, size_dict)
end
# inplace einsum, the fallback
function einsum!(ixs, iy, @nospecialize(xs::Tuple), @nospecialize(y), sx, sy, size_dict::Dict)
    @debug "fallback to loop_einsum" ixs => iy size.(xs)
    loop_einsum!(ixs, iy, (xs...,), y, sx, sy, size_dict)
end

struct UnaryOperation
    type
    ix::Tuple
    iy::Tuple
end
# for unary operations
# overhead ~ 2.3us
# @benchmark OMEinsum.einsum(DefaultRule(), $((('a', 'a', 'b'),)), $(('c', 'b','a')), (x,), $(Dict('a'=>1, 'b'=>1, 'c'=>1))) setup=(x=randn(1,1,1))
function unary_pipeline(::Val{ix}, ::Val{iy}) where {ix,iy}
    ix_unique = _unique(ix)
    iy_unique = _unique(iy)
    iy_a = filter(i -> i ∈ ix, iy_unique)

    step1 = if length(ix_unique) != length(ix)  # diag
        UnaryOperation(Diag(), ix, ix_unique)
    else nothing end

    step2 = if length(ix_unique) != length(iy_a)  # sum
        UnaryOperation(Sum(), ix_unique, iy_a)
    elseif ix_unique != iy_a   # permute, high freq
        UnaryOperation(Permutedims(), ix_unique, iy_a)
    else nothing end

    step3 = if length(iy_unique) != length(iy_a)  # repeat
        UnaryOperation(Repeat(), iy_a, iy_unique)
    else nothing end

    step4 = if length(iy_unique) != length(iy)  # duplicate
        UnaryOperation(Duplicate(), iy_unique, iy)
    else nothing end

    return filter(!isnothing, (step1, step2, step3, step4))
end

function einsum!(ixs, iy, @nospecialize(xs::NTuple{1,Any}), @nospecialize(y), sx, sy, size_dict::Dict{LT}) where {LT}
    @debug "compiling unary" ixs[1] => iy size(xs[1])
    ix1 = (ixs[1]...,)
    iy = (iy...,)
    pipeline = unary_pipeline(Val(ix1), Val(iy))
    lasttensor = xs[1]
    for (k, op) in enumerate(pipeline)
        if k == length(pipeline)  # last operation
            unary_einsum!(op.type, op.ix, op.iy, lasttensor, y, sx, sy)
        else
            cache = similar(y, ([size_dict[l] for l in op.iy]...,))
            unary_einsum!(op.type, op.ix, op.iy, lasttensor, cache, true, false)
            lasttensor = cache
        end
    end
    if length(pipeline) == 0
        @flatten_addmul! sy * y + sx * lasttensor
    end
    return y
end

# there are too many combination in the binary case, so nospecialize
function einsum!(ixs, iy, @nospecialize(xs::NTuple{2,Any}), @nospecialize(y), sx, sy, size_dict::Dict{LT}) where {LT}
    iyv = _collect(LT, iy)
    ix1v, ix2v = _collect.(Ref(LT), ixs)
    @debug "compiling binary" ixs => iyv size.(xs)
    x1, x2 = xs
    c1, c2, cy, s1, s2, s3, i1, i2, iyb = analyze_binary(ix1v, ix2v, iyv, size_dict)
    rule = SimpleBinaryRule{(i1...,),(i2...,),(iyb...,)}()
    xs1 = simplifyto(ix1v, c1, x1, size_dict)
    xs2 = simplifyto(ix2v, c2, x2, size_dict)
    x1_ = safe_reshape(xs1, s1)
    x2_ = safe_reshape(xs2, s2)
    @debug rule size.((x1_, x2_))
    if cy != iyv
        y_ = similar(y, (s3...,))
        y_ = reshape(binary_einsum!(rule, x1_, x2_, y_, true, false), [size_dict[x] for x in cy]...)
        return einsum!((cy,), iyv, (y_,), y, sx, sy, size_dict)
    else
        binary_einsum!(rule, x1_, x2_, safe_reshape(y, s3), sx, sy)
        return y
    end
end
safe_reshape(x, sz) = reshape(x, (sz...,))

function simplifyto(ix1, c1, x1, size_dict::Dict{LT}) where {LT}
    if c1 != ix1
        xs1 = similar(x1, ([size_dict[l] for l in c1]...,))
        return einsum!((_collect(LT, ix1),), c1, (x1,), xs1, true, false, size_dict)
    else
        return x1
    end
end

"""
Get the expected labels.
"""
function analyze_binary(ix1::Vector{T}, ix2::Vector{T}, iy::Vector{T}, size_dict::Dict{T,Int}) where {T}
    ix_inner, ix1_outer, ix2_outer, batch = _analyze_binary_input(ix1, ix2, iy)
    c1 = vcat(ix1_outer, ix_inner, batch)
    c2 = vcat(ix_inner, ix2_outer, batch)
    cy = vcat(ix1_outer, ix2_outer, batch)
    si = prod(map(x -> size_dict[x], ix1_outer))
    sj = prod(map(x -> size_dict[x], ix_inner))
    sk = prod(map(x -> size_dict[x], ix2_outer))
    sl = prod(map(x -> size_dict[x], batch))
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

function _analyze_binary_input(ix1::Vector{T}, ix2::Vector{T}, iy::Vector{T}) where {T}
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