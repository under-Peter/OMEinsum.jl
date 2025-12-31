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

struct UnaryOperation{LT}
    type
    ix::Vector{LT}
    iy::Vector{LT}
end
# for unary operations
# overhead ~ 2.3us
# @benchmark OMEinsum.einsum(DefaultRule(), $((('a', 'a', 'b'),)), $(('c', 'b','a')), (x,), $(Dict('a'=>1, 'b'=>1, 'c'=>1))) setup=(x=randn(1,1,1))
function unary_pipeline(ix::Vector{LT}, iy::Vector{LT}) where {LT}
    ix_unique = _unique(LT, ix)
    iy_unique = _unique(LT, iy)
    iy_a = filter(i -> i ∈ ix, iy_unique)

    operations = UnaryOperation[]
    if length(ix_unique) != length(ix)  # diag
        push!(operations, UnaryOperation(Diag(), ix, ix_unique))
    end
    if length(ix_unique) != length(iy_a)  # sum
        push!(operations, UnaryOperation(Sum(), ix_unique, iy_a))
    elseif ix_unique != iy_a   # permute, high freq
        push!(operations, UnaryOperation(Permutedims(), ix_unique, iy_a))
    end

    if length(iy_a) != length(iy_unique)  # repeat
        push!(operations, UnaryOperation(Repeat(), iy_a, iy_unique))
    end
    if length(iy_unique) != length(iy)  # duplicate
        push!(operations, UnaryOperation(Duplicate(), iy_unique, iy))
    end
    return operations
end

function einsum!(ixs, iy, @nospecialize(xs::NTuple{1,Any}), @nospecialize(y), sx, sy, size_dict::Dict{LT}) where {LT}
    @debug "compiling unary" ixs[1] => iy size(xs[1])
    pipeline = unary_pipeline(collect(LT, ixs[1]), collect(LT, iy))
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
    ix1 = (ixs[1]...,)
    ix2 = (ixs[2]...,)
    iy = (iy...,)
    @debug "compiling binary" ixs => iy size.(xs)
    c1, c2, cy, s1, s2, s3, i1, i2, iyb = analyze_binary(Val(ix1), Val(ix2), Val(iy), size_dict)
    rule = SimpleBinaryRule{i1,i2,iyb}()
    x1, x2 = xs
    xs1 = simplifyto(ix1, c1, x1, size_dict)
    xs2 = simplifyto(ix2, c2, x2, size_dict)
    x1_ = safe_reshape(xs1, s1)
    x2_ = safe_reshape(xs2, s2)
    @debug rule size.((x1_, x2_))
    if cy != iy
        y_ = similar(y, s3)
        y_ = reshape(binary_einsum!(rule, x1_, x2_, y_, true, false), ([size_dict[x] for x in cy]...,))
        return einsum!((cy,), iy, (y_,), y, sx, sy, size_dict)
    else
        binary_einsum!(rule, x1_, x2_, safe_reshape(y, s3), sx, sy)
        return y
    end
end
safe_reshape(x, sz) = reshape(x, sz) # Overloaded by CUDAExt.jl and AMDGPUExt.jl

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
function analyze_binary(::Val{ix1}, ::Val{ix2}, ::Val{iy}, size_dict::Dict{T,Int}) where {T, ix1, ix2, iy}
    ix_inner, ix1_outer, ix2_outer, batch = _analyze_binary_input(Val(ix1), Val(ix2), Val(iy))

    indices(::Val{label}) where label = begin
        if label === 'i'
            ix1_outer
        elseif label === 'j'
            ix_inner
        elseif label === 'k'
            ix2_outer
        elseif label === 'l'
            batch
        end
    end

    labels = filter((l)->!isempty(indices(Val(l))),
                    ('i','j','k','l'))
    sizes = NamedTuple{Symbol.(labels)}(
        ntuple(Val(length(labels))) do i
            dims = indices(Val(labels[i]))
            prod(map(x -> (@noinline size_dict[x]), dims))
        end
    )

    c1 = (ix1_outer..., ix_inner...,  batch...)
    c2 = (ix_inner...,  ix2_outer..., batch...)
    cy = (ix1_outer..., ix2_outer..., batch...)

    i1    = filter(in(('i','j','l')), labels)
    i2    = filter(in(('j','k','l')), labels)
    iyb   = filter(in(('i','k','l')), labels)

    s1    = values(Base.structdiff(sizes, NamedTuple{(:k,)})) # i j l
    s2    = values(Base.structdiff(sizes, NamedTuple{(:i,)})) # j k l
    s3    = values(Base.structdiff(sizes, NamedTuple{(:j,)})) # i k l

    return c1, c2, cy, s1, s2, s3, i1, i2, iyb
end

function _analyze_binary_input(::Val{ix1}, ::Val{ix2}, ::Val{iy}) where {ix1, ix2, iy}
    # These functions are carefully chosen to be eligible for compile-time execution
    ix1_batch = _unique(filter((l1) -> l1 ∈ ix2 && l1 ∈ iy, ix1))
    ix1_inner = _unique(filter((l1) -> l1 ∈ ix2 && l1 ∉ iy, ix1))
    ix1_outer = _unique(filter((l1) -> l1 ∉ ix2 && l1 ∈ iy, ix1))
    ix2_outer = _unique(filter((l2) -> l2 ∉ ix1 && l2 ∈ iy, ix2))

    ix1_inner, ix1_outer, ix2_outer, ix1_batch
end
