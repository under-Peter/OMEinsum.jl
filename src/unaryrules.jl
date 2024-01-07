# A general unary operation uses the following pipeline
# 0. special rules `Identity` and `Tr`,
# 1. rules reducing dimensions `Diag` and `Sum`
# 2. `Permutedims`,
# 3. `Repeat` and `Duplicate`,

# `NT` for number of tensors
abstract type EinRule{NT} end

struct Tr <: EinRule{1} end
struct Sum <: EinRule{1} end
struct Repeat <: EinRule{1} end
struct Permutedims <: EinRule{1} end
struct Identity <: EinRule{1} end
struct Duplicate <: EinRule{1} end
struct Diag <: EinRule{1} end
struct DefaultRule <: EinRule{Any} end

# trace
# overhead ~ 0.07us
# @benchmark OMEinsum.einsum(Tr(), $(('a', 'a')), $(()), x, $(Dict('a'=>1, 'b'=>1))) setup=(x=randn(1,1))
function unary_einsum!(::Tr, ix, iy, x, y::AbstractArray, sx, sy)
    @debug "Tr" size(x)
    y .= sy .* y .+ sx * tr(x)
    return y
end

# overhead ~ 0.55us
# @benchmark OMEinsum.einsum(Sum(), $(('a', 'b')), $(('b',)), x, $(Dict('a'=>1, 'b'=>1))) setup=(x=randn(1,1))
function unary_einsum!(::Sum, ix, iy, x::AbstractArray, y::AbstractArray, sx, sy)
    @debug "Sum" ix => iy size(x)
    dims = (findall(i -> i ∉ iy, ix)...,)::NTuple{length(ix)-length(iy),Int}
    res = dropdims(sum(x, dims=dims), dims=dims)
    ix1f = filter(i -> i ∈ iy, ix)::typeof(iy)
    if ix1f != iy
        return unary_einsum!(Permutedims(), (ix1f...,), iy, res, y, sx, sy)
    else
        return @addmul! sy * y + sx * res
    end
end

# overhead ~ 0.53us
# @benchmark OMEinsum.einsum(OMEinsum.Repeat(), $(('a',)), $(('a', 'b',)), x, $(Dict('a'=>1, 'b'=>1))) setup=(x=randn(1))
function unary_einsum!(::Repeat, ix, iy, x::AbstractArray, y::AbstractArray, sx, sy)
    @debug "Repeat" ix => iy size(x)
    ix1f = filter(i -> i ∈ ix, iy)
    shape1 = [s for (l, s) in zip(iy, size(y)) if l ∈ ix]
    shape2 = [l ∈ ix ? s : 1 for (l, s) in zip(iy, size(y))]
    repeat_dims = [l ∈ ix ? 1 : s for (l, s) in zip(iy, size(y))]
    # TODO: avoid copy
    if ix1f != ix
        y1 = similar(x, shape1...)
        unary_einsum!(Permutedims(), ix, ix1f, x, y1, true, false)
    else
        y1 = x
    end
    @addmul! sy * y + sx * repeat(reshape(y1, shape2...), repeat_dims...)
end

# overhead ~ 0.28us
# @benchmark OMEinsum.einsum(Diag(), $(('a', 'a')), $(('a',)), x, $(Dict('a'=>1, 'b'=>1))) setup=(x=randn(1,1))
function unary_einsum!(::Diag, ix, iy, x::AbstractArray, y::AbstractArray, sx, sy)
    @debug "Diag" ix => iy size.(x)
    compactify!(y, x, ix, iy, sx, sy)
end

function compactify!(y, x, ix, iy, sx, sy)
    x_in_y_locs = (Int[findfirst(==(x), iy) for x in ix]...,)
    @assert size(x) == map(loc->size(y, loc), x_in_y_locs)
    indexer = dynamic_indexer(x_in_y_locs, size(x))
    _compactify!(y, x, indexer, sx, sy)
end

function _compactify!(y, x, indexer, sx, sy)
    @inbounds for ci in CartesianIndices(y)
        y[ci] = sy * y[ci] + sx * x[subindex(indexer, ci.I)]
    end
    return y
end

function duplicate!(y, x, ix, iy, sx, sy)
    # compute same locs
    x_in_y_locs = (Int[findfirst(==(l), ix) for l in iy]...,)
    indexer = dynamic_indexer(x_in_y_locs, size(y))
    lmul!(sy, y)
    _duplicate!(y, x, indexer, sx)
end

@noinline function _duplicate!(y, x, indexer, sx)
    map(CartesianIndices(x)) do ci
        @inbounds y[subindex(indexer, ci.I)] += sx * x[ci]
    end
    return y
end

# e.g. 'ij'->'iij', left indices are unique, right are not
# overhead ~ 0.29us
# @benchmark OMEinsum.einsum(Duplicate(), $((('a', ),)), $(('a','a')), (x,), $(Dict('a'=>1, 'b'=>1))) setup=(x=randn(1))
function unary_einsum!(::Duplicate, ix, iy, x::AbstractArray, y::AbstractArray, sx, sy)
    @debug "Duplicate" ix => iy size(x)
    duplicate!(y, x, ix, iy, sx, sy)
end

# overhead ~ 0.15us
# @benchmark OMEinsum.einsum(Permutedims(), $((('a', 'b'),)), $(('b','a')), (x,), $(Dict('a'=>1, 'b'=>1))) setup=(x=randn(1,1))
function unary_einsum!(::Permutedims, ix, iy, x::AbstractArray, y::AbstractArray, sx, sy)
    perm = ntuple(i -> findfirst(==(iy[i]), ix)::Int, length(iy))
    @debug "Permutedims" ix => iy size(x) perm
    return tensorpermute!(y, x, perm, sx, sy)
end

# overhead ~0.04us
# @benchmark OMEinsum.einsum(Identity(), $((('a', 'b'),)), $(('a','b')), (x,), $(Dict('a'=>1, 'b'=>1))) setup=(x=randn(1,1))
function unary_einsum!(::Identity, ix, iy, x::AbstractArray, y::AbstractArray)
    @debug "Identity" ix => iy size(x)
    @addmul! sy * y + sx * x  # NOTE: copy can not be avoided, otherwise AD may fail!
end