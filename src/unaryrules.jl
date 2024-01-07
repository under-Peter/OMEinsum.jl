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

@doc raw"
    match_rule(ixs, iy)
    match_rule(code::EinCode)

Returns the rule that matches, otherwise use `DefaultRule` - the slow `loop_einsum` backend.
"
function match_rule(ixs, iy)
    if length(ixs) == 1
        return match_rule_unary(ixs[1], iy)
    elseif length(ixs) == 2
        return match_rule_binary(ixs[1], ixs[2], iy)
    else
        return DefaultRule()
    end
end

function match_rule_unary(ix, iy)
    Nx = length(ix)
    Ny = length(iy)
    # the first rule with the higher the priority
    if Ny == 0 && Nx == 2 && ix[1] == ix[2]
        return Tr()
    elseif allunique(iy)
        if ix == iy
            return Identity()
        elseif allunique(ix)
            if Nx == Ny
                if all(i -> i in iy, ix)
                    return Permutedims()
                else  # e.g. (abcd->bcde)
                    return DefaultRule()
                end
            else
                if all(i -> i in ix, iy)
                    return Sum()
                elseif all(i -> i in iy, ix)  # e.g. ij->ijk
                    return Repeat()
                else  # e.g. ijkxc,ijkl
                    return DefaultRule()
                end
            end
        else  # ix is not unique
            if all(i -> i in ix, iy) && all(i -> i in iy, ix)   # ijjj->ij
                return Diag()
            else
                return DefaultRule()
            end
        end
    else  # iy is not unique
        if allunique(ix) && all(x->x∈iy, ix)
            if all(y->y∈ix, iy)  # e.g. ij->ijjj
                return Duplicate()
            else  # e.g. ij->ijjl
                return DefaultRule()
            end
        else
            return DefaultRule()
        end
    end
end

match_rule(code::EinCode) = match_rule(getixs(code), getiy(code))

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

# for unary operations
# overhead ~ 2.3us
# @benchmark OMEinsum.einsum(DefaultRule(), $((('a', 'a', 'b'),)), $(('c', 'b','a')), (x,), $(Dict('a'=>1, 'b'=>1, 'c'=>1))) setup=(x=randn(1,1,1))
function einsum!(ix, iy, x::AbstractArray, y::AbstractArray, sx, sy, size_dict::Dict{LT}) where LT
    @debug "DefaultRule unary" ix => iy size(x)
    ix_unique = _unique(LT, ix)
    iy_unique = _unique(LT, iy)
    iy_a = filter(i->i ∈ ix, iy_unique)
    do_diag = length(ix_unique) != length(ix)
    do_duplicate = length(iy_unique) != length(iy)
    do_repeat = length(iy_a) != length(iy_unique)

    # diag
    if do_diag
        x_unique = similar(x, [size_dict[l] for l in ix_unique]...)
        unary_einsum!(Diag(), ix, (ix_unique...,), x, x_unique, true, false)
    else
        x_unique = x
    end

    # sum/permute
    if length(ix_unique) != length(iy_a)
        y_a = similar(x, [size_dict[l] for l in iy_a]...)
        unary_einsum!(Sum(), (ix_unique...,), (iy_a...,), x_, iy_a, true, false)
    elseif ix_unique != iy_a
        y_a = similar(x, [size_dict[l] for l in iy_a]...)
        unary_einsum!(Permutedims(), (ix_unique...,), (iy_a...,), x_, iy_a, true, false)
    else
        y_a = x_
    end
    # repeat indices
    # TODO: fix, should copy to y
    if do_repeat
        y_unique = similar(y, [size_dict[l] for l in iy_unique]...)
        unary_einsum!(Repeat(), (iy_a...,), (iy_unique...,), y_a, y_unique, true, false)
    else
        y_unique = y_a
    end
    # duplicate dimensions
    if do_duplicate
        return unary_einsum!(Duplicate(), (iy_unique...,), iy, y_unique, y, sx, sy)
    else
        return @addmul! sy * y + sx * y_unique
    end
end