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
function einsum(::Tr, ixs, iy, xs::Tuple{<:AbstractArray}, size_dict::Dict)
    x = xs[1]
    @debug "Tr" size(x)
    asarray(tr(x), x)
end

# overhead ~ 0.55us
# @benchmark OMEinsum.einsum(Sum(), $(('a', 'b')), $(('b',)), x, $(Dict('a'=>1, 'b'=>1))) setup=(x=randn(1,1))
function einsum(::Sum, ixs, iy, xs::Tuple{<:AbstractArray}, size_dict::Dict{LT}) where LT
    ix, x = ixs[1], xs[1]
    @debug "Sum" ix => iy size(x)
    dims = (findall(i -> i ∉ iy, ix)...,)::NTuple{length(ix)-length(iy),Int}
    res = dropdims(sum(x, dims=dims), dims=dims)
    ix1f = filter(i -> i ∈ iy, ix)::typeof(iy)
    if ix1f != iy
        return einsum(Permutedims(), ((ix1f...,),), iy, (res,), size_dict)
    else
        return res
    end
end

# overhead ~ 0.53us
# @benchmark OMEinsum.einsum(OMEinsum.Repeat(), $(('a',)), $(('a', 'b',)), x, $(Dict('a'=>1, 'b'=>1))) setup=(x=randn(1))
function einsum(::Repeat, ixs, iy, xs::Tuple{<:AbstractArray}, size_dict::Dict)
    ix, x = ixs[1], xs[1]
    @debug "Repeat" ix => iy size(x)
    ix1f = filter(i -> i ∈ ix, iy)
    res = if ix1f != ix
        einsum(Permutedims(), (ix,), ix1f, (x,), size_dict)
    else
        x
    end
    newshape = [l ∈ ix ? size_dict[l] : 1 for l in iy]
    repeat_dims = [l ∈ ix ? 1 : size_dict[l] for l in iy]
    repeat(reshape(res, newshape...), repeat_dims...)
end

# overhead ~ 0.28us
# @benchmark OMEinsum.einsum(Diag(), $(('a', 'a')), $(('a',)), x, $(Dict('a'=>1, 'b'=>1))) setup=(x=randn(1,1))
function einsum(::Diag, ixs, iy, xs::Tuple{<:AbstractArray}, size_dict::Dict)
    ix, x = ixs[1], xs[1]
    @debug "Diag" ix => iy size.(x)
    compactify!(get_output_array((x,), map(y->size_dict[y],iy); has_repeated_indices=false),x,ix, iy)
end

function compactify!(y, x, ix, iy)
    x_in_y_locs = (Int[findfirst(==(x), iy) for x in ix]...,)
    @assert size(x) == map(loc->size(y, loc), x_in_y_locs)
    indexer = dynamic_indexer(x_in_y_locs, size(x))
    _compactify!(y, x, indexer)
end

function _compactify!(y, x, indexer)
    @inbounds for ci in CartesianIndices(y)
        y[ci] = x[subindex(indexer, ci.I)]
    end
    return y
end

function duplicate(x, ix, iy, size_dict) where {Nx,Ny,T}
    y = get_output_array((x,), map(y->size_dict[y],iy); has_repeated_indices=true)
    # compute same locs
    x_in_y_locs = (Int[findfirst(==(l), ix) for l in iy]...,)
    indexer = dynamic_indexer(x_in_y_locs, size(y))
    _duplicate!(y, x, indexer)
end

@noinline function _duplicate!(y, x, indexer)
    map(CartesianIndices(x)) do ci
        @inbounds y[subindex(indexer, ci.I)] = x[ci]
    end
    return y
end

# e.g. 'ij'->'iij', left indices are unique, right are not
# overhead ~ 0.29us
# @benchmark OMEinsum.einsum(Duplicate(), $((('a', ),)), $(('a','a')), (x,), $(Dict('a'=>1, 'b'=>1))) setup=(x=randn(1))
function einsum(::Duplicate, ixs, iy, xs::Tuple{<:AbstractArray}, size_dict)
    ix, x = ixs[1], xs[1]
    @debug "Duplicate" ix => iy size(x)
    duplicate(x, ix, iy, size_dict)
end

# overhead ~ 0.15us
# @benchmark OMEinsum.einsum(Permutedims(), $((('a', 'b'),)), $(('b','a')), (x,), $(Dict('a'=>1, 'b'=>1))) setup=(x=randn(1,1))
function einsum(::Permutedims, ixs, iy, xs::Tuple{<:AbstractArray}, size_dict)
    ix, x = ixs[1], xs[1]
    perm = ntuple(i -> findfirst(==(iy[i]), ix)::Int, length(iy))
    @debug "Permutedims" ix => iy size(x) perm
    return tensorpermute(x, perm)
end

# overhead ~0.04us
# @benchmark OMEinsum.einsum(Identity(), $((('a', 'b'),)), $(('a','b')), (x,), $(Dict('a'=>1, 'b'=>1))) setup=(x=randn(1,1))
function einsum(::Identity, ixs, iy, xs::Tuple{<:AbstractArray}, size_dict)
    @debug "Identity" ixs[1] => iy size(xs[1])
    copy(xs[1])  # must copy, otherwise AD may fail!
end

# for unary operations
# overhead ~ 2.3us
# @benchmark OMEinsum.einsum(DefaultRule(), $((('a', 'a', 'b'),)), $(('c', 'b','a')), (x,), $(Dict('a'=>1, 'b'=>1, 'c'=>1))) setup=(x=randn(1,1,1))
function einsum(::DefaultRule, ixs, iy, xs::Tuple{<:AbstractArray}, size_dict::Dict{LT}) where LT
    ix, x = ixs[1], xs[1]
    @debug "DefaultRule unary" ix => iy size(x)
    # diag
    ix_ = _unique(LT, ix)
    x_ = length(ix_) != length(ix) ? einsum(Diag(), (ix,), (ix_...,), (x,), size_dict) : x
    # sum
    iy_b = _unique(LT, iy)
    iy_a = filter(i->i ∈ ix, iy_b)
    y_a = if length(ix_) != length(iy_a)
        einsum(Sum(), ((ix_...,),), (iy_a...,), (x_,), size_dict)
    elseif ix_ != iy_a
        einsum(Permutedims(), ((ix_...,),), (iy_a...,), (x_,), size_dict)
    else
        x_
    end
    # repeat
    y_b = length(iy_a) != length(iy_b) ? einsum(Repeat(), ((iy_a...,),), (iy_b...,), (y_a,), size_dict) : y_a
    # duplicate
    length(iy_b) != length(iy) ? einsum(Duplicate(), ((iy_b...,),), iy, (y_b,), size_dict) : y_b
end
