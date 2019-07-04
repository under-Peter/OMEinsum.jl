# `NT` for number of tensors
abstract type EinRule{NT} end

struct Sum <: EinRule{1} end
struct Tr <: EinRule{1} end
struct PairWise <: EinRule{Any} end
struct Permutedims <: EinRule{1} end
struct Hadamard <: EinRule{Any} end
struct PTrace <: EinRule{1} end
struct MatMul <: EinRule{2} end
struct Identity <: EinRule{Any} end
struct DefaultRule <: EinRule{Any} end


"""
a einsum code is trace
"""
function match_rule(::Type{Tr}, ixs, iy)
    iy == () &&
    length.(ixs) === (2,) &&
    ixs[1][1] == ixs[1][2]
end

"""
a einsum code is a pairwise graph.
"""
function match_rule(::Type{PairWise}, ixs::NTuple{N, NTuple{X,T} where X}, iy::NTuple{M, T}) where {N, M, T}
    allinds = TupleTools.vcat(ixs..., iy)
    all(i -> count(==(i), allinds) == 2, allinds) && allunique(iy)
end

"""
a einsum code is sum.
"""
function match_rule(::Type{Sum}, ixs, iy)
    length(ixs) != 1 && return false
    (ix,) = ixs
    allunique(ix) && allunique(iy) && nopermute(ix, iy)
end

"""
permutation rule
"""
function match_rule(::Type{Permutedims}, ixs, iy)
    length(ixs) == 1 || return false
    (ix,) = ixs
    length(ix) == length(iy) && allunique(ix) && allunique(iy) &&
     all(i -> i in iy, ix)
end

"""
Hadamard
"""
function match_rule(::Type{Hadamard}, ixs, iy)
    allunique(iy) && all(ix -> ix === iy, ixs)
end

"""
Ptrace rule if all indices of one ix in ixs all appear in iy or
appear twice and don't appear in iy
"""
function match_rule(::Type{PTrace}, ixs, iy)
    length(ixs) == 1 || return false
    (ix,) = ixs
    all(iy) do i
        count(==(i), ix) == 1 && count(==(i), iy) == 1
    end || return false
    all(ix) do i
        ciy, cix = count.(==(i), (iy,ix))
        (ciy == 0 && cix == 2) ||
        (ciy == 1 && cix == 1) ||
        (ciy > 1)
    end || return false
    nopermute(ix, iy)
end

function match_rule(::Type{MatMul}, ixs, iy)
    length.(ixs) == (2,2) && length(iy) == 2 &&
    iy[1] == ixs[1][1] && iy[2] == ixs[2][2] &&
    ixs[1][2] == ixs[2][1]
end

function match_rule(::Type{Identity}, ixs, iy)
    ixs === (iy,) && allunique(iy)
end

const einsum_rules = [
    Identity,
    MatMul,
    Permutedims,
    Hadamard,
    Tr,
    PTrace,
    Sum,
    PairWise,
    ]

"""Find the matched rule."""
function match_rule(ixs, iy)
    # the first rule with the higher the priority
    for T in einsum_rules
        match_rule(T, ixs, iy) && return T()
    end
    return DefaultRule()
end
