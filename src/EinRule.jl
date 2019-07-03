# `NT` for number of tensors
abstract type EinRule{NT} end

struct Sum <: EinRule{1} end
struct Tr <: EinRule{1} end
struct PairWise <: EinRule{Any} end
struct Permutedims <: EinRule{1} end
struct Hadamard <: EinRule{Any} end
struct PTrace <: EinRule{1} end
struct MatMul <: EinRule{2} end
struct DefaultRule <: EinRule{Any} end


"""
a einsum code is trace
"""
function match_rule(::Type{Tr}, ixs, iy)
    iy == () || return
    length.(ixs) === (2,) || return
    ixs[1][1] == ixs[1][2]  || return
    return Tr()
end

"""
a einsum code is a pairwise graph.
"""
function match_rule(::Type{PairWise}, ixs::NTuple{N, NTuple{X,T} where X}, iy::NTuple{M, T}) where {N, M, T}
    allinds = TupleTools.vcat(ixs..., iy)
    counts = map(x -> count(==(x), allinds), allinds)
    all(isequal(2), counts) || return
    allunique(iy) || return
    return PairWise()
end

"""
a einsum code is sum.
"""
function match_rule(::Type{Sum}, ixs, iy)
    length(ixs) != 1 && return
    (ix,) = ixs
    (allunique(ix) && allunique(iy)) || return
    nopermute(ix, iy) || return
    return Sum()
end

"""
permutation rule
"""
function match_rule(::Type{Permutedims}, ixs, iy)
    length(ixs) == 1 || return
    (ix,) = ixs
    length(ix) == length(iy) || return
    all(i -> count(==(i), iy) == 1, ix) || return

    return Permutedims()
end

"""
Hadamard
"""
function match_rule(::Type{Hadamard}, ixs, iy)
    allunique(iy) || return
    all(ix -> ix === iy, ixs) || return
    return Hadamard()
end

"""
Ptrace rule if all indices of one ix in ixs all appear in iy or
appear twice and don't appear in iy
"""
function match_rule(::Type{PTrace}, ixs, iy)
    length(ixs) == 1 || return
    (ix,) = ixs
    for i in iy
        count(==(i), ix) == 1 || return
        count(==(i), iy) == 1 || return
    end
    for i in ix
        ciy = count(==(i), iy)
        cix = count(==(i), ix)
        if ciy == 0
            cix == 2 || return
        elseif ciy == 1
            cix == 1 || return
        elseif ciy > 1
            return
        end
    end
    nopermute(ix, iy) || return

    return PTrace()
end

function match_rule(::Type{MatMul}, ixs, iy)
    length.(ixs) == (2,2) || return
    length(iy) == 2 || return
    iy[1] == ixs[1][1] || return
    iy[2] == ixs[2][2] || return
    ixs[1][2] == ixs[2][1] || return

    return MatMul()
end

const einsum_rules = [
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
        res = match_rule(T, ixs, iy)
        if res !== nothing
            return res
        end
    end
    return DefaultRule()
end
