# `NT` for number of tensors
abstract type EinRule{NT} end

struct Sum <: EinRule{1} end
struct Tr <: EinRule{1} end
struct PairWise <: EinRule{Any} end
struct Permutedims <: EinRule{1} end
struct Hadamard <: EinRule{Any} end
struct PTrace <: EinRule{1} end
struct DefaultRule <: EinRule{Any} end


"""
a einsum code is trace
"""
function match_rule(::Type{Tr}, ixs, iy)
    iy == () &&
    length(ixs) == 1 &&
    length(ixs[1]) == 2 &&
    ixs[1][1] == ixs[1][2] ? Tr() : nothing
end

"""
a einsum code is a pairwise graph.
"""
function match_rule(::Type{PairWise}, ixs::NTuple{N, NTuple{X,T} where X}, iy::NTuple{M, T}) where {N, M, T}
    all_indices = TupleTools.vcat(ixs..., iy)
    counts = Dict{T, Int}()
    for ind in all_indices
        counts[ind] = get(counts, ind, 0) + 1
    end
    all(isequal(2), counts |> values) && length(tunique(iy)) == M ? PairWise() : nothing
end

"""
a einsum code is sum.
"""
function match_rule(::Type{Sum}, ixs, iy)
    length(ixs) != 1 && return
    ix = ixs[1]
    length(ix) != length(tunique(ix)) && return
    dims = _sumed_dims(ix, iy)
    setdiff(ix, dims) == [iy...] ? Sum() : nothing
end

function _sumed_dims(ix, iy::NTuple{N,T}) where {N,T}
    dims = T[]
    for i in ix
        if !(i in iy)
            push!(dims, i)
        end
    end
    return (dims...,)
end

"""
permutation rule
"""
function match_rule(::Type{Permutedims}, ixs, iy)
    length(ixs) == 1 || return nothing
    (ix,) = ixs
    length(ix) == length(iy) || return nothing
    for i in ix
        count(==(i), iy) == 1 || return nothing
    end
    return Permutedims()
end

"""
Hadamard
"""
function match_rule(::Type{Hadamard}, ixs, iy)
    for i in iy
        count(==(i), iy) == 1 || return nothing
    end
    for ix in ixs
        ix === iy || return nothing
    end
    return Hadamard()
end

"""
Ptrace rule if all indices of one ix in ixs all appear in iy or
appear twice and don't appear in iy
"""
function match_rule(::Type{PTrace}, ixs, iy)
    length(ixs) == 1 || return nothing
    (ix,) = ixs
    for i in iy
        count(==(i), ix) == 1 || return nothing
        count(==(i), iy) == 1 || return nothing
    end
    for i in ix
        ciy = count(==(i), iy)
        cix = count(==(i), ix)
        if ciy == 0
            cix == 2 || return nothing
        elseif ciy == 1
            cix == 1 || return nothing
        elseif ciy > 1
            return nothing
        end
    end
    return PTrace()
end

global einsum_rules = [Tr, Sum, PairWise, Permutedims, Hadamard, PTrace]

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
