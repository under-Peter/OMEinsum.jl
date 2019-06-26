# `NT` for number of tensors
abstract type EinRule{NT} end

struct Sum <: EinRule{1} end
struct Tr <: EinRule{1} end
struct PairWise <: EinRule{Any} end
struct Permutedims <: EinRule{1} end
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

function _sumed_dims(ix, iy)
    dims = []
    for i in ix
        if !(i in iy)
            push!(dims, i)
        end
    end
    return (dims...,)
end

global einsum_rules = [Tr, Sum, PairWise]

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
