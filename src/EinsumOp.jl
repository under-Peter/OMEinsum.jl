# `NT` for number of tensors
abstract type EinsumOp{NT} end

struct Sum <: EinsumOp{1}
    dims
end
struct Trace <: EinsumOp{1} end
struct PairWise{NT} <: EinsumOp{NT} end
struct Permutedims <: EinsumOp{1} end


"""
a einsum code is trace
"""
function match_rule(::Type{Trace}, ixs, iy)
    iy == () &&
    length(ixs) == 1 &&
    length(ixs[1]) == 2 &&
    ixs[1][1] == ixs[1][2] ? Trace : nothing
end

"""
a einsum code is a pairwise graph.
"""
function match_rule(::Type{PairWise}, ixs::NTuple{N}, iy) where N
    all_indices = TupleTools.vcat(ixs..., iy)
    counts = Dict{Int, Int}()
    for ind in all_indices
        counts[ind] = get(counts, ind, 0) + 1
    end
    all(isequal(2), counts |> values) ? PairWise{N}() : nothing
end

"""
a einsum code is sum.
"""
function match_rule(::Type{Sum}, ixs, iy)
    length(ixs) != 1 && return
    ix = ixs[1]
    length(ix) != length(unique(ix)) && return
    dims = []
    for i in ix
        if !(i in iy)
            push!(dims, i)
        end
    end
    setdiff(ix, dims) == [iy...] ? Sum(Tuple(dims...)) : nothing
end

global einsum_rules = [Trace, Sum, PairWise]

"""Find the matched rule."""
function match_rule(ixs, iy)
    # the first rule with the higher the priority
    for T in einsum_rules
        res = match_rule(T, ixs, iy)
        if res !== nothing
            return res
        end
    end
end
