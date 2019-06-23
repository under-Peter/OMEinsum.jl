abstract type EinRule{NT} end
struct Sum <: EinRule{1}
    dims
end
struct Trace <: EinRule{1} end
struct PairWise{NT} <: EinRule{NT} end
struct Permutedims <: EinRule{1} end

"""
a einsum code is trace
"""
function match(::Type{Trace}, ixs, iy)
    iy == () &&
    length(ixs) == 1 &&
    length(ixs[1]) == 2 &&
    ixs[1][1] == ixs[1][2]
end

"""
a einsum code is sum.
"""
function match(::Type{Sum}, ixs, iy)
    length(ixs) != 1 && return false
    length(ixs[1]) != length(unique(ixs[1])) && return false
    dims = []
    for i in ixs[1]
        if !(i in iy)
            push!(dims, i)
        end
    end
    return Sum{Tuple(dims...)}()
end

"""
a einsum code is a pairwise graph.
"""
function match(::Type{PairWise}, ixs, iy)
    all_indices = TupleTools.vcat(ixs..., iy)
    counts = Dict{Int, Int}()
    for ind in all_indices
        counts[ind] = get(counts, ind, 0) + 1
    end
    all(isequal(2), counts |> values)
end
