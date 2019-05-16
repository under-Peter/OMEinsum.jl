# Ref
# * https://github.com/JuliaLang/julia/issues/2345#issuecomment-54537633
# * https://github.com/mauro3/SimpleTraits.jl

using TupleTools
using SimpleTraits, LinearAlgebra

export EinCode, is_pairwise, IsPairWise

struct EinCode{C} end

function EinCode(ixs::Tuple, iys::Tuple)
    # re-assign indices
    CODE = (ixs..., iys)
    EinCode{CODE}()
end

"""
a einsum code is a pairwise graph.
"""
function is_pairwise(code::Tuple)
    all_indices = TupleTools.vcat(code...)
    counts = Dict{Int, Int}()
    for ind in all_indices
        counts[ind] = get(counts, ind, 0) + 1
    end
    all(isequal(2), counts |> values)
end

@traitdef IsPairWise{CODE}
@traitimpl IsPairWise{CODE} <- is_pairwise(CODE)

"""The most general case as fall back"""
function einsum!(::Type{TP}, ::EinCode{C}, xs, y) where {TP, C}
    @show TP, C
    einsum!(C[1:end-1], xs, C[end], y)
    return "general"
end

"""Dispatch to trace."""
function einsum!(::EinCode{((1,1), ())}, xs, y)
    println("doing contraction using tr!")
    y[] = tr(xs)
    return "tr"
end

@traitfn function einsum!(::EinCode{X}, xs, y) where {X; IsPairWise{X}}
    println("doing contraction using @tensor!")
    return "@tensor"
end
