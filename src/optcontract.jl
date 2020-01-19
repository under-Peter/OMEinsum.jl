export optcontract, treecontract

"""
find the optimal contraction tree.
"""
function TensorOperations.optimaltree(network, size_dict::Dict{Int, Int}=Dict{Int, Int}())
    unique_tokens = union(network...)
    optimaltree(network, Dict(token=>Power{:χ}(get(size_dict, token, 1),1) for token in unique_tokens))
end

function leg_analysis(IVS...)
    IALL = union(IVS...)
    II = intersect(IVS...)
    IC = setdiff(IALL, II)
    IALL, II, IC
end

_treecontract(tree::Int, ixs, xs, iy::Nothing) = xs[tree], ixs[tree]
function _treecontract(tree::Int, ixs, xs, iy)
    iy0, y = ixs[tree], xs[tree]
    einsum(EinCode{(iy0,), iy}(), (C,)), iy
end

function _treecontract(tree, ixs, xs, iy)
    i, j = tree
    A, IA = _treecontract(i, ixs, xs, nothing)
    B, IB = _treecontract(j, ixs, xs, nothing)
    _iy = iy == nothing ? Tuple(leg_analysis(IA, IB)[3]) : iy
    ixs = (IA, IB)
    code = EinCode{ixs, _iy}()
    res = einsum(BatchedContract(), code, (A, B), get_size_dict(ixs, (A, B))), _iy
    return res
end

function treecontract(tree, ixs, xs, iy)
    _treecontract(tree, ixs, xs, iy) |> first
end

function get_size_dict(ixs, xs)
    nt = length(ixs)
    size_dict = Dict{Int, Int}()
    @inbounds for i = 1:nt
        for (N, leg) in zip(size(xs[i]), ixs[i])
            if haskey(size_dict, leg)
                size_dict[leg] == N || throw(DimensionMismatch("size of contraction leg $leg not match."))
            else
                size_dict[leg] = N
            end
        end
    end
    return size_dict
end

function optcontract(ixs, xs, iy)
    size_dict = get_size_dict(ixs, xs)
    tree, cost = optimaltree(ixs, Dict(size_dict))
    treecontract(tree, ixs, xs, iy)
end


