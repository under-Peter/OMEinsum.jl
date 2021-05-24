module ContractionOrder

export ContractionTree

struct ContractionTree
    left
    right
end

function log2sumexp2(s)
    ms = maximum(s)
    return log2(sum(x->exp2(x - ms), s)) + ms
end

include("incidencelist.jl")
include("greedy.jl")

end

using .ContractionOrder: IncidenceList, ContractionTree, contract_pair!, MinSpaceOut, MinSpaceDiff, tree_greedy, timespace_complexity
export parse_eincode, ContractionOrder, optimize_greedy

function parse_eincode!(incidence_list::IncidenceList, tree, vertices_order)
    ti = findfirst(==(tree), vertices_order)
    ti, ti
end

function parse_eincode!(incidence_list::IncidenceList, tree::ContractionTree, vertices_order)
    ti, codei = parse_eincode!(incidence_list, tree.left, vertices_order)
    tj, codej = parse_eincode!(incidence_list, tree.right, vertices_order)
    dummy = Dict([e=>0 for e in keys(incidence_list.e2v)])
    tc, sc, code = contract_pair!(incidence_list, vertices_order[ti], vertices_order[tj], dummy)
    ti, NestedEinsum((codei, codej), EinCode(Tuple.(code.first), Tuple(code.second)))
end

function parse_eincode(incidence_list::IncidenceList{VT,ET}, tree::ContractionTree; vertices = collect(keys(incidence_list.v2e))) where {VT,ET}
    parse_eincode!(copy(incidence_list), tree, vertices)[2]
end

function parse_tree(ein, vertices)
    if ein isa NestedEinsum
        if length(ein.args) != 2
            error("This eincode is not a binary tree.")
        end
        left, right = parse_tree.(ein.args, Ref(vertices))
        ContractionTree(left, right)
    else
        vertices[ein]
    end
end

"""
    optimize_greedy(eincode, size_dict; method=MinSpaceOut(), nrepeat=10)

Greedy optimizing the contraction order. Methods are
* `MinSpaceOut`, always choose the next contraction that produces the minimum output tensor.
* `MinSpaceDiff`, always choose the next contraction that minimizes the total space.
"""
function optimize_greedy(@nospecialize(code::EinCode{ixs, iy}), size_dict; method=MinSpaceOut(), nrepeat=10) where {ixs, iy}
    optimize_greedy(collect(ixs), collect(iy), size_dict; method=MinSpaceOut(), nrepeat=nrepeat)
end
function optimize_greedy(ixs::AbstractVector, iy::AbstractVector, size_dict; method=MinSpaceOut(), nrepeat=10)
    if length(ixs) < 2
        return code
    end
    T = promote_type(eltype.(ixs)...)
    log2_edge_sizes = empty(size_dict)
    for (k, v) in size_dict
        log2_edge_sizes[k] = log2(v)
    end
    incidence_list = ContractionOrder.IncidenceList(Dict([i=>collect(T,ixs[i]) for i=1:length(ixs)]); openedges=collect(T,iy))
    tree, _, _ = tree_greedy(incidence_list, log2_edge_sizes; method=method, nrepeat=nrepeat)
    parse_eincode!(incidence_list, tree, 1:length(ixs))[2]
end
optimize_greedy(code::Int, size_dict; method=MinSpaceOut()) = code
function optimize_greedy(code::NestedEinsum, size_dict; method=MinSpaceOut())
    args = optimize_greedy.(code.args, Ref(size_dict); method=method)
    if length(code.args) > 2
        # generate coarse grained hypergraph.
        nested = optimize_greedy(code.eins, size_dict; method=method)
        replace_args(nested, args)
    else
        NestedEinsum(args, code.eins)
    end
end

function replace_args(nested::NestedEinsum, trueargs)
    NestedEinsum(replace_args.(nested.args, Ref(trueargs)), nested.eins)
end
replace_args(nested::Int, trueargs) = trueargs[nested]

"""
    ContractionOrder.timespace_complexity(eincode, size_dict)

Return the time and space complexity of the einsum contraction.
The time complexity is defined as `log2(number of element multiplication)`.
The space complexity is defined as `log2(size of the maximum intermediate tensor)`.
"""
ContractionOrder.timespace_complexity(ei::Int, size_dict) = -Inf, -Inf
function ContractionOrder.timespace_complexity(ei::NestedEinsum, size_dict)
    tcs = Float64[]
    scs = Float64[]
    for arg in ei.args
        tc, sc = timespace_complexity(arg, size_dict)
        push!(tcs, tc)
        push!(scs, sc)
    end
    tc2, sc2 = timespace_complexity(collect(getixs(ei.eins)), collect(getiy(ei.eins)), size_dict)
    tc = ContractionOrder.log2sumexp2([tcs..., tc2])
    sc = max(reduce(max, scs), sc2)
    return tc, sc
end
function ContractionOrder.timespace_complexity(@nospecialize(ei::EinCode{ixs, iy}), size_dict) where {ixs, iy}
    ContractionOrder.timespace_complexity(collect(ixs), collect(iy), size_dict)
end
function ContractionOrder.timespace_complexity(ixs::AbstractVector, iy::AbstractVector, size_dict)
    # remove redundant legs
    labels = vcat(collect.(ixs)..., iy)
    loop_inds = unique!(filter(l->count(==(l), labels)>=2, labels))

    tc = isempty(loop_inds) ? -Inf : sum(l->log2(size_dict[l]), loop_inds)
    sc = isempty(iy) ? 0.0 : sum(l->log2(size_dict[l]), iy)
    return tc, sc
end

function _flatten(code::NestedEinsum, iy=nothing)
    ixs = []
    for i=1:length(code.args)
        append!(ixs, _flatten(code.args[i], OMEinsum.getixs(code.eins)[i]))
    end
    return ixs
end
_flatten(i::Int, iy) = [i=>iy]

function Base.Iterators.flatten(code::NestedEinsum)
    ixd = Dict(_flatten(code))
    EinCode(([ixd[i] for i=1:length(ixd)]...,), OMEinsum.getiy(code.eins))
end
