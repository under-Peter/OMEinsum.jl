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

using .ContractionOrder: IncidenceList, ContractionTree, contract_pair!, MinSpaceOut, MinSpaceDiff, tree_greedy
export parse_eincode, ContractionOrder, optimize_greedy

function parse_eincode!(::IncidenceList, tree, vertices_order, level=0)
    ti = findfirst(==(tree), vertices_order)
    ti, ti
end

function parse_eincode!(incidence_list::IncidenceList, tree::ContractionTree, vertices_order, level=0)
    ti, codei = parse_eincode!(incidence_list, tree.left, vertices_order, level+1)
    tj, codej = parse_eincode!(incidence_list, tree.right, vertices_order, level+1)
    dummy = Dict([e=>0 for e in keys(incidence_list.e2v)])
    _, _, code = contract_pair!(incidence_list, vertices_order[ti], vertices_order[tj], dummy)
    ti, NestedEinsum((codei, codej), EinCode(Tuple.(code.first), Tuple(level==0 ? incidence_list.openedges : code.second)))
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

Greedy optimizing the contraction order and return a `NestedEinsum` object. Methods are
* `MinSpaceOut`, always choose the next contraction that produces the minimum output tensor.
* `MinSpaceDiff`, always choose the next contraction that minimizes the total space.
"""
function optimize_greedy(@nospecialize(code::EinCode{ixs, iy}), size_dict::Dict{L,T}; method=MinSpaceOut(), nrepeat=10) where {ixs, iy, L, T}
    optimize_greedy(collect(ixs), collect(iy), size_dict; method=MinSpaceOut(), nrepeat=nrepeat)
end
function optimize_greedy(ixs::AbstractVector, iy::AbstractVector, size_dict::Dict{L,TI}; method=MinSpaceOut(), nrepeat=10) where {L, TI}
    if length(ixs) <= 2
        return NestedEinsum((1:length(ixs)...,), EinCode{(ixs...,), (iy...,)}())
    end
    T = promote_type(eltype.(ixs)...)
    log2_edge_sizes = Dict{L,Float64}()
    for (k, v) in size_dict
        log2_edge_sizes[k] = log2(v)
    end
    incidence_list = ContractionOrder.IncidenceList(Dict([i=>collect(T,ixs[i]) for i=1:length(ixs)]); openedges=collect(T,iy))
    tree, _, _ = tree_greedy(incidence_list, log2_edge_sizes; method=method, nrepeat=nrepeat)
    parse_eincode!(incidence_list, tree, 1:length(ixs))[2]
end
optimize_greedy(code::Int, size_dict; method=MinSpaceOut(), nrepeat=10) = code
function optimize_greedy(code::NestedEinsum, size_dict; method=MinSpaceOut(), nrepeat=10)
    args = optimize_greedy.(code.args, Ref(size_dict); method=method, nrepeat=nrepeat)
    if length(code.args) > 2
        # generate coarse grained hypergraph.
        nested = optimize_greedy(code.eins, size_dict; method=method, nrepeat=nrepeat)
        replace_args(nested, args)
    else
        NestedEinsum(args, code.eins)
    end
end

function replace_args(nested::NestedEinsum, trueargs)
    NestedEinsum(replace_args.(nested.args, Ref(trueargs)), nested.eins)
end
replace_args(nested::Int, trueargs) = trueargs[nested]

export timespace_complexity
"""
    timespace_complexity(eincode, size_dict)

Return the time and space complexity of the einsum contraction.
The time complexity is defined as `log2(number of element multiplication)`.
The space complexity is defined as `log2(size of the maximum intermediate tensor)`.
"""
function timespace_complexity(ei::NestedEinsum, size_dict)
    log2_sizes = Dict([k=>log2(v) for (k,v) in size_dict])
    _timespace_complexity(ei, log2_sizes)
end

function timespace_complexity(@nospecialize(ei::EinCode{ixs, iy}), size_dict) where {ixs, iy}
    log2_sizes = Dict([k=>log2(v) for (k,v) in size_dict])
    _timespace_complexity(collect(ixs), collect(iy), log2_sizes)
end

function _timespace_complexity(ei::NestedEinsum, log2_sizes)
    tcs = Float64[]
    scs = Float64[]
    for arg in ei.args
        tc, sc = _timespace_complexity(arg, log2_sizes)
        push!(tcs, tc)
        push!(scs, sc)
    end
    tc2, sc2 = _timespace_complexity(collect(getixs(ei.eins)), collect(getiy(ei.eins)), log2_sizes)
    tc = ContractionOrder.log2sumexp2([tcs..., tc2])
    sc = max(reduce(max, scs), sc2)
    return tc, sc
end
_timespace_complexity(ei::Int, size_dict) = -Inf, -Inf

function _timespace_complexity(ixs::AbstractVector, iy::AbstractVector{T}, log2_sizes::Dict{L}) where {T, L}
    # remove redundant legs
    counts = Dict{L,Int}()
    for ix in ixs
        for l in ix
            if haskey(counts, l)
                counts[l] += 1
            else
                counts[l] = 1
            end
        end
    end
    for l in iy
        if haskey(counts, l)
            counts[l] += 1
        else
            counts[l] = 1
        end
    end
    loop_inds = L[]
    for ix in ixs
        for l in ix
            c = count(==(l), ix)
            if counts[l] > c && l ∉ loop_inds
                push!(loop_inds, l)
            end
        end
    end
    tc = isempty(loop_inds) ? -Inf : sum(l->log2_sizes[l], loop_inds)
    sc = isempty(iy) ? 0.0 : sum(l->log2_sizes[l], iy)
    return tc, sc
end
