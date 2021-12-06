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
export parse_nested, ContractionOrder, optimize_greedy, ContractionTree, label_elimination_order

function parse_eincode!(::Type{ET}, ::IncidenceList, tree, vertices_order, level=0) where ET
    ti = findfirst(==(tree), vertices_order)
    ti, NestedEinsum{ET}(ti)
end

function parse_eincode!(::Type{EIT}, incidence_list::IncidenceList, tree::ContractionTree, vertices_order, level=0) where EIT
    ti, codei = parse_eincode!(EIT, incidence_list, tree.left, vertices_order, level+1)
    tj, codej = parse_eincode!(EIT, incidence_list, tree.right, vertices_order, level+1)
    dummy = Dict([e=>0 for e in keys(incidence_list.e2v)])
    _, _, code = contract_pair!(incidence_list, vertices_order[ti], vertices_order[tj], dummy)
    ti, NestedEinsum((codei, codej), _eincode(EIT, [code.first...], level==0 ? incidence_list.openedges : code.second))
end
_eincode(::Type{DynamicEinCode{LT}}, ixs::Vector, iy::Vector) where LT = DynamicEinCode(ixs, iy)
_eincode(::Type{StaticEinCode}, ixs::Vector, iy::Vector) = StaticEinCode{(Tuple.(ixs)...,), (iy...,)}()

function parse_eincode(::Type{EIT}, incidence_list::IncidenceList{VT,ET}, tree::ContractionTree; vertices = collect(keys(incidence_list.v2e))) where {EIT,VT,ET}
    parse_eincode!(EIT, copy(incidence_list), tree, vertices)[2]
end

parse_nested(code::StaticEinCode, tree::ContractionTree) = parse_nested(StaticEinCode, code, tree)
parse_nested(code::DynamicEinCode{LT}, tree::ContractionTree) where LT = parse_nested(DynamicEinCode{LT}, code, tree)
function parse_nested(::Type{ET}, code::EinCode, tree::ContractionTree) where ET
    ixs, iy = OMEinsum.getixsv(code), OMEinsum.getiyv(code)
    incidence_list = ContractionOrder.IncidenceList(Dict([i=>ixs[i] for i=1:length(ixs)]); openedges=iy)
    parse_eincode!(ET, incidence_list, tree, 1:length(ixs))[2]
end

function parse_tree(ein, vertices)
    if isleaf(ein)
        vertices[ein.tensorindex]
    else
        if length(ein.args) != 2
            error("This eincode is not a binary tree.")
        end
        left, right = parse_tree.(ein.args, Ref(vertices))
        ContractionTree(left, right)
    end
end

"""
    optimize_greedy(eincode, size_dict; method=MinSpaceOut(), nrepeat=10)

Greedy optimizing the contraction order and return a `NestedEinsum` object. Methods are
* `MinSpaceOut`, always choose the next contraction that produces the minimum output tensor.
* `MinSpaceDiff`, always choose the next contraction that minimizes the total space.
"""
function optimize_greedy(code::DynamicEinCode{L}, size_dict::Dict; method=MinSpaceOut(), nrepeat=10) where {L}
    optimize_greedy(DynamicEinCode{L}, getixsv(code), getiyv(code), size_dict; method=method, nrepeat=nrepeat)
end
function optimize_greedy(code::StaticEinCode, size_dict::Dict; method=MinSpaceOut(), nrepeat=10)
    optimize_greedy(StaticEinCode, getixsv(code), getiyv(code), size_dict; method=method, nrepeat=nrepeat)
end
function optimize_greedy(::Type{ET}, ixs::AbstractVector{<:AbstractVector}, iy::AbstractVector, size_dict::Dict{L,TI}; method=MinSpaceOut(), nrepeat=10) where {ET, L, TI}
    if length(ixs) <= 2
        return NestedEinsum((1:length(ixs)...,), _eincode(ET, ixs, iy))
    end
    log2_edge_sizes = Dict{L,Float64}()
    for (k, v) in size_dict
        log2_edge_sizes[k] = log2(v)
    end
    incidence_list = ContractionOrder.IncidenceList(Dict([i=>ixs[i] for i=1:length(ixs)]); openedges=iy)
    tree, _, _ = tree_greedy(incidence_list, log2_edge_sizes; method=method, nrepeat=nrepeat)
    parse_eincode!(ET, incidence_list, tree, 1:length(ixs))[2]
end
function optimize_greedy(code::NestedEinsum, size_dict; method=MinSpaceOut(), nrepeat=10)
    isleaf(code) && return code
    args = optimize_greedy.(code.args, Ref(size_dict); method=method, nrepeat=nrepeat)
    if length(code.args) > 2
        # generate coarse grained hypergraph.
        nested = optimize_greedy(code.eins, size_dict; method=method, nrepeat=nrepeat)
        replace_args(nested, args)
    else
        NestedEinsum(args, code.eins)
    end
end

function replace_args(nested::NestedEinsum{ET}, trueargs) where ET
    isleaf(nested) && return trueargs[nested.tensorindex]
    NestedEinsum(replace_args.(nested.args, Ref(trueargs)), nested.eins)
end

export timespace_complexity
"""
    timespace_complexity(eincode, size_dict)

Returns the time and space complexity of the einsum contraction.
The time complexity is defined as `log2(number of element multiplication)`.
The space complexity is defined as `log2(size of the maximum intermediate tensor)`.
"""
function timespace_complexity(ei::NestedEinsum, size_dict)
    log2_sizes = Dict([k=>log2(v) for (k,v) in size_dict])
    _timespace_complexity(ei, log2_sizes)
end

function timespace_complexity(ei::EinCode, size_dict)
    log2_sizes = Dict([k=>log2(v) for (k,v) in size_dict])
    _timespace_complexity(getixsv(ei), getiyv(ei), log2_sizes)
end

function _timespace_complexity(ei::NestedEinsum, log2_sizes::Dict{L,VT}) where {L,VT}
    isleaf(ei) && return (VT(-Inf), VT(-Inf))
    tcs = VT[]
    scs = VT[]
    for arg in ei.args
        tc, sc = _timespace_complexity(arg, log2_sizes)
        push!(tcs, tc)
        push!(scs, sc)
    end
    tc2, sc2 = _timespace_complexity(getixsv(ei.eins), getiyv(ei.eins), log2_sizes)
    tc = ContractionOrder.log2sumexp2([tcs..., tc2])
    sc = max(reduce(max, scs), sc2)
    return tc, sc
end

function _timespace_complexity(ixs::AbstractVector, iy::AbstractVector{T}, log2_sizes::Dict{L,VT}) where {T, L, VT}
    loop_inds = get_loop_inds(ixs, iy)
    tc = isempty(loop_inds) ? VT(-Inf) : sum(l->log2_sizes[l], loop_inds)
    sc = isempty(iy) ? zero(VT) : sum(l->log2_sizes[l], iy)
    return tc, sc
end

function get_loop_inds(ixs::AbstractVector, iy::AbstractVector{LT}) where {LT}
    # remove redundant legs
    counts = Dict{LT,Int}()
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
    loop_inds = LT[]
    for ix in ixs
        for l in ix
            c = count(==(l), ix)
            if counts[l] > c && l ∉ loop_inds
                push!(loop_inds, l)
            end
        end
    end
    return loop_inds
end


export flop
"""
    flop(eincode, size_dict)

Returns the number of iterations, which is different with the true floating point operations (FLOP) by a factor of 2.
"""
function flop(ei::EinCode, size_dict::Dict{LT,VT}) where {LT,VT}
    loop_inds = uniquelabels(ei)
    return isempty(loop_inds) ? zero(VT) : prod(l->size_dict[l], loop_inds)
end

function flop(ei::NestedEinsum, size_dict::Dict{L,VT}) where {L,VT}
    isleaf(ei) && return zero(VT)
    return sum(ei.args) do arg
        flop(arg, size_dict)
    end + flop(ei.eins, size_dict)
end

"""
    label_elimination_order(code)

Returns a vector of labels sorted by the order they are eliminated in the contraction tree.
The contraction tree is specified by `code`, which e.g. can be a `NestedEinsum` instance.
"""
label_elimination_order(code::NestedEinsum) = label_elimination_order!(code, labeltype(code)[])
function label_elimination_order!(code, eliminated_vertices)
    OMEinsum.isleaf(code) && return eliminated_vertices
    for arg in code.args
        label_elimination_order!(arg, eliminated_vertices)
    end
    append!(eliminated_vertices, setdiff(vcat(getixsv(code.eins)...), getiyv(code.eins)))
    return eliminated_vertices
end