export tree_greedy, MinSpaceOut, MinSpaceDiff
export timespace_complexity

struct MinSpaceOut end
struct MinSpaceDiff end

struct LegInfo{ET}
    l1::Vector{ET}
    l2::Vector{ET}
    l12::Vector{ET}
    l01::Vector{ET}
    l02::Vector{ET}
    l012::Vector{ET}
end

"""
    tree_greedy(incidence_list, log2_sizes; method=MinSpaceOut())

Compute greedy order, and the time and space complexities, the rows of the `incidence_list` are vertices and columns are edges.
`log2_sizes` are defined on edges.

```jldoctest; setup = :(using OMEinsum)
julia> code = ein"(abc,cde),(ce,sf,j),ak->ael"
aec, ec, ak -> ael
├─ ce, sf, j -> ec
│  ├─ sf
│  ├─ j
│  └─ ce
├─ ak
└─ abc, cde -> aec
   ├─ cde
   └─ abc


julia> optimize_greedy(code, Dict([c=>2 for c in "abcdefjkls"]))
ae, ak -> ea
├─ ak
└─ aec, ec -> ae
   ├─ ce,  -> ce
   │  ├─ sf, j -> 
   │  │  ├─ j
   │  │  └─ sf
   │  └─ ce
   └─ abc, cde -> aec
      ├─ cde
      └─ abc
```
"""
function tree_greedy(incidence_list::IncidenceList{VT,ET}, log2_edge_sizes; method=MinSpaceOut(), nrepeat=10) where {VT,ET}
    @assert nrepeat >= 1
    best_tree, best_tcs, best_scs = _tree_greedy(incidence_list, log2_edge_sizes; method=method)
    best_tc, best_sc = log2sumexp2(best_tcs), maximum(best_scs)
    for _ = 1:nrepeat-1
        tree, tcs, scs = _tree_greedy(incidence_list, log2_edge_sizes; method=method)
        tc, sc = log2sumexp2(tcs), maximum(scs)
        if sc < best_sc || (sc <= best_sc && tc < best_tc)
            best_tcs, best_scs, best_tc, best_sc, best_tree = tcs, scs, tc, sc, tree
        end
    end
    return best_tree, best_tcs, best_scs
end

function _tree_greedy(incidence_list::IncidenceList{VT,ET}, log2_edge_sizes; method=MinSpaceOut()) where {VT,ET}
    incidence_list = copy(incidence_list)
    n = nv(incidence_list)
    if n == 0
        return nothing
    elseif n == 1
        return collect(vertices(incidence_list))[1]
    end
    #tree_dict = collect(Any, 1:n)
    log2_tcs = Float64[] # time complexity
    log2_scs = Float64[]

    tree = Dict{VT,Any}([v=>v for v in vertices(incidence_list)])
    cost_values = evaluate_costs(method, incidence_list, log2_edge_sizes)
    while true
        if length(cost_values) == 0
            vpool = collect(vertices(incidence_list))
            pair = minmax(vpool[1], vpool[2])  # to prevent empty intersect
        else
            pair = find_best_cost(cost_values)
        end
        log2_tc_step, sc, code = contract_pair!(incidence_list, pair..., log2_edge_sizes)
        push!(log2_tcs, log2_tc_step)
        push!(log2_scs, space_complexity(incidence_list, log2_edge_sizes))
        if nv(incidence_list) > 1
            tree[pair[1]] = ContractionTree(tree[pair[1]], tree[pair[2]])
        else
            return ContractionTree(tree[pair[1]], tree[pair[2]]), log2_tcs, log2_scs
        end
        update_costs!(cost_values, pair..., method, incidence_list, log2_edge_sizes)
    end
end

function contract_pair!(incidence_list, vi, vj, log2_edge_sizes)
    log2dim(legs) = sum(l->log2_edge_sizes[l], legs, init=0)
    @assert vj > vi
    # compute time complexity and output tensor
    legsets = analyze_contraction(incidence_list, vi, vj)
    D12,D01,D02,D012 = log2dim.(getfield.(Ref(legsets),3:6))
    tc = D12+D01+D02+D012  # dangling legs D1 and D2 do not contribute

    # einsum code
    eout = legsets.l01 ∪ legsets.l02 ∪ legsets.l012
    code = (edges(incidence_list, vi), edges(incidence_list, vj)) => eout
    sc = log2dim(eout)

    # change incidence_list
    delete_vertex!(incidence_list, vj)
    change_edges!(incidence_list, vi, eout)
    for e in eout
        replace_vertex!(incidence_list, e, vj=>vi)
    end
    remove_edges!(incidence_list, legsets.l1 ∪ legsets.l2 ∪ legsets.l12)
    return tc, sc, code
end

function evaluate_costs(method, incidence_list::IncidenceList{VT,ET}, log2_edge_sizes) where {VT,ET}
    # initialize cost values
    cost_values = Dict{Tuple{VT,VT},Float64}()
    for vi = vertices(incidence_list)
        for vj in neighbors(incidence_list, vi)
            if vj > vi
                cost_values[(vi,vj)] = greedy_loss(method, incidence_list, log2_edge_sizes, vi, vj)
            end
        end
    end
    return cost_values
end

function update_costs!(cost_values, va, vb, method, incidence_list::IncidenceList{VT,ET}, log2_edge_sizes) where {VT,ET}
    for vj in neighbors(incidence_list, va)
        vx, vy = minmax(vj, va)
        cost_values[(vx,vy)] = greedy_loss(method, incidence_list, log2_edge_sizes, vx, vy)
    end
    for k in keys(cost_values)
        if vb ∈ k
            delete!(cost_values, k)
        end
    end
end

function find_best_cost(cost_values::Dict{PT}) where PT
    length(cost_values) < 1 && error("cost value information missing")
    minval = minimum(Base.values(cost_values))
    pairs = PT[]
    for (k, v) in cost_values
        if v == minval
            push!(pairs, k)
        end
    end
    return rand(pairs)
end

function analyze_contraction(incidence_list::IncidenceList{VT,ET}, vi::VT, vj::VT) where {VT,ET}
    ei = edges(incidence_list, vi)
    ej = edges(incidence_list, vj)
    leg012,leg12,leg1,leg2,leg01,leg02 = ET[], ET[], ET[], ET[], ET[], ET[]
    # external legs
    for leg in ei ∪ ej
        isext = leg ∈ incidence_list.openedges || !all(x->x==vi || x==vj, vertices(incidence_list, leg))
        if isext
            if leg ∈ ei
                if leg ∈ ej
                    push!(leg012, leg)
                else
                    push!(leg01, leg)
                end
            else
                push!(leg02, leg)
            end
        else
            if leg ∈ ei
                if leg ∈ ej
                    push!(leg12, leg)
                else
                    push!(leg1, leg)
                end
            else
                push!(leg2, leg)
            end
        end
    end
    return LegInfo(leg1, leg2, leg12, leg01, leg02, leg012)
end

function greedy_loss(::MinSpaceOut, incidence_list, log2_edge_sizes, vi, vj)
    log2dim(legs) = sum(l->log2_edge_sizes[l], legs, init=0)
    legs = analyze_contraction(incidence_list, vi, vj)
    log2dim(legs.l01)+log2dim(legs.l02)+log2dim(legs.l012)
end

function greedy_loss(::MinSpaceDiff, incidence_list, log2_edge_sizes, vi, vj)
    log2dim(legs) = sum(l->log2_edge_sizes[l], legs, init=0)
    legs = analyze_contraction(incidence_list, vi, vj)
    D1,D2,D12,D01,D02,D012 = log2dim.(getfield.(Ref(legs), 1:6))
    exp2(D01+D02+D012) - exp2(D1+D01+D12) - exp2(D2+D02+D12)  # out - in
end

function space_complexity(incidence_list, log2_sizes)
    sc = 0.0
    for v in vertices(incidence_list)
        for e in edges(incidence_list, v)
            sc += log2_sizes[e]
        end
    end
    return sc
end

function contract_tree!(incidence_list::IncidenceList, tree::ContractionTree, log2_edge_sizes, tcs, scs)
    vi = tree.left isa ContractionTree ? contract_tree!(incidence_list, tree.left, log2_edge_sizes, tcs, scs) : tree.left
    vj = tree.right isa ContractionTree ? contract_tree!(incidence_list, tree.right, log2_edge_sizes, tcs, scs) : tree.right
    tc, sc, code = contract_pair!(incidence_list, vi, vj, log2_edge_sizes)
    push!(tcs, tc)
    push!(scs, sc)
    return vi
end

function timespace_complexity(incidence_list::IncidenceList, tree::ContractionTree, log2_edge_sizes)
    tcs, scs = Float64[], Float64[]
    contract_tree!(copy(incidence_list), tree, log2_edge_sizes, tcs, scs)
    return log2sumexp2(tcs), maximum(scs)
end