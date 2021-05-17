export IncidenceList

struct IncidenceList{VT,ET}
    v2e::Dict{VT,Vector{ET}}
    e2v::Dict{ET,Vector{VT}}
    openedges::Vector{ET}
end

function IncidenceList(v2e::Dict{VT,Vector{ET}}; openedges=ET[]) where {VT,ET}
    e2v = Dict{ET,Vector{VT}}()
    for (v, es) in v2e
        for e in es
            if haskey(e2v, e)
                push!(e2v[e], v)
            else
                e2v[e] = [v]
            end
        end
    end
    IncidenceList(v2e, e2v, openedges)
end

Base.copy(il::IncidenceList) = IncidenceList(deepcopy(il.v2e), deepcopy(il.e2v), copy(il.openedges))

function neighbors(il::IncidenceList{VT}, v) where VT
    res = VT[]
    for e in il.v2e[v]
        for v in il.e2v[e]
            push!(res, v)
        end
    end
    return unique!(res)
end
vertices(il::IncidenceList) = keys(il.v2e)
vertices(il::IncidenceList, e) = il.e2v[e]
vertex_degree(il::IncidenceList, v) = length(il.v2e[v])
edge_degree(il::IncidenceList, e) = length(il.e2v[v])
edges(il::IncidenceList, v) = il.v2e[v]
nv(il::IncidenceList) = length(il.v2e)
ne(il::IncidenceList) = length(il.e2v)

function delete_vertex!(incidence_list::IncidenceList{VT,ET}, vj::VT) where {VT,ET}
    edges = pop!(incidence_list.v2e, vj)
    for e in edges
        vs = vertices(incidence_list, e)
        res = findfirst(==(vj), vs)
        if res !== nothing
            deleteat!(vs, res)
        end
    end
    return incidence_list
end

function change_edges!(incidence_list, vi, es)
    incidence_list.v2e[vi] = es
    return incidence_list
end

function remove_edges!(incidence_list, es)
    for e in es
        delete!(incidence_list.e2v, e)
    end
    return incidence_list
end

function replace_vertex!(incidence_list, e, pair)
    el = incidence_list.e2v[e]
    if pair.first ∈ el
        if pair.second ∈ el
            deleteat!(el, findfirst(==(pair.first), el))
        else
            replace!(el, pair)
        end
    else
        if pair.second ∉ el
            push!(el, pair.second)
        end
    end
    return incidence_list
end