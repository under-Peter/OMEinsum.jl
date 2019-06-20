using TensorOperations, TupleTools, Combinatorics

@doc raw"
    edgesfrominds(ixs,iy)
return the edges of the ixs that imply an operation e.g.
in ixs = ((1,2),(2,3)), iy = (1,3), edge 2
requires a tensor contraction
"
function edgesfrominds(ixs,iy)
    allixs = TupleTools.vcat(ixs...)
    pred(e) = !(count(==(e), allixs) == 1 && e in iy)
    filter!(pred, unique(allixs))
end


@doc raw"
    EinsumOp{N}
abstract supertype of all einsum operations involving `N` edges
or `N` tensors (for `OuterProduct{N}`).
      "
abstract type EinsumOp{N} end

(::Type{T})(i::S) where {T<:EinsumOp, S<:Union{Integer,AbstractChar}} = T((i,))

@doc raw"
    TensorContract{N,T}
is a type that represents a tensorcontraction of `N` edges
of type `T` which are stored in its `edges` field, e.g. `'ij,jk -> ik'`
is represented by `TensorContract{1,Char}((j,))`.
"
struct TensorContract{N,T} <: EinsumOp{N}
    edges::NTuple{N,T}
end

@doc raw"
    Trace{N,T}
is a type that represents a trace operation of `N` edges
of type `T`, i.e. 2`N` indices, which are stored in its `edges` field,
e.g. `'ijjk -> ik'` is represented by `Trace{1,Char}((j,))`
"
struct Trace{N,T} <: EinsumOp{N}
    edges::NTuple{N,T}
end

@doc raw"
    StarContract{N,T}
is a type that represents a star-contraction of `N` edges of type `T`
which are stored in its `edges` field.
A `StarContract{N}` results from `N` tensors sharing at least one index
but *no* tensor has duplicate shared indices, e.g. `'ij,ik,il -> jkl'`
is represented by `StarContract{1,Char}((i,))`.
"
struct StarContract{N,T} <: EinsumOp{N}
    edges::NTuple{N,T}
end

@doc raw"
    MixedStarContract{N,T}
is a type that represents a mixed star-contraction of `N` edges of type `T`
which are stored in its `edges` field.
A `MixedStarContract{N}` results from `N` tensors sharing at least one index
and at least one tensor has duplicate shared indices, e.g. `'ij,ik,iil -> jkl'`
is represented by `MixedStarContract{1,Char}((i,))`.
"
struct MixedStarContract{N,T} <: EinsumOp{N}
    edges::NTuple{N,T}
end

@doc raw"
    Diag{N,T}
is a type that represents a (generalized) diagonal of `N` edges of type `T`
of one tensor which are stored in its `edges` field, e.g. `'iij -> ij'` is
represented by `Diag{1}((i,))`
"
struct Diag{N,T} <: EinsumOp{N}
    edges::NTuple{N,T}
end

@doc raw"

    MixedDiag{N,T}
is a type that represents a (generalized) mixed diagonal of `N` edges
of type `T` of more than one tensor which are stored in its `edges` field,
 e.g. `'iij, ik -> ijk'` is represented by `MixedDiag{1,Char}((i,))`
"
struct MixedDiag{N,T} <: EinsumOp{N}
    edges::NTuple{N,T}
end

@doc raw"
    IndexReduction{N,T}
is a type that represents an index reduction of `N` edges/indices of type `T`
which are stored in its `edges` field, e.g. `'ij -> i'` is
represented by `IndexReduction{1,Char}((j,))`.
"
struct IndexReduction{N,T} <: EinsumOp{N}
    edges::NTuple{N,T}
end

@doc raw"
    Permutation{N,T}
is a type that represents a permutation of `N` indices
which are stored in its `perm` field as a tuple of
`N` integers of type `T`.
"
struct Permutation{N,T} <: EinsumOp{1}
    perm::NTuple{N,T}
end

@doc raw"
    OuterProduct{N}
is a type that represents an outer product of `N` tensors.
"
struct OuterProduct{N} <: EinsumOp{N}
end

@doc raw"
    Fallback{N,T}
is a type that represents an `einsum` resulting in `N` indices of type `T`,
which are stored in its `iy` field.
It's used as a general fallback if no more efficient method is available.
"
struct Fallback{N,T} <: EinsumOp{N}
    iy::NTuple{N,T}
end

@doc raw"
    operatorfromedge(edge, ixs, iy)
returns a subtype of `EinsumOp` which specifies
the kind of operation that the reduction of `edge`
corresponds to.
"
function operatorfromedge(edge, ixs, iy)
    #it would be nice if this could be user extendible, maybe traits?
    edge == () &&  ArgumentError("empty edge provided")
    allixs = TupleTools.vcat(ixs...)
    ce      = count(==(edge), allixs)
    ceiniy  = count(==(edge), iy)
    ceinixs = count.(==(edge), ixs)
    if ce == 2 && ceiniy == 0
        all(x -> x == 0 || x == 1, ceinixs) && return TensorContract(edge)
        return Trace(edge)
    elseif  ce == 1 && ceiniy == 0
        return IndexReduction(edge)
    elseif ce > 1 && ceiniy >= 1
        any(x -> x > 1, ceinixs) && return MixedDiag(edge)
        return Diag(edge)
    elseif ce > 2 && ceiniy == 0
        any(x -> x > 1, ceinixs) && return MixedStarContract(edge)
        return StarContract(edge)
    end
end

operatorfromedge(op::EinsumOp{1}, ixs, iy) = operatorfromedge(op.edges[1], ixs, iy)

@doc raw"
    iscombineable(a,b)
return `true` if `EinsumOp`s `a` and `b` can be combined into one operator.
"
iscombineable(::T, ::S) where {T <: EinsumOp, S <: EinsumOp} = T.name == S.name

@doc raw"
    combineops(op1, op2)
return an operator that combines the operations of `op1` and `op2`.
"
function combineops(op1::T, op2::S) where {T <: EinsumOp, S <: EinsumOp}
    T.name == S.name && return T.name.wrapper((op1.edges..., op2.edges...))
    throw(ArgumentError("Can not combine $op1 and $op2"))
end



@doc raw"
    operatorsfromedges(ixs, sxs, edges, iy)
given a list of  edges `edges`, return a list of operations where
consecutive operations are combined if possible.
"
function operatorsfromedges(ixs, edges, iy)
    isempty(edges) && return appendfinalops(ixs, (), iy)

    opi = operatorfromedge(first(edges), ixs, iy)
    ops, ixs, op, = foldl((x,z) -> _operatorsfromedgeshelper(x,z,iy),
                        edges[2:end],
                        init = ((), ixs, opi, supportinds(opi, ixs)))
    ops = (ops..., op)
    ixs = indicesafteroperation(op, ixs)
    appendfinalops(ixs,ops, iy)
end

function _operatorsfromedgeshelper((ops, ixs, op2, sop2), edge, iy)
    sop1 = supportinds(edge, ixs)
    op1  = operatorfromedge(edge, ixs, iy)

    if iscombineable(op2,op1) && sop1 == sop2
        nop = combineops(op2, op1)
        return (ops, ixs, nop, sop2)
    else
        nixs = indicesafteroperation(op2, ixs)
        op1  = operatorfromedge(edge, nixs, iy)
        sop1 = supportinds(op1, nixs)
        return ((ops..., op2), nixs, op1, sop1)
    end
    return appendfinalops(ixs, (), iy)
end

function appendfinalops(ixs, ops, iy)
    if length(ixs) != 1
        op = OuterProduct{length(ixs)}()
        ops = (ops..., op)
        ixs = indicesafteroperation(op, ixs)
    end
    if all(x -> count(==(x), iy) == 1, iy)
        if ixs != (iy,)
            op = Permutation(map(x -> findfirst(==(x), ixs[1]), iy))
            ixs = (TupleTools.permute(ixs[1], op.perm),)
            ops = (ops..., op)
        end
    else
        ops = (ops..., Fallback(iy))
    end
    return ops
end


function _modifyhelper((ops, ixs, op2, sop2), edge, iy)
    sop1 = supportinds(edge, ixs)
    op1  = operatorfromedge(edge, ixs, iy)

    if iscombineable(op1,op2) && sop1 == sop2
        nop = combineops(op1, op2)
        return (ops, ixs, nop, sop2)
    else
        nixs = indicesafteroperation(op2, ixs)
        op1  = operatorfromedge(op1, nixs, iy)
        sop1 = supportinds(op1, nixs)
        return ((ops..., op2), nixs, op1, sop1)
    end
end


supportinds(op::EinsumOp, ixs) = map(x -> op.edges[1] in x, ixs)
supportinds(edge::Int, ixs) = map(x -> edge in x, ixs)

function opsfrominds(ixs, iy)
    edges = edgesfrominds(ixs, iy)
    sort!(edges)
    return operatorsfromedges(ixs, edges, iy)
end

function indicesafterop(op::EinsumOp, ixs)
    e = op.edges
    Tuple(i for i in TupleTools.vcat(ixs...) if i ∉ e)
end

function indicesafterop(op::Union{MixedDiag,Diag}, ixs)
    e = op.edges
    (Tuple(i for i in TupleTools.vcat(ixs...) if i ∉ e)...,e...)
end

indicesafterop(op::OuterProduct{N}, ixs) where N = TupleTools.vcat(ixs...)
indicesafterop(op::Permutation, ixs) = TupleTools.permute(ixs, op.perm)

@doc raw"
    opcost(op, ocost, allixs, allsxs)
returns the cost (in number of iterations it would require in a for loop)
of evaluating `op` with arguments `allixs` and `allsxs` plus `ocost`
as well as the new indices and sizes after evaluation.

`allsxs` is a tuple of tuples of Ints - the sizes of the respective arrays

"
function opcost(op::EinsumOp, cost, allixs, allsxs::NTuple{M,NTuple{N,Int} where N} where M)
    e = op.edges
    inds = Tuple(i for (i, ix) in enumerate(allixs) if overlap(e,ix))

    ixs, nallixs = pickfromtup(allixs, inds)
    sxs, nallsxs = pickfromtup(allsxs, inds)

    nix = indicesafterop(op, ixs)

    allinds  = TupleTools.vcat(ixs...)
    allsizes = TupleTools.vcat(sxs...)

    l = length(allinds)
    dims = map(ntuple(identity,l), allinds) do k,i
                ifelse(any(x -> allinds[x] == i, 1:(k-1)), 1, allsizes[k])
            end
    cost += prod(dims)
    nsx = map(i -> allsizes[findfirst(==(i), allinds)::Int], nix)

    return (cost, (nix, nallixs...), (nsx, nallsxs...))
end

function opcost(::Union{Fallback, OuterProduct, Permutation}, cost, allixs,
     allsxs::NTuple{M,NTuple{N,Int} where N} where M)
     (cost, (), ())
 end


function pickfromtup(things, inds)
    (TupleTools.getindices(things, inds), TupleTools.deleteat(things, inds))
end

@doc raw"
    overlap(s1, s2)
return true if `s1` and `s2` share any element.
"
overlap(s1, s2) = any(x -> x in s1, s2)



@doc raw"
    indicesafteroperation(op, allixs)
returns all indices of tensors after operation `op` was applied.
"
function indicesafteroperation(op::EinsumOp, allixs)
    e = op.edges
    inds = Tuple(i for (i, ix) in enumerate(allixs) if overlap(ix,e))
    ixs, nallixs = pickfromtup(allixs, inds)
    nix = indicesafterop(op, ixs)
    return (nix, nallixs...)
end

indicesafteroperation(op::OuterProduct{N}, allixs) where N = (TupleTools.vcat(allixs...),)


@doc raw"

    einsumcost(ixs, sxs, ops)
returns the cost of evaluating the einsum of `ixs`, `sxs` according to the
sequence in ops, where `sxs` is a tuple of sizes.
"
function einsumcost(ixs, sxs, ops)
    foldl((args, op) -> opcost(op, args...), ops, init = (0, ixs, sxs))[1]
end

@doc raw"
    optimalorder(ixs, xs, iy)
return a tuple of operations that represents the (possibly nonunique) optimal
order of reduction-operations.
"
function optimalorder(ixs, xs, iy)
    edges = edgesfrominds(ixs,iy)
    sxs = size.(xs)
    optimiseorder(ixs, sxs, edges, iy)[2]
end

@doc raw"
    optimiseorder(ixs, sxs, edges, iy)
return a tuple of operations that represents the (possibly nonunique) optimal
order of reduction-operations in `ops` and its cost.
"
function optimiseorder(ixs, sxs, edges,iy)
    isempty(edges) && return (0, appendfinalops(ixs, (), iy))
    foldl(permutations(edges), init = (typemax(Int), operatorsfromedges(ixs,edges,iy))) do (cost, op1), op2
        op2p = operatorsfromedges(ixs,op2,iy)
        ncost = einsumcost(ixs, sxs, op2p)
        if ncost == cost
            # if cost is the same, prefer less operations
            length(op2p) < length(op1) ? (ncost, op2p) : (cost, op1)
        else
            ncost < cost ? (ncost, op2p) : (cost, op1)
        end
    end
end
