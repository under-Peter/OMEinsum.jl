using TensorOperations, TupleTools, Combinatorics

@doc raw"
    edgesfrominds(ixs,iy)
return the edges of the ixs that imply an operation e.g.
in ixs = ((1,2),(2,3)), iy = (1,3), edge 2
requires a tensor contraction
"
function edgesfrominds(ixs,iy)
    allixs = TupleTools.vcat(ixs...)
    pred(i,e) = !(count(==(e), allixs) == 1 && e in iy) && # not trivial
                all(j -> allixs[j] != e, 1:(i-1)) # not seen before
    Tuple(e for (i,e) in enumerate(allixs) if pred(i,e))
end


@doc raw"
    EinsumOp
abstract supertype of all einsum operations
"
abstract type EinsumOp{N} end

(::Type{T})(i::S) where {T<:EinsumOp, S<:Union{Integer,AbstractChar}} = T((i,))

struct PlaceHolder{T} <: EinsumOp{1}
    edges::Tuple{T}
end

struct TensorContract{N,T} <: EinsumOp{N}
    edges::NTuple{N,T}
end

struct Trace{N,T} <: EinsumOp{N}
    edges::NTuple{N,T}
end

struct StarContract{N,T} <: EinsumOp{N}
    edges::NTuple{N,T}
end

struct MixedStarContract{N,T} <: EinsumOp{N}
    edges::NTuple{N,T}
end

struct Diag{N,T} <: EinsumOp{N}
    edges::NTuple{N,T}
end

struct MixedDiag{N,T} <: EinsumOp{N}
    edges::NTuple{N,T}
end

struct IndexReduction{N,T} <: EinsumOp{N}
    edges::NTuple{N,T}
end

struct Permutation{N,T} <: EinsumOp{1}
    perm::NTuple{N,T}
end

struct OuterProduct{N} <: EinsumOp{N}
end

struct Fallback{N,T} <: EinsumOp{N}
    iy::NTuple{N,T}
end

@doc raw"
    placeholdersfrominds(ixs, iy)
return all indices in `ixs` that imply an operation, wrapped in a `PlaceHolder`.
"
placeholdersfrominds(ixs, iy) = map(PlaceHolder, edgesfrominds(ixs, iy))

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
iscombineable(::Any,::Any) = false
iscombineable(::T, ::T) where {T <: EinsumOp} = true

@doc raw"
    combineops(op1, op2)
return an operator that combines the operations of `op1` and `op2`.
"
combineops(op1::T, op2::T) where {T <: EinsumOp} = T.name.wrapper((op1.edges..., op2.edges...))



@doc raw"
    modifyops(ixs, sxs, ops, iy)
given a list of placeholders `ops`, return a list of operations where
consecutive operations are combined if possible.
"
function modifyops(ixs, ops, iy)
    if !isempty(ops)
        opi = operatorfromedge(first(ops), ixs, iy)
        ops, ixs, op, = foldl((x,z) -> _modifyhelper(x,z,iy),
                            ops[2:end],
                            init = ((), ixs, opi, supportinds(opi, ixs)))
        ops = (ops..., op)
        ixs = indicesafteroperation(op, ixs)
    end
    ops = appendfinalops(ixs,ops, iy)
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

function _modifyhelper((ops, ixs, op2, sop2), op, iy)
    sop1 = supportinds(op, ixs)
    op1  = operatorfromedge(op, ixs, iy)

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

supportinds(op, ixs) = Tuple(i for (i,ix) in enumerate(ixs) if op.edges[1] in ix)

function opsfrominds(ixs, iy)
    tmp = placeholdersfrominds(ixs, iy)
    tmp = TupleTools.sort(tmp, by = x -> x.edges[1])
    return modifyops(ixs, tmp, iy)
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

`allscs` is a tuple of tuples of Ints - the sizes of the respective arrays

"
function opcost(op::EinsumOp, cost, allixs, allsxs::NTuple{M,NTuple{N,Int} where N} where M)
    e = op.edges
    inds = Tuple(i for (i, ix) in enumerate(allixs) if !isempty(intersect(e,ix)))

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
    nsx = map(nix) do i
        j = findfirst(==(i), allinds)::Int
        allsizes[j]
    end
    return (cost, (nix, nallixs...), (nsx, nallsxs...))
end

function pickfromtup(things, inds)
    (TupleTools.getindices(things, inds), TupleTools.deleteat(things, inds))
end

function indicesafteroperation(op::EinsumOp, allixs)
    e = op.edges
    inds = Tuple(i for (i, ix) in enumerate(allixs) if !isempty(intersect(ix, e)))
    ixs, nallixs = pickfromtup(allixs, inds)
    nix = indicesafterop(op, ixs)
    return (nix, nallixs...)
end

indicesafteroperation(op::OuterProduct{N}, allixs) where N = (TupleTools.vcat(allixs...),)


@doc raw"
    meinsumcost(ixs, xs, ops)
returns the cost of evaluating the einsum of `ixs`, `xs` according to the
sequence in ops.
"
function meinsumcost(ixs, xs, ops)
    foldl((args, op) -> opcost(op, args...), ops, init = (0, ixs, xs))[1]
end

@doc raw"
    optimalorder(ixs, xs, iy)
return a tuple of operations that represents the (possibly nonunique) optimal
order of reduction-operations.
"
function optimalorder(ixs, xs, iy)
    tmp = placeholdersfrominds(ixs, iy)
    sxs = size.(xs)
    optimiseorder(ixs, sxs, tmp, iy)[2]
end

@doc raw"
    optimiseorder(ixs, sxs, ops, iy)
return a tuple of operations that represents the (possibly nonunique) optimal
order of reduction-operations in `ops` and its cost.
"
function optimiseorder(ixs, sxs, ops,iy)
    isempty(ops) && return (0, appendfinalops(ixs, ops, iy))
    foldl(permutations(ops), init = (typemax(Int), modifyops(ixs,ops,iy))) do (cost, op1), op2
        op2p = modifyops(ixs,op2,iy)
        ncost = meinsumcost(ixs, sxs, op2p)
        if ncost == cost
            #prefer less operations even if same cost
            length(op2p) < length(op1) ? (ncost, op2p) : (cost, op1)
        else
            ncost < cost ? (ncost, op2p) : (cost, op1)
        end
    end
end
