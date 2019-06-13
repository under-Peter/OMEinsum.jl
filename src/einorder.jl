using OMEinsum
using TensorOperations
#= einorder
1) identify edges & their corresponding operations:
    - tensorcontraction
    - (partial) trace
    - (partial) diagonal
    - star-contraction between
    - star-contraction mixed
    - index reduction
    - perm
   and then construct list of all operations implied by the indices
2. optimise order of operations for minimum total iterations (in the naive loop view)

3. evaluate einsum in optimized order
=#

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

struct PlaceHolder{T} <: EinsumOp{0}
    edge::T
    keep::Bool
end

struct TensorContract{N,T} <: EinsumOp{N}
    edges::NTuple{N,T}
end
TensorContract(e::Int) = TensorContract((e,))

struct Trace{N,T} <: EinsumOp{N}
    edges::NTuple{N,T}
end
Trace(e::Int) = Trace((e,))

struct StarContract{N,T} <: EinsumOp{N}
    edges::NTuple{N,T}
end
StarContract(e::Int) = StarContract((e,))

struct MixedStarContract{N,T} <: EinsumOp{N}
    edges::NTuple{N,T}
end
MixedStarContract(e::Int) = MixedStarContract((e,))

struct Diag{N,T} <: EinsumOp{N}
    edges::NTuple{N,T}
end
Diag(e::Int) = Diag((e,))

struct MixedDiag{N,T} <: EinsumOp{N}
    edges::NTuple{N,T}
end
MixedDiag(e::Int) = MixedDiag((e,))

struct IndexReduction{N,T} <: EinsumOp{N}
    edges::NTuple{N,T}
end
IndexReduction(e::Int) = IndexReduction((e,))

placeholderfromedge(edge, iy) = PlaceHolder(edge, edge in iy)
function placeholdersfrominds(ixs, iy)
    edges = edgesfrominds(ixs,iy)
    map(x -> placeholderfromedge(x, iy), edges)
end

@doc raw"
    operatorfromedge(edge, ixs, iy)
returns a subtype of `EinsumOp` which specifies
the kind of operation that `edge` implies.
"
function operatorfromedge(edge, ixs, iy)
    #it would be nice if this could be user extendible, maybe traits?
    edge == () &&  error()
    allixs = TupleTools.vcat(ixs...)
    ce = count(==(edge), allixs)
    ceiniy = count(==(edge), iy)
    ceinixs = count.(==(edge), ixs)
    if ce == 2 && ceiniy == 0
        all(x -> x == 0 || x >= 1, ceinixs) && return TensorContract(edge)
        return Trace(edge)
    elseif  ce == 1 && ceiniy == 0
        return IndexReduction(edge)
    elseif ce > 1 && ceiniy >= 1
        #diagonal
        count(x -> x > 1, ceinixs) > 1 && return MixedDiag(edge)
        return Diag(edge)
    elseif ce > 2 && ceiniy == 0
        any(x -> x > 1, ceinixs) && return MixedStarContract(edge)
        return StarContract(edge)
    end
end

@doc raw"
    modifyops(ixs, sxs, ops, iy)
returns a list of operations with correct types.
This is a quick fix while initial operation-assignment doesn't work,
due to how the sequence of operations influences what operations are evaluated,
e.g. a tensorcontraction might turn into a trace if the corresponding tensors
are contracted earlier.
This function just passes over all operations and returns a list of the
real operations that are evaluated.
"
iscompatible(::EinsumOp, ::EinsumOp) = false
iscompatible(::TensorContract, ::TensorContract) = true
iscompatible(::MixedDiag, ::MixedDiag) = true
iscompatible(::IndexReduction, ::IndexReduction) = true
iscompatible(::Diag, ::Diag) = true
function foo((ops, ixs, op2, sop2), op, iy)
    sop1 = supportinds(op, ixs)
    op1 = operatorfromedge(op.edge, ixs, iy)
    if iscompatible(op1,op2) && sop1 == sop2
        #if compatible and same support - keep and go on
        nop = combineops(op1, op2)
        return (ops, ixs, nop, sop2)
    else
        nixs = indicesafteroperation(op2, ixs)
        return ((ops..., op2), nixs, op1, sop1)
    end
end

combineops(op1::TensorContract, op2::TensorContract) =
    TensorContract((op1.edges...,op2.edges...))
combineops(op1::Trace, op2::Trace) =
    Trace((op1.edges...,op2.edges...))
combineops(op1::IndexReduction, op2::IndexReduction) =
    IndexReduction((op1.edges...,op2.edges...))
combineops(op1::Diag, op2::Diag) =
    Diag((op1.edges...,op2.edges...))

function modifyops(ixs, ops, iy)
    ops == () && return ()
    opi = operatorfromedge(ops[1].edge, ixs, iy)
    ops, _, op, = foldl((x,z) -> foo(x,z,iy),
                        ops[2:end],
                        init = ((), ixs, opi, supportinds(opi, ixs)))
    return (ops..., op)
end

supportinds(op, ixs) =
    Tuple(i for (i,ix) in enumerate(ixs) if op.edges[1] in ix)
supportinds(op::PlaceHolder, ixs) = Tuple(i for (i,ix) in enumerate(ixs) if op.edge[1] in ix)




#in reality, outer products between all tensors should be included as ops
function operatorsfrominds(ixs,iy)
    edges = edgesfrominds(ixs,iy)
    map(x -> operatorfromedge(x, ixs, iy), edges)
end

using TupleTools
function indicesafterop(op::PlaceHolder, ixs)
    e = op.edge
    Tuple(i for i in TupleTools.vcat(ixs...) if i != e)
end
function indicesafterop(op::EinsumOp, ixs)
    e = op.edges
    Tuple(i for i in TupleTools.vcat(ixs...) if i ∉ e)
end

function indicesafterop(op::Diag, ixs)
    e = op.edges
    (Tuple(i for i in TupleTools.vcat(ixs...) if i ∉ e)...,e...)
end

function indicesafterop(op::MixedDiag, ixs)
    e = op.edges
    (Tuple(i for i in TupleTools.vcat(ixs...) if i ∉ e)...,e...)
end

@doc raw"
    evaluate(op::EinsumOp, allixs, allxs)
returns a tuple of xs and a tuple of ixs that includes the result
of the operation `op`.
"
function evaluate(op::EinsumOp, allixs, allxs)
    e = op.edges
    # println("$op w/ einsum")
    inds = Tuple(i for (i, ix) in enumerate(allixs) if !isempty(intersect(e,ix)))
    ixs = TupleTools.getindices(allixs,inds)
    xs  = TupleTools.getindices(allxs,inds)

    nallixs =  TupleTools.deleteat(allixs, inds)
    nallxs  =  TupleTools.deleteat(allxs,  inds)

    nix = indicesafterop(op, ixs)
    nx = einsumexp(ixs, xs, nix)

    return (nix, nallixs...), (nx, nallxs...)
end

function evaluate(op::IndexReduction, allixs, allxs)
    e = op.edges[1]
    # println("$op w/ sum")
    ind = findfirst(x -> e in x, allixs)
    ix, x = allixs[ind], allxs[ind]

    nallixs =  TupleTools.deleteat(allixs, (ind,))
    nallxs  =  TupleTools.deleteat(allxs,  (ind,))

    inds = Tuple(findall(x -> x in op.edges, ix))
    nix = TupleTools.deleteat(ix, inds)
    nx = dropdims(sum(x, dims = inds), dims = inds)

    return (nallixs..., nix), (nallxs..., nx)
end

# # overload evaluate for special types! multiple dispatch for separate functions!
# function evaluate(op::TensorContract, allixs, allxs)
#     # println("$op w/ TensorOperations")
#     e = op.edge
#     inds = Tuple(i for (i, ix) in enumerate(allixs) if e in ix)
#     ixs = TupleTools.getindices(allixs,inds)
#     xs  = TupleTools.getindices(allxs,inds)
#
#     nallixs =  TupleTools.deleteat(allixs, inds)
#     nallxs  =  TupleTools.deleteat(allxs,  inds)
#
#     a, b = xs
#     ia, ib = ixs
#     i2change =  [i for (i,j) in enumerate(ia) if j in ib && j != e]
#     for i in i2change
#         ia = TupleTools.insertat(ia, i, (ia[i] + 1000,))
#     end
#     nix = indicesafterop(op, (ia,ib))
#     nx = tensorcontract(a,ia,b,ib,nix)
#     nix = map(x -> x > 500 ? x - 1000 : x, nix)
#
#     return (nix, nallixs...), (nx, nallxs...)
# end
#
# function evaluate(op::Trace, allixs, allxs)
#     # println("Trace w/ TensorOperations")
#     e = op.edge
#     inds = Tuple(i for (i, ix) in enumerate(allixs) if e in ix)
#     ixs = TupleTools.getindices(allixs,inds)
#     xs  = TupleTools.getindices(allxs,inds)
#
#     nallixs =  TupleTools.deleteat(allixs, inds)
#     nallxs  =  TupleTools.deleteat(allxs,  inds)
#
#     a, = xs
#     ia, = ixs
#     i2change =  [i for (i,j) in enumerate(ia) if count(==(j),ia) > 1 && j != e]
#     c = 1
#     for i in i2change
#         ia = TupleTools.insertat(ia, i, (ia[i] + 1000+c,))
#         c += 1
#     end
#     nix = indicesafterop(op, (ia,))
#     nx = tensortrace(a,ia,nix)
#     c = 1
#     for i in 1:length(nix)
#         if nix[i] > 500
#             nix = TupleTools.insertat(nix, i, (nix[i] - 1000-c,))
#             c += 1
#         end
#     end
#     return (nix, nallixs...), (nx, nallxs...)
# end


#Optimiziation

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
    ixs = TupleTools.getindices(allixs,inds)
    sxs = TupleTools.getindices(allsxs,inds)
    nix = indicesafterop(op, ixs)

    nallixs = TupleTools.deleteat(allixs, inds)
    nallsxs = TupleTools.deleteat(allsxs,  inds)

    allinds  = TupleTools.vcat(ixs...)
    allsizes = TupleTools.vcat(sxs...)

    cost += prod(map(ntuple(i -> (i, allinds[i]), length(allinds))) do (k,i)
        ifelse(any(x -> allinds[x] == i, 1:(k-1)), 1, allsizes[k])
    end)
    nsx = map(nix) do i
        j = findfirst(==(i), allinds)::Int
        allsizes[j]
    end
    return (cost, (nix, nallixs...), (nsx, nallsxs...))
end

function indicesafteroperation(op::EinsumOp, allixs)
    e = op.edges
    inds = Tuple(i for (i, ix) in enumerate(allixs) if !isempty(intersect(ix, e)))
    ixs = TupleTools.getindices(allixs,inds)
    nix = indicesafterop(op, ixs)
    nallixs = TupleTools.deleteat(allixs, inds)
    return (nix, nallixs...)
end

@doc raw"
    meinsumcost(ixs, xs, ops)
returns the cost of evaluating the einsum of `ixs`, `xs` according to the
sequence in ops.
"
function meinsumcost(ixs, xs, ops)
    foldl((args, op) -> opcost(op, args...), ops, init = (0, ixs, xs))[1]
end



using Combinatorics

function optimiseorder(ixs, sxs, ops,iy)
    isempty(ops) && return (0, ())
    foldl(permutations(ops), init = (typemax(Int), modifyops(ixs,ops,iy))) do (cost, op1), op2
        op2p = modifyops(ixs,op2,iy)
        ncost = meinsumcost(ixs, sxs, op2p)
        if ncost == cost
            @show  op2p, op1
            @show cost
            length(op2p) < length(op1) ? (ncost, op2p) : (cost, op1)
        else
            ncost < cost ? (ncost, op2p) : (cost, op1)
        end
    end
end
