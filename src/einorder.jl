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

#=
# TODO
#
# 1. find optimal order with PlaceHolder op (only includes edge and whether to keep or not -> sufficient)
# 2. when optimal order is found, step through and label the ops correctly
# 3. then evaluate
=#
inds =
    [
        (((1,2),(2,3)),(1,3)), #matmul
        (((1,2),(1,3),(1,4)),(2,3,4)), #star between
        (((1,2),(1,3),(1,4),(3,4)),(2,)), #star&contract
        (((1,2,3),(3,4,5),(5,6,7),(7,8,1)), (2,4,6,8)), #tensor
        (((1,2),), (1,)), #index reduction
        (((1,2,2),), (1,)), #partial trace
        (((1,2,2),), (1,2)), #partial diagonal
        (((1,2,2),(2,3)), (1,3)), #star mixed
    ]

@doc raw"
    edgesfrominds(ixs,iy)
return the edges of the ixs that imply an operation e.g.
in ixs = ((1,2),(2,3)), iy = (1,3), edge 2
requires a tensor contraction
"
function edgesfrominds(ixs,iy)
    allixs = reduce(vcat, collect.(ixs))
    uniqueallixs = unique(allixs)
    # edges are indices that are not trivial
    # indices are trivial if they appear once in the input and once in the output
    pred(x) = !(count(==(x), allixs) == 1 && x in iy)
    Tuple(filter(pred, uniqueallixs))
end

@doc raw"
    EinsumOp
abstract supertype of all einsum operations
"
abstract type EinsumOp end

struct PlaceHolder{T} <: EinsumOp
    edge::T
    keep::Bool
end

struct TensorContract{T} <: EinsumOp
    edge::T
end

struct Trace{T} <: EinsumOp
    edge::T
end

struct StarContract{N,T} <: EinsumOp
    edge::T
end
StarContract{N}(e) where N = StarContract{N,typeof(e)}(e)

struct MixedStarContract{N,T} <: EinsumOp
    edge::T
end
MixedStarContract{N}(e) where N = MixedStarContract{N,typeof(e)}(e)

struct Diag{N,T} <: EinsumOp
    edge::T
end
Diag{N}(e) where N = Diag{N,typeof(e)}(e)

struct MixedDiag{N,T} <: EinsumOp
    edge::T
end
MixedDiag{N}(e) where N = MixedDiag{N,typeof(e)}(e)

struct IndexReduction{T} <: EinsumOp
    edge::T
end

struct Permute <: EinsumOp
end

@doc raw"
    operatorfromedge(edge, ixs, iy)
returns a subtype of `EinsumOp` which specifies
the kind of operation that `edge` implies.
"
function operatorfromedge(edge, ixs, iy)
    #it would be nice if this could be user extendible, maybe traits?
    edge == () &&  error()
    allixs = reduce(vcat, collect.(ixs))
    ce = count(==(edge), allixs)
    ceiniy = count(==(edge), iy)
    ceinixs = count.(==(edge), ixs)
    if ce == 2 && ceiniy == 0
        all(x -> x == 0 || x == 1, ceinixs) && return TensorContract(edge)
        return Trace(edge)
    elseif  ce == 1 && ceiniy == 0
        return IndexReduction(edge)
    elseif ce > 1 && ceiniy == 1
        #diagonal
        any(ceinixs .> 1) && return MixedDiag{count(ceinixs .> 0)}(edge)
        return Diag{ce}(edge)
    elseif ce > 2 && ceiniy == 0
        any(ceinixs .> 1) && return MixedStarContract{ce}(edge)
        return StarContract{ce}(edge)
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
function modifyops(ixs, sxs, ops, iy)
    ops, = foldl(ops, init = (EinsumOp[], ixs, sxs)) do (nops, oixs, osxs), op
        nop = operatorfromedge(op.edge, oixs, iy)
        push!(nops,nop)
        _ , nixs, nsxs = evaluatebutdont(op, 0, oixs, osxs)
        (nops, nixs, nsxs)
    end
    return ops
end


#in reality, outer products between all tensors should be included as ops
function operatorsfrominds(ixs,iy)
    edges = edgesfrominds(ixs,iy)
    map(x -> operatorfromedge(x, ixs, iy), edges)
end

using TupleTools
function indicesafterop(op::EinsumOp, ixs)
    e = op.edge
    Tuple(i for i in TupleTools.vcat(ixs...) if i != e)
end

function indicesafterop(op::Diag, ixs)
    e = op.edge
    (Tuple(i for i in TupleTools.vcat(ixs...) if i != e)...,e)
end

function indicesafterop(op::MixedDiag, ixs)
    e = op.edge
    (Tuple(i for i in TupleTools.vcat(ixs...) if i != e)...,e)
end

@doc raw"
    evaluate(op::EinsumOp, allixs, allxs)
returns a tuple of xs and a tuple of ixs that includes the result
of the operation `op`.
"
function evaluate(op::EinsumOp, allixs, allxs)
    e = op.edge
    println("$op w/ einsum")
    inds = Tuple(i for (i, ix) in enumerate(allixs) if e in ix)
    ixs = TupleTools.getindices(allixs,inds)
    xs  = TupleTools.getindices(allxs,inds)

    nallixs =  TupleTools.deleteat(allixs, inds)
    nallxs  =  TupleTools.deleteat(allxs,  inds)

    nix = indicesafterop(op, ixs)
    nx = einsumexp(ixs, xs, nix)

    return (nix, nallixs...), (nx, nallxs...)
end

# overload evaluate for special types! multiple dispatch for separate functions!
function evaluate(op::TensorContract, allixs, allxs)
    println("$op w/ TensorOperations")
    e = op.edge
    inds = Tuple(i for (i, ix) in enumerate(allixs) if e in ix)
    ixs = TupleTools.getindices(allixs,inds)
    xs  = TupleTools.getindices(allxs,inds)

    nallixs =  TupleTools.deleteat(allixs, inds)
    nallxs  =  TupleTools.deleteat(allxs,  inds)

    a, b = xs
    ia, ib = ixs
    i2change =  [i for (i,j) in enumerate(ia) if j in ib && j != e]
    for i in i2change
        ia = TupleTools.insertat(ia, i, (ia[i] + 1000,))
    end
    nix = indicesafterop(op, (ia,ib))
    nx = tensorcontract(a,ia,b,ib,nix)
    nix = map(x -> x > 500 ? x - 1000 : x, nix)

    return (nix, nallixs...), (nx, nallxs...)
end

function evaluate(op::Trace, allixs, allxs)
    println("Trace w/ TensorOperations")
    e = op.edge
    inds = Tuple(i for (i, ix) in enumerate(allixs) if e in ix)
    ixs = TupleTools.getindices(allixs,inds)
    xs  = TupleTools.getindices(allxs,inds)

    nallixs =  TupleTools.deleteat(allixs, inds)
    nallxs  =  TupleTools.deleteat(allxs,  inds)

    a, = xs
    ia, = ixs
    i2change =  [i for (i,j) in enumerate(ia) if count(==(j),ia) > 1 && j != e]
    c = 1
    for i in i2change
        ia = TupleTools.insertat(ia, i, (ia[i] + 1000+c,))
        c += 1
    end
    nix = indicesafterop(op, (ia,))
    nx = tensortrace(a,ia,nix)
    c = 1
    for i in 1:length(nix)
        if nix[i] > 500
            nix = TupleTools.insertat(nix, i, (nix[i] - 1000-c,))
            c += 1
        end
    end
    return (nix, nallixs...), (nx, nallxs...)
end

@doc raw"
    evaluatebutdont(op, ocost, allixs, allsxs)
returns the cost (in number of iterations it would require in a for loop)
of evaluating `op` with arguments `allixs` and `allsxs` plus `ocost`
as well as the new indices and sizes after evaluation.

`allscs` is a tuple of tuples of Ints - the sizes of the respective arrays

"
function evaluatebutdont(op::EinsumOp, cost, allixs, allsxs::NTuple{M,NTuple{N,Int} where N} where M)
    e = op.edge
    inds = Tuple(i for (i, ix) in enumerate(allixs) if e in ix)
    ixs = TupleTools.getindices(allixs,inds)
    sxs  = TupleTools.getindices(allsxs,inds)
    nix = indicesafterop(op, ixs)

    nallixs =  TupleTools.deleteat(allixs, inds)
    nallsxs  =  TupleTools.deleteat(allsxs,  inds)

    allinds = reduce(vcat, collect.(ixs))
    allsizes = reduce(vcat, collect.(sxs))
    cost += mapreduce(*, unique(allinds)) do i
        j = findfirst(==(i), allinds)
        allsizes[j]
    end
    nsx = map(nix) do i
        j = findfirst(==(i), allinds)
        allsizes[j]
    end
    return (cost, (nix, nallixs...), (nsx, nallsxs...))
end

@doc raw"
    meinsumcost(ixs, xs, ops)
returns the cost of evaluating the einsum of `ixs`, `xs` according to the
sequence in ops.
"
function meinsumcost(ixs, xs, ops)
    foldl((args, op) -> evaluatebutdont(op, args...), ops, init = (0, ixs, xs))[1]
end



using Combinatorics

function optimiseorder(ixs, sxs, ops)
    foldl(permutations(ops), init = (typemax(Int), ops)) do (cost, op1), op2
        ncost = meinsumcost(ixs, sxs, op2)
        ncost < cost ? (ncost, op2) : (cost, op1)
    end
end
