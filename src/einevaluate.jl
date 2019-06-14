@doc raw"
    evaluateall(ixs, xs, ops,iy)
evaluate the einsum specified by 'ixs -> iy' by going through all operations
in `ops` in order applying index-permutations, outer products and expansions
at the end.
"
function evaluateall(ixs, xs, ops, iy)
    _, (x,) = foldl(((ixs, xs), op) -> evaluate(op, ixs, xs), ops, init = (ixs, xs))
    return x
end

@doc raw"
    evaluate(op::EinsumOp, allixs, allxs)
returns a tuple of xs and ixs that result from the evaluation of the
operator `op`.
"
function evaluate(op::EinsumOp, allixs, allxs)
    #generic fallback
    e = op.edges
    inds = Tuple(i for (i, ix) in enumerate(allixs) if !isempty(intersect(e,ix)))
    ixs, nallixs = pickfromtup(allixs, inds)
    xs,  nallxs  = pickfromtup(allxs, inds)

    nix = indicesafterop(op, ixs)
    nx = einsumexp(ixs, xs, nix)

    return (nix, nallixs...), (nx, nallxs...)
end

function evaluate(op::OuterProduct{N}, allixs::NTuple{N,T where T}, allxs) where N
    nix = indicesafterop(op, allixs)
    nxs = map(allxs, allixs) do x, ix
        s = map(nix) do i
            j = findfirst(==(i), ix)
            j === nothing ? 1 : size(x,j)
        end
        reshape(x, s...)
    end
    return (nix,), (broadcast(*, nxs...),)
end

function evaluate(op::Permutation, allixs::NTuple{1,T where T}, allxs)
    (x,)  = allxs
    (ix,) = allixs
    perm = op.perm
    return (TupleTools.permute(ix, perm),), (permutedims(x, perm),)
end

evaluate(op::Fallback{N,T}, allixs, allxs) where {N,T} =
    ((op.iy,), (einsumexp(allixs, allxs, op.iy),))


function evaluate(op::IndexReduction{N}, allixs, allxs) where N
    e = op.edges[1]

    ind = findfirst(x -> e in x, allixs)::Int
    (ix,), nallixs = pickfromtup(allixs, (ind,))
    (x,),  nallxs  = pickfromtup(allxs,  (ind,))

    inds = map(e -> findfirst(==(e), ix), op.edges)
    nix = TupleTools.deleteat(ix, inds)
    nx = dropdims(sum(x, dims = inds), dims = inds)

    return (nallixs..., nix), (nallxs..., nx)
end

function evaluate(op::TensorContract, allixs, allxs)
    e = op.edges
    i1 = findfirst(x -> e[1] in x, allixs)::Int
    i2 = findnext( x -> e[1] in x, allixs, i1+1)::Int
    inds = (i1, i2)

    (ia, ib), nallixs = pickfromtup(allixs, inds)
    (a, b),   nallxs  = pickfromtup(allxs, inds)

    ia, ib, rev = tcdups(ia, ib, e)
    nix = indicesafterop(op, (ia,ib))
    nx = tensorcontract(a,ia,b,ib,nix)
    nix = undotcdups(nix, rev)

    return (nix, nallixs...), (nx, nallxs...)
end

function tcdups(ia::NTuple{N}, ib::NTuple{M}, e) where {N,M}
    # if there are duplicate indices in either ia or ib that are not tensor-contracted,
    # we need to change labels because TensorOperations doesn't allow duplicate output labels
    # the third return value encodes all information needed to undo this deduplication
    imax = max(maximum(ia), maximum(ib))
    ra = ntuple(identity,N)
    rb = ntuple(identity,M)
    ia2 = map(ra, ia) do i, a
        ifelse(count(==(a), ia) > 1 || a in ib && a ∉ e, imax + i, a)
    end
    ib2 = map(rb, ib) do i, a
        ifelse(count(==(a), ib) > 1 || a in ia && a ∉ e, imax + N + i, a)
    end

    ia2, ib2, (TupleTools.vcat(ia,ib), TupleTools.vcat(ia2, ib2))
end

undotcdups(nix, (iaib, ia2ib2)) = map(i -> iaib[findfirst(==(i), ia2ib2)], nix)

function evaluate(op::Trace, allixs, allxs)
    e = op.edges
    inds = Tuple(i for (i, ix) in enumerate(allixs) if !isempty(intersect(e, ix)))
    (ia, ), nallixs = pickfromtup(allixs, inds)
    (a, ),  nallxs  = pickfromtup(allxs, inds)


    ia, rev = tracedups(ia, e)

    nix = indicesafterop(op, (ia,))
    nx = tensortrace(a,ia,nix)

    nix = undotracedups(nix, rev)

    return (nix, nallixs...), (nx, nallxs...)
end

function tracedups(ia::NTuple{N}, e) where N
    # if there are duplicate indices in ia that are not tensor-traced,
    # we need to change labels because TensorOperations doesn't allow duplicate output labels
    # the second return value encodes all information needed to undo this deduplication
    iamax = maximum(ia)
    r = ntuple(identity,N)
    ib = map(r, ia) do i, a
        ifelse(count(==(a), ia) > 1 && a ∉ e, iamax + i, a)
    end
    return ib, (ia,ib)
end

undotracedups(ix, (ia, ib)) = map(i -> ib[findfirst(==(i), ia)], ix)
