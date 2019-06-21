@doc raw"
    evaluateall(ixs, xs, ops,iy)
evaluate the einsum specified by 'ixs -> iy' by going through all operations
in `ops` in order.
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
    nix = indicesafterop(op, ix)
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

    (ia,ib), rev = dedup((ia,ib), op)
    nix = indicesafterop(op, (ia,ib))
    nx  = tensorcontract(a,ia,b,ib,nix)
    nix = redup(nix, rev)

    return (nix, nallixs...), (nx, nallxs...)
end

function evaluate(op::Trace, allixs, allxs)
    e = op.edges
    inds = Tuple(i for (i, ix) in enumerate(allixs) if !isempty(intersect(e, ix)))
    (ia, ), nallixs = pickfromtup(allixs, inds)
    (a, ),  nallxs  = pickfromtup(allxs, inds)

    (ia,), rev = dedup((ia,), op)

    nix = indicesafterop(op, (ia,))
    nx = tensortrace(a,ia,nix)

    nix = redup(nix, rev)

    return (nix, nallixs...), (nx, nallxs...)
end

function evaluate(op::Diag, allixs, allxs)
    e = op.edges
    inds = Tuple(i for (i, ix) in enumerate(allixs) if !isempty(intersect(e,ix)))
    ixs, nallixs = pickfromtup(allixs, inds)
    xs,  nallxs  = pickfromtup(allxs, inds)

    ixs, rev = dedup(ixs, op)
    nix = indicesafterop(op, ixs)

    nxs = map(ixs, xs) do ix, x
        permuteandreshape(nix, x, ix)
    end
    nx = broadcast(*, nxs...)
    nix = redup(nix, rev)

    return (nix, nallixs...), (nx, nallxs...)
end

function evaluate(op::StarContract, allixs, allxs)
    # evaluate star-contraction as a Diagonal followed by an index-reduction
    opd = Diag(op.edges)
    opr = IndexReduction(op.edges)
    evaluate(opr, evaluate(opd, allixs, allxs)...)
end

@doc raw"
    dedup(ixs, op)
changes all duplicate indices in ixs that are not directly acted on by `op`.
This is needed when inputs are evaluated by functions that would otherwise
error or evaluate not just the `op`, e.g. in `iijj -> j`,
`tensortrace` is called for the `Trace(i)` but will try to evaluate a trace
over `j` althought that's not a trace.

`dedup` returns the new, renamed indices as well as an argument to `redup`,
a function to undo the index renaming.
"
function dedup(ixs, op)
    allixs = TupleTools.vcat(ixs...)
    edges = op.edges

    imax = maximum(allixs)
    offsets = cumsum((0, length.(ixs[1:end-1])...)) .+ imax

    # associate with each element of ixs a unique (in the indexes) candidate replacement
    iys = map(ixs, offsets) do ix, offset
        ntuple(identity, length(ix)) .+ offset
    end

    nixs = map(ixs, iys) do ix, iy
        map(ix, iy) do i,j
            i in edges && return i
            #pick candidate replacement if index is duplicate and not in edges
            (count(==(i), ix) > 1 || count(z -> i in z, ixs) > 1) ? j : i
        end
    end

    return nixs, (TupleTools.vcat(iys...), TupleTools.vcat(ixs...))
end

@doc raw"
    redup(ix, rev)
undoes the index-renaming of `nixs, rev = dedup(ixs, op)`.
"
function redup(ix, (alliys, allixs))
    map(ix) do i
        j = findfirst(==(i), alliys)
        j === nothing ? i : allixs[j]
    end
end

@doc raw"
    permuteandreshape(nix, x, ix)
reshape and permute the tensor `x` such that its indices `ix`
transform into indices `nix` where singleton-dimensions are
introduced for missing indices (i.e. in nix but not ix).
"
function permuteandreshape(nix, x, ix)
    nixinix = filter!(in(ix), collect(nix))
    p = map(i -> findfirst(==(i), nixinix), ix)
    rs = map(nix) do i
            j = findfirst(==(i), ix)
            j === nothing ? 1 : size(x,j)
        end
    if isempty(rs)
        return x
    elseif isempty(p)
        return reshape(x,rs...)
    else
        return reshape(permutedims(x,p),rs...)
    end
end
