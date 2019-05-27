function einsum(cs, ts)
    allins  = reduce(vcat, collect.(cs))
    outinds = sort(filter(x -> count(==(x), allins) == 1, allins))
    einsum(cs, ts, tuple(outinds...))
end

@doc raw"
    einsum(cs, ts, out)
return the tensor that results from contracting the tensors `ts` according
to their indices `cs`, where twice-appearing indices are contracted.
The result is permuted according to `out`.

- `cs` - tuple of tuple of integers that label all indices of a tensor.
       Indices that appear twice (in different tensors) are summed over

- `ts` - tuple of tensors

- `out` - tuple of integers that should correspond to remaining indices in `cs` after contractions.

This implementation has space requirements that are exponential in the number of unique indices.

# example
```jldoctest; setup = :(using OMEinsum)
julia> a = rand(2,2);

julia> b = rand(2,2);

julia> einsum(((1,2),(2,3)), (a, b), (1,3)) ≈ a * b
true

julia> einsum(((1,2),(2,3)), (a, b), (3,1)) ≈ permutedims(a * b, (2,1))
true
```
"
function einsum(contractions::NTuple{N, NTuple{M, T} where M},
                tensors::NTuple{N, AbstractArray{<:Any,M} where M},
                outinds::NTuple{<:Any,T}) where {N,T}
    out = outputtensor(tensors, contractions, outinds)
    einsum!(contractions, tensors, outinds, out)
    return out
end

function outputtensor(tensors, contractions, outinds)
    T = mapreduce(eltype, promote_type, tensors)
    sizes   = reduce(TupleTools.vcat, size.(tensors))
    indices = reduce(TupleTools.vcat, contractions)
    outdims = map(x -> sizes[findfirst(==(x), indices)], outinds)
    return zeros(T,outdims...)
end

@doc raw"
    diagonals(tensors, contractions)

for each tensor in `tensors`, check whether there are
any duplicates in `contractions`.
If there are duplicates, take the diagonal w.r.t to the duplicates
and return these diagonals and the new contractions.
# Example
```jldoctest; setup = :(using OMEinsum)
julia> OMEinsum.diagonals(([1 2; 3 4],), ((1,1),))
(([1, 4],), ((1,),))
```
"
function diagonals(ts, cs)
    tcs = map(ts, cs) do t,c
        diagonal(t,c)
    end
    nts = getindex.(tcs,1)
    ncs = getindex.(tcs,2)
    return nts, ncs
end

@doc raw"
    diagonal(tensor::AbstractArray{<:Any,N}, inds::NTuple{N})

if there are any duplicate labels in `inds`, take those elements
of `tensor` for which the indices for the duplicate labels are
the same, i.e. if `ind = (1,2,2)`, make a new tensor
from the slices `tensor[i,j,j]`.
If multiple duplicate labels are in `inds` recursively calls itself
until none are left.

# Example
```jldoctest; setup = :(using OMEinsum)
julia> OMEinsum.diagonal([1 2; 3 4], (1,1))
([1, 4], (1,))
```
"
function diagonal(t::AbstractArray{<:Any,N}, c::NTuple{N}) where N
    idup = findfirst(i -> count(==(i), c) > 1, c)
    idup === nothing && return (t,c)

    dup = c[idup]
    dinds = findall(==(dup), c)
    oinds = findall(x -> x != dup, c)
    l = length(dinds)
    perm = vcat(oinds, dinds)

    s = vcat([size(t,i) for i in oinds], prod(x -> size(t,x), dinds))

    nt = reshape(permutedims(t,perm),s...)
    ds = size(t, idup)
    stride = sum(x -> ds^x, 0:(l-1))
    nt = nt[fill(:,length(oinds))..., 1:stride:size(nt)[end]]
    nc = ((x->c[x]).(oinds)..., dup)

    return diagonal(nt, nc)
end


function einsum!(cons::NTuple{N, NTuple{M,T} where M},
                tens::NTuple{N, AbstractArray{<:Any,M} where M},
                oinds::NTuple{L,T},
                out::AbstractArray{<:Any,L}) where {N,L,T}

    # reduce duplicate indices within tensors to diagonals
    tens, cons = diagonals(tens, cons)

    # combine and contract
    oindspre = tuple(unique(oinds)...) ∩ vcat(collect.(cons)...)
    outpre = outputtensor(tens, cons, oindspre)
    contractcombine!(outpre, oindspre, oinds, cons, tens)

    # expand duplicate indices in output
    expandall!(out, oinds, outpre, oindspre)
    return out
end

@doc raw"
    permuteandreshape(uniqueallins, tensor, inds)
permute `tensor` such that its indices in `inds` are in the same
order as in `uniqueallins`.
Then reshape the permuted `tensor` such that the indices of
the resulting tensors for indices in `inds` is conserved
while singleton-dimensions are inserted for indices in
`uniqueallins` that are not in `inds`.

# Example
Here, the array `[1 2;3 4]` is permuted according to the indices in `uniqueallins`,
where label `1` comes before label `4` and then reshape it to have a shape that
could be indexed with four labels.

```jldoctest; setup = :(using OMEinsum)
julia> OMEinsum.permuteandreshape((1,2,3,4), [1 2; 3 4], (4,1))
2×1×1×2 Array{Int64,4}:
[:, :, 1, 1] =
 1
 2

[:, :, 1, 2] =
 3
 4
```
"
function permuteandreshape(uniqueallins, t, c)
    x = [i for i in uniqueallins if i in c]
    p = map(i -> findfirst(==(i),x), c)
    rs = map(uniqueallins) do i
            j = findfirst(==(i), c)
            j === nothing && return 1
            return size(t,j)
        end
    if isempty(rs)
        return t
    elseif isempty(p)
        return reshape(t,rs...)
    else
        return reshape(permutedims(t,p),rs...)
    end
end

@doc raw"
    contractcombine!(out, outindspre, outinds, contractions, tensors)
take the tensorproduct of all tensors in `tensors` according to the
specification in `contractions` and save the result in `out`.
"
function contractcombine!(outpre, outindspre, outinds, contractions, tensors)
    allins = reduce(vcat, collect.(contractions))
    uniqueallins = unique(allins)
    ntensors = permuteandreshape.(Ref(uniqueallins), tensors, contractions)

    ds = unique([i for i in setdiff(allins, outindspre)])
    ds = map(i -> findfirst(==(i), uniqueallins), ds)

    if isempty(ds)
        tf = broadcast(*, ntensors...)
        if isempty(outindspre ∩ allins)
            copyto!(outpre, tf)
        else
            p = map(i -> findfirst(==(i),uniqueallins), outindspre)
            copyto!(outpre, permutedims(tf,p))
        end
    else
        t = sum(broadcast(*, ntensors...), dims=ds)
        tf = dropdims(t, dims = tuple(ds...))
        if isempty(outinds)
            copyto!(outpre, tf)
        else
            x = [i for i in uniqueallins if i in outinds]
            p = map(i -> findfirst(==(i),x), outinds)
            permutedims!(outpre, tf, p)
        end
    end
    return outpre
end

function deltastride(ns)
    stride, dt = 0, 1
    for n in ns
        stride += dt
        dt *= n
    end
    return stride
end

function densedelta(::Type{T}, ns::Vararg{Int,N}) where {T,N}
    id = zeros(T,ns...)
    o = one(T)
    stride = deltastride(ns)
    for i in 1:stride:length(id)
        @inbounds id[i] = o
    end
    return id
end

using LazyArrays
_lazydelta(::Type{T}, c::CartesianIndex{N}) where {T,N} = ifelse(all(==(c.I[1]), c.I), one(T), zero(T))
lazydelta(::Type{T}, ns...) where T = @~ _lazydelta.(T,CartesianIndices(ns))

@doc raw"
    expandall!(b::AbstractArray{T,N}, indsb::NTuple{<:Any,N},
               a::AbstractArray{T,M}, indsa::NTuple{<:Any,M})
expands `a` into `b` by finding all index-labels in `indsa` in `indsb`
and in case an label appears more in `indsb` than in `indsa`,
the corresponding index is expanded.
This enables operations such as
```jldoctest; setup = :(using OMEinsum)
julia> OMEinsum.expandall!(zeros(2,2), (1,1), reshape([2]), ())
2×2 Array{Float64,2}:
 2.0  0.0
 0.0  2.0
```
where the zero-dimensional array `reshape([2])` is expanded onto
the diagonal of `zeros(2,2)` as in the backward action of taking
the diagonal.
Expansions are done by contracting a dirac-delta function with
the indices to expand.
"
function expandall!(b, indsb, a, indsa)
    inds2expand = [i for i in unique(indsb) if count(==(i), indsb) > count(==(i), indsa)]
    sizes = [size(b,findfirst(==(i),indsb)) for i in inds2expand]
    ns = [count(==(i), indsb) for i in inds2expand]
    #construct dirac deltas - one for each index that needs expansion
    deltas = [lazydelta(eltype(b), fill(s,n)...) for (s,n) in zip(sizes,ns)]
    indsainb = map(i -> findfirst(==(i), indsa), indsb)
    perm = unique!([i for i in indsainb if i != nothing])

    ap = isempty(perm) ? a : permutedims(a,perm)
    sa = []
    for (j,i) in enumerate(indsainb)
        if i == nothing
            push!(sa, 1)
        elseif !(i in indsainb[1:(j-1)])
            push!(sa, size(a,i))
        end
    end
    rap = reshape(ap, sa...)
    nids = []
    for i in 1:length(inds2expand)
        sb = fill(1, ndims(b))
        inds = findall(==(inds2expand[i]), indsb)
        sb[inds] .= sizes[i]
        push!(nids,reshape(deltas[i], sb...))
    end
    broadcast!(*, b, rap, nids...)
    return b
end
