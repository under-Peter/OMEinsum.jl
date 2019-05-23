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
function einsum(contractions::NTuple{N, NTuple{M, Int} where M},
                tensors::NTuple{N, Array{<:Any,M} where M},
                outinds::NTuple{<:Any,Int}) where N
    tensors, contractions = getdiagonals(tensors, contractions, outinds)
    out = outputtensor(tensors, contractions, outinds)
    einsum!(contractions, tensors, outinds, out)
    return out
end

function outputtensor(tensors, contractions, outinds)
    T = mapreduce(eltype, promote_type, tensors)
    sizes = reduce(TupleTools.vcat,size.(tensors))
    indices = reduce(TupleTools.vcat, contractions)
    outdims = map(x -> sizes[findfirst(==(x), indices)], outinds)
    return Array{T}(undef,outdims...)
end

getdiagonals(tensors, contractions, outinds) =
    (x -> (first.(x), last.(x)))(map((t,c) -> getdiagonal(t,c,outinds), tensors, contractions))

function getdiagonal(t, c, outinds)
    idup = findfirst(i -> count(==(i), c) > 1, c)
    idup === nothing && return (t,c)

    dup = c[idup]
    dinds = findall(==(dup), c)
    oinds = findall(x -> x != dup, c)
    l = length(dinds)
    perm = vcat(oinds, dinds)

    s = [size(t, i) for i in oinds]
    s = vcat(s, prod(x -> size(t,x), dinds))

    nt = reshape(permutedims(t,perm),s...)
    ds = size(t, idup)
    stride = sum(x -> ds^x, 0:(l-1))
    nt = nt[fill(:,length(oinds))..., 1:stride:size(nt)[end]]
    nc = ((x->c[x]).(oinds)..., dup)

    return getdiagonal(nt, nc, outinds)
end

function einsum!(contractions::NTuple{N, NTuple{M, Int} where M},
                tensors::NTuple{N, Array{<:Any,M} where M},
                outinds::NTuple{L,Int},
                out::Array{<:Any,L}) where {N,L}
    allins = reduce(vcat, collect.(contractions))
    uniqueallins = unique(allins)
    ntensors = permuteandreshape.(Ref(uniqueallins), tensors, contractions)

    ds = unique([i for i in setdiff(allins, outinds)])
    ds = map(i -> findfirst(==(i), uniqueallins), ds)

    t = sum(broadcast(*, ntensors...), dims=ds)
    tf = dropdims(t, dims = tuple(ds...))
    if isempty(outinds)
        copyto!(out, tf)
    else
        x = [i for i in uniqueallins if i in outinds]
        p = map(i -> findfirst(==(i),x), outinds)
        permutedims!(out, tf, p)
    end
    return out
end

function permuteandreshape(uniqueallins, t, c)
    x = [i for i in uniqueallins if i in c]
    p = map(i -> findfirst(==(i),x), c)
    rs = map(uniqueallins) do i
            j = findfirst(==(i), c)
            j === nothing && return 1
            return size(t,j)
        end
    return reshape(permutedims(t,p),rs...)
end
