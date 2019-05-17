module OMEinsum
export einsum

function einsum(cs, ts)
    allins  = reduce(vcat, collect.(cs))
    outinds = sort(filter(x -> count(==(x), allins) == 1, allins))
    einsum(cs, ts, outinds)
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
function einsum(contractions, tensors, outinds)
    T = mapreduce(eltype, promote_type, tensors)
    l = length(tensors)
    allins = reduce(vcat, collect.(contractions))
    uniqueallins = unique(allins)
    ntensors = permuteandreshape.(Ref(uniqueallins), tensors, contractions)

    ds = unique([i for i in setdiff(allins, outinds)])
    ds = map(i -> findfirst(==(i), uniqueallins), ds)

    t = sum(broadcast(*, ntensors...), dims=ds)
    tf = dropdims(t, dims = tuple(ds...))
    x = [i for i in uniqueallins if i in outinds]
    p = map(i -> findfirst(==(i),x), outinds)
    tff = permutedims(tf,p)
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

end # module
