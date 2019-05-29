using TupleTools, Base.Cartesian

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
    T = mapreduce(eltype, promote_type, tensors)
    sizes = reduce(TupleTools.vcat,size.(tensors))
    indices = reduce(TupleTools.vcat, contractions)
    outdims = map(x -> sizes[findfirst(==(x), indices)], outinds)
    out = Array{T}(undef,outdims...)

    einsum!(contractions, tensors, outinds, out)
    return out
end


function einsum!(ixs::NTuple{N, NTuple{M, Int} where M},
                xs::NTuple{N, Array{<:Any,M} where M},
                iy::NTuple{L,Int},
                y::Array{T,L}) where {N,L,T}
    foreach(i -> y[i] = zero(T), eachindex(y))
    all_indices = TupleTools.vcat(ixs..., iy)
    all_sizes = TupleTools.vcat(size.(xs)..., size(y))
    indices = unique(all_indices)
    sizes = Tuple(all_sizes[i] for i in indexin(indices, collect(all_indices)))

    ci = CartesianIndices(sizes)
    locs_xs = map(ixs) do ix
        map(i -> findfirst(==(i), indices)::Int, ix)
    end
    locs_y = map(i -> findfirst(==(i), indices)::Int, iy)
    loop!(locs_xs, xs, locs_y, y, ci)
end

"""loop and accumulate products to y"""
function loop!(locs_xs::NTuple{N, NTuple{M} where M}, xs, locs_y, y::AbstractArray{T}, ci::CartesianIndices) where {N, T, S}
    @simd for ind in ci
        # equivalent to doing `mapreduce`, but `mapreduce` is much slower due to the allocation
        # @inbounds y[index_map(ind, locs_y)] += mapreduce(i -> mygetindex(xs[i], ind, locs_xs[i]), *, 1:N)
        @inbounds y[index_map(ind, locs_y)] += map_prod(T, xs, ind, locs_xs)
    end
    y
end

"""take an index subset from `ind`"""
index_map(ind::CartesianIndex, locs::Tuple) = CartesianIndex(TupleTools.getindices(Tuple(ind), locs))

"""indiex tensors, and return the product of elements"""
@inline @generated function map_prod(::Type{T}, xs::Tuple,
        ind::CartesianIndex, locs_xs::NTuple{N, NTuple{M} where M}) where {N, T}
    quote
        p = one(T)
        @nexprs $N i -> @inbounds p *= xs[i][index_map(ind, locs_xs[i])]
    end
end
