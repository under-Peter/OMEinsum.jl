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
function einsum(ixs, xs, iy)
    ops = operatorsfrominds(ixs, iy)
    ops = modifyops(ixs, size.(xs), ops, iy)
    einsumexp(foldl(((ixs,xs), op) -> evaluate(op, ixs,xs), ops, init = (ixs,xs))..., iy)
end

@doc raw"
    meinsumopt(ixs, xs, iy)
returns the result of the einsum operation implied by `ixs`, `iy` but
evaluated in the optimal order according to `meinsumcost`.
"
function einsumopt(ixs, xs, iy)
    ops = operatorsfrominds(ixs, iy)
    ops = optimiseorder(ixs, size.(xs), ops)[2]
    ops = modifyops(ixs, size.(xs), ops, iy)
    @show ops
    res = foldl(((ixs, xs), op) -> evaluate(op, ixs, xs),
                 ops, init = (ixs, xs))
    return einsumexp(res...  ,iy)
end
function einsumexp(contractions::NTuple{N, NTuple{M, T} where M},
                tensors::NTuple{N, AbstractArray{<:Any,M} where M},
                outinds::NTuple{<:Any,T}) where {N,T}
    out = outputtensor(tensors, contractions, outinds)
    einsumexp!(contractions, tensors, outinds, out)
end

function outputtensor(tensors, contractions, outinds)
    T = mapreduce(eltype, promote_type, tensors)
    sizes = TupleTools.vcat(size.(tensors)...)
    indices = TupleTools.vcat(contractions...)
    outdims = map(x -> sizes[findfirst(==(x), indices)], outinds)
    zeros(T,outdims...)
end


function einsumexp!(ixs::NTuple{N, NTuple{M, Int} where M},
                xs::NTuple{N, AbstractArray{<:Any,M} where M},
                iy::NTuple{L,Int},
                y::AbstractArray{T,L}) where {N,L,T}
    all_indices = TupleTools.vcat(ixs..., iy)
    indices = unique(all_indices)
    size_dict = get_size_dict((ixs..., iy), (xs..., y))
    sizes = Tuple(size_dict[i] for i in indices)

    ci = CartesianIndices(sizes)
    locs_xs = map(ixs) do ix
        map(i -> findfirst(==(i), indices)::Int, ix)
    end
    locs_y = map(i -> findfirst(==(i), indices)::Int, iy)

    loop!(locs_xs, xs, locs_y, y, ci)
end

"""loop and accumulate products to y"""
function loop!(locs_xs::NTuple{N, NTuple{M} where M}, xs,
              locs_y, y::AbstractArray{T}, ci::CartesianIndices) where {N, T, S}
    @simd for ind in ci
        @inbounds y[index_map(ind, locs_y)] += prod(map(i -> xs[i][index_map(ind, locs_xs[i])], ntuple(identity,N)))
    end
    y
end


"""take an index subset from `ind`"""
index_map(ind::CartesianIndex, locs::Tuple) = CartesianIndex(TupleTools.getindices(Tuple(ind), locs))

"""get the dictionary of `index=>size`, error if there are conflicts"""
function get_size_dict(ixs, xs)
    nt = length(ixs)
    size_dict = Dict{Int, Int}()
    @inbounds for i = 1:nt
        for (N, leg) in zip(size(xs[i]), ixs[i])
            if haskey(size_dict, leg)
                size_dict[leg] == N || throw(DimensionMismatch("size of index($leg) does not match."))
            else
                size_dict[leg] = N
            end
        end
    end
    return size_dict
end
