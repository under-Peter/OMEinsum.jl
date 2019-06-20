using TupleTools, Base.Cartesian

function outindsfrominput(ixs)
    allixs = vcat(collect.(ixs)...)
    iy = sort!(filter!(x -> count(==(x), TupleTools.vcat(ixs...)) == 1, allixs))
    return tuple(iy...)
end

einsum(ixs, xs) = einsum(ixs, xs, outindsfrominput(ixs))

@doc raw"
    einsum(ixs, xs, out)
return the tensor that results from contracting the tensors `xs` according
to their indices `ixs`, where all indices that do not appear in the output are
summed over. The indices are contracted in the order implied by their numerical value,
smaller first.
The result is permuted according to `out`.

- `ixs` - tuple of tuple of integers that label all indices of a tensor.
       Indices that appear twice (in different tensors) are summed over

- `xs` - tuple of tensors

- `out` - tuple of integers that should correspond to remaining indices in `ixs` after contractions.


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
    ops = operatorsfromindices(ixs, iy)
    evaluateall(ixs, xs, ops, iy)
end

einsumopt(ixs, xs) = einsumopt(ixs, xs, outindsfrominput(ixs))
@doc raw"
    meinsumopt(ixs, xs, iy)
returns the result of the einsum operation implied by `ixs`, `iy` but
evaluated in the optimal order according to `meinsumcost`.
"
function einsumopt(ixs, xs, iy)
    ops = optimalorder(ixs, xs, iy)
    evaluateall(ixs, xs, ops, iy)
end


function einsumexp(ixs::NTuple{N, NTuple{M, T} where M},
                xs::NTuple{N, AbstractArray{<:Any,M} where M},
                iy::NTuple{<:Any,T}) where {N,T}
    out = outputtensor(xs, ixs, iy)
    einsumexp!(ixs, xs, iy, out)
end

function outputtensor(xs, ixs, iy)
    T = mapreduce(eltype, promote_type, xs)
    sizes = TupleTools.vcat(size.(xs)...)
    indices = TupleTools.vcat(ixs...)
    outdims = map(x -> sizes[findfirst(==(x), indices)], iy)
    zeros(T, outdims...)
end


function einsumexp!(ixs::NTuple{N, NTuple{M, IT} where M},
                xs::NTuple{N, AbstractArray{<:Any,M} where M},
                iy::NTuple{L,IT},
                y::AbstractArray{T,L}) where {N,L,T,IT <: Union{AbstractChar,Integer}}
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
function get_size_dict(ixs::NTuple{N, NTuple{M, T} where M} where N, xs) where T
    nt = length(ixs)
    size_dict = Dict{T,Int}()
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
