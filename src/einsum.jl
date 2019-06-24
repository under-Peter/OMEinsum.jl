using TupleTools, Base.Cartesian

function parseeinsumsstring(s::AbstractString)
    s = replace(s, " " => "")
    m = match(r"([a-z,]+)->([a-z]*)", s)
    m == nothing && throw(ArgumentError("invalid einsum specification $s"))
    sixs, siy = m.captures
    iy  = Tuple(siy)
    ixs = Tuple(Tuple(ix) for ix in split(sixs,','))
    return (ixs, iy)
end

function einsum(s::AbstractString, xs)
    ixs, iy = parseeinsumsstring(s)
    return einsum(ixs, xs, iy)
end

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
    checkargs(ixs, xs, iy)
    ops = opsfrominds(ixs, iy)
    evaluateall(ixs, xs, ops, iy)
end

@doc raw"
    meinsumopt(ixs, xs, iy)
returns the result of the einsum operation implied by `ixs`, `iy` but
evaluated in the optimal order according to `meinsumcost`.
"
function einsumopt(ixs, xs, iy)
    checkargs(ixs, xs, iy)
    ops = optimalorder(ixs, xs, iy)
    evaluateall(ixs, xs, ops, iy)
end

function checkargs(ixs, xs, iy)
    length(ixs) == length(xs) || throw(
        ArgumentError("Number of indices and tensors not the same"))
    foreach(ixs, xs) do ix, x
        length(ix) == ndims(x) || throw(
        ArgumentError("Indices $ix are invalid for a tensor with ndims = $(ndims(x))"))
    end
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
    # outer legs and inner legs
    outer_indices = unique(iy)
    inner_indices = setdiff(TupleTools.vcat(ixs...), outer_indices)

    # find size for each leg
    all_indices = TupleTools.vcat(ixs..., iy)
    all_sizes = TupleTools.vcat(size.(xs)..., size(y))
    outer_sizes = Tuple(all_sizes[i] for i in indexin(outer_indices, [all_indices...]))
    inner_sizes = Tuple(all_sizes[i] for i in indexin(inner_indices, [all_indices...]))

    # cartesian indices for outer and inner legs
    outer_ci = CartesianIndices(outer_sizes)
    inner_ci = CartesianIndices(inner_sizes)

    # for indexing tensors (leg binding)
    indices = (outer_indices..., inner_indices...)
    locs_xs = Tuple(Tuple(findfirst(isequal(i), indices) for i in ix) for ix in ixs)
    locs_y = Tuple(findfirst(isequal(i), outer_indices) for i in iy)

    loop!(locs_xs, xs, locs_y, y, outer_ci, inner_ci)
end

"""indiex tensors, and return the product of elements"""
@inline @generated function map_prod(::Type{T}, xs::Tuple, ind::CartesianIndex, locs_xs::NTuple{N}) where {N, T}
    quote
        p = one(T)
        @nexprs $N i -> @inbounds p *= xs[i][index_map(ind, locs_xs[i])]
    end
end

"""
loop and accumulate products to y, the CPU version.
"""
function loop!(locs_xs::NTuple{N}, xs::NTuple{N, AbstractArray}, locs_y, y::AbstractArray{T}, outer_ci::CartesianIndices, inner_ci::CartesianIndices) where {N, T}
    @simd for i in outer_ci
        @inbounds ind_y = outer_ci[i]
        iy = index_map(ind_y, locs_y)
        for ind_x in inner_ci
            ind_xy = CartesianIndex(TupleTools.vcat(ind_y.I, ind_x.I))
            @inbounds y[iy] += map_prod(T, xs, ind_xy, locs_xs)
        end
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
