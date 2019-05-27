using TupleTools
using Base.Cartesian

"""
A naive implementation of `einsum!`
    * `ixs`: input tensor indices,
    * `xs`: input tensors,
    * `iy`: output tensor indices,
    * `y`: accumulated tensor, notice it is initialized to 0 as output!
"""
function naive_einsum!(ixs::Tuple, xs, iy::NTuple{NO, Int}, y) where NO
    all_indices = TupleTools.vcat(ixs..., iy)
    all_sizes = TupleTools.vcat(size.(xs)..., size(y))
    indices = unique(all_indices)
    sizes = Tuple(all_sizes[i] for i in indexin(indices, [all_indices...]))

    ci = CartesianIndices(sizes)
    locs_xs = Tuple(Tuple(findfirst(isequal(i), indices) for i in ix) for ix in ixs)
    locs_y = Tuple(findfirst(isequal(i), indices) for i in iy)

    loop!(locs_xs, xs, locs_y, y, ci)
end

"""loop and accumulate products to y"""
function loop!(locs_xs::NTuple{N}, xs, locs_y, y::AbstractArray{T}, ci::CartesianIndices) where {N, T}
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
@inline @generated function map_prod(::Type{T}, xs::Tuple, ind::CartesianIndex, locs_xs::NTuple{N}) where {N, T}
    quote
        p = one(T)
        @nexprs $N i -> @inbounds p *= xs[i][index_map(ind, locs_xs[i])]
    end
end

using Test
a = randn(3, 3)
@test naive_einsum!(((1,2), (2,3)), (a,a), (1,3), zeros(3,3)) ≈ a*a
@test naive_einsum!(((1,2),), (a,), (), fill(0.0, ())) ≈ fill(sum(a), ())
@test naive_einsum!(((),), (fill(1.0, ()),), (1,3), zeros(3, 3)) ≈ ones(3,3)

a = randn(50, 50)
c = zeros(50, 50)
using BenchmarkTools
@benchmark ein!(((1,2), (2,3)), (a,a), (1,3), c) seconds = 1
@benchmark a*a seconds = 1
