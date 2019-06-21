using TupleTools
using Base.Cartesian
# using CuArrays, CUDAnative

"""
A naive implementation of `einsum!`
    * `ixs`: input tensor indices,
    * `xs`: input tensors,
    * `iy`: output tensor indices,
    * `y`: accumulated tensor, notice it is initialized to 0 as output!
"""
function einsumexp!(ixs::Tuple, xs, iy::NTuple{NO, Int}, y) where NO
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

"""take an index subset from `ind`"""
index_map(ind::CartesianIndex, locs::Tuple) = CartesianIndex(TupleTools.getindices(Tuple(ind), locs))

"""indiex tensors, and return the product of elements"""
@inline @generated function map_prod(::Type{T}, xs::Tuple, ind::CartesianIndex, locs_xs::NTuple{N}) where {N, T}
    quote
        p = one(T)
        @nexprs $N i -> @inbounds p *= xs[i][index_map(ind, locs_xs[i])]
    end
end

"""decide the number of threads and blocks to be launched."""
@inline function cudiv(x::Int)
    max_threads = 256
    threads_x = min(max_threads, x)
    threads_x, ceil(Int, x/threads_x)
end

"""
loop and accumulate products to y, the GPU version.

## References
    * CUDAnative.jl: https://github.com/JuliaGPU/CUDAnative.jl
"""
# function loop!(locs_xs::NTuple{N}, xs::NTuple{N, CuArray}, locs_y, y::CuArray{T}, outer_ci::CartesianIndices, inner_ci::CartesianIndices) where {N, T}
#     function loop_kernel(locs_xs, xs, locs_y, y, outer_ci, inner_ci)
#         i = (blockIdx().x-1) * blockDim().x + threadIdx().x
#         i > length(outer_ci) && return nothing
#         @inbounds ind_y = outer_ci[i]
#         iy = index_map(ind_y, locs_y)
#         # To avoid race condition (https://mc.stanford.edu/cgi-bin/images/3/34/Darve_cme343_cuda_3.pdf),
#         # inner loops (cumulative operations) can not be avoided inside a single CUDA core,
#         # which means, reduction of dimensions can be slow.
#         # to increase the parallism, we should use a different strategy described in
#         # http://people.cs.vt.edu/yongcao/teaching/cs5234/spring2013/slides/Lecture10.pdf
#         for ind_x in inner_ci
#             ind_xy = CartesianIndex(TupleTools.vcat(ind_y.I, ind_x.I))
#             @inbounds y[iy] += map_prod(T, xs, ind_xy, locs_xs)
#         end
#         nothing
#     end
#     X, Y = cudiv(length(outer_ci))
#     @cuda threads=X blocks=Y loop_kernel(locs_xs, xs, locs_y, y, outer_ci, inner_ci)
#     y
# end

"""
loop and accumulate products to y, the GPU version, the CPU version.
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
