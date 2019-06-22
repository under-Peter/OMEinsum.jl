using CuArrays, CUDAnative

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
function loop!(locs_xs::NTuple{N}, xs::NTuple{N, CuArray}, locs_y, y::CuArray{T}, outer_ci::CartesianIndices, inner_ci::CartesianIndices) where {N, T}
    function loop_kernel(locs_xs, xs, locs_y, y, outer_ci, inner_ci)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        i > length(outer_ci) && return nothing
        @inbounds ind_y = outer_ci[i]
        iy = index_map(ind_y, locs_y)
        # To avoid race condition (https://mc.stanford.edu/cgi-bin/images/3/34/Darve_cme343_cuda_3.pdf),
        # inner loops (cumulative operations) can not be avoided inside a single CUDA core,
        # which means, reduction of dimensions can be slow.
        # to increase the parallism, we should use a different strategy described in
        # http://people.cs.vt.edu/yongcao/teaching/cs5234/spring2013/slides/Lecture10.pdf
        for ind_x in inner_ci
            ind_xy = CartesianIndex(TupleTools.vcat(ind_y.I, ind_x.I))
            @inbounds y[iy] += map_prod(T, xs, ind_xy, locs_xs)
        end
        nothing
    end
    X, Y = cudiv(length(outer_ci))
    @cuda threads=X blocks=Y loop_kernel(locs_xs, xs, locs_y, y, outer_ci, inner_ci)
    y
end
