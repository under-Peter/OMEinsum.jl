using TupleTools
using CUDAnative, GPUArrays, CuArrays

function loop_kn(xs, y, outer_ci, inner_ci) where {IT, NX, T}
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    i > length(outer_ci) && return nothing
    @inbounds iy = outer_ci[i]
    @inbounds for ind_x in inner_ci
        y[iy] += y[CUDAnative.mod1(iy,iy+1)]*11f-1
    end
    nothing
end

function loop_sum(x::CuArray{T, 4}) where T
    out_size = map(i->size(x,i),(1,2,3))
    outer_ci = CartesianIndices(out_size)
    inner_ci = CartesianIndices((size(x,4),))
    X, Y = GPUArrays.thread_blocks_heuristic(length(outer_ci))
    out = CuArrays.cuzeros(T, out_size...)
    xs = (x,)
    @cuda threads=X blocks=Y loop_kn(xs, out, outer_ci, inner_ci)
    out
end

x = randn(100,100,100,100) |> cu
@benchmark CuArrays.@sync loop_sum(x)
