using CuArrays, CUDAnative, GPUArrays

println("CUDA: YOU FIND ME!")

asarray(x::Number, arr::CuArray) where T = CuArray(fill(x, ()))

"""decide the number of threads and blocks to be launched."""
@inline function cudiv(x::Int)
    max_threads = 256
    num_threads = min(max_threads, x)
    num_blocks = ceil(Int, x/num_threads)
    num_threads, num_blocks
end

"""
loop and accumulate products to y, the GPU version.
## References
    * CUDAnative.jl: https://github.com/JuliaGPU/CUDAnative.jl
"""
function loop!(x_indexers::NTuple{N,Any}, xs::NTuple{N, CuArray{T}}, y_indexer, y::CuArray{T}, outer_ci::CartesianIndices, inner_ci::CartesianIndices) where {N, T}
    X, Y = GPUArrays.thread_blocks_heuristic(length(outer_ci))
    @cuda threads=Y blocks=X loop_kernel(x_indexers, xs, y_indexer, y, outer_ci, inner_ci)
    y
end

function loop_kernel(x_indexers::IT, xs::NTuple{NX, AbstractArray{T}}, y_indexer, y::AbstractArray{T}, outer_ci, inner_ci) where {IT, NX, T}
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    i > length(outer_ci) && return nothing
    @inbounds ind_y = outer_ci[i]
    @inbounds iy = subindex(y_indexer, ind_y)
    for ind_x = inner_ci
        ind_xy = TupleTools.vcat(ind_x.I,ind_y.I)
        y[iy] += map_prod(xs, ind_xy, x_indexers)
    end
    nothing
end

function loop_einsum(code::EinCode{ixs, iy},
                xs::NTuple{N, CuArray{<:Any,M} where M},
                size_dict) where {N,T, ixs, iy}
    TO = mapreduce(eltype, promote_type, xs)
    out = CuArrays.zeros(TO, getindex.(Ref(size_dict), iy))
    loop_einsum!(code, xs, out, size_dict)
end

# unfortunately, TensorOperations does not support CUDA at the moment.
function einsum(::PairWise, code::EinCode{ixs, iy}, xs::NTuple{NT,CuArray{T} where T<:Union{Complex, Real}}, size_dict) where {ixs, iy, NT}
    loop_einsum(code, xs, size_dict)
end
