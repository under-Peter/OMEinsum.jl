using CuArrays, CUDAnative

println("CUDA: YOU FIND ME!")

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
    X, Y = cudiv(length(outer_ci))
    @cuda threads=X blocks=Y loop_kernel(x_indexers, xs, y_indexer, y, outer_ci, inner_ci)
    y
end

@generated function loop_kernel(x_indexers::IT, xs, y_indexer, y, outer_ci, inner_ci) where IT
    quote
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    i > length(outer_ci) && return nothing
    @inbounds ind_y = outer_ci[i]
    @inbounds iy = subindex(y_indexer, ind_y)
    @inbounds for ind_x in inner_ci
        ind_xy = TupleTools.vcat(ind_x.I,ind_y.I)
        #y[iy] += 1f1#map_prod(xs, ind_xy, x_indexers)
        y[iy] += map_prod(xs, ind_xy, x_indexers)
    end
    nothing
    end
end

"""indiex tensors, and return the product of elements"""
@inline function gmap_prod(::Type{T}, xs::XT, ind, indexers::TT) where {N, T, TT<:NTuple{N,Any}, XT<:Tuple}
    res = one(T)
    #subindex(indexers[1], ind)
    for i=1:N
        #@inbounds res *= xs[i][1]#subindex(indexers[i], ind)]
        @inbounds res*=subindex(indexers[i], ind)
    end
    res
end

function loop_einsum(code::EinCode{ixs, iy},
                xs::NTuple{N, CuArray{<:Any,M} where M},
                size_dict) where {N,T, ixs, iy}
    TO = mapreduce(eltype, promote_type, xs)
    out = CuArrays.zeros(TO, getindex.(Ref(size_dict), iy))
    loop_einsum!(code, xs, out, size_dict)
end
