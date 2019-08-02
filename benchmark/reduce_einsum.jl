# `]add OMEinsum#master`
# `]add TupleTools`
using OMEinsum
using CuArrays
using CUDAdrv

using CUDAnative, TupleTools
using OMEinsum: index_map, map_prod, einindexer, subindex
using Base.Cartesian
using GPUArrays
import CuArrays: @cuindex
CuArrays.allowscalar(false)

CUDAnative.cudaconvert(A::EinArray) = EinArray(cudaconvert.(A.xs), A.x_indexers, A.y_indexer, A.size, A.ICIS, A.OCIS)
CuArrays.cu(A::EinArray) = EinArray(cu.(A.xs), A.x_indexers, A.y_indexer, A.size, A.ICIS, A.OCIS)

function CuArrays.mapreducedim_kernel_parallel(f, op, R::CuDeviceArray{T}, A,
                             CIS, Rlength, Slength) where {T}
    for Ri_base in 0:(gridDim().x * blockDim().y):(Rlength-1)
        Ri = Ri_base + (blockIdx().x - 1) * blockDim().y + threadIdx().y
        Ri > Rlength && return
        RI = Tuple(CartesianIndices(R)[Ri])
        S = @cuStaticSharedMem(T, 512)
        Si_folded_base = (threadIdx().y - 1) * blockDim().x
        Si_folded = Si_folded_base + threadIdx().x
        # serial reduction of A into S by Slength ÷ xthreads
        for Si_base in 0:blockDim().x:(Slength-1)
            Si = Si_base + threadIdx().x
            Si > Slength && break
            SI = Tuple(CIS[Si])
            AI = ifelse.(size(R) .== 1, SI, RI)
            if Si_base == 0
                S[Si_folded] = f(A[AI...])
            else
                S[Si_folded] = op(S[Si_folded], f(A[AI...]))
            end
        end
        # block-parallel reduction of S to S[1] by xthreads
        CuArrays.reduce_block(view(S, (Si_folded_base + 1):512), op)
        # reduce S[1] into R
        threadIdx().x == 1 && (R[Ri] = op(R[Ri], S[Si_folded]))
    end
    return
end

function Base._mapreducedim!(f, op, R::CuArray{T}, A::EinArray{T}) where {T}
    # the kernel as generated from `f` and `op` can require lots of registers (eg. #160),
    # so we need to be careful about how many threads we launch not to run out of them.
    Rlength = length(R)
    Ssize = ifelse.(size(R) .== 1, size(A), 1)
    Slength = prod(Ssize)
    CIS = CartesianIndices(Ssize)

    parallel_args = (f, op, R, A, CIS, Rlength, Slength)
    # NOTE: why is GC.@preserve ?
    GC.@preserve parallel_args begin
        # NOTE: why not using `@cuda` here?
        parallel_kargs = cudaconvert.(parallel_args)  # CuArray -> DevicePtr
        parallel_tt = Tuple{Core.Typeof.(parallel_kargs)...}
        parallel_kernel = cufunction(CuArrays.mapreducedim_kernel_parallel, parallel_tt)

        # we are limited in how many threads we can launch...
        ## by the kernel
        kernel_threads = CUDAnative.maxthreads(parallel_kernel)
        ## by the device
        dev = CUDAdrv.device()
        block_threads = (x=attribute(dev, CUDAdrv.MAX_BLOCK_DIM_X),
                         y=attribute(dev, CUDAdrv.MAX_BLOCK_DIM_Y),
                         total=attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK))

        # figure out a legal launch configuration
        y_thr = min(nextpow(2, Rlength ÷ 512 + 1), 512, block_threads.y, kernel_threads)
        x_thr = min(512 ÷ y_thr, Slength, block_threads.x,
                    ceil(Int, block_threads.total/y_thr),
                    ceil(Int, kernel_threads/y_thr))
        #@show kernel_threads, y_thr, x_thr, Rlength, Slength
        #@show 512 ÷ y_thr, Slength, block_threads.x,
                    #block_threads.total/y_thr,
                    #kernel_threads/y_thr

        if x_thr >= 8
            blk, thr = (Rlength - 1) ÷ y_thr + 1, (x_thr, y_thr, 1)
            parallel_kernel(parallel_kargs...; threads=thr, blocks=blk)
        else
            # not enough work, fall back to serial reduction
            range = ifelse.(length.(axes(R)) .== 1, axes(A), nothing)
            blk, thr = CuArrays.cudims(R)
            @cuda(blocks=blk, threads=thr, CuArrays.mapreducedim_kernel_serial(f, op, R, A, range))
        end
    end

    return R
end

function CuArrays.mapreducedim_kernel_serial(f, op, R, A::EinArray, range)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    i > length(A.OCIS) && return nothing
    @inbounds ind_y = A.OCIS[i]
    iy = subindex(A.locs_y, ind_y.I)
    #invoke(CuArrays.mapreducedim_kernel_serial, Tuple{Any,Any,Any,Any,Any}, f, op, R, A, range)
    @inbounds for ind_x in A.ICIS
        ind_xy = TupleTools.vcat(ind_x.I,ind_y.I)
        R[iy] = op(R[iy], f(map_prod(A.xs, ind_xy, A.locs_xs)))
    end
    return nothing
end

function OMEinsum.loop_einsum!(code::EinCode{ixs, iy},
                xs::NTuple{N, CuArray{<:Any,M} where M},
                y::CuArray{T,L}, size_dict) where {N,L,T,IT <: Union{AbstractChar,Integer}, ixs, iy}
    Ny = ndims(y)
    A = EinArray(code, xs, size_dict)
    y = reshape(y, fill(1, ndims(A)-Ny)...,size(y)...)
    dropdims(Base._mapreducedim!(x->x, +, y, A), dims=(1:ndims(A)-Ny...,))
end
