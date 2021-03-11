using .CUDA

println("OMEinsum: YOU FIND CUDA!")

#include("cudapatch.jl")

asarray(x::Number, arr::CuArray) where T = CuArray(fill(x, ()))

Base.Array(x::Base.ReshapedArray{T,0,<:CuArray}) where T = Array(x.parent)

function get_output_array(xs::NTuple{N, DenseCuArray{<:Any,M} where M}, size; has_repeated_indices=true) where N
    out = CUDA.zeros(promote_type(map(eltype,xs)...), size)
end

CUDA.cudaconvert(A::EinArray{T}) where T = EinArray{T}(cudaconvert.(A.xs), A.x_indexers, A.y_indexer, A.size, A.ICIS, A.OCIS)
CUDA.cu(A::EinArray{T}) where T = EinArray{T}(cu.(A.xs), A.x_indexers, A.y_indexer, A.size, A.ICIS, A.OCIS)

#=
function CUDA.mapreducedim_kernel_serial(f, op, R, A::EinArray, range)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    i > length(A.OCIS) && return nothing
    @inbounds ind_y = A.OCIS[i]
    iy = subindex(A.y_indexer, ind_y.I)
    #invoke(CUDA.mapreducedim_kernel_serial, Tuple{Any,Any,Any,Any,Any}, f, op, R, A, range)
    @inbounds for ind_x in A.ICIS
        ind_xy = TupleTools.vcat(ind_x.I,ind_y.I)
        R[iy] = op(R[iy], f(map_prod(A.xs, ind_xy, A.x_indexers)))
    end
    return nothing
end
=#

for TP in [:Diag, :Repeat, :Duplicate, :DefaultRule]
    @eval function einsum(::$TP, code::EinCode{ixs, iy}, xs::Tuple{<:DenseCuArray}, size_dict) where {ixs, iy}
        loop_einsum(code, xs, size_dict)
    end
end

function loop_einsum!(code::EinCode{ixs, iy},
                xs::NTuple{N, DenseCuArray{<:Any,M} where M},
                y::DenseCuArray{T,L}, size_dict) where {N,L,T, ixs, iy}
    iy_ = tunique(iy)
    NO = length(iy_)
    A = einarray(code, xs, size_dict)
    if NO == length(iy)
        y = reshape(y, fill(1, ndims(A)-NO)...,size(y)...)
        dropdims(Base.mapreducedim!(x->x, +, y, A), dims=(1:ndims(A)-NO...,))
    else
        y_ = CUDA.zeros(T, size(A)[end-NO+1:end]...)
        y_ = reshape(y_, fill(1, ndims(A)-NO)...,size(y_)...)
        raw = Base.mapreducedim!(x->x, +, y_, A)
        out = dropdims(raw, dims=(1:ndims(A)-NO...,))
        expanddims!(EinCode{((iy_...,),), iy}(), out, y)
    end
end

@generated function expandind(::EinCode{ixs,iy}, ind) where {ixs, iy}
    ix = ixs[1]
    ids = map(ii->:(ind[$(findfirst(==(ii), ix))]), iy)
    Expr(:tuple, ids...)
end

function expanddims!(code::EinCode{ixs, iy}, x, y) where {ixs, iy}
    nthreads = 256
    nblocks = cld(prod(size(x)), nthreads)
    ix = tunique(iy)
    CIS = CartesianIndices(x)
    @inline function kernel(code, y, x)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        i > length(x) && return nothing
        @inbounds yi = expandind(code, CIS[i].I)
        @inbounds y[CartesianIndex(yi)] = x[i]
        nothing
    end
    @cuda(blocks=nblocks, threads=nthreads, kernel(code, y, x))
    return y
end

function _batched_gemm(C1::Char, C2::Char, A::DenseCuArray{T1,3}, B::DenseCuArray{T2,3}) where {T1<:Number, T2<:Number}
    CUDA.CUBLAS.gemm_strided_batched(C1, C2, align_eltypes(A,B)...)
end

tensorpermute(A::DenseCuArray, perm) = permutedims(A, perm)
tensorpermute(A::DenseCuArray, perm::Tuple{}) = A

function einsum(::SimpleBinaryRule{(),(), ()}, xs::NTuple{2, DenseCuArray})
    asarray(Array(xs[1])[] * Array(xs[2])[], xs[1])
end

