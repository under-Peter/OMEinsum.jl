using .CUDA

println("OMEinsum: YOU FIND CUDA!")

asarray(x, arr::CuArray) where T = CuArray(fill(x, ()))
asarray(x::AbstractArray, y::CuArray) = x

Base.Array(x::Base.ReshapedArray{T,0,<:CuArray}) where T = Array(x.parent)

function get_output_array(xs::NTuple{N, DenseCuArray{<:Any,M} where M}, size; has_repeated_indices=true) where N
    out = CUDA.zeros(promote_type(map(eltype,xs)...), size)
end

CUDA.cudaconvert(A::EinArray{T}) where T = EinArray{T}(cudaconvert.(A.xs), A.x_indexers, A.y_indexer, A.size, A.ICIS, A.OCIS)
CUDA.cu(A::EinArray{T}) where T = EinArray{T}(cu.(A.xs), A.x_indexers, A.y_indexer, A.size, A.ICIS, A.OCIS)

for TP in [:Diag, :Repeat, :Duplicate, :DefaultRule]
    @eval function einsum(::$TP, ix, iy, x::DenseCuArray, size_dict::Dict{LT}) where LT
        loop_einsum(EinCode{((ix...,),),(iy...,)}(), (x,), size_dict)
    end
end

function loop_einsum!(code::EinCode{ixs, iy},
                xs::NTuple{N, DenseCuArray{<:Any,M} where M},
                y::DenseCuArray{T,L}, size_dict::Dict{LT}) where {N,L,T, ixs, iy, LT}
    iy_ = _unique(LT,iy)
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

function expanddims!(code::EinCode{ixs, iy}, x, y) where {LT,ixs, iy}
    nthreads = 256
    nblocks = cld(prod(size(x)), nthreads)
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

function _batched_gemm(C1::Char, C2::Char, A::DenseCuArray{T1,3}, B::DenseCuArray{T2,3}) where {T1<:CuBlasFloat, T2<:CuBlasFloat}
    CUDA.CUBLAS.gemm_strided_batched(C1, C2, align_eltypes(A,B)...)
end

tensorpermute(A::DenseCuArray, perm) = permutedims(A, perm)
tensorpermute(A::DenseCuArray, perm::Tuple{}) = A

function einsum(::SimpleBinaryRule{(),(), ()}, xs::NTuple{2, DenseCuArray})
    asarray(Array(xs[1])[] * Array(xs[2])[], xs[1])
end

