using .CUDA

asarray(x, arr::CuArray) where T = CuArray(fill(x, ()))
asarray(x::AbstractArray, y::CuArray) = x
asscalar(x::DenseCuArray) = Array(x)[]

Base.Array(x::Base.ReshapedArray{T,0,<:CuArray}) where T = Array(x.parent)

function get_output_array(xs::NTuple{N, DenseCuArray{<:Any,M} where M}, size; has_repeated_indices=true) where N
    CUDA.zeros(promote_type(map(eltype,xs)...), size...)
end

CUDA.cudaconvert(A::EinArray{T}) where T = EinArray{T}(cudaconvert.(A.xs), A.x_indexers, A.y_indexer, A.size, A.ICIS, A.OCIS)
CUDA.cu(A::EinArray{T}) where T = EinArray{T}(cu.(A.xs), A.x_indexers, A.y_indexer, A.size, A.ICIS, A.OCIS)

for TP in [:Diag, :Repeat, :Duplicate, :DefaultRule]
    @eval function einsum(::$TP, ixs, iy, xs::Tuple{<:DenseCuArray}, size_dict::Dict{LT}) where LT
        @debug "cueinsum fallback to loop_einsum" rule ixs => iy size.(xs)
        loop_einsum(EinCode(ixs, iy), xs, size_dict)
    end
end

function einsum(::SimpleBinaryRule{('j',), ('j',), ()}, xs::NTuple{2, DenseCuArray})
    dropdims(reshape(xs[1],1,:) * xs[2]; dims=1)
end

function loop_einsum!(code::EinCode,
                xs::NTuple{N, DenseCuArray{<:Any,M} where M},
                y::DenseCuArray{T,L}, size_dict::Dict{LT}) where {N,L,T, LT}
    iy = (getiy(code)...,)
    ixs = (Tuple.(getixs(code))...,)
    iy_ = _unique(LT,iy)
    NO = length(iy_)
    A = einarray(Val(ixs), Val(iy), xs, size_dict)
    if NO == length(iy)
        y = reshape(y, fill(1, ndims(A)-NO)...,size(y)...)
        raw = Base.mapreducedim!(x->x, +, y, A)
        if ndims(A)-NO > 0  # fix 1.7 compatibility
            raw = dropdims(raw, dims=(1:ndims(A)-NO...,))
        end
        return raw
    else
        y_ = CUDA.zeros(T, size(A)[end-NO+1:end]...)
        y_ = reshape(y_, fill(1, ndims(A)-NO)...,size(y_)...)
        raw = Base.mapreducedim!(x->x, +, y_, A)
        if ndims(A)-NO > 0  # fix 1.7 compatibility
            raw = dropdims(raw, dims=(1:ndims(A)-NO...,))
        end
        return expanddims!(Val{((iy_...,),)}(), Val{iy}(), raw, y)
    end
end

@generated function expandind(::Val{ixs}, ::Val{iy}, ind) where {ixs, iy}
    ix = ixs[1]
    ids = map(ii->:(ind[$(findfirst(==(ii), ix))]), iy)
    Expr(:tuple, ids...)
end

function expanddims!(::Val{ixs}, ::Val{iy}, x, y) where {LT,ixs,iy}
    nthreads = 256
    nblocks = cld(prod(size(x)), nthreads)
    CIS = CartesianIndices(x)
    @inline function kernel(y, x)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        i > length(x) && return nothing
        @inbounds yi = expandind(Val{ixs}(), Val{iy}(), CIS[i].I)
        @inbounds y[CartesianIndex(yi)] = x[i]
        nothing
    end
    @cuda(blocks=nblocks, threads=nthreads, kernel(y, x))
    return y
end

function _batched_gemm(C1::Char, C2::Char, A::DenseCuArray{T1,3}, B::DenseCuArray{T2,3}) where {T1<:CuBlasFloat, T2<:CuBlasFloat}
    CUDA.CUBLAS.gemm_strided_batched(C1, C2, align_eltypes(A,B)...)
end

function einsum(::SimpleBinaryRule{(),(), ()}, xs::NTuple{2, DenseCuArray})
    asarray(Array(xs[1])[] * Array(xs[2])[], xs[1])
end

Base.ndims(::Base.Broadcast.Broadcasted{CUDA.CuArrayStyle{0}}) = 0

#=
using .CUDA: @cartesianidx, AbstractGPUArray, gpu_call, @linearidx

@inline @generated function permute_linearindex(size::NTuple{N,T}, l::Integer, strides::NTuple{N,T}) where {N,T}
    quote
        l -= one(T)
        res = one(T)
        @nexprs $(N-1) i->begin
            @inbounds l, s = divrem(l, size[i])
            @inbounds res += s * strides[i]
        end
        return @inbounds res + strides[N] * l
    end
end
function LinearAlgebra.permutedims!(dest::AbstractGPUArray, src::AbstractGPUArray,
                                    perm::NTuple{N}) where N
    Base.checkdims_perm(dest, src, perm)
    dest_strides = ntuple(k->k==1 ? 1 : prod(i->size(dest, i), 1:k-1), N)
    dest_strides_perm = ntuple(i->dest_strides[findfirst(==(i), perm)], N)
    LEN = length(src)
    function permutedims_kernel(dest, src, dest_strides_perm, LEN)
        LI = (blockIdx().x-1) * blockDim().x + threadIdx().x
        LI > LEN && return
        dest_index = permute_linearindex(size(src), LI, dest_strides_perm)
        @inbounds dest[dest_index] = src[LI]
        return
    end
    NTHREADS = 256
    @cuda threads=NTHREADS blocks=ceil(Int, length(dest)/NTHREADS) permutedims_kernel(dest, src, dest_strides_perm, LEN)
    return dest
end
=#

@info("OMEinsum loaded the CUDA module successfully")
