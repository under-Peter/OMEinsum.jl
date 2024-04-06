module AMDGPUExt

import OMEinsum: asarray, get_output_array, einsum, loop_einsum!, _batched_gemm!, asscalar, @flatten_addmul!
using OMEinsum: EinArray, Diag, Repeat, Duplicate, DefaultRule, EinCode, DynamicEinCode, StaticEinCode, NestedEinsum, SimpleBinaryRule, match_rule, loop_einsum, getiy, getixs, _unique, einarray, align_eltypes, siblings, isleaf, tensorindex, _safe_set, rootcode
import OMEinsum
using LinearAlgebra
import LinearAlgebra: BlasFloat
using AMDGPU

const ROCBlasFloat = Union{BlasFloat,Float16,ComplexF16}
const ROCArrayTypes{T,N} = Union{LinearAlgebra.Transpose{T,<:ROCArray{T,N}},DenseROCArray{T,N},LinearAlgebra.Adjoint{T,<:ROCArray{T,N}}}
_unwrap(x::LinearAlgebra.Adjoint{T,<:ROCArray{T}}) where {T} = ROCArray(x)
_unwrap(x::LinearAlgebra.Transpose{T,<:ROCArray{T}}) where {T} = ROCArray(x)
_unwrap(x::ROCArray) = x

asarray(x, arr::ROCArray) = ROCArray(fill(x, ()))
asarray(x::AbstractArray, y::ROCArray) = x
asscalar(x::ROCArrayTypes) = Array(x)[]

# to avoid returning a ReshapedArray
OMEinsum.safe_reshape(x::ROCArray, sz) = reshape(x, (sz...,))
OMEinsum.safe_reshape(x::Adjoint{T,<:ROCArray{T}} where {T}, sz) = reshape(ROCArray(x), (sz...,))
OMEinsum.safe_reshape(x::Transpose{T,<:ROCArray{T}} where {T}, sz) = reshape(ROCArray(x), (sz...,))

Base.Array(x::Base.ReshapedArray{T,0,<:ROCArray}) where {T} = Array(x.parent)

function get_output_array(xs::NTuple{N,ROCArrayTypes{<:Any,M} where M}, size; fillzero=true) where {N}
    AMDGPU.zeros(promote_type(map(eltype, xs)...), size...)
end
function get_output_array(xs::NTuple{N,ROCArrayTypes{T,M} where M}, size; fillzero=true) where {T,N}
    AMDGPU.zeros(T, size...)
end


AMDGPU.rocconvert(A::EinArray{T}) where {T} = EinArray{T}(rocconvert.(A.xs), A.x_indexers, A.y_indexer, A.size, A.ICIS, A.OCIS)
AMDGPU.roc(A::EinArray{T}) where {T} = EinArray{T}(cu.(A.xs), A.x_indexers, A.y_indexer, A.size, A.ICIS, A.OCIS)

for TP in [:Diag, :Repeat, :Duplicate]
    @eval function OMEinsum.unary_einsum!(::$TP, ix, iy, x::ROCArrayTypes, y::ROCArrayTypes, sx, sy)
        @debug "cueinsum fallback to loop_einsum" rule ix => iy size(x)
        size_dict = OMEinsum.get_size_dict((ix, iy), (x, y))
        loop_einsum!((ix,), iy, (x,), y, sx, sy, size_dict)
    end
end

function loop_einsum!(ixs0, iy0,
    xs::NTuple{N,ROCArrayTypes{<:Any,M} where M},
    y::ROCArrayTypes{T,L}, sx, sy, size_dict::Dict{LT}) where {N,L,T,LT}
    iy = (iy0...,)
    ixs = (Tuple.(ixs0)...,)
    iy_ = _unique(LT, iy)
    NO = length(iy_)
    A = einarray(Val(ixs), Val((iy_...,)), xs, size_dict)
    if NO == length(iy)
        raw_ = similar(y, (fill(1, ndims(A) - NO)..., size(y)...,))
        fill!(raw_, zero(T))
        Base.mapreducedim!(x -> x, +, raw_, A)
        if ndims(A) - NO > 0  # fix 1.7 compatibility
            raw = dropdims(raw_, dims=(1:ndims(A)-NO...,))
        else
            raw = raw_
        end
        return @flatten_addmul! sy * y + sx * raw
    else
        if iszero(sy)
            fill!(y, zero(T))
        else
            lmul!(sy, y)
        end
        y_ = similar(y, (fill(1, ndims(A) - NO)..., [size_dict[l] for l in iy_]...))
        fill!(y_, zero(T))
        raw = Base.mapreducedim!(x -> x, +, y_, A)
        if ndims(A) - NO > 0  # fix 1.7 compatibility
            raw = dropdims(raw, dims=(1:ndims(A)-NO...,))
        end
        return expanddims!(Val{((iy_...,),)}(), Val{iy}(), raw, y, sx)
    end
end

@generated function expandind(::Val{ixs}, ::Val{iy}, ind) where {ixs,iy}
    ix = ixs[1]
    ids = map(ii -> :(ind[$(findfirst(==(ii), ix))]), iy)
    Expr(:tuple, ids...)
end

function expanddims!(::Val{ixs}, ::Val{iy}, x, y, sx) where {ixs,iy}
    ngroupsize = 256
    ngridsize = cld(prod(size(x)), ngroupsize)
    CIS = CartesianIndices(x)
    @inline function kernel(y, x)
        i = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
        i > length(x) && return nothing
        @inbounds yi = expandind(Val{ixs}(), Val{iy}(), CIS[i].I)
        @inbounds y[CartesianIndex(yi)] += sx * x[i]
        nothing
    end

    @roc(gridsize = ngridsize, groupsize = ngroupsize, kernel(y, x))
    return y
end

function _batched_gemm!(C1::Char, C2::Char, alpha, A::ROCArrayTypes{T1,3}, B::ROCArrayTypes{T2,3}, beta, C::ROCArrayTypes{T3,3}) where {T1<:ROCBlasFloat,T2<:ROCBlasFloat,T3<:ROCBlasFloat}
    println("type of alpha", typeof(alpha))
    AMDGPU.rocBLAS.gemm_strided_batched!(C1, C2, alpha, T1 == T3 ? A : T3.(A), T2 == T3 ? B : T3.(B), beta, C)
end

Base.ndims(::Base.Broadcast.Broadcasted{AMDGPU.ROCArrayStyle{0}}) = 0

function einsum!(neinsum::NestedEinsum, @nospecialize(xs::NTuple{N,ROCArrayTypes} where {N}), @nospecialize(y::ROCArrayTypes), sx, sy, size_dict::Dict; active_free=false)
    # do not use map because the static overhead is too large
    # do not use `setindex!` because we need to make the AD work
    mxs = Vector{AbstractArray}(undef, length(siblings(neinsum)))
    for (i, arg) in enumerate(siblings(neinsum))
        mxs = _safe_set(mxs, i, isleaf(arg) ? xs[tensorindex(arg)] : einsum(arg, xs, similar(y, ([size_dict[l] for l in getiy(rootcode(arg))]...,)), true, false, size_dict; active_free=active_free))
    end
    res = einsum!(rootcode(neinsum), (mxs...,), y, sx, sy, size_dict)
    active_free && for mx in mxs  # free ROCArray aggressively.
        AMDGPU.unsafe_free!(mx)
    end
    return res
end

function einsum(neinsum::NestedEinsum, @nospecialize(xs::NTuple{N,ROCArrayTypes} where {N}), size_dict::Dict; active_free=false)
    # do not use map because the static overhead is too large
    # do not use `setindex!` because we need to make the AD work
    mxs = Vector{AbstractArray}(undef, length(siblings(neinsum)))
    for (i, arg) in enumerate(siblings(neinsum))
        mxs = _safe_set(mxs, i, isleaf(arg) ? xs[tensorindex(arg)] : einsum(arg, xs, size_dict; active_free=active_free))
    end
    res = einsum(rootcode(neinsum), (mxs...,), size_dict)
    active_free && for mx in mxs  # free ROCArray aggressively.
        AMDGPU.unsafe_free!(mx)
    end
    return res
end

@info("OMEinsum loaded the AMDGPU module successfully")

end