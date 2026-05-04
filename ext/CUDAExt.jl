module CUDAExt

import OMEinsum: asarray, get_output_array, einsum, loop_einsum!, _batched_gemm!, asscalar, @flatten_addmul!
using OMEinsum: EinArray, Diag, Repeat, Duplicate, DefaultRule, EinCode, DynamicEinCode, StaticEinCode, NestedEinsum, SimpleBinaryRule, match_rule, loop_einsum, getiy, getixs, _unique, einarray, align_eltypes, siblings, isleaf, tensorindex, _safe_set, rootcode
using OMEinsum: EinsumBackend, DefaultBackend, CuTensorBackend, get_einsum_backend, CuTensorSupportedTypes, _CUTENSOR_AVAILABLE, _cutensor_einsum!
import OMEinsum
using LinearAlgebra
import LinearAlgebra: BlasFloat
using CUDA

# -----------------------------------------------------------------------------
# Workspace pool for intermediate CuArrays produced during a contraction.
#
# OMEinsum allocates a fresh CuArray for every binary einsum's output (the
# `get_output_array` call) and, when `active_free=true`, frees them via
# `CUDA.unsafe_free!` once the parent `einsum`/`einsum!` returns. For large
# contraction trees (~10^4 binaries per slice) this allocation churn dominates
# wall-clock even though CUDA.jl's allocator already pools backing storage —
# every call still does Julia-side bookkeeping, a finalizer registration, and
# a `Base.similar`-style code path.
#
# The pool below sits in front of CUDA.jl's allocator and recycles freed
# intermediate CuArrays keyed by `(eltype, size)`. Users opt in by:
#   1. Passing `active_free=true` to `code(xs...; active_free=true)` (or the
#      lower-level `einsum`/`einsum!` API), so freed intermediates flow into
#      the pool instead of `unsafe_free!`.
#   2. Calling `cuda_workspace_pool_drain!()` between contractions when they
#      want the pool's GPU memory released (e.g. between CRT primes that may
#      use a different element type).
#
# When `active_free=false` (the default) the pool simply remains empty, every
# `get_output_array` falls through to the underlying allocator, and behavior
# is unchanged. The pool is single-process and assumes single-threaded
# contraction execution; reentry from concurrent tasks would corrupt the
# `Dict` and counters.
# -----------------------------------------------------------------------------
const _POOL_CAP = Ref{Int}(typemax(Int))                            # bytes
const _POOL_BYTES = Ref{Int}(0)
const _POOL = Dict{Tuple{DataType, Vector{Int}}, Vector{CuArray}}()
const _POOL_HITS = Ref{Int}(0)
const _POOL_MISSES = Ref{Int}(0)
const _POOL_PUSHED = Ref{Int}(0)
const _POOL_DROPPED = Ref{Int}(0)   # buffers that didn't fit and were unsafe_free!d

"""
    OMEinsum.cuda_workspace_pool_drain!()

Free every CuArray held in the CUDAExt workspace pool and reset its counters.
Use between contractions whose intermediate shapes / element types differ.
Buffers are released to the GPU via `CUDA.unsafe_free!`; the function then
runs an incremental `GC.gc(false)` and a `CUDA.synchronize()`.
"""
function cuda_workspace_pool_drain!()
    for (_, v) in _POOL
        for buf in v
            CUDA.unsafe_free!(buf)
        end
        empty!(v)
    end
    empty!(_POOL)
    _POOL_BYTES[] = 0
    _POOL_HITS[] = 0
    _POOL_MISSES[] = 0
    _POOL_PUSHED[] = 0
    _POOL_DROPPED[] = 0
    GC.gc(false)
    CUDA.synchronize()
    return nothing
end

"""
    OMEinsum.cuda_workspace_pool_stats() -> NamedTuple

Returns `(hits, misses, pushed, dropped, live_bytes, n_buckets, n_buffers)`
counters describing the pool's current state since the last `drain!`.
`hits` = bucket pops (allocations served from the pool); `misses` =
fresh `CuArray{T}(undef, ...)` allocations under the cap; `pushed` =
intermediates pushed in via `active_free`; `dropped` = pushes that
exceeded the cap and were `unsafe_free!`d immediately.
"""
function cuda_workspace_pool_stats()
    n_buffers = sum(length, values(_POOL); init = 0)
    return (hits = _POOL_HITS[], misses = _POOL_MISSES[],
            pushed = _POOL_PUSHED[], dropped = _POOL_DROPPED[],
            live_bytes = _POOL_BYTES[],
            n_buckets = length(_POOL), n_buffers = n_buffers)
end

"""
    OMEinsum.cuda_workspace_pool_set_cap!(bytes)

Set the maximum number of bytes the workspace pool may hold. Pushes that
would exceed the cap are routed straight to `unsafe_free!` (counted as
`dropped`). Default cap is `typemax(Int)` (effectively unbounded — the
pool is naturally bounded by the largest intermediate per contraction).
"""
function cuda_workspace_pool_set_cap!(bytes::Integer)
    _POOL_CAP[] = Int(bytes)
    return nothing
end

# Module-private helpers.
function _pool_take!(::Type{T}, sz::NTuple{N, Int}) where {T, N}
    sz_vec = collect(Int, sz)
    bucket = get(_POOL, (T, sz_vec), nothing)
    bucket === nothing && return nothing
    isempty(bucket) && return nothing
    out = pop!(bucket)
    nbytes = length(out) * sizeof(T)
    _POOL_BYTES[] -= nbytes
    _POOL_HITS[] += 1
    return out::CuArray{T}
end

function _pool_put!(buf::CuArray{T}) where T
    sz_vec = collect(Int, size(buf))
    nbytes = length(buf) * sizeof(T)
    if _POOL_BYTES[] + nbytes > _POOL_CAP[]
        CUDA.unsafe_free!(buf)
        _POOL_DROPPED[] += 1
        return nothing
    end
    bucket = get!(_POOL, (T, sz_vec), CuArray[])
    push!(bucket, buf)
    _POOL_BYTES[] += nbytes
    _POOL_PUSHED[] += 1
    return nothing
end

# Re-export pool entry points through OMEinsum's namespace so users say
# `OMEinsum.cuda_workspace_pool_drain!()` without depending on the extension
# module name.
OMEinsum.cuda_workspace_pool_drain!()       = cuda_workspace_pool_drain!()
OMEinsum.cuda_workspace_pool_stats()        = cuda_workspace_pool_stats()
OMEinsum.cuda_workspace_pool_set_cap!(b::Integer) = cuda_workspace_pool_set_cap!(b)

@static if pkgversion(CUDA) >= v"6"
    using CUDA.CUDACore: CuArrayStyle
else
    using CUDA: CuArrayStyle
end

const CuBlasFloat = Union{BlasFloat, Float16, ComplexF16}
const CUDAArrayTypes{T,N} = Union{LinearAlgebra.Transpose{T,<:CuArray{T,N}}, DenseCuArray{T,N}, LinearAlgebra.Adjoint{T,<:CuArray{T,N}}}
_unwrap(x::LinearAlgebra.Adjoint{T,<:CuArray{T}}) where T = CuArray(x)
_unwrap(x::LinearAlgebra.Transpose{T,<:CuArray{T}}) where T = CuArray(x)
_unwrap(x::CuArray) = x

asarray(x, arr::CuArray) = CuArray(fill(x, ()))
asarray(x::AbstractArray, y::CuArray) = x
asscalar(x::CUDAArrayTypes) = Array(x)[]

# to avoid returning a ReshapedArray
OMEinsum.safe_reshape(x::CuArray, sz) = reshape(x, (sz...,))
OMEinsum.safe_reshape(x::Adjoint{T, <:CuArray{T}} where T, sz) = reshape(CuArray(x), (sz...,))
OMEinsum.safe_reshape(x::Transpose{T, <:CuArray{T}} where T, sz) = reshape(CuArray(x), (sz...,))

Base.Array(x::Base.ReshapedArray{T,0,<:CuArray}) where T = Array(x.parent)

function get_output_array(xs::NTuple{N, CUDAArrayTypes{<:Any,M} where M}, size, fillzero::Bool) where N
    T = promote_type(map(eltype, xs)...)
    return _alloc_output(T, NTuple{length(size), Int}(size), fillzero)
end
function get_output_array(xs::NTuple{N, CUDAArrayTypes{T,M} where M}, size, fillzero::Bool) where {T,N}
    return _alloc_output(T, NTuple{length(size), Int}(size), fillzero)
end

# Single allocation entry point used by both `get_output_array` methods.
# Checks the workspace pool first (zero-cost when the pool is empty); on
# miss falls through to the underlying allocator.
function _alloc_output(::Type{T}, sz::NTuple{N, Int}, fillzero::Bool) where {T, N}
    out = _pool_take!(T, sz)
    if out === nothing
        _POOL_MISSES[] += 1
        out = fillzero ? CUDA.zeros(T, sz...) : CuArray{T}(undef, sz...)
    elseif fillzero
        fill!(out, zero(T))
    end
    return out
end

CUDA.cudaconvert(A::EinArray{T}) where T = EinArray{T}(cudaconvert.(A.xs), A.x_indexers, A.y_indexer, A.size, A.ICIS, A.OCIS)
CUDA.cu(A::EinArray{T}) where T = EinArray{T}(cu.(A.xs), A.x_indexers, A.y_indexer, A.size, A.ICIS, A.OCIS)

for TP in [:Diag, :Repeat, :Duplicate]
    @eval function OMEinsum.unary_einsum!(::$TP, ix, iy, x::CUDAArrayTypes, y::CUDAArrayTypes, sx, sy)
        @debug "cueinsum fallback to loop_einsum" rule ix => iy size(x)
        size_dict = OMEinsum.get_size_dict((ix, iy), (x, y))
        loop_einsum!((ix,), iy, (x,), y, sx, sy, size_dict)
    end
end

function loop_einsum!(ixs0, iy0,
                xs::NTuple{N, CUDAArrayTypes{<:Any,M} where M},
                y::CUDAArrayTypes{T,L}, sx, sy, size_dict::Dict{LT}) where {N,L,T, LT}
    iy = (iy0...,)
    ixs = (Tuple.(ixs0)...,)
    iy_ = _unique(LT,iy)
    NO = length(iy_)
    A = einarray(Val(ixs), Val((iy_...,)), xs, size_dict)
    if NO == length(iy)
        raw_ = similar(y, (fill(1, ndims(A)-NO)...,size(y)...,))
        fill!(raw_, zero(T))
        Base.mapreducedim!(x->x, +, raw_, A)
        if ndims(A)-NO > 0  # fix 1.7 compatibility
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
        y_ = similar(y, (fill(1, ndims(A)-NO)...,[size_dict[l] for l in iy_]...))
        fill!(y_, zero(T))
        raw = Base.mapreducedim!(x->x, +, y_, A)
        if ndims(A)-NO > 0  # fix 1.7 compatibility
            raw = dropdims(raw, dims=(1:ndims(A)-NO...,))
        end
        return expanddims!(Val{((iy_...,),)}(), Val{iy}(), raw, y, sx)
    end
end

@generated function expandind(::Val{ixs}, ::Val{iy}, ind) where {ixs, iy}
    ix = ixs[1]
    ids = map(ii->:(ind[$(findfirst(==(ii), ix))]), iy)
    Expr(:tuple, ids...)
end

function expanddims!(::Val{ixs}, ::Val{iy}, x, y, sx) where {ixs,iy}
    nthreads = 256
    nblocks = cld(prod(size(x)), nthreads)
    CIS = CartesianIndices(x)
    @inline function kernel(y, x)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        i > length(x) && return nothing
        @inbounds yi = expandind(Val{ixs}(), Val{iy}(), CIS[i].I)
        @inbounds y[CartesianIndex(yi)] += sx * x[i]
        nothing
    end
    @cuda(blocks=nblocks, threads=nthreads, kernel(y, x))
    return y
end

function _batched_gemm!(C1::Char, C2::Char, alpha, A::CUDAArrayTypes{T1,3}, B::CUDAArrayTypes{T2,3}, beta, C::CUDAArrayTypes{T3,3}) where {T1<:CuBlasFloat, T2<:CuBlasFloat, T3<:CuBlasFloat}
    CUDA.CUBLAS.gemm_strided_batched!(C1, C2, alpha, T1 == T3 ? A : T3.(A), T2 == T3 ? B : T3.(B), beta, C)
end

Base.ndims(::Base.Broadcast.Broadcasted{CuArrayStyle{0}}) = 0

function einsum!(neinsum::NestedEinsum, @nospecialize(xs::NTuple{N,CUDAArrayTypes} where N), @nospecialize(y::CUDAArrayTypes), sx, sy, size_dict::Dict; active_free=false)
    # do not use map because the static overhead is too large
    # do not use `setindex!` because we need to make the AD work
    n = length(siblings(neinsum))
    mxs = Vector{AbstractArray}(undef, n)
    leaf_flags = Vector{Bool}(undef, n)
    for (i, arg) in enumerate(siblings(neinsum))
        leaf = isleaf(arg)
        leaf_flags[i] = leaf
        mxs = _safe_set(mxs, i, leaf ? xs[tensorindex(arg)] : einsum(arg, xs, similar(y, ([size_dict[l] for l in getiy(rootcode(arg))]...,)), true, false, size_dict; active_free=active_free))
    end
    res = einsum!(rootcode(neinsum), (mxs...,), y, sx, sy, size_dict)
    if active_free
        for (i, mx) in enumerate(mxs)
            leaf_flags[i] && continue            # never recycle leaf user inputs
            mx isa CuArray && _pool_put!(mx)     # pool intermediates; ignore wrapped views
        end
    end
    return res
end

function einsum(neinsum::NestedEinsum, @nospecialize(xs::NTuple{N,CUDAArrayTypes} where N), size_dict::Dict; active_free=false)
    # do not use map because the static overhead is too large
    # do not use `setindex!` because we need to make the AD work
    n = length(siblings(neinsum))
    mxs = Vector{AbstractArray}(undef, n)
    leaf_flags = Vector{Bool}(undef, n)
    for (i, arg) in enumerate(siblings(neinsum))
        leaf = isleaf(arg)
        leaf_flags[i] = leaf
        mxs = _safe_set(mxs, i, leaf ? xs[tensorindex(arg)] : einsum(arg, xs, size_dict; active_free=active_free))
    end
    res = einsum(rootcode(neinsum), (mxs...,), size_dict)
    if active_free
        for (i, mx) in enumerate(mxs)
            leaf_flags[i] && continue
            mx isa CuArray && _pool_put!(mx)
        end
    end
    return res
end

#####################################################################
# Binary einsum with backend dispatch
#####################################################################

"""
Default binary einsum using CUBLAS (existing implementation path).
"""
function binary_einsum_cublas!(
    ixs, iy,
    xs::NTuple{2,CuArray{T}}, y::CuArray{T},
    sx, sy, size_dict::Dict{LT}
) where {T, LT}
    iyv = OMEinsum._collect(LT, iy)
    ix1v, ix2v = OMEinsum._collect.(Ref(LT), ixs)
    x1, x2 = xs
    c1, c2, cy, s1, s2, s3, i1, i2, iyb = OMEinsum.analyze_binary(ix1v, ix2v, iyv, size_dict)
    rule = SimpleBinaryRule{(i1...,),(i2...,),(iyb...,)}()
    xs1 = OMEinsum.simplifyto(ix1v, c1, x1, size_dict)
    xs2 = OMEinsum.simplifyto(ix2v, c2, x2, size_dict)
    x1_ = OMEinsum.safe_reshape(xs1, s1)
    x2_ = OMEinsum.safe_reshape(xs2, s2)
    if cy != iyv
        y_ = similar(y, (s3...,))
        y_ = reshape(OMEinsum.binary_einsum!(rule, x1_, x2_, y_, true, false), [size_dict[x] for x in cy]...)
        return OMEinsum.einsum!((cy,), iyv, (y_,), y, sx, sy, size_dict)
    else
        OMEinsum.binary_einsum!(rule, x1_, x2_, OMEinsum.safe_reshape(y, s3), sx, sy)
        return y
    end
end

# Override einsum! for binary CuArray operations with backend dispatch
function OMEinsum.einsum!(
    ixs, iy,
    xs::NTuple{2,CuArray{T}}, y::CuArray{T},
    sx, sy, size_dict::Dict{LT}
) where {T, LT}
    # Use cuTENSOR if: backend is CuTensorBackend, extension is loaded, and type is supported
    if get_einsum_backend() isa CuTensorBackend && _CUTENSOR_AVAILABLE[] && T <: CuTensorSupportedTypes
        return _cutensor_einsum!(ixs, iy, xs, y, sx, sy, size_dict)
    end
    # Default: use CUBLAS path
    return binary_einsum_cublas!(ixs, iy, xs, y, sx, sy, size_dict)
end

@debug("OMEinsum loaded the CUDA module successfully")

end