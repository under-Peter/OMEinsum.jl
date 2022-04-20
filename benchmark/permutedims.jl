using CUDA, Random, LinearAlgebra
using CUDA: AbstractGPUArray, @linearidx, gpu_call
using Base.Cartesian: @nexprs
using BenchmarkTools

function LinearAlgebra.permutedims!(dest::AbstractGPUArray, src::AbstractGPUArray,
                                    perm::NTuple{N}) where N
    @assert length(src) < typemax(Int32)
    Base.checkdims_perm(dest, src, perm)

    # get the new strides of destination tensor
    dest_strides = ntuple(k->k==1 ? 1 : prod(i->size(dest, i), 1:k-1), N)
    dest_strides_perm = ntuple(i->Int32(dest_strides[findfirst(==(i), perm)]), N)
    size_src = Int32.(size(src))

    function permutedims_kernel(ctx, dest, src, size_src, dest_strides_perm)
        # find the cartesian index in source tensor
        LI = @linearidx src
        I = @inbounds CartesianIndices(size_src)[LI]

        # the corresponding linear index in the destination tensor
        dest_index = map_index(I.I, dest_strides_perm)
        @inbounds dest[dest_index] = src[LI]
        return
    end
    gpu_call(permutedims_kernel, vec(dest), vec(src), size_src, dest_strides_perm)
    return dest
end

# get linear index from cartesian indices and strides.
@inline @generated function map_index(I::NTuple{N,T}, dest_strides::NTuple{N,T}) where {N,T}
    Expr(:call, :+, one(T), [:(@inbounds (I[$i]-one(T)) * dest_strides[$i]) for i in 1:N]...)
end

@inline @generated function permute_linearindex(size::NTuple{N,T}, l::T, strides::NTuple{N,T}) where {N,T}
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
function permutedims2!(dest::AbstractGPUArray, src::AbstractGPUArray,
                                    perm::NTuple{N}) where N
    @assert length(src) < typemax(Int32)
    Base.checkdims_perm(dest, src, perm)
    dest_strides = ntuple(k->k==1 ? 1 : prod(i->size(dest, i), 1:k-1), N)
    dest_strides_perm = ntuple(i->Int32(dest_strides[findfirst(==(i), perm)]), N)
    size_src = Int32.(size(src))
    LEN = Int32(length(src))
    function permutedims_kernel(ctx, dest, src, size_src, dest_strides_perm, LEN)
        LI = (blockIdx().x-one(Int32)) * blockDim().x + threadIdx().x
        LI > LEN && return
        dest_index = permute_linearindex(size_src, LI, dest_strides_perm)
        @inbounds dest[dest_index] = src[LI]
        return
    end
    gpu_call(permutedims_kernel, vec(dest), vec(src), size_src, dest_strides_perm, LEN)
    return dest
end

let D = 24
    a = CUDA.randn(fill(2, D)...);
    b = CUDA.randn(fill(2, D)...);
    pm = (randperm(D)...,);
    @benchmark CUDA.@sync permutedims2!($b, $a, $pm)
    #CUDA.@sync permutedims!(b, a, pm)
    #CUDA.@sync permutedims!(b, a, pm)
end

let D = 4
    a = CUDA.randn(fill(2, D)...);
    b = CUDA.randn(fill(2, D)...);
    pm = (randperm(D)...,);
    CUDA.@device_code_llvm permutedims2!(b, a, pm)
end
