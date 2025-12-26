module CuTENSORExt

using OMEinsum: EinsumBackend, DefaultBackend, CuTensorBackend, get_einsum_backend, CuTensorSupportedTypes, _CUTENSOR_AVAILABLE, _CUTENSOR_EINSUM_IMPL
import OMEinsum
using cuTENSOR
using cuTENSOR.CUDA

# Set flags at module initialization (runtime), not precompilation time
function __init__()
    _CUTENSOR_AVAILABLE[] = true
    _CUTENSOR_EINSUM_IMPL[] = cutensor_einsum!
end

"""
    cutensor_einsum!(ixs, iy, xs::NTuple{2,CuArray{T}}, y::CuArray{T}, sx, sy, size_dict) where T

Perform binary tensor contraction using cuTENSOR.jl.
This provides native tensor contraction without reshape/permute overhead.
"""
function cutensor_einsum!(
    ixs, iy,
    xs::NTuple{2,CuArray{T}}, y::CuArray{T},
    sx, sy, size_dict
) where {T<:CuTensorSupportedTypes}
    if length(ixs) != 2
        error("cuTENSOR backend only supports binary contractions")
    end
    
    A, B = xs
    ix1, ix2 = ixs[1], ixs[2]
    
    # Convert labels to vectors (cuTENSOR accepts Char or Integer indices)
    modes_A = collect(ix1)
    modes_B = collect(ix2)
    modes_C = collect(iy)
    
    # cuTENSOR.contract!(α, A, Ainds, opA, B, Binds, opB, β, C, Cinds, opC, opOut)
    cuTENSOR.contract!(
        T(sx), A, modes_A, cuTENSOR.OP_IDENTITY,
        B, modes_B, cuTENSOR.OP_IDENTITY,
        T(sy), y, modes_C, cuTENSOR.OP_IDENTITY,
        cuTENSOR.OP_IDENTITY
    )
    
    return y
end

@debug "OMEinsum: cuTENSOR extension loaded"

end

