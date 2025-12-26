"""
    EinsumBackend

Abstract type for einsum computation backends.

OMEinsum supports multiple backends for tensor contractions, allowing users
to choose the most appropriate implementation for their hardware and use case.

## Available Backends
- [`DefaultBackend`](@ref): Uses BLAS/CUBLAS with reshape/permute operations
- [`CuTensorBackend`](@ref): Uses NVIDIA cuTENSOR for native tensor contractions

## Usage
```julia
# Check current backend
get_einsum_backend()

# Change backend globally
set_einsum_backend!(CuTensorBackend())
```

See also: [`set_einsum_backend!`](@ref), [`get_einsum_backend`](@ref)
"""
abstract type EinsumBackend end

"""
    DefaultBackend <: EinsumBackend

Default backend using BLAS/CUBLAS for tensor contractions.

This backend reduces tensor contractions to matrix multiplications (GEMM) by:
1. Analyzing the contraction pattern to identify inner/outer/batch indices
2. Permuting tensors to canonical GEMM form if needed
3. Reshaping tensors to 2D/3D matrices
4. Calling optimized BLAS routines (`gemm!`, `gemm_strided_batched!`)
5. Reshaping and permuting the result back

## Pros
- Highly optimized for matrix-like contractions
- Works with any array type that supports `mul!`
- No additional library dependencies

## Cons
- Overhead from permute/reshape for non-GEMM patterns
- May allocate intermediate arrays

## Example
```julia
set_einsum_backend!(DefaultBackend())
ein"ij,jk->ik"(A, B)  # Uses BLAS gemm
```
"""
struct DefaultBackend <: EinsumBackend end

"""
    CuTensorBackend <: EinsumBackend

Backend using NVIDIA cuTENSOR library for native tensor contractions on GPU.

This backend calls cuTENSOR's `contract!` function directly, which:
- Handles arbitrary tensor contraction patterns natively
- Eliminates reshape/permute overhead
- Optimizes memory access patterns internally

## Requirements
- NVIDIA GPU with compute capability ≥ 6.0
- CUDA.jl v5.0 or later with cuTENSOR support

## Supported Types
- `Float16`, `Float32`, `Float64`
- `ComplexF16`, `ComplexF32`, `ComplexF64`

For unsupported types, automatically falls back to `DefaultBackend`.

## Pros
- No intermediate allocations for non-GEMM patterns
- Better performance for tensor network contractions
- Native support for arbitrary index patterns

## Cons
- Only available on NVIDIA GPUs with cuTENSOR
- Slightly higher overhead for simple GEMM patterns
- Limited to BLAS-compatible numeric types

## Example
```julia
using CUDA, cuTENSOR, OMEinsum

# Enable cuTENSOR backend
set_einsum_backend!(CuTensorBackend())

# Tensor contraction (benefits most from cuTENSOR)
A = CUDA.rand(Float32, 64, 64, 64)
B = CUDA.rand(Float32, 64, 64, 64)
C = ein"ijk,jkl->il"(A, B)  # Direct cuTENSOR call, no reshape needed
```

## When to Use
- Tensor network contractions (MPS, PEPS, etc.)
- High-dimensional tensor operations
- Contractions with complex index patterns like `ein"ijkl,klmn->ijmn"`

See also: [`DefaultBackend`](@ref), [`set_einsum_backend!`](@ref)
"""
struct CuTensorBackend <: EinsumBackend end

# Global backend configuration
const EINSUM_BACKEND = Ref{EinsumBackend}(DefaultBackend())

"""
    set_einsum_backend!(backend::EinsumBackend) -> EinsumBackend

Set the global backend for einsum operations.

!!! note
    This sets a global state. For thread-safe usage in concurrent code,
    consider using the same backend throughout or synchronizing access.

## Arguments
- `backend::EinsumBackend`: The backend to use.
  - `DefaultBackend()`: Use BLAS/CUBLAS (default)
  - `CuTensorBackend()`: Use cuTENSOR for GPU contractions

## Returns
The backend that was set.

## Example
```julia
using OMEinsum, CUDA

# Check current backend
get_einsum_backend()  # DefaultBackend()

# Switch to cuTENSOR
set_einsum_backend!(CuTensorBackend())

# Perform contractions with cuTENSOR
A = CUDA.rand(Float32, 100, 200)
B = CUDA.rand(Float32, 200, 300)
C = ein"ij,jk->ik"(A, B)

# Reset to default
set_einsum_backend!(DefaultBackend())
```

## Performance Comparison
```julia
using BenchmarkTools

A = CUDA.rand(Float32, 64, 64, 64)
B = CUDA.rand(Float32, 64, 64, 64)

# DefaultBackend: requires permute + reshape + GEMM + reshape
set_einsum_backend!(DefaultBackend())
@btime CUDA.@sync ein"ijk,jkl->il"(\$A, \$B)

# CuTensorBackend: direct tensor contraction
set_einsum_backend!(CuTensorBackend())
@btime CUDA.@sync ein"ijk,jkl->il"(\$A, \$B)
```

See also: [`get_einsum_backend`](@ref), [`EinsumBackend`](@ref)
"""
function set_einsum_backend!(backend::EinsumBackend)
    EINSUM_BACKEND[] = backend
    return backend
end

# Hook for extensions to check backend availability - returns (available::Bool, message::String)
check_backend_availability(::EinsumBackend) = (true, "")

# Registry for backend availability (used to avoid method overwriting during precompilation)
const _CUTENSOR_AVAILABLE = Ref{Bool}(false)

# Function reference for cuTENSOR einsum - set by CuTENSORExt when loaded
const _CUTENSOR_EINSUM_IMPL = Ref{Any}(nothing)

function _cutensor_einsum!(ixs, iy, xs, y, sx, sy, size_dict)
    impl = _CUTENSOR_EINSUM_IMPL[]
    if impl === nothing
        error("cuTENSOR extension not loaded. Please run `using cuTENSOR`.")
    end
    return impl(ixs, iy, xs, y, sx, sy, size_dict)
end

function check_backend_availability(::CuTensorBackend)
    if _CUTENSOR_AVAILABLE[]
        return (true, "")
    end
    return (false, "cuTENSOR.jl not loaded - run `using cuTENSOR` first")
end

function set_einsum_backend!(backend::CuTensorBackend)
    available, msg = check_backend_availability(backend)
    available || @warn "CuTensorBackend: $msg. Will fall back to CUBLAS."
    EINSUM_BACKEND[] = backend
end

"""
    get_einsum_backend() -> EinsumBackend

Get the current global einsum backend.

## Returns
The currently active `EinsumBackend` instance.

## Example
```julia
backend = get_einsum_backend()
if backend isa CuTensorBackend
    println("Using cuTENSOR")
else
    println("Using default BLAS/CUBLAS")
end
```

See also: [`set_einsum_backend!`](@ref), [`EinsumBackend`](@ref)
"""
get_einsum_backend() = EINSUM_BACKEND[]

"""
    CuTensorSupportedTypes

Union of numeric types supported by the cuTENSOR backend.

```julia
CuTensorSupportedTypes = Union{Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64}
```

Arrays with element types not in this union will automatically fall back to
the default backend when `CuTensorBackend` is active.
"""
const CuTensorSupportedTypes = Union{Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64}
