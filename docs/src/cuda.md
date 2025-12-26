# CUDA Acceleration

OMEinsum supports GPU acceleration through CUDA.jl. By uploading your data to the GPU, you can significantly accelerate tensor contractions.

## Basic CUDA Usage

```julia repl
julia> using CUDA, OMEinsum

julia> code = ein"ij,jk,kl,li->"  # the einsum notation
ij, jk, kl, li -> 

julia> A, B, C, D = rand(1000, 1000), rand(1000, 300), rand(300, 800), rand(800, 1000);

julia> size_dict = OMEinsum.get_size_dict(getixsv(code), (A, B, C, D))  # get the size of the labels
Dict{Char, Int64} with 4 entries:
  'j' => 1000
  'i' => 1000
  'k' => 300
  'l' => 800

julia> optcode = optimize_code(code, size_dict, TreeSA())  # optimize the contraction order
SlicedEinsum{Char, DynamicNestedEinsum{Char}}(Char[], kl, kl -> 
├─ ki, li -> kl
│  ├─ jk, ij -> ki
│  │  ├─ jk
│  │  └─ ij
│  └─ li
└─ kl
)
```

The contraction order is optimized. Now, let's benchmark the contraction on the CPU.

```julia repl
julia> using BenchmarkTools

julia> @btime optcode($A, $B, $C, $D)  # the contraction on CPU
  6.053 ms (308 allocations: 20.16 MiB)
0-dimensional Array{Float64, 0}:
1.4984046443610943e10
```

The contraction on the CPU takes about 6 ms. Now, let's upload the data to the GPU and perform the contraction on the GPU.
```julia repl
julia> cuA, cuB, cuC, cuD = CuArray(A), CuArray(B), CuArray(C), CuArray(D);

julia> @btime CUDA.@sync optcode($cuA, $cuB, $cuC, $cuD)  # the contraction on GPU
  243.888 μs (763 allocations: 28.56 KiB)
0-dimensional CuArray{Float64, 0, CUDA.DeviceMemory}:
1.4984046443610939e10
```

## GPU Backends

OMEinsum provides two backends for GPU tensor contractions:

| Backend | Library | Best For |
|---------|---------|----------|
| `DefaultBackend()` | CUBLAS | Matrix-like contractions (GEMM patterns) |
| `CuTensorBackend()` | cuTENSOR | General tensor contractions |

### DefaultBackend (CUBLAS)

The default backend uses NVIDIA's CUBLAS library. It works by:
1. Analyzing the contraction pattern
2. Reshaping tensors to fit matrix multiplication (GEMM) format
3. Calling `CUBLAS.gemm_strided_batched!`
4. Reshaping the result back

This approach works well for contractions that naturally map to matrix multiplication, such as:
- `ein"ij,jk->ik"` (matrix multiplication)
- `ein"ijl,jkl->ikl"` (batched matrix multiplication)

### CuTensorBackend (cuTENSOR)

The cuTENSOR backend uses NVIDIA's cuTENSOR library, which provides **native tensor contraction** without the reshape/permute overhead. This is especially beneficial for:

- **Non-GEMM patterns**: Contractions like `ein"ijk,jkl->il"` that don't naturally fit GEMM
- **High-dimensional tensors**: Avoids costly permutations
- **Complex index patterns**: Direct support for arbitrary contraction patterns

## Using the cuTENSOR Backend

### Prerequisites

cuTENSOR requires:
- NVIDIA GPU with compute capability ≥ 6.0
- CUDA.jl v5.0 or later
- cuTENSOR.jl package (install with `Pkg.add("cuTENSOR")`)

### Check Availability

```julia
using CUDA
using cuTENSOR  # Must be loaded to enable the cuTENSOR backend
using OMEinsum

# Check if cuTENSOR is available
if cuTENSOR.has_cutensor()
    println("cuTENSOR is available!")
else
    println("cuTENSOR is not available")
end
```

!!! important "Loading cuTENSOR"
    cuTENSOR.jl is a **separate package**. You must:
    1. Install it: `Pkg.add("cuTENSOR")`
    2. Load it **before** or alongside OMEinsum: `using cuTENSOR`
    
    This triggers OMEinsum's `CuTENSORExt` extension, which provides the cuTENSOR backend functionality.

### Enable cuTENSOR Backend

```julia
using CUDA
using cuTENSOR  # ← Required! This loads the CuTENSORExt extension
using OMEinsum

# Set the backend globally
set_einsum_backend!(CuTensorBackend())

# Now all GPU einsum operations will use cuTENSOR
A = CUDA.rand(Float32, 100, 200, 300)
B = CUDA.rand(Float32, 200, 300, 400)
C = ein"ijk,jkl->il"(A, B)  # Uses cuTENSOR

# Reset to default backend
set_einsum_backend!(DefaultBackend())
```

!!! warning "Forgetting to load cuTENSOR"
    If you set `CuTensorBackend()` without loading cuTENSOR.jl, you'll see:
    ```
    ┌ Warning: CuTensorBackend: cuTENSOR.jl not loaded - run `using cuTENSOR` first. Will fall back to CUBLAS.
    ```
    The computation will still work (using CUBLAS), but you won't get the cuTENSOR optimization.

### Supported Data Types

The cuTENSOR backend supports BLAS-compatible types:
- `Float16`, `Float32`, `Float64`
- `ComplexF16`, `ComplexF32`, `ComplexF64`

For other types (e.g., `Double64`, custom number types), the backend automatically falls back to the loop-based implementation.

### Example: Performance Comparison

```julia
using CUDA
using cuTENSOR  # Required for cuTENSOR backend
using OMEinsum, BenchmarkTools

# Create test tensors (non-GEMM pattern)
A = CUDA.rand(Float32, 64, 64, 64)
B = CUDA.rand(Float32, 64, 64, 64)

# Benchmark with DefaultBackend (CUBLAS)
set_einsum_backend!(DefaultBackend())
@btime CUDA.@sync ein"ijk,jkl->il"($A, $B)
# Requires: permute → reshape → GEMM → reshape

# Benchmark with CuTensorBackend
set_einsum_backend!(CuTensorBackend())
@btime CUDA.@sync ein"ijk,jkl->il"($A, $B)
# Direct tensor contraction - no intermediate steps!
```

### When to Use cuTENSOR

| Use Case | Recommended Backend |
|----------|-------------------|
| Matrix multiplication `ij,jk->ik` | Either (similar performance) |
| Batched matmul `ijl,jkl->ikl` | Either (similar performance) |
| Tensor contraction `ijk,jkl->il` | **CuTensorBackend** |
| High-dimensional `ijkl,klmn->ijmn` | **CuTensorBackend** |
| MPS/PEPS contractions | **CuTensorBackend** |
| Non-BLAS types (Double64, etc.) | DefaultBackend |

### Best Practices

1. **Use cuTENSOR for tensor networks**: If you're doing MPS, PEPS, or general tensor network contractions, cuTENSOR typically provides better performance.

2. **Profile your workload**: The relative performance depends on tensor sizes and contraction patterns. Use `BenchmarkTools` to measure.

3. **Keep data on GPU**: Minimize CPU-GPU transfers by keeping intermediate results on the GPU.

4. **Batch operations**: When contracting many small tensors, consider batching them together.

```julia
# Good: Keep data on GPU throughout
cuA, cuB, cuC = CuArray.((A, B, C))
result = ein"(ij,jk),kl->il"(cuA, cuB, cuC)

# Avoid: Repeated transfers
result = ein"ij,jk->ik"(CuArray(A), CuArray(B))  # Transfer every call
```

## Nested Contractions with cuTENSOR

The cuTENSOR backend works seamlessly with optimized contraction orders:

```julia
using CUDA, cuTENSOR, OMEinsum

set_einsum_backend!(CuTensorBackend())

# Define a tensor network
code = ein"ij,jk,kl,lm->im"

# Optimize contraction order
size_dict = Dict('i'=>100, 'j'=>100, 'k'=>100, 'l'=>100, 'm'=>100)
optcode = optimize_code(code, size_dict, TreeSA())

# Execute with cuTENSOR (each pairwise contraction uses cuTENSOR)
A, B, C, D = [CUDA.rand(Float32, 100, 100) for _ in 1:4]
result = optcode(A, B, C, D)
```

## Troubleshooting

### cuTENSOR not detected

If `has_cutensor()` returns `false`:

1. **Check CUDA.jl version**: Ensure you have CUDA.jl v5.0+
   ```julia
   using Pkg
   Pkg.status("CUDA")
   ```

2. **Check GPU compatibility**: cuTENSOR requires compute capability ≥ 6.0
   ```julia
   using CUDA
   CUDA.capability(CUDA.device())
   ```

3. **Reinstall CUDA artifacts**:
   ```julia
   using CUDA
   CUDA.versioninfo()  # Check if cuTENSOR is listed
   ```

### Performance not as expected

1. **Ensure synchronization**: Use `CUDA.@sync` when benchmarking
2. **Check tensor sizes**: cuTENSOR has more overhead for very small tensors
3. **Verify backend is active**: Check `get_einsum_backend()`

## Video Tutorial

To learn more about using GPU and autodiff, please check out the following asciinema video.
[![asciicast](https://asciinema.org/a/wE4CtIzWUC3R0GkVV28rVBRFb.svg)](https://asciinema.org/a/wE4CtIzWUC3R0GkVV28rVBRFb)

