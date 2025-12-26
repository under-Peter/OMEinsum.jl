using OMEinsum
using cuTENSOR, cuTENSOR.CUDA
using BenchmarkTools
using Printf

# Benchmark configuration
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5

"""
Run benchmark for a specific einsum code and tensor sizes.
"""
function benchmark_einsum(code, sizes, T::Type; warmup=true)
    # Create input tensors
    xs = Tuple(CUDA.rand(T, s...) for s in sizes)
    
    results = Dict{String, Any}()
    
    # Warmup
    if warmup
        code(xs...)
        CUDA.synchronize()
    end
    
    # DefaultBackend (CUBLAS)
    set_einsum_backend!(DefaultBackend())
    CUDA.synchronize()
    
    default_bench = @benchmark begin
        $code($xs...)
        CUDA.synchronize()
    end
    results["default"] = default_bench
    
    # CuTensorBackend
    set_einsum_backend!(CuTensorBackend())
    CUDA.synchronize()
    
    cutensor_bench = @benchmark begin
        $code($xs...)
        CUDA.synchronize()
    end
    results["cutensor"] = cutensor_bench

    # Reset to default
    set_einsum_backend!(DefaultBackend())

    return results
end

"""
Format benchmark results for display.
"""
function format_result(bench)
    med = median(bench)
    return @sprintf("%.3f ms (±%.3f)", med.time/1e6, std(bench.times)/1e6)
end

"""
Print benchmark comparison table.
"""
function print_comparison(name, results)
    default_med = median(results["default"]).time
    
    print(@sprintf("%-40s", name))
    print(@sprintf(" | %-20s", format_result(results["default"])))
    
    if haskey(results, "cutensor")
        cutensor_med = median(results["cutensor"]).time
        speedup = default_med / cutensor_med
        print(@sprintf(" | %-20s", format_result(results["cutensor"])))
        print(@sprintf(" | %.2fx", speedup))
    else
        print(" | N/A                  | N/A")
    end
    println()
end

# ============================================================================
# Benchmark Suite
# ============================================================================

function run_benchmarks()
    println("=" ^ 100)
    println("cuTENSOR vs CUBLAS Benchmark")
    println("=" ^ 100)
    println()
    
    # GPU Info
    println("GPU: ", CUDA.name(CUDA.device()))
    println("CUDA: ", CUDA.runtime_version())
    println("cuTENSOR: ", cuTENSOR.version())
    println()
    
    for T in [Float32, Float64]
        println("-" ^ 100)
        println("Data Type: $T")
        println("-" ^ 100)
        println(@sprintf("%-40s | %-20s | %-20s | %s", "Operation", "CUBLAS", "cuTENSOR", "Speedup"))
        println("-" ^ 100)
        
        # 1. Matrix multiplication (GEMM-like)
        for n in [128, 256, 512, 1024, 2048]
            name = "matmul ($n×$n)"
            results = benchmark_einsum(ein"ij,jk->ik", [(n, n), (n, n)], T)
            print_comparison(name, results)
        end
        println()
        
        # 2. Rectangular matrix multiplication
        results = benchmark_einsum(ein"ij,jk->ik", [(512, 1024), (1024, 768)], T)
        print_comparison("matmul (512×1024, 1024×768)", results)
        println()
        
        # 3. Batched matrix multiplication
        for (m, n, k, b) in [(64, 64, 64, 32), (128, 128, 128, 16), (256, 256, 256, 8)]
            name = "batched_matmul ($m×$n×$k, batch=$b)"
            results = benchmark_einsum(ein"ijl,jkl->ikl", [(m, n, b), (n, k, b)], T)
            print_comparison(name, results)
        end
        println()
        
        # 4. Tensor contraction (non-GEMM patterns)
        # These are where cuTENSOR should shine vs CUBLAS (which needs reshape/permute)
        
        # 4a. ijk,jkl->il (contract middle two indices)
        for d in [16, 32, 64]
            name = "contract ijk,jkl->il ($(d)³)"
            results = benchmark_einsum(ein"ijk,jkl->il", [(d, d, d), (d, d, d)], T)
            print_comparison(name, results)
        end
        println()
        
        # 4b. ijkl,klmn->ijmn (contract k,l)
        for d in [8, 16, 24]
            name = "4D contract ijkl,klmn->ijmn ($d⁴)"
            results = benchmark_einsum(ein"ijkl,klmn->ijmn", [(d, d, d, d), (d, d, d, d)], T)
            print_comparison(name, results)
        end
        println()
        
        # 4c. Trace-like contraction: ij,ji->
        for n in [256, 512, 1024, 2048]
            name = "trace ij,ji-> ($n×$n)"
            results = benchmark_einsum(ein"ij,ji->", [(n, n), (n, n)], T)
            print_comparison(name, results)
        end
        println()
        
        # 4d. Outer product: i,j->ij
        for n in [1024, 2048, 4096]
            name = "outer i,j->ij ($n)"
            results = benchmark_einsum(ein"i,j->ij", [(n,), (n,)], T)
            print_comparison(name, results)
        end
        println()
        
        # 5. More complex patterns (tensor network style)
        # 5a. Typical MPS-style contraction
        D, d = 64, 4  # bond dimension, physical dimension
        name = "MPS-style aib,bjc->aijc (D=$D, d=$d)"
        results = benchmark_einsum(ein"aib,bjc->aijc", [(D, d, D), (D, d, D)], T)
        print_comparison(name, results)
        
        # 5b. PEPS-style contraction
        D = 16
        name = "PEPS-style ijab,jkbc->ikac (D=$D)"
        results = benchmark_einsum(ein"ijab,jkbc->ikac", [(D, D, D, D), (D, D, D, D)], T)
        print_comparison(name, results)
        
        println()
    end
    
    println("=" ^ 100)
    println("Benchmark complete!")
end

# Run benchmarks
run_benchmarks()

