using CUDAnative: device!
device!(0)
using BenchmarkTools, OMEinsum, CuArrays
CuArrays.allowscalar(false)

function bfunc_cpu_matmul(N::Int)
    a = randn(Float32, N, N)
    ein"ij,jk->ik"(a,a)
end

function bfunc_cpu_star(N::Int)
    a = randn(Float32, N, N)
    ein"ij,lk,il->jkl"(a,a,a)
end

function bfunc_gpu_matmul(N::Int)
    a = randn(Float32, N, N)
    a = a |> CuArray
    @CuArrays.sync ein"ij,jk->ik"(a,a)
end

function bfunc_gpu_star(N::Int)
    a = randn(Float32, N, N)
    a = a |> CuArray
    @CuArrays.sync ein"ij,lk,il->jkl"(a,a,a)
end

@benchmark bfunc_cpu_matmul(100)
@benchmark bfunc_cpu_star(100)

@benchmark bfunc_gpu_matmul(100)
@benchmark bfunc_gpu_star(100)
