using CUDAnative: device!
device!(0)
using BenchmarkTools, OMEinsum, CuArrays
CuArrays.allowscalar(false)

function bfunc(N::Int)
    a = randn(Float32, N, N)
    t = randn(Float32, N, N, N)
    ca = a |> CuArray
    ct = t |> CuArray
    res = CuArrays.@sync ein"ji,kl,li->jkl"(ca,ca,ca)
end

include("reduce_einsum.jl")
@benchmark bfunc(100) seconds = 1
