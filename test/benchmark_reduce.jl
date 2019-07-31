using CUDAnative: device!
device!(4)
using BenchmarkTools, OMEinsum, CuArrays
CuArrays.allowscalar(false)

include("reduce_einsum.jl")

function bfunc(N::Int)
    a = randn(Float32, N, N)
    ca = a |> CuArray
    res = CuArrays.@sync ein"ij,ik,il->jkl"(ca,ca,ca)
    #@show Array(res) - ein"ij,ik,il->jkl"(a,a,a)
end
@benchmark bfunc(100) seconds = 1
