using CUDAnative: device!
device!(4)
using BenchmarkTools, OMEinsum, CuArrays
CuArrays.allowscalar(false)

include("reduce_einsum.jl")

function bfunc(N::Int)
    a = randn(Float32, N, N)
    ca = a |> CuArray
    res = CuArrays.@sync ein"ji,ki,li->jkl"(ca,ca,ca)
    #@show Array(res) - ein"ij,ik,il->jkl"(a,a,a)
end
display(@benchmark bfunc(300) seconds = 1)
