using CUDAnative: device!
device!(0)
using BenchmarkTools, OMEinsum, CuArrays
CuArrays.allowscalar(false)

function bfunc(N::Int)
    a = randn(Float32, N, N)
    ca = a |> CuArray
    res = CuArrays.@sync ein"ji,ki,li->jkl"(ca,ca,ca)
end
@benchmark bfunc(100) seconds = 1
