using CUDAnative: device!
device!(0)
using BenchmarkTools, OMEinsum, CuArrays
CuArrays.allowscalar(false)

function bfunc_star(N::Int)
    ca = CuArrays.curandn(Float32, N, N)
    CuArrays.@sync ein"ji,kl,li->jkl"(ca,ca,ca)
end

function bfunc_t3(N::Int)
    ca = CuArrays.curandn(Float32, N, N)
    ct = CuArrays.curandn(Float32, N, N, N)
    CuArrays.@sync ein"ji,kli,li->jkl"(ca,ct,ca)
end

function bfunc_psum(N::Int)
    cq = CuArrays.curandn(Float32, N, N, N, N)
    CuArrays.@sync ein"iikl->"(cq)
end

display(@benchmark bfunc_star(300) seconds = 1)
display(@benchmark bfunc_t3(300) seconds = 1)
display(@benchmark bfunc_psum(100) seconds = 1)
include("reduce_einsum.jl")
display(@benchmark bfunc_star(300) seconds = 1)
display(@benchmark bfunc_t3(300) seconds = 1)
display(@benchmark bfunc_psum(100) seconds = 1)
