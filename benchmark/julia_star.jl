using CUDA: device!
device!(0)
using BenchmarkTools, OMEinsum, CUDA
CUDA.allowscalar(false)

function bfunc_star(N::Int)
    ca = CUDA.curandn(Float32, N, N)
    res = CUDA.@sync ein"ji,kl,li->jkl"(ca,ca,ca)
    #res ≈ ein"ji,kl,li->jkl"(Matrix(ca),Matrix(ca), Matrix(ca))
end

function bfunc_t3(N::Int)
    ca = CUDA.curandn(Float32, N, N)
    ct = CUDA.curandn(Float32, N, N, N)
    CUDA.@sync ein"ji,kli,li->jkl"(ca,ct,ca)
end

function bfunc_psum(N::Int)
    cq = CUDA.curandn(Float32, N, N, N, N)
    CUDA.@sync ein"iikl->"(cq)
end

display(@benchmark bfunc_star(300) seconds = 1)
display(@benchmark bfunc_t3(300) seconds = 1)
display(@benchmark bfunc_psum(100) seconds = 1)
