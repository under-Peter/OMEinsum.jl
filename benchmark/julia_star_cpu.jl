using OMEinsum

N = 100
ca = randn(Float32, N, N)
ct = randn(Float32, N, N, N)
cq = randn(Float32, N, N, N, N)

function bfunc_star_cpu()
    ein"ji,kl,li->jkl"(ca,ca,ca)
end

function bfunc_t3_cpu()
    ein"ji,kli,li->jkl"(ca,ct,ca)
end

function bfunc_psum_cpu()
    ein"iikl->"(cq)
end

using BenchmarkTools
display(@benchmark bfunc_star_cpu() seconds = 1)
display(@benchmark bfunc_t3_cpu() seconds = 1)
display(@benchmark bfunc_psum_cpu() seconds = 1)
