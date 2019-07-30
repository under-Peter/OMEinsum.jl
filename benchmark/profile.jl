using BenchmarkTools, OMEinsum, CuArrays
using CUDAdrv
CuArrays.allowscalar(false)

a = randn(Float32, 100, 100)
a = a |> CuArray
CUDAdrv.Profile.start()
y = CuArrays.@sync ein"ij,ik,il->jkl"(a,a,a)
CUDAdrv.Profile.stop()
z = ein"ij,jk->ik"(a,a)
print(z |> typeof)
