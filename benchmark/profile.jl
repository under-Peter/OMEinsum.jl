using BenchmarkTools, OMEinsum, CuArrays
using CUDAdrv
CuArrays.allowscalar(false)

# NOTE: to excute this profile, run `nvprof --profile-from-start off julia benchmark/profile.jl`
a = randn(Float32, 100, 100)
a = a |> CuArray
CUDAdrv.synchronize()
CUDAdrv.Profile.start()
y = CuArrays.@sync ein"ij,ik,il->jkl"(a,a,a)
CUDAdrv.Profile.stop()
