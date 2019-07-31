using BenchmarkTools, OMEinsum, CuArrays
using CUDAdrv
CuArrays.allowscalar(false)

# NOTE: to excute this profile, run `nvprof --profile-from-start off julia benchmark/profile.jl`
a = randn(Float32, 100, 100)
ca = a |> CuArray
CUDAdrv.synchronize()
CUDAdrv.Profile.start()
CUDAdrv.@profile begin
    @sync ein"ij,ik,il->jkl"(ca,ca,ca)
    CUDAdrv.synchronize()
end
CUDAdrv.Profile.stop()
