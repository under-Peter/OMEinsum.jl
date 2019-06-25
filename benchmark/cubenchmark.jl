using BenchmarkTools, OMEinsum, CuArrays
CuArrays.allowscalar(false)

a = randn(Float32, 50, 50)
@benchmark ein"ij,jk->ik"($a,$a) seconds = 1
@benchmark ein"ij,ik,il->jkl"($a,$a,$a) seconds = 1
@benchmark a*a seconds = 1

a = a |> CuArray
y = y |> CuArray
@benchmark ein"ij,jk->ik"($a,$a) seconds = 1
@benchmark ein"ij,ik,il->jkl"($a,$a,$a) seconds = 1
@benchmark a*a seconds = 1
