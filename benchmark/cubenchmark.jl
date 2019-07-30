using BenchmarkTools, OMEinsum, CuArrays
CuArrays.allowscalar(false)

a = randn(Float32, 100, 100)
@benchmark ein"ij,jk->ik"($a,$a) seconds = 1
@benchmark ein"ij,ik,il->jkl"($a,$a,$a) seconds = 1
@benchmark a*a seconds = 1

a = a |> CuArray
@benchmark (CuArrays.@sync ein"ij,jk->ik"($a,$a)) seconds = 1
@benchmark (CuArrays.@sync ein"ij,ik,il->jkl"($a,$a,$a)) seconds = 1
@benchmark (CuArrays.@sync $a*$a) seconds = 1
