using BenchmarkTools, OMEinsum

a = randn(Float32, 50, 50)
c = zeros(Float32, 50, 50)
y = zeros(Float32, 50, 50, 50)
@benchmark naive_einsum!(((1,2), (2,3)), (a,a), (1,3), c) seconds = 1
@benchmark naive_einsum!(((1,2), (1,3), (1,4)), ($a,$a,$a), (2,3,4), $y) seconds = 1
@benchmark a*a seconds = 1

a = a |> CuArray
c = c |> CuArray
y = y |> CuArray
@benchmark naive_einsum!(((1,2), (2,3)), (a,a), (1,3), c) seconds = 1
@benchmark naive_einsum!(((1,2), (1,3), (1,4)), ($a,$a,$a), (2,3,4), $y) seconds = 1
@benchmark a*a seconds = 1
