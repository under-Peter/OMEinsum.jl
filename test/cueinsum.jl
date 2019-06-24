using Test
using OMEinsum: einsumexp!
using CuArrays

CuArrays.allowscalar(false)

@testset "cuda einsum" begin
    a = randn(3, 3) |> CuArray
    @test einsumexp!(((1,2), (2,3)), (a, a), (1,3), zeros(3,3) |> CuArray) ≈ a*a
end
