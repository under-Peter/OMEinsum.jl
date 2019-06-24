using Test
using OMEinsum
using CuArrays

CuArrays.allowscalar(false)

@testset "cuda einsum" begin
    a = randn(3, 3) |> CuArray
    @test einsumexp!(EinCode(((1,2), (2,3)), (1,3)), (a, a), zeros(3,3) |> CuArray) ≈ a*a
end
