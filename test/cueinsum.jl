using Test
using OMEinsum
using CuArrays

CuArrays.allowscalar(false)

@testset "cuda einsum" begin
    a = randn(3, 3) |> CuArray
    @test ein"ij,jk->ik"(a, a) ≈ a*a
    @test ein"ij,jk->ik"(a, a) isa CuArray
end
