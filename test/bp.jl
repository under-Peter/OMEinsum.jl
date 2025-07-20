using OMEinsum, Test, Zygote

@testset "bp check" begin
    A, B, C = randn(2, 3), randn(3, 4), randn(4, 2)
    cost0 = ein"(ij, jk), ki->"(A, B, C)[]
    zg = Zygote.gradient((a, b, c)->ein"(ij, jk), ki->"(a, b, c)[], A, B, C)
    cost, mg = OMEinsum.cost_and_gradient(ein"(ij, jk), ki->", (A, B, C))
    @test cost[] ≈ cost0
    @test all(zg .≈ mg)

    code = OMEinsum.optimize_code(ein"ij, jk, ki->", uniformsize(ein"ij, jk, ki->", 2), TreeSA())
    cost0 = code(A, B, C)[]
    zg = Zygote.gradient((a, b, c)->code(a, b, c)[], A, B, C)
    cost, mg = OMEinsum.cost_and_gradient(code, (A, B, C))
    @test cost[] ≈ cost0
    @test all(zg .≈ mg)
end