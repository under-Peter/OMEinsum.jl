using Test, OMEinsum, SimpleTraits

@testset "eincode" begin
    @test EinCode(((1,2), (2,3)), (1,3)) isa EinCode{((1,2), (2,3), (1,3))}
end

@testset "trait function" begin
    @test is_pairwise(((1,2),(2,3),(1,3)))
    @test !is_pairwise(((1,2),(1,3), (1,4),(2,3,4)))
end

@testset "trait dispatching" begin
    # for dispatching
    a = randn(3,3)
    @test istrait(IsPairWise{((1,2), (2,3), (1,3))})
    @test einsum!(EinCode(((1,1),), ()), a, fill(0.0, ())) == "tr"
    @test einsum!(EinCode(((1,2), (2,3)), (1,3)), (a,a), randn(3,3)) == "@tensor"
    @test einsum!(EinCode(((1,2), (1,3), (1,4)), (2,3,4)), (a,a,a), randn(3,3,3)) == "general"
end
