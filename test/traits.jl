using Test, OMEinsum, SimpleTraits
using OMEinsum: einmagic!, einsum!

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
    args_list = [(((1,1),), (a,), (), fill(0.0, ())),
        (((1,2), (2,3)), (a,a), (1,3), randn(3,3)),
        (((1,2), (1,3), (1,4)), (a,a,a), (2,3,4), randn(3,3,3))
    ]
    for args in args_list
        @test einmagic!(args...) ≈ einsum!(args...)
    end
end
