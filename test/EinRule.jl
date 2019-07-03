using Test, OMEinsum
using OMEinsum: match_rule, PairWise, Sum, Tr, DefaultRule,
                Permutedims, PTrace, Hadamard, MatMul, nopermute

@testset "match rule" begin
    ixs = ((1,2), (2,3))
    iy = (1,3)
    @test match_rule(ixs, iy) == MatMul()

    ixs = ((1,2), (2,3), (3,4))
    iy = (1,4)
    @test match_rule(ixs, iy) == PairWise()

    ixs = ((1,1),)
    iy = ()
    @test match_rule(ixs, iy) == Tr()

    ixs = ((1,1,2),)
    iy = (2,)
    @test match_rule(ixs, iy) == PTrace()

    ixs = ((1,2),)
    iy = (1,)
    @test match_rule(ixs, iy) == Sum()

    ixs = ((1,2,3),)
    iy = (2,1)
    @test match_rule(ixs, iy) != Sum()

    ixs = ((1,2),(1,2),(1,2))
    iy = (1,2)
    @test match_rule(ixs, iy) == Hadamard()

    ixs = ((1,2,3),)
    iy = (1,3,2)
    @test match_rule(ixs, iy) == Permutedims()

    ixs = ((1,2),)
    iy = (1,1)
    @test match_rule(ixs, iy) == DefaultRule()

    ix = (1,2,3)
    iy = (1,2)
    @test nopermute(ix,iy)

    ix = (1,2,3)
    iy = (2,1)
    @test !nopermute(ix,iy)
end
