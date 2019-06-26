using Test, OMEinsum
using OMEinsum: match_rule, PairWise, Sum, Tr, DefaultRule, Permutedims

@testset "match rule" begin
    ixs = ((1,2), (2,3))
    iy = (1,3)
    @test match_rule(ixs, iy) == PairWise()

    ixs = ((1,1),)
    iy = ()
    @test match_rule(ixs, iy) == Tr()

    ixs = ((1,2),)
    iy = (1,)
    @test match_rule(ixs, iy) == Sum()

    ixs = ((1,2),)
    iy = (1,1)
    @test match_rule(ixs, iy) == DefaultRule()
end
