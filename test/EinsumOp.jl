using Test, OMEinsum
using OMEinsum: match_rule, PairWise, Sum, Trace

@testset "match rule" begin
    ixs = ((1,2), (2,3))
    iy = (1,3)
    @test match_rule(ixs, iy) == PairWise()

    ixs = ((1,1),)
    iy = ()
    @test match_rule(ixs, iy) == Trace()

    ixs = ((1,2),)
    iy = (1,)
    @test match_rule(ixs, iy) == Sum((2,))
end
