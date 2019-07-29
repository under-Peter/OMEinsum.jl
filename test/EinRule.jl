using Test, OMEinsum
using OMEinsum: match_rule, PairWise, Sum, Tr, DefaultRule,
                Permutedims, PTrace, Hadamard, MatMul, nopermute,
                Identity

@testset "match rule" begin
    ixs = ((1,2), (2,3))
    iy = (1,3)
    @isdefined(pmobj) && next!(pmobj)
    @test match_rule(ixs, iy) == MatMul()

    ixs = ((1,2), (2,3), (3,4))
    iy = (1,4)
    @isdefined(pmobj) && next!(pmobj)
    @test match_rule(ixs, iy) == PairWise()

    ixs = ((1,1),)
    iy = ()
    @isdefined(pmobj) && next!(pmobj)
    @test match_rule(ixs, iy) == Tr()

    ixs = ((1,1,2),)
    iy = (2,)
    @isdefined(pmobj) && next!(pmobj)
    @test match_rule(ixs, iy) == PTrace()

    ixs = ((1,2),)
    iy = (1,)
    @isdefined(pmobj) && next!(pmobj)
    @test match_rule(ixs, iy) == Sum()

    ixs = ((1,2,3),)
    iy = (2,1)
    @isdefined(pmobj) && next!(pmobj)
    @test match_rule(ixs, iy) != Sum()

    ixs = ((1,2),(1,2),(1,2))
    iy = (1,2)
    @isdefined(pmobj) && next!(pmobj)
    @test match_rule(ixs, iy) == Hadamard()

    ixs = ((1,2,3),)
    iy = (1,3,2)
    @isdefined(pmobj) && next!(pmobj)
    @test match_rule(ixs, iy) == Permutedims()

    ixs = ((1,2),)
    iy = (1,1)
    @isdefined(pmobj) && next!(pmobj)
    @test match_rule(ixs, iy) == DefaultRule()

    ix = (1,2,3)
    iy = (1,2)
    @isdefined(pmobj) && next!(pmobj)
    @test nopermute(ix,iy)

    ix = (1,2,3)
    iy = (2,1)
    @isdefined(pmobj) && next!(pmobj)
    @test !nopermute(ix,iy)

    # 3-argument match_rule
    @isdefined(pmobj) && next!(pmobj)
    @test match_rule(Tr, ((1,1),), ())
    @isdefined(pmobj) && next!(pmobj)
    @test !match_rule(Tr, ((1,2),), ())
    @isdefined(pmobj) && next!(pmobj)
    @test !match_rule(Tr, ((1,1),), (1,))

    @isdefined(pmobj) && next!(pmobj)
    @test match_rule(PairWise, ((1,2),(2,3),(3,4),(4,5)), (1,5))
    @isdefined(pmobj) && next!(pmobj)
    @test match_rule(PairWise, ((1,2),(2,3),(5,4),(3,4)), (1,5))
    @isdefined(pmobj) && next!(pmobj)
    @test !match_rule(PairWise, ((1,2),(2,3),(3,4),(3,5)), (1,5))

    @isdefined(pmobj) && next!(pmobj)
    @test match_rule(Sum, ((1,2,3,4),), (1,2))
    @isdefined(pmobj) && next!(pmobj)
    @test match_rule(Sum, ((1,2,3,4),), (1,2))
    @isdefined(pmobj) && next!(pmobj)
    @test match_rule(Sum, ((1,2,3,4),), (1,2,3,4))
    @isdefined(pmobj) && next!(pmobj)
    @test !match_rule(Sum, ((1,2,3,4),), (2,1))

    @isdefined(pmobj) && next!(pmobj)
    @test match_rule(Permutedims, ((1,2),), (2,1))
    @isdefined(pmobj) && next!(pmobj)
    @test !match_rule(Permutedims, ((1,2,3),), (2,1))
    @isdefined(pmobj) && next!(pmobj)
    @test !match_rule(Permutedims, ((3,3),), (1,3))

    @isdefined(pmobj) && next!(pmobj)
    @test match_rule(Hadamard, ((1,2),(1,2)), (1,2))
    @isdefined(pmobj) && next!(pmobj)
    @test !match_rule(Hadamard, ((1,2),(2,1)), (1,2))
    @isdefined(pmobj) && next!(pmobj)
    @test !match_rule(Hadamard, ((1,2),(1,2)), (1,2,1))

    @isdefined(pmobj) && next!(pmobj)
    @test match_rule(PTrace, ((1,1,2),), (2,))
    @isdefined(pmobj) && next!(pmobj)
    @test match_rule(PTrace, ((1,1,2,3),), (2,3))
    @isdefined(pmobj) && next!(pmobj)
    @test !match_rule(PTrace, ((1,1,2,3),), (3,2))
    @isdefined(pmobj) && next!(pmobj)
    @test !match_rule(PTrace, ((1,1,2),), (1,2,))

    @isdefined(pmobj) && next!(pmobj)
    @test match_rule(MatMul, ((1,2),(2,3)), (1,3))
    @isdefined(pmobj) && next!(pmobj)
    @test !match_rule(MatMul, ((1,2),(2,3)), (3,1))
    @isdefined(pmobj) && next!(pmobj)
    @test !match_rule(MatMul, ((1,2,3),(2,3,4)), (1,4))

    @isdefined(pmobj) && next!(pmobj)
    @test match_rule(Identity, ((1,2,3),), (1,2,3))
    @isdefined(pmobj) && next!(pmobj)
    @test !match_rule(Identity, ((1,2,3),), (1,3,2))

end