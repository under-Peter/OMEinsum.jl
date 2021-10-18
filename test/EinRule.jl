using Test, OMEinsum
using OMEinsum: match_rule, Sum, Tr, DefaultRule, Diag, Duplicate,
                Permutedims, nopermute,
                Identity, SimpleBinaryRule

@testset "match rule" begin
    ixs = ((1,2), (2,3))
    iy = (1,3)
    @test match_rule(ixs, iy) == SimpleBinaryRule(ein"ij,jk->ik")
    @test match_rule(EinCode(ixs, iy)) == SimpleBinaryRule(ein"ij,jk->ik")

    ixs = ((1,2), (2,3), (3,4))
    iy = (1,4)
    @test match_rule(ixs, iy) == DefaultRule()

    ixs = ((1,1),)
    iy = ()
    @test match_rule(ixs, iy) == Tr()

    ixs = ((1,1,2),)
    iy = (2,)
    @test match_rule(ixs, iy) == DefaultRule()

    ixs = ((1,2),)
    iy = (1,)
    @test match_rule(ixs, iy) == Sum()

    ixs = ((1,2,3),)
    iy = (2,1,3)
    @test match_rule(ixs, iy) != Sum()

    ixs = ((1,2),(1,2),(1,2))
    iy = (1,2)
    @test match_rule(ixs, iy) == DefaultRule()

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

    # 3-argument match_rule
    @test match_rule(((1,1),), ()) == Tr()
    @test match_rule(((1,2),), ()) == Sum()
    @test match_rule(((1,1),), (1,)) == Diag()

    @test match_rule(((1,2),(2,3),(3,4),(4,5)), (1,5)) == DefaultRule()
    @test match_rule(((1,2),(2,3),(5,4),(3,4)), (1,5)) == DefaultRule()
    @test match_rule(((1,2),(2,3),(3,4),(3,5)), (1,5)) == DefaultRule()

    @test match_rule(((1,2,3,4),), (1,2)) == Sum()
    @test match_rule(((1,2,3,4),), (1,2)) == Sum()
    @test match_rule(((1,2,3,4),), (1,2,3,4)) == Identity()
    @test match_rule(((1,2,3,4),), (2,1)) == Sum()
    @test match_rule(((1,2,3,4),), (2,1,1)) == DefaultRule()

    @test match_rule(((1,2),), (2,1)) == Permutedims()
    @test match_rule(((1,2,3),), (2,1)) == Sum()
    @test match_rule(((3,3),), (1,3)) == DefaultRule()

    @test match_rule(((1,2),(1,2)), (1,2)) == DefaultRule()
    @test match_rule(((1,2),(2,1)), (1,2)) == DefaultRule()
    @test match_rule(((1,2),(1,2)), (1,2,1)) == DefaultRule()

    @test match_rule(((1,1,2),), (2,)) == DefaultRule()
    @test match_rule(((1,1,2,3),), (2,3)) == DefaultRule()
    @test match_rule(((1,1,2,3),), (3,2)) == DefaultRule()
    @test match_rule(((1,1,2),), (1,2,)) == Diag()

    @test match_rule(((1,2),(2,3)), (1,3)) == SimpleBinaryRule(ein"ij,jk->ik")
    @test match_rule(((1,2),(2,3)), (3,1)) == SimpleBinaryRule(ein"ij,jk->ki")
    @test match_rule(((2,1),(2,3)), (3,1)) == SimpleBinaryRule(ein"ji,jk->ki")
    @test match_rule(((2,1),(3,2)), (3,1)) == SimpleBinaryRule(ein"ji,kj->ki")
    @test match_rule(((1,2,3),(2,3,4)), (1,4)) == DefaultRule()

    @test match_rule(((1,2,3),), (1,2,3)) == Identity()
    @test match_rule(((1,2,3),), (1,3,2)) == Permutedims()

    @test match_rule(((1,2,1),), (1,2)) == Diag()
    @test match_rule(((1,2),), (1,2,1)) == Duplicate()
end

@testset "isbatchmul" begin
    @test match_rule(((1,2), (2,3)), (1,3)) == SimpleBinaryRule(ein"ij,jk->ik") # matmul
    @test match_rule(((1,2,3), (2,3)), (1,3)) == SimpleBinaryRule(ein"ijl,jl->il") # matvec-batched
    @test match_rule(((7,1,2,3), (2,4,3,7)), (1,4,3)) == DefaultRule() # matmat-batched
    @test match_rule(((3,), (3,)), (3,)) == SimpleBinaryRule(ein"l,l->l")  # pure batch
    @test match_rule(((3,1), (3,)), (3,1)) == DefaultRule()  # batched vector-vector
    @test match_rule(((2,2), (2,3)), (1,3)) == DefaultRule()
    @test match_rule(((2,1), (2,3)), (1,1)) == DefaultRule()
    @test match_rule(((1,2,3), (2,4,3)), (1,3)) == DefaultRule()
end

@testset "match_rule eye candies" begin
    @test match_rule(ein"ij,jk,kl->il") == DefaultRule()
end
