using Test, OMEinsum
using OMEinsum: match_rule, PairWise, Sum, Tr, DefaultRule,
                Permutedims, PTrace, Hadamard, MatMul, nopermute,
                Identity, BatchedContract

@testset "match rule" begin
    ixs = ((1,2), (2,3))
    iy = (1,3)
    @test match_rule(ixs, iy) == MatMul()
    @test match_rule(EinCode(ixs, iy)) == MatMul()

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
    iy = (2,1,3)
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

    # 3-argument match_rule
    @test match_rule(Tr, ((1,1),), ())
    @test !match_rule(Tr, ((1,2),), ())
    @test !match_rule(Tr, ((1,1),), (1,))

    @test match_rule(PairWise, ((1,2),(2,3),(3,4),(4,5)), (1,5))
    @test match_rule(PairWise, ((1,2),(2,3),(5,4),(3,4)), (1,5))
    @test !match_rule(PairWise, ((1,2),(2,3),(3,4),(3,5)), (1,5))

    @test match_rule(Sum, ((1,2,3,4),), (1,2))
    @test match_rule(Sum, ((1,2,3,4),), (1,2))
    @test match_rule(Sum, ((1,2,3,4),), (1,2,3,4))
    @test match_rule(Sum, ((1,2,3,4),), (2,1))
    @test !match_rule(Sum, ((1,2,3,4),), (2,1,1))

    @test match_rule(Permutedims, ((1,2),), (2,1))
    @test !match_rule(Permutedims, ((1,2,3),), (2,1))
    @test !match_rule(Permutedims, ((3,3),), (1,3))

    @test match_rule(Hadamard, ((1,2),(1,2)), (1,2))
    @test !match_rule(Hadamard, ((1,2),(2,1)), (1,2))
    @test !match_rule(Hadamard, ((1,2),(1,2)), (1,2,1))

    @test match_rule(PTrace, ((1,1,2),), (2,))
    @test match_rule(PTrace, ((1,1,2,3),), (2,3))
    @test match_rule(PTrace, ((1,1,2,3),), (3,2))
    @test !match_rule(PTrace, ((1,1,2),), (1,2,))

    @test match_rule(MatMul, ((1,2),(2,3)), (1,3))
    @test match_rule(MatMul, ((1,2),(2,3)), (3,1))
    @test match_rule(MatMul, ((2,1),(2,3)), (3,1))
    @test match_rule(MatMul, ((2,1),(3,2)), (3,1))
    @test !match_rule(MatMul, ((1,2,3),(2,3,4)), (1,4))

    @test match_rule(Identity, ((1,2,3),), (1,2,3))
    @test !match_rule(Identity, ((1,2,3),), (1,3,2))

end

@testset "isbatchmul" begin
    @test match_rule(BatchedContract,((1,2), (2,3)), (1,3)) # matmul
    @test match_rule(BatchedContract,((1,2,3), (2,3)), (1,3)) # matvec-batched
    @test match_rule(BatchedContract,((7,1,2,3), (2,4,3,7)), (1,4,3)) # matmat-batched
    @test match_rule(BatchedContract,((3,), (3,)), (3,))  # pure batch
    @test match_rule(BatchedContract,((3,1), (3,)), (3,1))  # batched vector-vector
    @test !match_rule(BatchedContract,((2,2), (2,3)), (1,3))
    @test !match_rule(BatchedContract,((2,1), (2,3)), (1,1))
    @test !match_rule(BatchedContract,((1,2,3), (2,4,3)), (1,3))
end

@testset "match_rule eye candies" begin
    @test match_rule(ein"ij,jk,kl->il") == PairWise()
    @test match_rule(ein"(ij,jk),kl->il") == (OMEinsum.MatMul(), ((OMEinsum.MatMul(), (1, 2)), 3))
end
