using Test
using OMEinsum
using OMEinsum: edgesfrominds, operatorfromedge, operatorsfromedges, supportinds
using OMEinsum: TensorContract, Trace, StarContract, MixedStarContract, Diag,
                MixedDiag, IndexReduction, Permutation, OuterProduct

@testset "einorder" begin
    @testset "edges" begin
        @test edgesfrominds(((1,2),(2,3)), (1,3)) == [2]
        @test edgesfrominds(((1,2),(2,3)), (1,)) == [2,3]
        @test edgesfrominds(((1,2),(2,3)), ()) == [1,2,3]

        @test edgesfrominds(((1,2,3),(2,3,4),(4,5,5,6)), (1,6)) == [2,3,4,5]
        @test edgesfrominds(((1,2,3),(2,3,4),(4,5,5,6)), (1,5,6)) == [2,3,4,5]
    end

    @testset "operator from edge" begin
        @test operatorfromedge(2, ((1,2),(2,3)), (1,3)) isa TensorContract
        @test operatorfromedge(2, ((1,2,2),), (1,)) isa Trace
        @test operatorfromedge(1, ((1,2),(1,3),(1,4)), (2,3,4)) isa StarContract
        @test operatorfromedge(1, ((1,2),(1,3),(1,1)), (2,3,4)) isa MixedStarContract
        @test operatorfromedge(1, ((1,2),(1,3)), (1,2,3)) isa Diag
        @test operatorfromedge(1, ((1,2),(1,3),(1,1)), (1,2,3)) isa MixedDiag
        @test operatorfromedge(3, ((1,2),(1,3),(1,1)), (1,2)) isa IndexReduction
    end

    @testset "supportinds" begin
        @test supportinds(1, ((1,2),(1,3),(1,4))) == (true, true, true)
        @test supportinds(2, ((1,2),(1,3),(1,4))) == (true, false, false)
        @test supportinds(3, ((1,2),(1,3),(1,4))) == (false, true, false)
        @test supportinds(4, ((1,2),(1,3),(1,4))) == (false, false, true)
    end

    @testset "operators from edges" begin
        @test operatorsfromedges(((1,2,3),(2,3,4)), [2,3], (1,4)) == (TensorContract((2,3)),)
        @test operatorsfromedges(((1,2,3),(2,3,4)), [2,3], (4,1)) == (TensorContract((2,3)), Permutation((2,1)))
        @test operatorsfromedges(((1,),(2,),(3,)), [], (1,2,3)) == (OuterProduct{3}(),)
        @test operatorsfromedges(((1,1,2),(2,3)), [1,2], (3,)) == (Trace(1), TensorContract(2))
        @test operatorsfromedges(((1,1,2),(2,3)), [2,1], (3,)) == (TensorContract(2), Trace(1))
        @test operatorsfromedges(((1,1,2),(2,3)), [1,2], (1,3)) == (MixedDiag(1), TensorContract(2))
        @test operatorsfromedges(((1,2),(2,3,4),(1,5,6),(3,5,7,8)), [1,2,3,5], (4,6,7,8)) ==
            (TensorContract(1), TensorContract(2), TensorContract((3,5)), Permutation((2,1,3,4)))
        @test operatorsfromedges(((1,2,3),), [], (1,3,2)) == (Permutation((1,3,2)),)
    end
end
