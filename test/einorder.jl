using Test
using OMEinsum
using OMEinsum: edgesfrominds, operatorfromedge
using OMEinsum: TensorContract, Trace, StarContract, MixedStarContract, Diag,
                MixedDiag, IndexReduction

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

end
