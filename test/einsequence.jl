using Test, OMEinsum
using OMEinsum: IndexGroup, NestedEinsum, parse_nested
@testset "einsequence" begin
    @test push!(IndexGroup([],1), 'c').inds == IndexGroup(['c'], 1).inds
    @test isempty(IndexGroup([],1))

    @test_throws ArgumentError parse_nested("((ij,jk),km")
    @test_throws ArgumentError parse_nested("((ij,jk),k1)")

    a, b, c = rand(2,2), rand(2,2), rand(2,2)
    abc1 = ein"(ij,jk),km -> im"(a,b,c)
    abc2 = ein"((ij,jk),km) -> im"(a,b,c)
    abc3 = ein"ij,jk,km -> im"(a,b,c)

    @test abc1 ≈ abc2 ≈ abc3
end
