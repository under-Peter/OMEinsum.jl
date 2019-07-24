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

@testset "macro" begin
    b, c, d = rand(2,2), rand(2,2,2), rand(2,2,2,2)
    @ein a[i,j] := b[i,k] * c[k,k,l] * d[l,m,m,j]
    @ein a2[i,j] := b[i,k] * (c[k,k,l] * d[l,m,m,j])
    @test a ≈ a2
end
