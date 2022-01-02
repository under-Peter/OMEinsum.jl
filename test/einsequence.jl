using Test, OMEinsum
using OMEinsum: IndexGroup, NestedEinsum, parse_nested, DynamicEinCode, isleaf, getixsv, getiyv
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
    size_info = Dict('k'=>2)
    a, b, c, d = randn(2), randn(2,2), randn(2), randn(2)
    @test ein"((i,ij),i),j->ik"(a, b, c, d; size_info=size_info) ≈ ein"i,ij,i,j->ik"(a, b, c, d; size_info=size_info)
    @test getixsv(ein"((i,ij),i),j->ik") == getixsv(ein"i,ij,i,j->ik") == getixsv(DynamicEinCode(ein"i,ij,i,j->ik")) == [['i'], ['i','j'], ['i'], ['j']]
    @test getiyv(ein"((i,ij),i),j->ik") == getiyv(ein"i,ij,i,j->ik") == getiyv(DynamicEinCode(ein"i,ij,i,j->ik")) == ['i','k']
end

@testset "macro" begin
    b, c, d = rand(2,2), rand(2,2,2), rand(2,2,2,2)
    @ein a[i,j] := b[i,k] * c[k,k,l] * d[l,m,m,j]
    @ein a2[i,j] := b[i,k] * (c[k,k,l] * d[l,m,m,j])
    @test a ≈ a2
end

@testset "flatten" begin
    ne = ein"(ij,j),k->"
    @test OMEinsum.flatten(ne) === ein"ij,j,k->"
    todynamic(ne::NestedEinsum) = isleaf(ne) ? NestedEinsum{DynamicEinCode{Char}}(ne.tensorindex) : NestedEinsum(todynamic.(ne.args), DynamicEinCode(ne.eins))
    ne2 = todynamic(ne)
    @test OMEinsum.flatten(ne2) isa DynamicEinCode && OMEinsum.flatten(ne2) == DynamicEinCode(ein"ij,j,k->")
end

@testset "time, space, rw complexity" begin
    ne = ein"(ij,jkc),klc->il"
    tc, sc, rw = timespacereadwrite_complexity(ne, Dict([l=>10 for l in "ijklc"]))
    @test tc ≈ log2(10000+10000)
    @test sc ≈ log2(1000)
    @test rw ≈ log2(100+1000+1000+1000+1000+100)
end
