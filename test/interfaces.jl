using Test
using OMEinsum
using OMEinsum: get_size_dict

@testset "unspecified index sizes" begin
    v = randn(5)
    a = randn(6,6)
    xs = (v, a)
    ixs = (('j',), ('k', 'k'))
    @test einsum(ein"j,kk->j", xs, get_size_dict(ixs, xs)) ≈ ein"j,kk->j"(xs...)
    res = ein"j,kk->iij"(xs...; size_info=Dict('i'=>9))
    @test size(res) == (9,9,5)
    @test ein"ijk,
    ijk->
    ikl" == ein"ijk,ijk->ikl"
end

@testset "unicode index labels" begin
    # basic Greek letters (within α-ω range)
    ec1 = ein"aζ,ζβ->aβ"
    @test OMEinsum.getixs(ec1) == (('a', 'ζ'), ('ζ', 'β'))
    @test OMEinsum.getiy(ec1) == ('a', 'β')

    # Greek letters outside α-ω range (e.g. ϵ = U+03F5)
    ec2 = ein"abc,ζαfa,ϵβeb,δγdc,def,βγ,ϵζ->δα"
    @test length(OMEinsum.getixs(ec2)) == 7
    @test OMEinsum.getixs(ec2)[3] == ('ϵ', 'β', 'e', 'b')
    @test OMEinsum.getiy(ec2) == ('δ', 'α')

    # correctness: Greek indices contract properly
    A = rand(3, 3)
    B = rand(3, 3)
    @test ein"αβ,βγ->αγ"(A, B) ≈ A * B
    @test ein"αi,iγ->αγ"(A, B) ≈ A * B

    # nested einsum with Unicode
    ec3 = ein"(αβ,βγ),γδ->αδ"
    @test ec3 isa NestedEinsum
    C = rand(3, 3)
    @test ec3(A, B, C) ≈ A * B * C

    # invalid characters should error, not silently truncate
    @test_throws ArgumentError OMEinsum.ein("ab,b1->a1")
end

@testset "opein" begin
    code = optein"ij,jk,ki->"
    @test code isa NestedEinsum
end