using Test
using OMEinsum
using OMEinsum: subindex, dynamic_indexer, DynamicEinCode, StaticEinCode, getixs, getiy, labeltype, einarray

@testset "EinCode" begin
    code = EinCode(((1,2), (2,3)), (1,3))
    @test code isa EinCode
    @test OMEinsum.getixs(code) == [[1,2], [2,3]]
    @test OMEinsum.getiy(code) == [1,3]

    code1 = ein"ab,bc->ac"
    code2 = EinCode((('a', 'b'), ('b', 'c')), ('a', 'c'))
    @test code2 isa DynamicEinCode
    @test code1 isa StaticEinCode
    @test DynamicEinCode(code1) == code2
    @test StaticEinCode(code2) === code1
    @test collect(collect.(getixs(code1))) == getixs(code2)
    @test collect(getiy(code1)) == getiy(code2)
    @test labeltype(code1) == labeltype(code2)

    code1 = ein",->"
    code2 = EinCode(((), ()), ())
    @test collect(collect.(getixs(code1))) == getixs(code2)
    @test collect(getiy(code1)) == getiy(code2)
    @test labeltype(code1) == Char
    @test labeltype(code2) == Union{}

    code1 = ein"->"
    code2 = EinCode(((),), ())
    @test collect(collect.(getixs(code1))) == getixs(code2)
    @test collect(getiy(code1)) == getiy(code2)
    @test labeltype(code1) == Char
    @test labeltype(code2) == Union{}

    @test_throws ErrorException EinCode((), ())
end

@testset "indexer" begin
    si = EinIndexer{()}(())
    @test subindex(si, (1,2,3)) == 1
    si = EinIndexer{(3,2)}((7,6))
    @test OMEinsum.getlocs(si) == (3,2)
    a = randn(7,6)
    @test a[subindex(si, (4,5,2))] == a[2,5]
end

@testset "dynamic indexer" begin
    si = dynamic_indexer((), ())
    @test subindex(si, (1,2,3)) == 1
    si = dynamic_indexer((3,2), (7,6))
    @test OMEinsum.getlocs(si) == (3,2)
    a = randn(7,6)
    @test a[subindex(si, (4,5,2))] == a[2,5]
end

@testset "EinArray" begin
    locs_xs = (EinIndexer{(1,2)}((8,8)), EinIndexer{(2,3)}((8,8)))
    ixs = ((1,2), (2,3))
    iy = (1,3)
    x1 = randn(8, 8)
    x2 = randn(8, 8)
    arr = einarray(Val(ixs), Val(iy), (x1, x2), OMEinsum.get_size_dict!(ixs, (x1, x2), Dict{Int,Int}()))
    @test size(arr) == (8,8,8)
    # inner first, then outer
    @test arr[CartesianIndex((3,4,2))] == x1[4,3]*x2[3,2]
    @test arr[3,4,2] == x1[4,3]*x2[3,2]
end
